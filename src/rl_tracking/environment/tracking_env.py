import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete
from rl_tracking.physics.track import Helix
from rl_tracking.physics.hit_holder import HitHolder
from rl_tracking.utils.logger import get_logger

logger = get_logger()

COLS = ['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id',
       'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight', 'vx',
       'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits', 'r', 'pt',
        'unique_layer_id', 'phi', 'theta']


class TrackingEnv:
    def __init__(
        self, 
        data_loader,
        particle_filters=None,  # Dict of particle-level filters
        hit_filters=None,  # Dict of hit-level filters
        use_distance_reward=False,  # Use distance-based reward instead of binary
        use_truth_path_plan=False,  # Use truth-level path plan (all layers in track) instead of lookup
        max_layer_skip=4,  # Max layers to skip ahead when looking for compatible hits
        tolerance_r=2.0,  # Distance tolerance in cm for finding compatible hits
        deterministic=False,  # If True, use deterministic particle ordering (for evaluation)
    ):
        """
        Initialize the environment with a data loader.
        
        Args:
            data_loader: DataLoader that yields event data
            particle_filters: Dict with particle filtering criteria (e.g., {'pt': 2, 'nhits_min': 3, 'nhits_max': 20})
            hit_filters: Dict with hit filtering criteria (e.g., {'exclude_volumes': [7,8,9]})
                         By default, applies no filtering to hits.
            max_layer_skip: Maximum number of layers to skip ahead when looking for compatible hits.
                           If no compatible hits are found in the next max_layer_skip layers,
                           propagation will fail. Default is 4. Increase to be more persistent.
            tolerance_r: Distance tolerance in cm for finding compatible hits.
                       Hits within this 3D distance from the propagated position are considered compatible.
                       Default is 1.0 cm. Increase to find more hits (but may include more false positives).
        """
        self.data_loader = iter(data_loader)  # Streaming file list
        self.all_hits = None  # Stores ALL hits from the current file (for hit finding)
        self.pids_to_explore = []  # Tracks remaining unique values for steps
        self.particle_index = 0  # Tracks current step within the file
        
        # Store filter configurations
        self.particle_filters = particle_filters or {}
        self.hit_filters = hit_filters or {}
        self.use_distance_reward = use_distance_reward
        self.use_truth_path_plan = use_truth_path_plan
        self.max_layer_skip = max_layer_skip
        self.tolerance_r = tolerance_r
        self.deterministic = deterministic  # For deterministic evaluation
        
        # Track distances for statistics
        self.distance_history = []
        self.distance_sum = 0.0
        self.distance_count = 0
        
        self.load_next_file()  # Load the first file
        self.observation_space = Box(low=0, high=120, shape=(4,), dtype=np.float32)  # [z, r, x, y]
        self.action_space = Discrete(3)  # Actions: 0=best hit, 1=second best, 2=third best
        self.hit_number = 0
        
        # Store current comp_hits and rewards for action selection
        self.current_comp_hits = None
        self.current_rewards = None
        self.current_info = {}
        
        # Track accuracy metrics
        self.total_selections = 0
        self.correct_selections = 0
        self.correct_hit_ids_set = set()  # For quick lookup
        
        # Store last selected hit for evaluation tracking
        self.last_selected_hit = None
        self.last_selected_hit_id = None
        self.last_selected_hit_correct = False
        self.last_selected_hit_layer = None
        self.last_selected_hit_correct_hits_df = None

    def _apply_particle_selections(self):
        """Select particles based on criteria, keeping them separate from hit filtering."""
        hits = self.all_hits
        
        # Apply particle-level filters
        if self.particle_filters.get('pt', None) is not None:
            hits = hits[hits['pt'] > self.particle_filters['pt']]
            
        if self.particle_filters.get('nhits_min', None) is not None:
            hits = hits[hits['nhits'] >= self.particle_filters['nhits_min']]
            
        if self.particle_filters.get('nhits_max', None) is not None:
            hits = hits[hits['nhits'] <= self.particle_filters['nhits_max']]
        
        # Filter particles to only those with at least 3 unique layers (required for seeding)
        if 'unique_layer_id' in hits.columns:
            particle_unique_layers = hits.groupby('particle_id')['unique_layer_id'].nunique()
            particles_with_3plus_layers = particle_unique_layers[particle_unique_layers >= 3].index
            hits = hits[hits['particle_id'].isin(particles_with_3plus_layers)]
            logger.debug(f"Filtered to {len(particles_with_3plus_layers)} particles with >= 3 unique layers (from {len(particle_unique_layers)} total)")
        
        # Get unique particles that pass the selection
        selected_particles = hits['particle_id'].unique()
        
        self.pids_to_explore = list(selected_particles)
        # CRITICAL: For deterministic evaluation, sort instead of shuffle
        # Shuffling causes different results on each evaluation run
        # For training, shuffling is fine (adds randomness), but for evaluation we need determinism
        if self.deterministic:
            self.pids_to_explore.sort()  # Deterministic sorting for evaluation
        else:
            np.random.shuffle(self.pids_to_explore)  # Random shuffle for training
    
    def _apply_hit_filters(self, hits):
        """
        Apply hit-level filters to the hits DataFrame.
        By default, applies no filtering.
        
        Args:
            hits: DataFrame of hits
            
        Returns:
            Filtered DataFrame of hits
        """
        filtered_hits = hits.copy()
        
        # Apply hit-level filters
        if self.hit_filters.get('exclude_volumes', None) is not None:
            filtered_hits = filtered_hits[~filtered_hits['volume_id'].isin(self.hit_filters['exclude_volumes'])]
        
        if self.hit_filters.get('exclude_layers', None) is not None:
            filtered_hits = filtered_hits[~filtered_hits['unique_layer_id'].isin(self.hit_filters['exclude_layers'])]
        
        # You can add more hit-level filters here as needed
        
        return filtered_hits

    def load_next_file(self):
        """Load the next file and compute the number of steps from the column."""
        try:
            data = next(self.data_loader)  # Get new file's data
            hits = pd.DataFrame(
                data.view(data.shape[1], data.shape[2]),
                columns=COLS)
            
            logger.info(f"Loaded event with {len(hits)} raw hits")
            logger.debug(f"Unique layer IDs: {hits['unique_layer_id'].unique()}")
            
            # Apply hit-level filters (if any)
            #self.all_hits = self._apply_hit_filters(hits)
            self.all_hits = hits
            logger.debug(f"After hit filters: {len(self.all_hits)} hits")
            
            # Apply particle-level selections (for which tracks to reconstruct)
            self._apply_particle_selections()
            
            logger.info(f"Selected {len(self.pids_to_explore)} particles to explore")
            self.particle_index = 0  # Reset index
        except StopIteration:
            self.all_hits = None
            self.pids_to_explore = []  # End of stream

    def reset(self):
        """Reset environment to the start of the current file."""
        if len(self.pids_to_explore) == 0 or self.particle_index >= len(self.pids_to_explore):
            # If the file is exhausted, load a new one
            self.load_next_file()
            self.particle_index = 0
        if self.all_hits is None:
            self.current_comp_hits = None
            self.current_rewards = None
            return None, {}  # No more data

        #if self.hit_number > len(self.current_track):
        self.get_new_track()
        self.hit_number = 0
        # Reset comp_hits storage
        self.current_comp_hits = None
        self.current_rewards = None
        # Reset accuracy tracking for new episode
        episode_accuracy = self.correct_selections / self.total_selections if self.total_selections > 0 else 0.0
        if self.total_selections > 0:
            logger.info(f"Episode accuracy: {episode_accuracy:.3f} ({self.correct_selections}/{self.total_selections})")
        self.total_selections = 0
        self.correct_selections = 0
        # Clear selected hit tracking
        self.last_selected_hit = None
        self.last_selected_hit_id = None
        self.last_selected_hit_correct = False
        self.last_selected_hit_layer = None
        self.last_selected_hit_correct_hits_df = None
        return self.get_current_state()

    def get_new_track(self):
        # Get the current particle's hits from all_hits
        current_pid = self.pids_to_explore[self.particle_index]
        self.current_track = self.all_hits[self.all_hits['particle_id'] == current_pid].copy()
        self.particle_index += 1
        logger.info(f"Loading track for particle {current_pid} with {len(self.current_track)} hits")
        self.helix = Helix(self.current_track, use_truth_path_plan=self.use_truth_path_plan)
        logger.debug(f"Helix start position: x={self.helix.x:.2f}, y={self.helix.y:.2f}, z={self.helix.z:.2f}")

    def get_current_state(self):
        """Return the current step's data."""
        # Check if path_plan is complete (more reliable than hit_number vs track length)
        # The path_plan determines which layers we should explore
        if self.helix is None or len(self.current_track) == 0:
            return None, {}
        
        # Check if we've exhausted the path plan
        unexplored_layers = self.helix.path_plan[self.helix.path_plan > self.helix.current_layer]
        if len(unexplored_layers) == 0:
            # No more layers in path plan - track is complete
            return None, {}
        
        # For backward compatibility, still check hit_number (but path_plan is primary)
        if self.hit_number >= len(self.current_track):
            return None, {}
        
        current_pos = self.current_track.iloc[self.hit_number][['z', 'r']] if self.hit_number < len(self.current_track) else self.current_track.iloc[-1][['z', 'r']]
        # Wrap the DataFrame in HitHolder - use ALL hits for hit finding
        hit_holder = HitHolder(self.all_hits)
        comp_hits, rewards, done, correct_in_comp, correct_is_best = (
            self.helix.propagate_one_layer(hit_holder, use_distance_reward=self.use_distance_reward, 
                                         max_layer_skip=self.max_layer_skip, tolerance_r=self.tolerance_r))
        
        # Store comp_hits and rewards for action selection in step()
        self.current_comp_hits = comp_hits
        self.current_rewards = rewards
        
        # Track distances if using distance-based reward
        if self.use_distance_reward and len(comp_hits) > 0:
            correct_hits_df = self.helix.correct_hits_df_in_layer
            for i in range(len(comp_hits)):
                hit = comp_hits.iloc[i]
                distance = hit_holder.get_distance_to_correct(hit, correct_hits_df)
                if distance != np.inf:
                    self.distance_history.append(distance)
                    self.distance_sum += distance
                    self.distance_count += 1
                    
                    # Print average distance periodically (every 100 measurements)
                    if self.distance_count % 100 == 0:
                        avg_distance = self.distance_sum / self.distance_count
                        logger.info(f"Average distance to correct hits: {avg_distance:.3f} cm (over {self.distance_count} measurements)")
        
        # Return state (helix position) and hit features for pointer network
        # State: current helix position [z, r, x, y]
        if len(comp_hits) > 0:
            # Use helix's current position (not the first hit)
            obs = np.array([
                self.helix.z,
                self.helix.r0,  # Current r position
                self.helix.x,
                self.helix.y
            ], dtype=np.float32)
        else:
            # Return current position if no compatible hits
            obs = np.array([current_pos['z'], current_pos['r'], 0, 0], dtype=np.float32)
        
        # Extract hit features for all candidate hits: [num_hits, hit_dim]
        # Use features: [x, y, z, r] for each hit
        max_hits = 20  # Maximum number of hits to consider (pad if fewer)
        hit_dim = 4  # x, y, z, r
        
        if len(comp_hits) > 0:
            # Always truncate to max_hits to ensure consistent shapes
            num_hits_to_use = min(len(comp_hits), max_hits)
            hit_features_list = []
            
            for idx in range(num_hits_to_use):
                hit = comp_hits.iloc[idx]
                hit_feat = np.array([
                    float(hit.get('x', 0.0)),
                    float(hit.get('y', 0.0)),
                    float(hit.get('z', 0.0)),
                    float(hit.get('r', 0.0))
                ], dtype=np.float32)
                
                # Check for NaN/inf and replace with 0
                if np.isnan(hit_feat).any() or np.isinf(hit_feat).any():
                    logger.warning(f"Found NaN/Inf in hit features at idx {idx}, replacing with 0")
                    hit_feat = np.nan_to_num(hit_feat, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Clamp to reasonable values to prevent extreme features
                hit_feat = np.clip(hit_feat, -1e4, 1e4)
                
                hit_features_list.append(hit_feat)
            
            # Always pad to exactly max_hits to ensure consistent shapes
            # Truncate first if we have more than max_hits
            hit_features_list = hit_features_list[:max_hits]
            num_hits_to_use = min(num_hits_to_use, max_hits)
            
            # Then pad if we have fewer than max_hits
            while len(hit_features_list) < max_hits:
                hit_features_list.append(np.zeros(hit_dim, dtype=np.float32))
            
            # Final truncation to ensure exactly max_hits
            hit_features_list = hit_features_list[:max_hits]
            hit_features = np.array(hit_features_list, dtype=np.float32)  # [max_hits, hit_dim]
            
            # Create mask: True for actual hits, False for padding
            hit_mask = np.array([True] * num_hits_to_use + [False] * (max_hits - num_hits_to_use), dtype=bool)
            
            # Ensure mask has exactly max_hits elements
            assert len(hit_mask) == max_hits, f"Mask length {len(hit_mask)} != max_hits {max_hits}"
            assert hit_features.shape == (max_hits, hit_dim), f"Hit features shape {hit_features.shape} != ({max_hits}, {hit_dim})"
        else:
            # No hits available
            hit_features = np.zeros((max_hits, hit_dim), dtype=np.float32)
            hit_mask = np.zeros(max_hits, dtype=bool)
        
        # Store correct hit IDs for accuracy tracking
        correct_hit_ids_set = set()
        if hasattr(self.helix, 'correct_hit_ids_in_layer'):
            correct_ids = self.helix.correct_hit_ids_in_layer
            if isinstance(correct_ids, set):
                correct_hit_ids_set = correct_ids
            elif hasattr(correct_ids, '__iter__'):
                correct_hit_ids_set = set(correct_ids)
            else:
                correct_hit_ids_set = {correct_ids} if correct_ids is not None else set()
        
        info = {
            'rewards': rewards,
            'correct_in_comp': correct_in_comp,
            'correct_is_best': correct_is_best,
            'num_comp_hits': len(comp_hits),
            'correct_hit_ids': correct_hit_ids_set,
            'is_correct': [int(comp_hits.iloc[i]['hit_id']) in correct_hit_ids_set 
                          for i in range(len(comp_hits))] if len(comp_hits) > 0 else [],
            'hit_features': hit_features,  # [max_hits, hit_dim]
            'hit_mask': hit_mask,  # [max_hits] - True for valid hits
        }
        self.current_info = info
        
        return obs, info

    def step(self, action):
        """Take an action and move to the next step.
        
        Args:
            action: Hit index (0=first hit, 1=second hit, etc.) - selected by pointer network
        
        Returns:
            next_state, reward, done
        """
        # Select hit based on action from PREVIOUS state's comp_hits
        reward = 0.0
        selected_hit = None
        is_correct_selection = False
        
        if self.current_comp_hits is not None and len(self.current_comp_hits) > 0:
            # Action is the index of the selected hit (from pointer network scoring)
            action_idx = min(action, len(self.current_comp_hits) - 1)
            if action_idx < 0:
                action_idx = 0
            selected_hit = self.current_comp_hits.iloc[action_idx]
            
            # Get reward for selected hit (from previous state)
            if isinstance(self.current_rewards, list) and len(self.current_rewards) > action_idx:
                reward = float(self.current_rewards[action_idx])
            elif isinstance(self.current_rewards, (int, float)):
                reward = float(self.current_rewards)
            else:
                reward = 0.0
            
            # Track if selected hit is correct (for accuracy metrics)
            selected_hit_id = int(selected_hit['hit_id'])
            
            # Get seed hit IDs to exclude from accuracy tracking (matches evaluation logic)
            seed_hit_ids = set()
            if hasattr(self, 'helix') and self.helix is not None:
                if hasattr(self.helix, 'seed_hits') and self.helix.seed_hits is not None:
                    seed_df = self.helix.seed_hits
                    if isinstance(seed_df, pd.DataFrame) and 'hit_id' in seed_df.columns:
                        seed_hit_ids = set(seed_df['hit_id'].values.astype(int))
                    elif isinstance(seed_df, pd.DataFrame) and 'hit_id' in seed_df.index:
                        seed_hit_ids = set(seed_df.index.values.astype(int))
            
            # Get correct hit IDs from the layer where this hit was selected
            if hasattr(self, 'current_info') and 'correct_hit_ids' in self.current_info:
                correct_ids = self.current_info['correct_hit_ids']
            else:
                # Fallback: use helix's correct hit IDs if available
                correct_ids = getattr(self.helix, 'correct_hit_ids_in_layer', set())
            
            is_correct_selection = selected_hit_id in correct_ids if isinstance(correct_ids, set) else selected_hit_id in list(correct_ids)
            
            # Update accuracy tracking - EXCLUDE seed hits to match evaluation
            # This ensures training accuracy matches evaluation accuracy
            if selected_hit_id not in seed_hit_ids:
                self.total_selections += 1
                if is_correct_selection:
                    self.correct_selections += 1
            
            # Store selected hit for evaluation tracking
            self.last_selected_hit = selected_hit
            self.last_selected_hit_id = selected_hit_id
            self.last_selected_hit_correct = is_correct_selection
            # Store the layer where this hit was selected (for distance calculation)
            if hasattr(self, 'helix') and self.helix is not None:
                # Store the layer from the selected hit itself
                if isinstance(selected_hit, pd.Series) and 'unique_layer_id' in selected_hit.index:
                    self.last_selected_hit_layer = float(selected_hit['unique_layer_id'])
                elif hasattr(selected_hit, 'unique_layer_id'):
                    self.last_selected_hit_layer = float(selected_hit.unique_layer_id)
                else:
                    # Fallback: use current layer (but this might be wrong if we've advanced)
                    self.last_selected_hit_layer = getattr(self.helix, 'current_layer', None)
                # Store the correct hits for this layer (before we advance)
                self.last_selected_hit_correct_hits_df = getattr(self.helix, 'correct_hits_df_in_layer', None)
            else:
                self.last_selected_hit_layer = None
                self.last_selected_hit_correct_hits_df = None
            
            # Update helix position to selected hit (this is critical for learning!)
            self.helix.x = float(selected_hit['x'])
            self.helix.y = float(selected_hit['y'])
            self.helix.z = float(selected_hit['z'])
            
            # Track distance for selected hit
            if self.use_distance_reward:
                correct_hits_df = self.helix.correct_hits_df_in_layer
                distance = HitHolder(self.all_hits).get_distance_to_correct(selected_hit, correct_hits_df)
                if distance != np.inf:
                    self.distance_history.append(distance)
                    self.distance_sum += distance
                    self.distance_count += 1
        else:
            # No compatible hits available from previous state
            reward = 0.0
            # Clear selected hit tracking
            self.last_selected_hit = None
            self.last_selected_hit_id = None
            self.last_selected_hit_correct = False
            self.last_selected_hit_layer = None
            self.last_selected_hit_correct_hits_df = None
        
        # Move to next step (increment hit_number to progress through layers)
        # Note: hit_number tracks which layer/step we're on, not necessarily how many hits were selected
        self.hit_number += 1
        
        # Check if we've finished this track
        # Primary check: path_plan is complete (more accurate than hit_number vs track length)
        if self.helix is not None:
            unexplored_layers = self.helix.path_plan[self.helix.path_plan > self.helix.current_layer]
            if len(unexplored_layers) == 0:
                # Path plan exhausted - track is complete
                done = True
                next_state = None
                self.current_comp_hits = None
                self.current_rewards = None
                return next_state, reward, done
        
        # Secondary check: hit_number exceeds track length (backward compatibility)
        if self.hit_number >= len(self.current_track):
            done = True
            next_state = None
            self.current_comp_hits = None
            self.current_rewards = None
            return next_state, reward, done
        
        # Get next state (this will compute new comp_hits for the next step)
        obs, info = self.get_current_state()
        
        # Handle reward from info dict if we didn't get it from action selection
        if obs is None:
            reward = 0.0
            done = True
            self.current_comp_hits = None
            self.current_rewards = None
        else:
            done = False

        return obs, reward, done

    def calculate_reward(self, state):
        """Placeholder for reward logic (adjust based on your task)."""
        return 1 if state is not None else 0  # Example: reward for valid step
