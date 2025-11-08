"""Multi-agent tracking environment for simultaneous track propagation.

This environment manages multiple agents that propagate nearby tracks in lockstep,
sharing a compatible hit pool. This helps with hit assignment when multiple tracks
have overlapping candidate hits.
"""
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


class MultiAgentTrackingEnv:
    """Environment for multiple agents propagating tracks simultaneously.
    
    Key features:
    - Multiple agents propagate different tracks in lockstep
    - Agents share a compatible hit pool (hits are removed once selected)
    - Agents can observe each other's states
    - Coordination through shared observation space
    """
    
    def __init__(
        self, 
        data_loader,
        num_agents=3,  # Number of simultaneous tracks/agents
        particle_filters=None,
        hit_filters=None,
        use_distance_reward=False,
        use_truth_path_plan=False,
        max_layer_skip=4,
        tolerance_r=2.0,
        deterministic=False,
        proximity_threshold=50.0,  # Group tracks within this distance (cm)
    ):
        """
        Initialize multi-agent environment.
        
        Args:
            data_loader: DataLoader that yields event data
            num_agents: Number of agents (tracks) to propagate simultaneously
            particle_filters: Dict with particle filtering criteria
            hit_filters: Dict with hit filtering criteria
            use_distance_reward: Use distance-based reward
            use_truth_path_plan: Use truth-level path plan
            max_layer_skip: Max layers to skip ahead
            tolerance_r: Distance tolerance for finding compatible hits
            deterministic: Deterministic particle ordering for evaluation
            proximity_threshold: Group tracks within this distance (cm)
        """
        self.data_loader = iter(data_loader)
        self.num_agents = num_agents
        self.all_hits = None
        self.pids_to_explore = []
        self.particle_index = 0
        
        # Store filter configurations
        self.particle_filters = particle_filters or {}
        self.hit_filters = hit_filters or {}
        self.use_distance_reward = use_distance_reward
        self.use_truth_path_plan = use_truth_path_plan
        self.max_layer_skip = max_layer_skip
        self.tolerance_r = tolerance_r
        self.deterministic = deterministic
        self.proximity_threshold = proximity_threshold
        
        # Multi-agent state
        self.current_tracks = []  # List of tracks (one per agent)
        self.helices = []  # List of helices (one per agent)
        self.agent_active = []  # Which agents are still active
        self.hit_number = 0  # Global step counter
        
        # Shared hit pool - hits that have been selected are removed
        self.selected_hit_ids = set()  # Track which hits have been selected
        
        # Per-agent state storage
        self.current_comp_hits = [None] * num_agents
        self.current_rewards = [None] * num_agents
        self.current_info = [{} for _ in range(num_agents)]
        
        # Accuracy tracking per agent
        self.total_selections = [0] * num_agents
        self.correct_selections = [0] * num_agents
        
        # Coordination metrics
        self.hit_overlap_count = 0  # Times agents had overlapping candidate hits
        self.total_steps = 0  # Total steps for percentage calculation
        self.conflict_attempts = 0  # Times agent tried to select already-used hit
        self.shared_hits_per_step = []  # List of shared hit counts per step
        
        self.load_next_file()
        # Observation space: [z, r, x, y] for own helix + [z, r, x, y] * (num_agents-1) for other agents
        obs_dim = 4 + 4 * (num_agents - 1)  # Own state + other agents' states
        self.observation_space = Box(low=-200, high=200, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Discrete(20)  # Hit index selection

    def _apply_particle_selections(self):
        """Select particles based on criteria."""
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
        
        # Get unique particles that pass the selection
        selected_particles = hits['particle_id'].unique()
        
        self.pids_to_explore = list(selected_particles)
        if self.deterministic:
            self.pids_to_explore.sort()
        else:
            np.random.shuffle(self.pids_to_explore)
    
    def _group_nearby_particles(self, start_idx):
        """Group nearby particles that should be propagated together.
        
        Uses multiple criteria:
        1. Spatial proximity of trajectories (not just seed hits)
        2. Overlapping detector layers
        3. Similar phi angles (azimuthal direction)
        
        Returns list of particle IDs that are spatially close.
        """
        if start_idx >= len(self.pids_to_explore):
            return []
        
        # Start with first particle
        particle_ids = [self.pids_to_explore[start_idx]]
        
        # Get first particle's track
        first_pid = particle_ids[0]
        first_track = self.all_hits[self.all_hits['particle_id'] == first_pid]
        if len(first_track) == 0:
            return particle_ids
        
        # Get reference trajectory properties
        # Use multiple hits along trajectory, not just seed
        first_track_sorted = first_track.sort_values('r')
        ref_layers = set(first_track_sorted['unique_layer_id'].values)
        
        # Get phi angle distribution (azimuthal direction)
        ref_phi_mean = first_track_sorted['phi'].mean() if 'phi' in first_track_sorted.columns else 0
        ref_phi_std = first_track_sorted['phi'].std() if 'phi' in first_track_sorted.columns else 0
        
        # Get spatial extent of trajectory (min/max in each dimension)
        ref_r_range = (first_track_sorted['r'].min(), first_track_sorted['r'].max())
        ref_z_range = (first_track_sorted['z'].min(), first_track_sorted['z'].max())
        
        # Find nearby particles using multiple criteria
        candidates = []
        for i in range(start_idx + 1, min(start_idx + self.num_agents * 10, len(self.pids_to_explore))):
            if len(particle_ids) >= self.num_agents:
                break
                
            pid = self.pids_to_explore[i]
            track = self.all_hits[self.all_hits['particle_id'] == pid]
            if len(track) == 0:
                continue
            
            track_sorted = track.sort_values('r')
            track_layers = set(track_sorted['unique_layer_id'].values)
            
            # Criterion 1: Layer overlap (do they pass through same detector layers?)
            layer_overlap = len(ref_layers.intersection(track_layers))
            layer_overlap_ratio = layer_overlap / min(len(ref_layers), len(track_layers)) if min(len(ref_layers), len(track_layers)) > 0 else 0
            
            # Criterion 2: Phi angle proximity (are they going in similar direction?)
            track_phi_mean = track_sorted['phi'].mean() if 'phi' in track_sorted.columns else 0
            phi_diff = abs(ref_phi_mean - track_phi_mean)
            # Handle phi wraparound (phi is in [-pi, pi])
            if phi_diff > np.pi:
                phi_diff = 2 * np.pi - phi_diff
            
            # Criterion 3: Spatial trajectory overlap (do their paths overlap in space?)
            track_r_range = (track_sorted['r'].min(), track_sorted['r'].max())
            track_z_range = (track_sorted['z'].min(), track_sorted['z'].max())
            
            # Check if r ranges overlap
            r_overlap = min(ref_r_range[1], track_r_range[1]) - max(ref_r_range[0], track_r_range[0])
            r_overlap_ratio = r_overlap / (ref_r_range[1] - ref_r_range[0]) if r_overlap > 0 else 0
            
            # Check if z ranges overlap
            z_overlap = min(ref_z_range[1], track_z_range[1]) - max(ref_z_range[0], track_z_range[0])
            z_overlap_ratio = z_overlap / (ref_z_range[1] - ref_z_range[0]) if z_overlap > 0 else 0
            
            # Compute proximity score (0 = not close, 1 = very close)
            # Weight layer overlap most heavily since that's where coordination matters
            score = (
                0.5 * layer_overlap_ratio +  # Most important: do they share layers?
                0.2 * (1.0 - phi_diff / np.pi) +  # Going in similar direction?
                0.15 * r_overlap_ratio +  # Trajectories overlap in r?
                0.15 * z_overlap_ratio  # Trajectories overlap in z?
            )
            
            # Add to candidates with score
            candidates.append((pid, score, layer_overlap))
        
        # Sort candidates by score and take top ones
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Add particles with good enough scores
        # Threshold: at least 0.3 score (meaning some layer overlap + spatial proximity)
        for pid, score, layer_overlap in candidates:
            if len(particle_ids) >= self.num_agents:
                break
            if score >= 0.25 and layer_overlap >= 2:  # At least 2 shared layers and score > 0.25
                particle_ids.append(pid)
        
        # If we still don't have enough agents, just take next particles (better than nothing)
        if len(particle_ids) < self.num_agents:
            remaining_idx = start_idx + 1
            while len(particle_ids) < self.num_agents and remaining_idx < len(self.pids_to_explore):
                pid = self.pids_to_explore[remaining_idx]
                if pid not in particle_ids:
                    track = self.all_hits[self.all_hits['particle_id'] == pid]
                    if len(track) > 0:
                        particle_ids.append(pid)
                remaining_idx += 1
        
        logger.info(f"Grouped {len(particle_ids)} nearby particles for multi-agent propagation")
        if len(candidates) > 0 and len(particle_ids) > 1:
            # Log scores for debugging
            selected_scores = [score for pid, score, _ in candidates if pid in particle_ids[1:]]
            if selected_scores:
                logger.info(f"  Selected tracks with scores: {selected_scores[:5]}")
        
        return particle_ids

    def load_next_file(self):
        """Load the next file and compute the number of steps from the column."""
        try:
            data = next(self.data_loader)
            hits = pd.DataFrame(
                data.view(data.shape[1], data.shape[2]),
                columns=COLS)
            
            logger.info(f"Loaded event with {len(hits)} raw hits")
            self.all_hits = hits
            self._apply_particle_selections()
            logger.info(f"Selected {len(self.pids_to_explore)} particles to explore")
            self.particle_index = 0
        except StopIteration:
            self.all_hits = None
            self.pids_to_explore = []

    def reset(self):
        """Reset environment for a new multi-agent episode."""
        # Check if we need to load more particles
        if len(self.pids_to_explore) == 0 or self.particle_index >= len(self.pids_to_explore):
            self.load_next_file()
            self.particle_index = 0
        
        if self.all_hits is None:
            return None, {}
        
        # Get nearby particles to propagate together
        particle_ids = self._group_nearby_particles(self.particle_index)
        self.particle_index += len(particle_ids)
        
        # Initialize tracks and helices for all agents
        self.current_tracks = []
        self.helices = []
        self.agent_active = []
        
        for pid in particle_ids:
            track = self.all_hits[self.all_hits['particle_id'] == pid].copy()
            self.current_tracks.append(track)
            
            if len(track) > 0:
                helix = Helix(track, use_truth_path_plan=self.use_truth_path_plan)
                self.helices.append(helix)
                self.agent_active.append(True)
                logger.info(f"Agent {len(self.helices)-1}: particle {pid} with {len(track)} hits")
            else:
                self.helices.append(None)
                self.agent_active.append(False)
        
        # Pad with inactive agents if we don't have enough
        while len(self.helices) < self.num_agents:
            self.current_tracks.append(pd.DataFrame())
            self.helices.append(None)
            self.agent_active.append(False)
        
        self.hit_number = 0
        self.selected_hit_ids = set()
        
        # Reset per-agent storage
        self.current_comp_hits = [None] * self.num_agents
        self.current_rewards = [None] * self.num_agents
        self.current_info = [{} for _ in range(self.num_agents)]
        
        # Reset accuracy tracking
        for i in range(self.num_agents):
            self.total_selections[i] = 0
            self.correct_selections[i] = 0
        
        # Get initial states for all agents
        return self.get_current_state()

    def get_current_state(self):
        """Get current state for all agents.
        
        Returns:
            observations: List of observations (one per agent)
            info: Dict containing per-agent info
        """
        observations = []
        all_info = {'agents': []}
        
        # Check if any agents are still active
        any_active = any(self.agent_active)
        if not any_active:
            return None, {}
        
        # Hit holder for finding compatible hits
        hit_holder = HitHolder(self.all_hits)
        
        # Get state for each agent
        for agent_idx in range(self.num_agents):
            if not self.agent_active[agent_idx] or self.helices[agent_idx] is None:
                # Inactive agent - return zero observation
                obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
                observations.append(obs)
                all_info['agents'].append({
                    'active': False,
                    'rewards': [],
                    'num_comp_hits': 0,
                    'hit_features': np.zeros((20, 4), dtype=np.float32),
                    'hit_mask': np.zeros(20, dtype=bool),
                })
                continue
            
            helix = self.helices[agent_idx]
            
            # Check if path plan is complete
            unexplored_layers = helix.path_plan[helix.path_plan > helix.current_layer]
            if len(unexplored_layers) == 0:
                # This agent is done
                self.agent_active[agent_idx] = False
                obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
                observations.append(obs)
                all_info['agents'].append({
                    'active': False,
                    'rewards': [],
                    'num_comp_hits': 0,
                    'hit_features': np.zeros((20, 4), dtype=np.float32),
                    'hit_mask': np.zeros(20, dtype=bool),
                })
                continue
            
            # Propagate one layer for this agent
            comp_hits, rewards, done, correct_in_comp, correct_is_best = (
                helix.propagate_one_layer(hit_holder, use_distance_reward=self.use_distance_reward,
                                         max_layer_skip=self.max_layer_skip, tolerance_r=self.tolerance_r))
            
            # Filter out hits that have already been selected by other agents
            if len(comp_hits) > 0:
                available_mask = ~comp_hits['hit_id'].isin(self.selected_hit_ids)
                comp_hits = comp_hits[available_mask].copy()
                if isinstance(rewards, list):
                    rewards = [r for i, r in enumerate(rewards) if available_mask.iloc[i]]
                
            # Store comp_hits and rewards for this agent
            self.current_comp_hits[agent_idx] = comp_hits
            self.current_rewards[agent_idx] = rewards
            
            # Build observation: own helix position + other agents' positions
            own_state = np.array([helix.z, helix.r0, helix.x, helix.y], dtype=np.float32)
            
            # Add other agents' states
            other_states = []
            for other_idx in range(self.num_agents):
                if other_idx == agent_idx:
                    continue
                if self.agent_active[other_idx] and self.helices[other_idx] is not None:
                    other_helix = self.helices[other_idx]
                    other_state = np.array([other_helix.z, other_helix.r0, other_helix.x, other_helix.y], dtype=np.float32)
                else:
                    other_state = np.zeros(4, dtype=np.float32)
                other_states.append(other_state)
            
            # Concatenate own state with other agents' states
            obs = np.concatenate([own_state] + other_states)
            observations.append(obs)
            
            # Prepare hit features for this agent
            max_hits = 20
            hit_dim = 4
            
            if len(comp_hits) > 0:
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
                    
                    # Check for NaN/inf
                    if np.isnan(hit_feat).any() or np.isinf(hit_feat).any():
                        hit_feat = np.nan_to_num(hit_feat, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    hit_feat = np.clip(hit_feat, -1e4, 1e4)
                    hit_features_list.append(hit_feat)
                
                # Pad to max_hits
                while len(hit_features_list) < max_hits:
                    hit_features_list.append(np.zeros(hit_dim, dtype=np.float32))
                
                hit_features = np.array(hit_features_list[:max_hits], dtype=np.float32)
                hit_mask = np.array([True] * num_hits_to_use + [False] * (max_hits - num_hits_to_use), dtype=bool)
            else:
                hit_features = np.zeros((max_hits, hit_dim), dtype=np.float32)
                hit_mask = np.zeros(max_hits, dtype=bool)
            
            # Store correct hit IDs for accuracy tracking
            correct_hit_ids_set = set()
            if hasattr(helix, 'correct_hit_ids_in_layer'):
                correct_ids = helix.correct_hit_ids_in_layer
                if isinstance(correct_ids, set):
                    correct_hit_ids_set = correct_ids
                elif hasattr(correct_ids, '__iter__'):
                    correct_hit_ids_set = set(correct_ids)
                else:
                    correct_hit_ids_set = {correct_ids} if correct_ids is not None else set()
            
            info = {
                'active': True,
                'rewards': rewards,
                'correct_in_comp': correct_in_comp,
                'correct_is_best': correct_is_best,
                'num_comp_hits': len(comp_hits),
                'correct_hit_ids': correct_hit_ids_set,
                'is_correct': [int(comp_hits.iloc[i]['hit_id']) in correct_hit_ids_set 
                              for i in range(len(comp_hits))] if len(comp_hits) > 0 else [],
                'hit_features': hit_features,
                'hit_mask': hit_mask,
            }
            self.current_info[agent_idx] = info
            all_info['agents'].append(info)
        
        # Track coordination metrics: check for overlapping candidate hits
        self.total_steps += 1
        active_agents = [i for i in range(self.num_agents) if self.agent_active[i]]
        
        if len(active_agents) >= 2:
            # Collect all hit IDs for active agents
            agent_hit_sets = []
            for agent_idx in active_agents:
                comp_hits = self.current_comp_hits[agent_idx]
                if comp_hits is not None and len(comp_hits) > 0:
                    hit_ids = set(comp_hits['hit_id'].values)
                    agent_hit_sets.append(hit_ids)
            
            # Check for overlaps between any pair of agents
            if len(agent_hit_sets) >= 2:
                has_overlap = False
                shared_hits_count = 0
                for i in range(len(agent_hit_sets)):
                    for j in range(i + 1, len(agent_hit_sets)):
                        overlap = agent_hit_sets[i].intersection(agent_hit_sets[j])
                        if len(overlap) > 0:
                            has_overlap = True
                            shared_hits_count += len(overlap)
                
                if has_overlap:
                    self.hit_overlap_count += 1
                self.shared_hits_per_step.append(shared_hits_count)
        
        return observations, all_info

    def step(self, actions):
        """Take actions for all agents and move to next step.
        
        Args:
            actions: List of actions (one per agent)
        
        Returns:
            next_observations, rewards, dones (per agent), info
        """
        rewards = []
        dones = []
        
        # Process each agent's action
        for agent_idx in range(self.num_agents):
            if not self.agent_active[agent_idx]:
                rewards.append(0.0)
                dones.append(True)
                continue
            
            action = actions[agent_idx]
            comp_hits = self.current_comp_hits[agent_idx]
            agent_rewards = self.current_rewards[agent_idx]
            
            reward = 0.0
            is_correct = False
            
            if comp_hits is not None and len(comp_hits) > 0:
                # Select hit based on action
                action_idx = min(action, len(comp_hits) - 1)
                if action_idx < 0:
                    action_idx = 0
                
                selected_hit = comp_hits.iloc[action_idx]
                selected_hit_id = int(selected_hit['hit_id'])
                
                # Check if hit was already selected by another agent (shouldn't happen with filtering)
                if selected_hit_id in self.selected_hit_ids:
                    # Penalty for selecting already-used hit
                    reward = -0.5
                    self.conflict_attempts += 1  # Track coordination failure
                else:
                    # Mark hit as selected
                    self.selected_hit_ids.add(selected_hit_id)
                    
                    # Get reward
                    if isinstance(agent_rewards, list) and len(agent_rewards) > action_idx:
                        reward = float(agent_rewards[action_idx])
                    elif isinstance(agent_rewards, (int, float)):
                        reward = float(agent_rewards)
                    
                    # Update helix position
                    helix = self.helices[agent_idx]
                    helix.x = float(selected_hit['x'])
                    helix.y = float(selected_hit['y'])
                    helix.z = float(selected_hit['z'])
                    
                    # Track accuracy (exclude seed hits)
                    seed_hit_ids = set()
                    if hasattr(helix, 'seed_hits') and helix.seed_hits is not None:
                        seed_df = helix.seed_hits
                        if isinstance(seed_df, pd.DataFrame) and 'hit_id' in seed_df.columns:
                            seed_hit_ids = set(seed_df['hit_id'].values.astype(int))
                    
                    correct_ids = self.current_info[agent_idx].get('correct_hit_ids', set())
                    is_correct = selected_hit_id in correct_ids
                    
                    if selected_hit_id not in seed_hit_ids:
                        self.total_selections[agent_idx] += 1
                        if is_correct:
                            self.correct_selections[agent_idx] += 1
            
            rewards.append(reward)
            
            # Check if agent is done
            helix = self.helices[agent_idx]
            if helix is not None:
                unexplored_layers = helix.path_plan[helix.path_plan > helix.current_layer]
                if len(unexplored_layers) == 0:
                    self.agent_active[agent_idx] = False
                    dones.append(True)
                else:
                    dones.append(False)
            else:
                dones.append(True)
        
        # Increment global step counter
        self.hit_number += 1
        
        # Check if all agents are done
        all_done = all(dones)
        
        # Get next state
        if all_done:
            next_obs = None
            next_info = {}
        else:
            next_obs, next_info = self.get_current_state()
        
        return next_obs, rewards, dones, next_info

    def get_accuracy(self, agent_idx):
        """Get accuracy for a specific agent."""
        if self.total_selections[agent_idx] > 0:
            return self.correct_selections[agent_idx] / self.total_selections[agent_idx]
        return 0.0
    
    def get_coordination_metrics(self):
        """Get coordination metrics for agents.
        
        Returns dict with:
        - overlap_ratio: Percentage of steps where agents had overlapping candidate hits
        - avg_shared_hits: Average number of shared hits per step
        - conflict_rate: Percentage of selections that were conflicts
        """
        metrics = {}
        
        if self.total_steps > 0:
            metrics['overlap_ratio'] = self.hit_overlap_count / self.total_steps
        else:
            metrics['overlap_ratio'] = 0.0
        
        if len(self.shared_hits_per_step) > 0:
            metrics['avg_shared_hits'] = np.mean(self.shared_hits_per_step)
        else:
            metrics['avg_shared_hits'] = 0.0
        
        total_selections = sum(self.total_selections)
        if total_selections > 0:
            metrics['conflict_rate'] = self.conflict_attempts / total_selections
        else:
            metrics['conflict_rate'] = 0.0
        
        return metrics
    
    def reset_coordination_metrics(self):
        """Reset coordination metrics (call at start of episode)."""
        self.hit_overlap_count = 0
        self.total_steps = 0
        self.conflict_attempts = 0
        self.shared_hits_per_step = []

