"""Animation script to visualize track reconstruction step by step."""
import argparse
import itertools
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rl_tracking.lightning_modules.dqn import DQNLightning
from rl_tracking.preprocessing.hit_candidates import EventProcessor
from rl_tracking.utils.stream_loading import TrackingDataModule
from rl_tracking.utils.config_loader import load_config
from rl_tracking.utils.data_paths import resolve_data_directories
from rl_tracking.environment.tracking_env import TrackingEnv
from rl_tracking.environment.agent import Agent

DEFAULT_TRACKML_DIR = Path("/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/codalab-data/part_1")


class TrackAnimator:
    """Animates track reconstruction step by step."""
    
    def __init__(self, model_path, config_path=None, test_data_dir=None, highlight_hit_ids=None):
        """Initialize animator with model and data."""
        # Load configuration
        if config_path is None:
            # Try to find config in checkpoint directory
            checkpoint_dir = Path(model_path).parent
            config_path = checkpoint_dir / 'train_config.yaml'
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
        
        config = load_config(config_path)
        data_config = config.get('data', {})
        environment_config = config.get('environment', {})
        
        # Extract configurations
        data_config_override = dict(data_config) if data_config else {}
        if test_data_dir is not None:
            data_config_override['data_dir'] = test_data_dir

        self.data_root, self.data_directories = resolve_data_directories(
            data_config_override,
            default_dir=DEFAULT_TRACKML_DIR,
        )

        print("Using TrackML data directories for animation:")
        for directory in self.data_directories:
            print(f"  - {directory}")

        batch_size = data_config.get('batch_size', 32)
        val_split = data_config.get('val_split', 0.2)
        test_split = data_config.get('test_split', 0.1)
        num_workers = data_config.get('num_workers', 0)
        random_seed = data_config.get('random_seed', 42)
        event_processor_config = data_config.get('event_processor', {})
        n_neighbors = event_processor_config.get('n_neighbors', 20)
        
        particle_filters = environment_config.get('particle_filters', {})
        hit_filters = environment_config.get('hit_filters', {})
        use_distance_reward = environment_config.get('use_distance_reward', False)
        use_truth_path_plan = environment_config.get('use_truth_path_plan', False)
        
        # Initialize data module
        ep = EventProcessor(self.data_root, n_neighbors)
        self.dm = TrackingDataModule(
            file_paths=self.data_directories,
            batch_size=batch_size,
            event_processor=ep,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            random_seed=random_seed,
        )
        
        # Setup datamodule for both fit and test (model needs train_dataset during init)
        self.dm.setup(stage="fit")
        self.dm.setup(stage="test")
        
        # Load model
        self.model = DQNLightning.load_from_checkpoint(
            model_path,
            dm=self.dm,
            particle_filters=particle_filters,
            hit_filters=hit_filters,
            use_distance_reward=use_distance_reward,
            use_truth_path_plan=use_truth_path_plan,
        )
        self.model.eval()
        self.highlight_hit_ids = {
            int(hit_id) for hit_id in (highlight_hit_ids or [])
        }
        
        # Create test environment using test dataset
        # TrackingEnv expects an iterator that yields numpy arrays/tensors
        # The dataset returns individual events, so we just wrap it in an iterator
        # that yields the raw numpy array (not batched)
        test_dataset = self.dm.test_dataset
        
        # Create a simple iterator that yields events one at a time
        # TrackingEnv expects next(data_loader) to return a tensor with shape [1, hits, features]
        # It then does data.view(data.shape[1], data.shape[2]) to get [hits, features]
        def event_iterator():
            for event in test_dataset:
                # Convert to tensor if needed (event_processor returns numpy array)
                if isinstance(event, np.ndarray):
                    event_tensor = torch.from_numpy(event)
                elif isinstance(event, torch.Tensor):
                    event_tensor = event
                else:
                    # Convert pandas or other formats
                    event_tensor = torch.from_numpy(np.array(event))
                
                # Ensure it has the right shape: [1, hits, features]
                # If it's 2D [hits, features], add batch dimension
                if event_tensor.dim() == 2:
                    event_tensor = event_tensor.unsqueeze(0)  # [1, hits, features]
                
                yield event_tensor
        
        # Create a simple iterable wrapper
        class EventIterable:
            def __init__(self, dataset):
                self.dataset = dataset
            def __iter__(self):
                return event_iterator()
        
        event_iter = EventIterable(test_dataset)
        
        self.test_env = TrackingEnv(
            event_iter,
            particle_filters=particle_filters,
            hit_filters=hit_filters,
            use_distance_reward=use_distance_reward,
            use_truth_path_plan=use_truth_path_plan,
        )
        
        # Create agent (for evaluation, we don't need a replay buffer)
        from rl_tracking.replay.buffer import ReplayBuffer
        replay_buffer = ReplayBuffer(1000)  # Dummy buffer, not used for inference
        self.test_agent = Agent(self.test_env, replay_buffer)
        
        # Load layer info for visualization
        physics_dir = Path(__file__).parent.parent / 'physics'
        layer_info_path = physics_dir / 'layer_info.csv'
        # Layer info has multi-index columns (z, r) with (min, max, median) sub-columns
        self.layer_info = pd.read_csv(layer_info_path, index_col=0, header=[0, 1])
        
        # Storage for animation frames
        self.frames = []
        self.current_track = None
        
    def collect_track_data(self, track_index=0):
        """Collect all data for a single track reconstruction."""
        # Reset environment to get a track
        obs = self.test_env.reset()
        
        # Skip tracks until we get to the desired one
        for _ in range(track_index):
            obs = self.test_env.reset()
            if obs is None:
                return None
        
        if obs is None:
            return None
        
        # Get initial track data
        self.current_track = self.test_env.current_track.copy()
        
        # Get seed hits
        seed_hits = self.test_env.helix.seed_hits if hasattr(self.test_env.helix, 'seed_hits') else None
        
        # Get initial stats
        initial_start_layer = getattr(self.test_env.helix, 'initial_start_layer', None)
        if initial_start_layer is None:
            initial_start_layer = self.test_env.helix.current_layer
        
        # Get truth hits after seed
        if initial_start_layer is not None:
            truth_hits_after_seed = self.current_track[
                self.current_track['unique_layer_id'] > initial_start_layer
            ]
        else:
            truth_hits_after_seed = self.current_track
        
        truth_hit_ids = set(truth_hits_after_seed['hit_id'].values)
        seed_hit_ids = set()
        if seed_hits is not None and isinstance(seed_hits, pd.DataFrame) and 'hit_id' in seed_hits.columns:
            seed_hit_ids = set(seed_hits['hit_id'].values.astype(int))
        
        # Track reconstruction step by step
        frames = []
        selected_hit_ids = []
        correct_hits = 0
        predicted_hits = 0
        
        # Track layers where we found at least one correct hit (for efficiency calculation)
        # Efficiency: count one hit per layer, not all hits
        layers_with_correct_hit = set()
        
        done = False
        step = 0
        
        while not done:
            # Get current state
            if obs is None:
                break
            
            # Get action - use same device as model
            device = next(self.model.net.parameters()).device
            action = self.test_agent.get_action(self.model.net, epsilon=0.0, device=str(device))
            
            # Store frame data before step
            frame_data = {
                'step': step,
                'helix_pos': (self.test_env.helix.x, self.test_env.helix.y, self.test_env.helix.z),
                'helix_layer': float(self.test_env.helix.current_layer),
                'comp_hits': self.test_env.current_comp_hits.copy() if self.test_env.current_comp_hits is not None else pd.DataFrame(),
                'selected_hit': None,
                'selected_hit_correct': False,
                'selected_hit_id': None,
                'correct_hits_in_layer': set(),
                'accuracy': correct_hits / predicted_hits if predicted_hits > 0 else 0.0,
                'correct_hits': correct_hits,
                'predicted_hits': predicted_hits,
            }
            
            # Get correct hits for current layer
            if hasattr(self.test_env.helix, 'correct_hit_ids_in_layer'):
                frame_data['correct_hits_in_layer'] = set(self.test_env.helix.correct_hit_ids_in_layer)
            
            # Step environment
            step_result = self.test_env.step(action)
            
            if len(step_result) == 3:
                next_obs, reward, done = step_result
            else:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            
            # Store selected hit info
            if hasattr(self.test_env, 'last_selected_hit') and self.test_env.last_selected_hit is not None:
                selected_hit = self.test_env.last_selected_hit
                frame_data['selected_hit'] = selected_hit
                frame_data['selected_hit_id'] = int(selected_hit['hit_id'])
                frame_data['selected_hit_correct'] = self.test_env.last_selected_hit_correct
                
                # Update counters
                if frame_data['selected_hit_id'] not in seed_hit_ids:
                    if frame_data['selected_hit_id'] not in selected_hit_ids:
                        selected_hit_ids.append(frame_data['selected_hit_id'])
                        predicted_hits += 1
                        if frame_data['selected_hit_correct']:
                            correct_hits += 1
                            # Track that we found at least one correct hit in this layer
                            # Get the layer from the selected hit or current helix layer
                            selected_hit_layer_for_efficiency = None
                            if frame_data['selected_hit'] is not None:
                                selected_hit = frame_data['selected_hit']
                                if isinstance(selected_hit, pd.Series) and 'unique_layer_id' in selected_hit.index:
                                    selected_hit_layer_for_efficiency = float(selected_hit['unique_layer_id'])
                                elif hasattr(selected_hit, 'unique_layer_id'):
                                    selected_hit_layer_for_efficiency = float(selected_hit.unique_layer_id)
                            # Fallback to current layer
                            if selected_hit_layer_for_efficiency is None:
                                selected_hit_layer_for_efficiency = frame_data['helix_layer']
                            
                            if selected_hit_layer_for_efficiency is not None:
                                layers_with_correct_hit.add(selected_hit_layer_for_efficiency)
            
            # Calculate efficiency: one hit per layer (not all hits)
            # Count unique layers with truth hits after seed
            if initial_start_layer is not None:
                truth_hits_after_seed_filtered = truth_hits_after_seed[
                    truth_hits_after_seed['unique_layer_id'] > initial_start_layer
                ]
                expected_layers = len(truth_hits_after_seed_filtered['unique_layer_id'].unique())
            else:
                expected_layers = len(truth_hits_after_seed['unique_layer_id'].unique())
            
            # Efficiency = layers where we found at least one correct hit / total layers with truth hits
            frame_data['efficiency'] = len(layers_with_correct_hit) / expected_layers if expected_layers > 0 else 0.0
            frame_data['purity'] = correct_hits / predicted_hits if predicted_hits > 0 else 0.0
            frame_data['expected_hits'] = expected_layers  # Now represents expected layers, not hits
            frame_data['layers_with_correct'] = len(layers_with_correct_hit)
            
            frames.append(frame_data)
            obs = next_obs
            step += 1
            
            if done:
                break
        
        self.frames = frames
        self.truth_hits = self.current_track
        self.seed_hits = seed_hits
        self.seed_hit_ids = seed_hit_ids
        
        return frames
    
    def create_cylinder(self, r, z_min, z_max, n_points=20):
        """Create a cylinder mesh for layer visualization."""
        theta = np.linspace(0, 2 * np.pi, n_points)
        z = np.linspace(z_min, z_max, 2)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Create cylinder surface
        vertices = []
        for z_val in z:
            for i in range(len(theta)):
                vertices.append([x[i], y[i], z_val])
        
        return vertices
    
    def animate(self, output_path='track_animation.gif', fps=2, dpi=100, num_tracks=3):
        """Create animation of track reconstruction.
        
        Args:
            output_path: Path to save animation
            fps: Frames per second
            dpi: Resolution
            num_tracks: Number of tracks to animate (will cycle through them)
        """
        if len(self.frames) == 0:
            print("No frames to animate. Call collect_track_data() first.")
            return
        
        # Collect data for multiple tracks
        all_tracks_data = []
        all_tracks_data.append({
            'frames': self.frames,
            'truth_hits': self.truth_hits,
            'seed_hits': self.seed_hits,
            'seed_hit_ids': self.seed_hit_ids,
        })
        
        # Collect additional tracks
        for track_idx in range(1, num_tracks):
            print(f"Collecting data for track {track_idx}...")
            frames = self.collect_track_data(track_idx)
            if frames is not None and len(frames) > 0:
                all_tracks_data.append({
                    'frames': frames,
                    'truth_hits': self.truth_hits,
                    'seed_hits': self.seed_hits,
                    'seed_hit_ids': self.seed_hit_ids,
                })
            else:
                break
        
        print(f"Collected {len(all_tracks_data)} tracks. Creating animation...")
        
        # Set up figure with two subplots: 3D view (left) and 2D RZ view (right)
        fig = plt.figure(figsize=(20, 10), facecolor='black')
        
        # Create 3D subplot (left, larger) with black background
        ax_3d = fig.add_subplot(121, projection='3d')
        ax_3d.set_facecolor('black')
        
        # Create 2D RZ subplot (right, smaller) with black background
        ax_2d = fig.add_subplot(122)
        ax_2d.set_facecolor('black')
        
        # Track current track index
        current_track_idx = 0
        frame_counter = 0
        total_frames = sum(len(track['frames']) for track in all_tracks_data)
        
        def update(frame_idx):
            nonlocal current_track_idx, frame_counter
            
            # Determine which track and frame we're on
            track_start_frame = 0
            for i, track_data in enumerate(all_tracks_data):
                if frame_counter < track_start_frame + len(track_data['frames']):
                    current_track_idx = i
                    local_frame_idx = frame_counter - track_start_frame
                    break
                track_start_frame += len(track_data['frames'])
            else:
                # End of all tracks, reset
                frame_counter = 0
                current_track_idx = 0
                local_frame_idx = 0
                # Clear paths when cycling back to start
                if hasattr(update, 'track_paths'):
                    update.track_paths = {}
                if hasattr(update, 'current_track_key'):
                    update.current_track_key = None
            
            track_data = all_tracks_data[current_track_idx]
            frames = track_data['frames']
            truth_hits = track_data['truth_hits']
            seed_hits = track_data['seed_hits']
            seed_hit_ids = track_data['seed_hit_ids']
            
            if local_frame_idx >= len(frames):
                frame_counter += 1
                return
            
            frame = frames[local_frame_idx]
            
            # Clear both subplots
            ax_3d.clear()
            ax_2d.clear()
            
            # Plot truth hits first (so seed hits appear on top)
            # Coordinate transformation: new_x = old_y, new_y = old_z, new_z = old_x
            # Only plot post-seed truth hits (no overlap with seed)
            # Use brighter blue (cyan) for better visibility on black
            post_seed_truth = truth_hits[~truth_hits['hit_id'].isin(seed_hit_ids)]
            highlight_set = getattr(self, "highlight_hit_ids", set())
            highlight_truth = pd.DataFrame()
            base_truth = post_seed_truth
            if highlight_set and len(post_seed_truth) > 0:
                mask = post_seed_truth['hit_id'].apply(
                    lambda hit: False if pd.isna(hit) else int(hit) in highlight_set
                )
                highlight_truth = post_seed_truth[mask]
                base_truth = post_seed_truth[~mask]
            if len(base_truth) > 0:
                ax_3d.scatter(
                    base_truth['y'],
                    base_truth['z'],
                    base_truth['x'],
                    c='cyan',
                    s=50,
                    marker='o',
                    label='Truth hits',
                    alpha=0.6,
                    edgecolors='white',
                    linewidths=0.5,
                    zorder=1,
                )
            if len(highlight_truth) > 0:
                ax_3d.scatter(
                    highlight_truth['y'],
                    highlight_truth['z'],
                    highlight_truth['x'],
                    c='yellow',
                    s=120,
                    marker='o',
                    label='Highlighted truth hits',
                    alpha=0.9,
                    edgecolors='white',
                    linewidths=1.5,
                    zorder=4,
                )
            
            # Plot seed hits on top (higher zorder) so they're always visible
            if seed_hits is not None and len(seed_hits) > 0:
                seed_x = seed_hits['x'].values
                seed_y = seed_hits['y'].values
                seed_z = seed_hits['z'].values
                seed_r = np.sqrt(seed_x**2 + seed_y**2)
                # Transform coordinates: x->y, y->z, z->x
                ax_3d.scatter(seed_y, seed_z, seed_x, c='orange', s=120, marker='o', 
                          label='Seed hits', alpha=0.9, edgecolors='white', linewidths=1.5, zorder=3)
            
            # Plot all truth hits in 2D RZ view (seed and post-seed separately)
            # Convention: z on x-axis, r on y-axis
            # Plot truth hits first, then seed hits on top
            if len(base_truth) > 0:
                post_seed_r = np.sqrt(base_truth['x']**2 + base_truth['y']**2)
                post_seed_z = base_truth['z'].values
                ax_2d.scatter(
                    post_seed_z,
                    post_seed_r,
                    c='cyan',
                    s=40,
                    alpha=0.7,
                    label='Truth hits',
                    marker='o',
                    edgecolors='white',
                    linewidths=0.5,
                    zorder=1,
                )
            if len(highlight_truth) > 0:
                highlight_r = np.sqrt(highlight_truth['x']**2 + highlight_truth['y']**2)
                highlight_z = highlight_truth['z'].values
                ax_2d.scatter(
                    highlight_z,
                    highlight_r,
                    c='yellow',
                    s=70,
                    alpha=0.95,
                    label='Highlighted truth hits',
                    marker='o',
                    edgecolors='white',
                    linewidths=1.5,
                    zorder=4,
                )
            
            # Plot seed hits on top (higher zorder) so they're always visible
            if seed_hits is not None and len(seed_hits) > 0:
                seed_r = np.sqrt(seed_hits['x']**2 + seed_hits['y']**2)
                seed_z = seed_hits['z'].values
                ax_2d.scatter(seed_z, seed_r, c='orange', s=60, alpha=0.95, 
                             label='Seed hits', marker='o', edgecolors='white', linewidths=1.5, zorder=3)
            
            # Plot layer boundaries in 2D RZ view - make them lighter and more visible
            # Convention: z on x-axis, r on y-axis
            for layer_id in self.layer_info.index:
                try:
                    layer_row = self.layer_info.loc[layer_id]
                    r_min = layer_row[('r', 'min')]
                    r_max = layer_row[('r', 'max')]
                    z_min = layer_row[('z', 'min')]
                    z_max = layer_row[('z', 'max')]
                    
                    # Draw layer as a rectangle with lighter, more visible style
                    # Swap axes: z on x, r on y
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((z_min, r_min), z_max - z_min, r_max - r_min,
                                   linewidth=1.5, edgecolor='lightgray', facecolor='lightgray', 
                                   alpha=0.5)
                    ax_2d.add_patch(rect)
                except:
                    pass
            
            # Layer boundaries removed from 3D view as requested
            current_layer = frame['helix_layer']
            
            # Plot compatible hits search area (tolerance sphere around helix)
            comp_hits = frame['comp_hits']
            if len(comp_hits) > 0:
                # Get tolerance from helix (or use default)
                tolerance_r = getattr(self.test_env, 'tolerance_r', 2.0)
                helix_x, helix_y, helix_z = frame['helix_pos']
                
                # Draw a sphere showing the search area
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                sphere_x = helix_x + tolerance_r * np.outer(np.cos(u), np.sin(v))
                sphere_y = helix_y + tolerance_r * np.outer(np.sin(u), np.sin(v))
                sphere_z = helix_z + tolerance_r * np.outer(np.ones(np.size(u)), np.cos(v))
                
                # Transform coordinates: x->y, y->z, z->x
                ax_3d.plot_wireframe(sphere_y, sphere_z, sphere_x, alpha=0.2, color='cyan', linewidth=0.5,
                                 label='Search area' if len(comp_hits) > 0 else '')
            
            # Track paths for this track
            # Reset paths when we start a new track
            if not hasattr(update, 'track_paths'):
                update.track_paths = {}
            if not hasattr(update, 'current_track_key'):
                update.current_track_key = None
            
            # If we're on a new track, reset the paths
            if update.current_track_key != current_track_idx:
                update.current_track_key = current_track_idx
            
            track_key = current_track_idx
            if track_key not in update.track_paths:
                update.track_paths[track_key] = {
                    'helix_path_x': [],
                    'helix_path_y': [],
                    'helix_path_z': [],
                    'helix_path_r': [],
                    'selected_path_x': [],
                    'selected_path_y': [],
                    'selected_path_z': [],
                    'selected_path_r': [],
                }
            
            paths = update.track_paths[track_key]
            
            # Plot helix position in 3D
            helix_x, helix_y, helix_z = frame['helix_pos']
            helix_r = np.sqrt(helix_x**2 + helix_y**2)
            paths['helix_path_x'].append(helix_x)
            paths['helix_path_y'].append(helix_y)
            paths['helix_path_z'].append(helix_z)
            paths['helix_path_r'].append(helix_r)
            
            # Show current track position (red star) - no path line
            # Transform coordinates: x->y, y->z, z->x
            ax_3d.scatter([helix_y], [helix_z], [helix_x], c='red', s=200, marker='*', 
                      label='Track position' if local_frame_idx == 0 else '', 
                      edgecolors='white', linewidths=2)
            
            # Plot track position in 2D RZ (z on x, r on y)
            if len(paths['helix_path_r']) > 1:
                ax_2d.plot(paths['helix_path_z'], paths['helix_path_r'], 'r-', 
                          linewidth=2, alpha=0.7, label='Track position' if len(paths['helix_path_r']) == 2 else '')
            ax_2d.scatter([helix_z], [helix_r], c='red', s=150, marker='*', 
                         label='Track position' if local_frame_idx == 0 else '', 
                         edgecolors='white', linewidths=2)
            
            # Plot compatible hits in 3D
            comp_hits = frame['comp_hits']
            if len(comp_hits) > 0:
                comp_x = comp_hits['x'].values
                comp_y = comp_hits['y'].values
                comp_z = comp_hits['z'].values
                comp_r = np.sqrt(comp_x**2 + comp_y**2)
                
                # Color code compatible hits - separate correct and wrong
                correct_comp_hits = []
                wrong_comp_hits = []
                correct_comp_r = []
                correct_comp_z = []
                wrong_comp_r = []
                wrong_comp_z = []
                
                for idx, hit_id in enumerate(comp_hits['hit_id'].values):
                    if int(hit_id) in frame['correct_hits_in_layer']:
                        correct_comp_hits.append(idx)
                        correct_comp_r.append(comp_r[idx])
                        correct_comp_z.append(comp_z[idx])
                    else:
                        wrong_comp_hits.append(idx)
                        wrong_comp_r.append(comp_r[idx])
                        wrong_comp_z.append(comp_z[idx])
                
                # Plot correct compatible hits - make them more visible
                # Transform coordinates: x->y, y->z, z->x
                if len(correct_comp_hits) > 0:
                    correct_indices = comp_hits.index[correct_comp_hits]
                    ax_3d.scatter(comp_hits.loc[correct_indices, 'y'], 
                                comp_hits.loc[correct_indices, 'z'],
                                comp_hits.loc[correct_indices, 'x'],
                                c='green', s=100, marker='^', 
                                label='Compatible (correct)', alpha=0.85, edgecolors='white', linewidths=1, zorder=2)
                
                # Plot wrong compatible hits - make them more visible
                # Transform coordinates: x->y, y->z, z->x
                if len(wrong_comp_hits) > 0:
                    wrong_indices = comp_hits.index[wrong_comp_hits]
                    ax_3d.scatter(comp_hits.loc[wrong_indices, 'y'], 
                                comp_hits.loc[wrong_indices, 'z'],
                                comp_hits.loc[wrong_indices, 'x'],
                                c='gray', s=100, marker='^', 
                                label='Compatible (wrong)', alpha=0.75, edgecolors='white', linewidths=1, zorder=2)
                
                if highlight_set:
                    highlight_indices = []
                    for idx, hit_id in zip(comp_hits.index, comp_hits['hit_id'].values):
                        try:
                            if int(hit_id) in highlight_set:
                                highlight_indices.append(idx)
                        except Exception:
                            continue
                    if len(highlight_indices) > 0:
                        ax_3d.scatter(
                            comp_hits.loc[highlight_indices, 'y'],
                            comp_hits.loc[highlight_indices, 'z'],
                            comp_hits.loc[highlight_indices, 'x'],
                            c='yellow',
                            s=160,
                            marker='o',
                            label='Highlighted hits',
                            alpha=0.95,
                            edgecolors='black',
                            linewidths=1.2,
                            zorder=5,
                        )
                        highlight_r = np.sqrt(
                            comp_hits.loc[highlight_indices, 'x']**2
                            + comp_hits.loc[highlight_indices, 'y']**2
                        )
                        highlight_z = comp_hits.loc[highlight_indices, 'z']
                        ax_2d.scatter(
                            highlight_z,
                            highlight_r,
                            c='yellow',
                            s=180,
                            marker='o',
                            alpha=0.95,
                            edgecolors='black',
                            linewidths=1.5,
                            label='Highlighted hits',
                            zorder=5,
                        )
            
            # Plot selected hit in 3D
            if frame['selected_hit'] is not None:
                selected_hit = frame['selected_hit']
                sel_x = float(selected_hit['x'])
                sel_y = float(selected_hit['y'])
                sel_z = float(selected_hit['z'])
                sel_r = np.sqrt(sel_x**2 + sel_y**2)
                
                paths['selected_path_x'].append(sel_x)
                paths['selected_path_y'].append(sel_y)
                paths['selected_path_z'].append(sel_z)
                paths['selected_path_r'].append(sel_r)
                
                # Only show last 2-3 selected hits for zoomed view
                # Ensure we have the path in the correct order (most recent last)
                recent_sel_path_x = paths['selected_path_x'][-3:] if len(paths['selected_path_x']) > 3 else paths['selected_path_x']
                recent_sel_path_y = paths['selected_path_y'][-3:] if len(paths['selected_path_y']) > 3 else paths['selected_path_y']
                recent_sel_path_z = paths['selected_path_z'][-3:] if len(paths['selected_path_z']) > 3 else paths['selected_path_z']
                
                # Only plot if we have at least 2 points and they're in order
                if len(recent_sel_path_x) > 1:
                    # Transform coordinates: x->y, y->z, z->x
                    # Plot the path in the order it was selected (oldest to newest)
                    ax_3d.plot(recent_sel_path_y, recent_sel_path_z, recent_sel_path_x, 
                              'purple', linewidth=3, alpha=0.7, label='Selected path')
                
                # Color and marker based on correctness - add to legend properly
                sel_color = 'green' if frame['selected_hit_correct'] else 'red'
                sel_marker = 'D' if frame['selected_hit_correct'] else 'X'
                sel_label = 'Selected (correct)' if frame['selected_hit_correct'] else 'Selected (wrong)'
                if highlight_set and frame['selected_hit_id'] is not None and frame['selected_hit_id'] in highlight_set:
                    sel_color = 'yellow'
                    sel_marker = 'o'
                    sel_label = 'Selected (highlighted)'
                # Use smaller marker size so other hits are more visible
                # Transform coordinates: x->y, y->z, z->x
                ax_3d.scatter([sel_y], [sel_z], [sel_x], c=sel_color, s=150, marker=sel_marker,
                          label=sel_label, edgecolors='white', linewidths=1.5)
                
                # Plot selected path in 2D RZ (z on x, r on y)
                if len(paths['selected_path_r']) > 1:
                    ax_2d.plot(paths['selected_path_z'], paths['selected_path_r'], 
                              'purple', linewidth=2.5, alpha=0.8, label='Selected path')
                ax_2d.scatter([sel_z], [sel_r], c=sel_color, s=180, marker=sel_marker,
                            label=sel_label, edgecolors='white', linewidths=2)
            
            # Set labels and titles with even larger font
            # Coordinate system: x axis shows old y, y axis shows old z, z axis shows old x
            ax_3d.set_xlabel('Y (cm)', fontsize=16, color='white')
            ax_3d.set_ylabel('Z (cm)', fontsize=16, color='white')
            ax_3d.set_zlabel('X (cm)', fontsize=16, color='white')
            ax_3d.set_title(f'3D View - Track {current_track_idx+1}, Step {frame["step"]} (Layer {current_layer:.0f})', 
                          fontsize=16, color='white')
            ax_3d.tick_params(colors='white', labelsize=14)
            
            ax_2d.set_xlabel('Z (cm)', fontsize=16, color='white')
            ax_2d.set_ylabel('R (cm)', fontsize=16, color='white')
            ax_2d.set_title('R-Z View (All Layers)', fontsize=16, color='white')
            ax_2d.set_aspect('auto')
            ax_2d.grid(True, alpha=0.3, color='gray')
            ax_2d.tick_params(labelsize=14, colors='white')
            
            # Set zoomed-in view that shows only last 1-2 hits and compatible hits
            helix_x, helix_y, helix_z = frame['helix_pos']
            
            # Collect only recent points (last 2 helix positions + compatible hits + nearby truth)
            # Use transformed coordinates: x->y, y->z, z->x
            all_relevant_y = []  # old y values (will be new x axis)
            all_relevant_z = []  # old z values (will be new y axis)
            all_relevant_x = []  # old x values (will be new z axis)
            
            # Add last 2 helix positions
            if len(paths['helix_path_x']) > 0:
                recent_helix = paths['helix_path_x'][-2:]
                recent_helix_y = paths['helix_path_y'][-2:]
                recent_helix_z = paths['helix_path_z'][-2:]
                all_relevant_y.extend(recent_helix_y)
                all_relevant_z.extend(recent_helix_z)
                all_relevant_x.extend(recent_helix)
            
            # Add last 2 selected hits
            if len(paths['selected_path_x']) > 0:
                recent_sel = paths['selected_path_x'][-2:]
                recent_sel_y = paths['selected_path_y'][-2:]
                recent_sel_z = paths['selected_path_z'][-2:]
                all_relevant_y.extend(recent_sel_y)
                all_relevant_z.extend(recent_sel_z)
                all_relevant_x.extend(recent_sel)
            
            # Add current compatible hits (these are the key ones to see)
            if len(comp_hits) > 0:
                all_relevant_y.extend(comp_hits['y'].values)
                all_relevant_z.extend(comp_hits['z'].values)
                all_relevant_x.extend(comp_hits['x'].values)
            
            # Add very nearby truth hits only (tight tolerance for zoomed view)
            if len(truth_hits) > 0:
                tolerance = 10.0  # cm - very tight to see close hits
                nearby_truth = truth_hits[
                    (np.abs(truth_hits['x'] - helix_x) < tolerance) &
                    (np.abs(truth_hits['y'] - helix_y) < tolerance) &
                    (np.abs(truth_hits['z'] - helix_z) < tolerance)
                ]
                if len(nearby_truth) > 0:
                    all_relevant_y.extend(nearby_truth['y'].values)
                    all_relevant_z.extend(nearby_truth['z'].values)
                    all_relevant_x.extend(nearby_truth['x'].values)
            
            # Calculate zoom range to include all relevant points
            if len(all_relevant_x) > 0:
                all_relevant_x = np.array(all_relevant_x)
                all_relevant_y = np.array(all_relevant_y)
                all_relevant_z = np.array(all_relevant_z)
                
                # Add minimal padding for very tight zoom
                padding = 5.0  # cm padding - very minimal for tight zoom
                x_range = all_relevant_x.max() - all_relevant_x.min()
                y_range = all_relevant_y.max() - all_relevant_y.min()
                z_range = all_relevant_z.max() - all_relevant_z.min()
                
                # Use larger range to ensure everything is visible
                max_range = max([x_range, y_range, z_range]) / 2.0 + padding
                if max_range < 8.0:  # Minimum zoom range (very tight to see close hits)
                    max_range = 8.0
                if max_range > 15.0:  # Maximum zoom range (keep it very tight)
                    max_range = 15.0
                
                # Center on current helix position (follows the track)
                # Transform coordinates: x->y, y->z, z->x
                mid_x = helix_x  # old x -> new z
                mid_y = helix_y  # old y -> new x
                mid_z = helix_z  # old z -> new y
                
                ax_3d.set_xlim(mid_y - max_range, mid_y + max_range)  # new x axis = old y
                ax_3d.set_ylim(mid_z - max_range, mid_z + max_range)  # new y axis = old z
                ax_3d.set_zlim(mid_x - max_range, mid_x + max_range)  # new z axis = old x
            else:
                # Fallback to helix-centered view
                zoom_range = 10.0  # Very tight zoom to see close hits
                ax_3d.set_xlim(helix_y - zoom_range, helix_y + zoom_range)  # new x = old y
                ax_3d.set_ylim(helix_z - zoom_range, helix_z + zoom_range)  # new y = old z
                ax_3d.set_zlim(helix_x - zoom_range, helix_x + zoom_range)  # new z = old x
            
            # Set 2D RZ view limits to show all layers and hits (z on x, r on y)
            if len(truth_hits) > 0:
                truth_r = np.sqrt(truth_hits['x']**2 + truth_hits['y']**2)
                truth_z = truth_hits['z'].values
                
                # Include all hits and layer boundaries
                all_r = list(truth_r)
                all_z = list(truth_z)
                
                # Add layer boundaries
                for layer_id in self.layer_info.index:
                    try:
                        layer_row = self.layer_info.loc[layer_id]
                        all_r.extend([layer_row[('r', 'min')], layer_row[('r', 'max')]])
                        all_z.extend([layer_row[('z', 'min')], layer_row[('z', 'max')]])
                    except:
                        pass
                
                if len(all_r) > 0:
                    r_padding = (max(all_r) - min(all_r)) * 0.1
                    z_padding = (max(all_z) - min(all_z)) * 0.1
                    # Swap: z on x-axis, r on y-axis
                    ax_2d.set_xlim(min(all_z) - z_padding, max(all_z) + z_padding)
                    ax_2d.set_ylim(min(all_r) - r_padding, max(all_r) + r_padding)
            
            # Update stats text - position on right side to avoid overlap
            stats_str = (
                f"Step: {frame['step']}\n"
                f"Layer: {current_layer:.0f}\n"
                f"Accuracy: {frame['accuracy']:.3f}\n"
                f"Efficiency: {frame['efficiency']:.3f}\n"
                f"Purity: {frame['purity']:.3f}\n"
                f"Correct: {frame['correct_hits']}/{frame['predicted_hits']}\n"
                f"Layers: {frame['layers_with_correct']}/{frame['expected_hits']}\n"
                f"Selected: {'✓' if frame['selected_hit_correct'] else '✗'}"
            )
            # Stats text with larger font and white text on dark background
            ax_3d.text2D(0.98, 0.98, stats_str, transform=ax_3d.transAxes, 
                    fontsize=13, verticalalignment='top', horizontalalignment='right',
                    color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white', linewidth=1))
            
            # Position legend on left side, below stats - larger font and fix marker sizes
            # White text on dark background for legends
            ax_3d.legend(loc='upper left', bbox_to_anchor=(0, 0.85), fontsize=14, 
                        markerscale=0.7, framealpha=0.9, facecolor='black', edgecolor='white',
                        labelcolor='white')
            ax_2d.legend(loc='upper right', fontsize=14, markerscale=0.7, framealpha=0.9,
                        facecolor='black', edgecolor='white', labelcolor='white')
            
            frame_counter += 1
        
        # Create animation - loop through all frames
        anim = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, repeat=True)
        
        # Save animation
        print(f"Saving animation to {output_path}...")
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        print(f"Animation saved!")
        
        return anim


def main():
    parser = argparse.ArgumentParser(description='Animate track reconstruction')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: look in checkpoint directory)')
    parser.add_argument('--track', type=int, default=0,
                       help='Track index to animate (default: 0)')
    parser.add_argument('--output', type=str, default='track_animation.gif',
                       help='Output animation file (default: track_animation.gif)')
    parser.add_argument('--fps', type=int, default=2,
                       help='Frames per second (default: 2)')
    parser.add_argument('--num_tracks', '--num_particles', type=int, default=3,
                       help='Number of tracks/particles to animate (default: 3)')
    parser.add_argument('--highlight-hit-ids', type=str, default=None,
                       help='Path to file containing hit IDs to highlight (JSON array or comma/space-separated text)')
    
    args = parser.parse_args()
    
    highlight_hit_ids = None
    if args.highlight_hit_ids:
        highlight_path = Path(args.highlight_hit_ids)
        try:
            text = highlight_path.read_text(encoding='utf-8').strip()
            if not text:
                highlight_hit_ids = []
            elif highlight_path.suffix.lower() == '.json':
                highlight_hit_ids = json.loads(text)
            else:
                tokens = [tok for tok in re.split(r'[\s,]+', text) if tok]
                highlight_hit_ids = [int(tok) for tok in tokens]
        except Exception as exc:
            print(f"Failed to load highlight hit IDs from {highlight_path}: {exc}")
            highlight_hit_ids = []
    
    # Create animator
    animator = TrackAnimator(args.model, args.config, highlight_hit_ids=highlight_hit_ids)
    
    # Collect track data
    print(f"Collecting data for track {args.track}...")
    frames = animator.collect_track_data(args.track)
    
    if frames is None or len(frames) == 0:
        print("Failed to collect track data. Trying track 0...")
        frames = animator.collect_track_data(0)
    
    if frames is None or len(frames) == 0:
        print("No track data available.")
        return
    
    print(f"Collected {len(frames)} steps. Creating animation...")
    
    # Create animation
    animator.animate(args.output, fps=args.fps, num_tracks=args.num_tracks)


if __name__ == '__main__':
    main()

