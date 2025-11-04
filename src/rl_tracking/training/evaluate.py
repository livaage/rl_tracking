"""Evaluation script for trained DQN model on test set - track-by-track evaluation."""
from pathlib import Path
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from rl_tracking.lightning_modules.dqn import DQNLightning
from rl_tracking.preprocessing.hit_candidates import EventProcessor
from rl_tracking.utils.stream_loading import TrackingDataModule
from rl_tracking.utils.config_loader import load_config
from rl_tracking.environment.tracking_env import TrackingEnv
from rl_tracking.environment.agent import Agent
from rl_tracking.replay.buffer import ReplayBuffer
from rl_tracking.physics.hit_holder import HitHolder
import pandas as pd


def evaluate_model(model_path: str, config_path: str = None, test_data_dir: str = None):
    """Evaluate a trained model on the test set track-by-track.
    
    Args:
        model_path: Path to the trained model checkpoint (.ckpt file)
        config_path: Optional path to the training configuration YAML file.
                     If None, will try to load from checkpoint directory.
        test_data_dir: Optional path to test data directory (if different from config)
    """
    # If config_path not provided, try to find it in checkpoint directory
    if config_path is None:
        checkpoint_dir = Path(model_path).parent
        # Look for config file in checkpoint directory
        config_files = list(checkpoint_dir.glob('*.yaml')) + list(checkpoint_dir.glob('*.yml'))
        if config_files:
            config_path = str(config_files[0])
            print(f"\n{'='*60}")
            print(f"Found config file in checkpoint directory: {config_path}")
            print(f"Checkpoint directory: {checkpoint_dir}")
            print(f"Model path: {model_path}")
            print(f"{'='*60}\n")
        else:
            raise ValueError(
                f"No config file found in checkpoint directory {checkpoint_dir}. "
                f"Please provide --config argument."
            )
    else:
        print(f"\n{'='*60}")
        print(f"Using provided config file: {config_path}")
        print(f"Model path: {model_path}")
        print(f"{'='*60}\n")
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract data configuration
    data_config = config.get('data', {})
    if test_data_dir is None:
        test_data_dir = data_config.get('data_dir', '/Users/liv/trackML/train_1/')
    
    # Convert test_data_dir to Path if it's a string
    test_data_dir = Path(test_data_dir)
    
    batch_size = data_config.get('batch_size', 32)
    val_split = data_config.get('val_split', 0.2)
    test_split = data_config.get('test_split', 0.1)
    num_workers = data_config.get('num_workers', 0)
    event_processor_config = data_config.get('event_processor', {})
    n_neighbors = event_processor_config.get('n_neighbors', 20)
    
    # Initialize dataset
    ep = EventProcessor(test_data_dir, n_neighbors)
    dm = TrackingDataModule(
        file_paths=test_data_dir,
        batch_size=batch_size,
        event_processor=ep,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split
    )
    
    # Extract model and environment configuration
    model_config = config.get('model', {})
    environment_config = config.get('environment', {})
    particle_filters = environment_config.get('particle_filters', {})
    hit_filters = environment_config.get('hit_filters', {})
    use_distance_reward = environment_config.get('use_distance_reward', False)
    use_truth_path_plan = environment_config.get('use_truth_path_plan', False)
    
    # Ensure particle_filters and hit_filters are dictionaries (not None)
    # Empty dict {} means "no filtering", None means "use defaults"
    if particle_filters is None:
        # Default particle filters (same as training defaults) - only if not specified at all
        particle_filters = {'pt': 1, 'nhits_min': 3, 'nhits_max': 20}
        print(f"WARNING: particle_filters not specified in config, using defaults: {particle_filters}")
    elif particle_filters == {}:
        print(f"INFO: particle_filters is empty dict - no particle filtering will be applied")
    if hit_filters is None:
        hit_filters = {}
    
    # Verify filters are loaded from config
    print(f"\n{'='*60}")
    print(f"Configuration Summary (BEFORE loading checkpoint)")
    print(f"{'='*60}")
    print(f"Config file: {config_path}")
    print(f"Particle filters: {particle_filters}")
    print(f"Hit filters: {hit_filters}")
    print(f"Use distance reward: {use_distance_reward}")
    print(f"Use truth path plan: {use_truth_path_plan}")
    print(f"{'='*60}\n")
    
    # Setup for fit first (needed for model initialization which accesses train_dataset)
    # Then we'll switch to test for evaluation
    dm.setup(stage="fit")
    
    # Load model from checkpoint
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*60}\n")
    
    # Try to load hyperparameters from checkpoint first (most accurate)
    # If checkpoint has hyperparameters, prefer those over config file
    checkpoint_hparams = None
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'hyper_parameters' in checkpoint:
            checkpoint_hparams = checkpoint['hyper_parameters']
            print(f"Found hyperparameters in checkpoint:")
            print(f"  particle_filters: {checkpoint_hparams.get('particle_filters', 'N/A')}")
            print(f"  hit_filters: {checkpoint_hparams.get('hit_filters', 'N/A')}")
            print(f"  use_distance_reward: {checkpoint_hparams.get('use_distance_reward', 'N/A')}")
            print(f"  use_truth_path_plan: {checkpoint_hparams.get('use_truth_path_plan', 'N/A')}")
            print(f"\nUsing hyperparameters from checkpoint (most accurate)\n")
            # Use checkpoint hyperparameters if available
            particle_filters = checkpoint_hparams.get('particle_filters', particle_filters)
            hit_filters = checkpoint_hparams.get('hit_filters', hit_filters)
            use_distance_reward = checkpoint_hparams.get('use_distance_reward', use_distance_reward)
            use_truth_path_plan = checkpoint_hparams.get('use_truth_path_plan', use_truth_path_plan)
    except Exception as e:
        print(f"Could not load hyperparameters from checkpoint: {e}")
        print(f"Using config file values instead\n")
    
    model = DQNLightning.load_from_checkpoint(
        model_path,
        dm=dm,
        batch_size=model_config.get('batch_size', 32),
        lr=model_config.get('lr', 0.01),
        gamma=model_config.get('gamma', 0.99),
        sync_rate=model_config.get('sync_rate', 10),
        replay_size=model_config.get('replay_size', 1000),
        warm_start_size=model_config.get('warm_start_size', 1000),
        eps_last_frame=model_config.get('eps_last_frame', 1000),
        eps_start=model_config.get('eps_start', 1.0),
        eps_end=model_config.get('eps_end', 0.01),
        episode_length=model_config.get('episode_length', 200),
        warm_start_steps=model_config.get('warm_start_steps', 1000),
        particle_filters=particle_filters,
        hit_filters=hit_filters,
        use_distance_reward=use_distance_reward,
        use_truth_path_plan=use_truth_path_plan,
    )
    
    # Print final configuration that will be used for evaluation
    print(f"\n{'='*60}")
    print(f"Final Configuration for Evaluation")
    print(f"{'='*60}")
    print(f"Particle filters (used for test): {particle_filters}")
    print(f"Hit filters (used for test): {hit_filters}")
    print(f"Use distance reward: {use_distance_reward}")
    print(f"Use truth path plan: {use_truth_path_plan}")
    print(f"{'='*60}\n")
    
    # Now setup for test to get test dataset
    dm.setup(stage="test")
    
    # Set model to evaluation mode
    model.eval()
    model.net.eval()
    model.target_net.eval()
    
    # Create test environment
    test_env_dataloader = DataLoader(
        dm.test_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    test_env = TrackingEnv(
        test_env_dataloader,
        particle_filters=particle_filters,
        hit_filters=hit_filters,
        use_distance_reward=use_distance_reward,
        use_truth_path_plan=use_truth_path_plan
    )
    
    # Create test agent (no replay buffer needed for evaluation)
    test_buffer = ReplayBuffer(100)  # Small buffer, not used
    test_agent = Agent(test_env, test_buffer)
    
    # Determine device - check for CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Move model to device
    model = model.to(device)
    model.net = model.net.to(device)
    model.target_net = model.target_net.to(device)
    
    print(f"\n{'='*60}")
    print(f"Evaluating model on test set (track-by-track)")
    print(f"Particle filters: {particle_filters}")
    print(f"Hit filters: {hit_filters}")
    print(f"Using device: {device}")
    print(f"{'='*60}\n")
    
    # Track-by-track evaluation metrics
    track_results = []
    total_tracks = 0
    total_time = 0.0
    
    # Track efficiency metrics
    total_correct_hits = 0
    total_predicted_hits = 0
    total_expected_hits = 0
    
    epsilon = 0.0  # Pure exploitation - no exploration
    max_tracks = 1000  # Limit evaluation to reasonable number
    
    try:
        tracks_skipped = 0
        consecutive_none = 0
        while total_tracks < max_tracks:
            # Reset environment (starts new track)
            obs, info = test_env.reset()
            if obs is None:
                consecutive_none += 1
                if consecutive_none == 1:
                    # First time getting None - check if we have any particles
                    print(f"\nDEBUG: Got None from reset(). Checking environment state...")
                    print(f"  pids_to_explore: {len(test_env.pids_to_explore) if hasattr(test_env, 'pids_to_explore') else 'N/A'}")
                    print(f"  particle_index: {test_env.particle_index if hasattr(test_env, 'particle_index') else 'N/A'}")
                    print(f"  all_hits: {test_env.all_hits is not None if hasattr(test_env, 'all_hits') else 'N/A'}")
                if consecutive_none >= 3:
                    print(f"\nNo more tracks available after {consecutive_none} consecutive None resets.")
                    print(f"Evaluated {total_tracks} tracks, skipped {tracks_skipped} tracks.")
                    break  # No more events
                continue
            
            # Reset consecutive_none counter if we got a valid observation
            consecutive_none = 0
            
            # Check if track has valid data
            if not hasattr(test_env, 'current_track') or test_env.current_track is None or len(test_env.current_track) == 0:
                tracks_skipped += 1
                if tracks_skipped <= 5:
                    print(f"WARNING: Track {total_tracks + 1} has no data, skipping...")
                continue
            
            if not hasattr(test_env, 'helix') or test_env.helix is None:
                tracks_skipped += 1
                if tracks_skipped <= 5:
                    print(f"WARNING: Track {total_tracks + 1} has no helix, skipping...")
                continue
            
            # Track reconstruction metrics for this track
            track_start_time = time.time()
            track_correct_hits = 0
            track_predicted_hits = 0
            track_expected_hits = 0
            track_reward = 0.0
            track_steps = 0
            
            # Track layers where we found at least one correct hit (for efficiency calculation)
            # Efficiency: count one hit per layer, not all hits
            layers_with_correct_hit = set()
            selected_hit_ids = []
            correct_hit_ids = set()
            seed_hit_ids = set()
            termination_reason = 'unknown'  # Will be updated when track completes
            
            # Get track properties and seed hits
            track_pt = None
            track_eta = None
            if hasattr(test_env, 'current_track') and test_env.current_track is not None:
                # Get all correct hit IDs for this track
                correct_hit_ids = set(test_env.current_track['hit_id'].values)
                
                # Get seed hits (first 3 hits from first 3 layers) - these should be excluded
                if hasattr(test_env, 'helix') and test_env.helix is not None:
                    if hasattr(test_env.helix, 'seed_hits') and test_env.helix.seed_hits is not None:
                        # Get seed hit IDs - seed_hits is a DataFrame
                        seed_df = test_env.helix.seed_hits
                        if isinstance(seed_df, pd.DataFrame):
                            if 'hit_id' in seed_df.columns:
                                seed_hit_ids = set(seed_df['hit_id'].values.astype(int))
                            else:
                                # Try index if hit_id not in columns
                                seed_hit_ids = set(seed_df.index.astype(int))
                        else:
                            # If it's not a DataFrame, try to get hit_id attribute
                            seed_hit_ids = set()
                    
                    # Get track pt and eta from helix
                    if hasattr(test_env.helix, 'pt'):
                        track_pt = float(test_env.helix.pt)
                    if hasattr(test_env.helix, 'theta'):
                        # Convert theta to eta: eta = -ln(tan(theta/2))
                        theta = test_env.helix.theta
                        if theta > 0 and theta < np.pi:
                            track_eta = -np.log(np.tan(theta / 2.0))
                else:
                    # Fallback: assume first 3 hits in track are seed hits
                    # This is less accurate but works if helix.seed_hits is not accessible
                    if len(test_env.current_track) >= 3:
                        # Sort by layer to get first 3 layers
                        track_sorted = test_env.current_track.sort_values('unique_layer_id')
                        seed_hit_ids = set(track_sorted.head(3)['hit_id'].values.astype(int))
                
                # Expected hits: ALL hits after the seed (not just those in path_plan)
                # Start: After the last seed layer (all seed hits are in first 3 layers)
                # End: Include the last hit in the track
                # Use initial_start_layer (the final seed layer) to determine what "after seed" means
                if hasattr(test_env, 'helix') and test_env.helix is not None:
                    # Get initial starting layer (final seed layer) - this doesn't change as we progress
                    initial_start_layer = getattr(test_env.helix, 'initial_start_layer', None)
                    if initial_start_layer is None:
                        # Fallback: try to get from seed_hits (last seed layer)
                        if hasattr(test_env.helix, 'seed_hits') and test_env.helix.seed_hits is not None:
                            seed_df = test_env.helix.seed_hits
                            if isinstance(seed_df, pd.DataFrame) and 'unique_layer_id' in seed_df.columns:
                                seed_layers = seed_df['unique_layer_id'].values
                                initial_start_layer = float(max(seed_layers)) if len(seed_layers) > 0 else None
                    
                    # Get ALL truth hits after the seed (not just those in path_plan)
                    # Filter current_track to only include hits from layers AFTER the seed
                    if initial_start_layer is not None:
                        initial_start_layer_float = float(initial_start_layer)
                        # Get all hits from layers after the seed layer
                        track_hits_after_seed = test_env.current_track[
                            test_env.current_track['unique_layer_id'] > initial_start_layer_float
                        ]
                        all_truth_hits_after_seed = set(track_hits_after_seed['hit_id'].values)
                        
                        # Also get hits only from path_plan layers (for comparison/debugging)
                        # This shows how many truth hits are also in the path_plan (but don't use for efficiency)
                        if hasattr(test_env.helix, 'path_plan') and test_env.helix.path_plan is not None:
                            all_path_layers = [float(layer) for layer in test_env.helix.path_plan]
                            path_plan_layers = [layer for layer in all_path_layers if float(layer) > initial_start_layer_float]
                            path_plan_layers_set = set(path_plan_layers)
                            track_hits_in_path_plan = test_env.current_track[
                                (test_env.current_track['unique_layer_id'] > initial_start_layer_float) &
                                (test_env.current_track['unique_layer_id'].isin(path_plan_layers_set))
                            ]
                            path_plan_hits_after_seed = set(track_hits_in_path_plan['hit_id'].values)
                        else:
                            path_plan_hits_after_seed = all_truth_hits_after_seed
                        
                        # Expected hits: ALWAYS use ALL truth hits in the particle track (after seed)
                        # This ensures efficiency reflects how many of the actual truth hits were found,
                        # regardless of whether the model was trained with truth path plan or lookup path plan
                        correct_hit_ids_in_path = all_truth_hits_after_seed
                        expected_hits_source = "all truth hits in particle track (after seed)"
                        
                        # Debug logging for first few tracks
                        if total_tracks < 3:
                            print(f"DEBUG Track {total_tracks}: initial_start_layer={initial_start_layer}, "
                                  f"total_hits_in_track={len(test_env.current_track)}, "
                                  f"all_truth_hits_after_seed={len(all_truth_hits_after_seed)}, "
                                  f"path_plan_hits_after_seed={len(path_plan_hits_after_seed)}, "
                                  f"using: {expected_hits_source}, "
                                  f"seed_hit_ids={len(seed_hit_ids)}")
                    else:
                        # Fallback: exclude seed hits by ID
                        correct_hit_ids_in_path = correct_hit_ids - seed_hit_ids
                else:
                    # Fallback: use all hits if helix not available, just exclude seed hits
                    correct_hit_ids_in_path = correct_hit_ids - seed_hit_ids
                
                # Expected layers: Count unique layers with truth hits (after seed)
                # Efficiency = layers where we found at least one correct hit / total layers with truth hits
                # This allows one hit per layer - if there are double hits and we only find one, that's OK
                if initial_start_layer is not None:
                    truth_hits_after_seed_df = test_env.current_track[
                        test_env.current_track['unique_layer_id'] > initial_start_layer
                    ]
                else:
                    truth_hits_after_seed_df = test_env.current_track[
                        ~test_env.current_track['hit_id'].isin(seed_hit_ids)
                    ]
                
                # Count unique layers with truth hits
                track_expected_hits = len(truth_hits_after_seed_df['unique_layer_id'].unique())
                if track_expected_hits < 0:
                    track_expected_hits = 0
                
                # Get truth-level path (all layers in current_track, excluding seed)
                truth_path_layers = []
                if hasattr(test_env, 'current_track') and test_env.current_track is not None and len(test_env.current_track) > 0:
                    # Get all unique layers from current_track, sorted
                    all_track_layers = sorted(test_env.current_track['unique_layer_id'].unique())
                    # Filter to post-seed layers
                    if initial_start_layer is not None:
                        # Only include layers that come after the seed layer
                        truth_path_layers = [float(layer) for layer in all_track_layers if float(layer) > initial_start_layer]
                    else:
                        # Fallback: exclude seed layers by ID
                        if len(seed_layer_ids) > 0:
                            truth_path_layers = [float(layer) for layer in all_track_layers if layer not in seed_layer_ids]
                        else:
                            # If we can't identify seed layers, assume first 3 layers are seed
                            if len(all_track_layers) > 3:
                                truth_path_layers = [float(layer) for layer in all_track_layers[3:]]
                
                # Debug: if truth_path_layers is empty, it might mean the track only has seed hits
                # This can happen if nhits_min is 3 and all hits are in seed layers
            
            # Track the actual path taken vs expected path
            expected_path_layers = []
            actual_path_layers = []  # Layers where we actually selected hits
            skipped_layers = []  # Layers we visited but found no hits
            # truth_path_layers already computed above (before the loop)
            termination_reason = 'unknown'  # Why the track ended
            seed_layer_ids = set()
            if hasattr(test_env, 'helix') and test_env.helix is not None:
                # Get seed layer IDs from seed_hits
                if hasattr(test_env.helix, 'seed_hits') and test_env.helix.seed_hits is not None:
                    seed_df = test_env.helix.seed_hits
                    if isinstance(seed_df, pd.DataFrame) and 'unique_layer_id' in seed_df.columns:
                        seed_layer_ids = set(seed_df['unique_layer_id'].values.astype(int))
                    elif isinstance(seed_df, pd.DataFrame) and 'unique_layer_id' in seed_df.index:
                        seed_layer_ids = set(seed_df.index.values.astype(int))
                
                # Get the starting layer (final seed layer) - use initial_start_layer which doesn't change
                initial_start_layer = getattr(test_env.helix, 'initial_start_layer', None)
                if initial_start_layer is None:
                    # Fallback: get from current_layer at start (before it advances)
                    initial_start_layer = getattr(test_env.helix, 'current_layer', None)
                
                if hasattr(test_env.helix, 'path_plan') and test_env.helix.path_plan is not None:
                    # Filter path_plan to only include layers AFTER the seed (after initial_start_layer)
                    # initial_start_layer is the final seed layer, so we only want layers > initial_start_layer
                    all_path_layers = list(test_env.helix.path_plan)
                    if initial_start_layer is not None:
                        # Only include layers that come after the starting layer (post-seed)
                        expected_path_layers = [layer for layer in all_path_layers if layer > initial_start_layer]
                    else:
                        # Fallback: exclude seed layers by ID
                        expected_path_layers = [layer for layer in all_path_layers if layer not in seed_layer_ids]
                else:
                    initial_start_layer = None
            
            # Reconstruct track step by step
            done = False
            while not done:
                # Get action (inference timing)
                inference_start = time.time()
                action = test_agent.get_action(model.net, epsilon, device)
                inference_end = time.time()
                
                # Step environment
                env_start = time.time()
                step_result = test_env.step(action)
                env_end = time.time()
                
                # Track propagation state before step
                prev_hit_number = getattr(test_env, 'hit_number', 0)
                had_comp_hits = hasattr(test_env, 'current_comp_hits') and test_env.current_comp_hits is not None and len(test_env.current_comp_hits) > 0
                
                # Handle return format
                if len(step_result) == 3:
                    next_obs, reward, done = step_result
                else:
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                
                # Track why episode ended - distinguish between path complete vs no hits found
                if done:
                    # Check if path_plan is exhausted (end of expected path)
                    path_exhausted = False
                    if hasattr(test_env, 'helix') and test_env.helix is not None:
                        unexplored_layers = test_env.helix.path_plan[test_env.helix.path_plan > test_env.helix.current_layer]
                        path_exhausted = len(unexplored_layers) == 0
                    
                    # Check if we ran out of hits vs propagation failed
                    current_hit_number = getattr(test_env, 'hit_number', 0)
                    track_length = len(test_env.current_track) if hasattr(test_env, 'current_track') and test_env.current_track is not None else 0
                    
                    if path_exhausted:
                        termination_reason = 'path_complete'  # Reached end of path_plan
                    elif current_hit_number >= track_length:
                        termination_reason = 'completed_all_hits'  # All truth hits visited
                    elif next_obs is None:
                        termination_reason = 'no_compatible_hits'  # No hits found in compatible layers
                    else:
                        termination_reason = 'unknown'
                    # Update the outer scope variable
                    termination_reason = termination_reason
                
                # Track metrics
                track_reward += reward
                track_steps += 1
                
                # Get selected hit ID from environment's stored last_selected_hit
                # This is set during step() and represents the hit that was actually selected
                if hasattr(test_env, 'last_selected_hit_id') and test_env.last_selected_hit_id is not None:
                    selected_hit_id = test_env.last_selected_hit_id
                    
                    # Get the layer of the selected hit (not current_layer, which may have advanced)
                    selected_hit_layer = None
                    if hasattr(test_env, 'last_selected_hit') and test_env.last_selected_hit is not None:
                        selected_hit = test_env.last_selected_hit
                        if isinstance(selected_hit, pd.Series) and 'unique_layer_id' in selected_hit.index:
                            selected_hit_layer = float(selected_hit['unique_layer_id'])
                        elif hasattr(selected_hit, 'unique_layer_id'):
                            selected_hit_layer = float(selected_hit.unique_layer_id)
                    
                    # Fallback: use current_layer if we can't get it from the hit
                    if selected_hit_layer is None and hasattr(test_env, 'helix') and test_env.helix is not None:
                        selected_hit_layer = getattr(test_env.helix, 'current_layer', None)
                    
                    # Track which layer we selected a hit from (only post-seed layers)
                    if selected_hit_layer is not None and initial_start_layer is not None:
                        if selected_hit_layer > initial_start_layer:
                            # Only add once per layer
                            if selected_hit_layer not in actual_path_layers:
                                actual_path_layers.append(selected_hit_layer)
                    
                    # Exclude seed hits from predicted hits count
                    if selected_hit_id not in seed_hit_ids:
                        # Avoid double counting if we've already seen this hit
                        if selected_hit_id not in selected_hit_ids:
                            selected_hit_ids.append(selected_hit_id)
                            track_predicted_hits += 1
                            
                            # Check if selected hit is correct (using environment's tracking)
                            # The environment's last_selected_hit_correct already checks if the hit is in the correct hits for that layer
                            # We just need to verify it's not a seed hit (which we already excluded from track_predicted_hits)
                            # Note: We count ALL correct hits after the seed, not just those in path_plan
                            # If the environment says it's correct and it's not a seed hit, count it
                            if test_env.last_selected_hit_correct:
                                track_correct_hits += 1
                                # Track that we found at least one correct hit in this layer
                                if selected_hit_layer is not None and initial_start_layer is not None:
                                    if selected_hit_layer > initial_start_layer:
                                        layers_with_correct_hit.add(selected_hit_layer)
                            
                            # Calculate distance to correct hit for debugging
                            # Use the correct hits from the layer where the hit was actually selected
                            # (stored in environment before it advanced to next layer)
                            distance_to_correct = None
                            if hasattr(test_env, 'last_selected_hit_correct_hits_df') and test_env.last_selected_hit_correct_hits_df is not None:
                                correct_hits_df = test_env.last_selected_hit_correct_hits_df
                                if len(correct_hits_df) > 0:
                                    hit_holder = HitHolder(test_env.all_hits)
                                    distance_to_correct = hit_holder.get_distance_to_correct(selected_hit, correct_hits_df)
                            elif hasattr(test_env, 'helix') and test_env.helix is not None:
                                # Fallback: try to get correct hits for the selected hit's layer
                                if selected_hit_layer is not None:
                                    correct_hits_df = test_env.helix.get_correct_hits_df_in_layer(selected_hit_layer)
                                    if correct_hits_df is not None and len(correct_hits_df) > 0:
                                        hit_holder = HitHolder(test_env.all_hits)
                                        distance_to_correct = hit_holder.get_distance_to_correct(selected_hit, correct_hits_df)
                            
                            # Log distance for debugging (for first few tracks or incorrect selections)
                            if total_tracks < 5 or not test_env.last_selected_hit_correct:
                                layer_info = f"layer={selected_hit_layer}" if selected_hit_layer is not None else "layer=unknown"
                                correct_info = "CORRECT" if test_env.last_selected_hit_correct else "WRONG"
                                distance_info = f"distance={distance_to_correct:.3f}cm" if distance_to_correct is not None and distance_to_correct != np.inf else "distance=inf"
                                
                                # Show how many correct hits are in this layer (for debugging)
                                num_correct_in_layer = 0
                                if hasattr(test_env, 'last_selected_hit_correct_hits_df') and test_env.last_selected_hit_correct_hits_df is not None:
                                    num_correct_in_layer = len(test_env.last_selected_hit_correct_hits_df)
                                elif hasattr(test_env, 'helix') and test_env.helix is not None and selected_hit_layer is not None:
                                    correct_hits_df = test_env.helix.get_correct_hits_df_in_layer(selected_hit_layer)
                                    if correct_hits_df is not None:
                                        num_correct_in_layer = len(correct_hits_df)
                                
                                correct_hits_info = f"({num_correct_in_layer} correct hits in layer)" if num_correct_in_layer > 0 else "(no correct hits in layer)"
                                print(f"  Track {total_tracks}, Step {track_steps}: Selected hit_id={selected_hit_id} in {layer_info}, {correct_info}, {distance_info} {correct_hits_info}")
                else:
                    # No hit selected - track which layer we skipped (only post-seed layers)
                    # Only count as skipped if we're in a layer that's in the expected path
                    # and we haven't already selected a hit from this layer
                    if hasattr(test_env, 'helix') and test_env.helix is not None:
                        current_layer = getattr(test_env.helix, 'current_layer', None)
                        if current_layer is not None and initial_start_layer is not None:
                            # Only count if it's a post-seed layer in the expected path
                            # and we haven't already selected a hit from this layer
                            if current_layer > initial_start_layer and current_layer in expected_path_layers:
                                # Only mark as skipped if we haven't already selected a hit from this layer
                                if current_layer not in actual_path_layers:
                                    if current_layer not in skipped_layers:
                                        skipped_layers.append(current_layer)
                
                obs = next_obs
                
                if done:
                    break
            
            # Track complete - compute metrics
            track_time = time.time() - track_start_time
            total_time += track_time
            
            # Use environment's accuracy tracking (should match training)
            # This is the same calculation as training: correct_selections / total_selections
            if hasattr(test_env, 'total_selections') and test_env.total_selections > 0:
                track_accuracy = test_env.correct_selections / test_env.total_selections
                
                # Debug: Compare with our manual tracking to identify discrepancies
                if total_tracks < 5:
                    manual_accuracy = track_correct_hits / track_predicted_hits if track_predicted_hits > 0 else 0.0
                    print(f"DEBUG Track {total_tracks} accuracy comparison:")
                    print(f"  Env accuracy (matches training): {track_accuracy:.3f} ({test_env.correct_selections}/{test_env.total_selections})")
                    print(f"  Manual accuracy (purity): {manual_accuracy:.3f} ({track_correct_hits}/{track_predicted_hits})")
                    if abs(track_accuracy - manual_accuracy) > 0.01:
                        print(f"  WARNING: Mismatch! Env counts may include seed hits or double-counting.")
                
                # Reset for next track
                test_env.total_selections = 0
                test_env.correct_selections = 0
            else:
                # Fallback: compute from our tracking
                track_accuracy = track_correct_hits / track_predicted_hits if track_predicted_hits > 0 else 0.0
                if total_tracks < 5:
                    print(f"DEBUG Track {total_tracks}: Using fallback accuracy (env counters not available)")
            
            # Track efficiency: fraction of expected layers where we found at least one correct hit
            # Efficiency = layers with correct hit / total layers with truth hits
            # This allows one hit per layer - if there are double hits and we only find one, that's OK
            track_efficiency = len(layers_with_correct_hit) / track_expected_hits if track_expected_hits > 0 else 0.0
            
            # Purity: fraction of predicted hits that were correct
            track_purity = track_correct_hits / track_predicted_hits if track_predicted_hits > 0 else 0.0
            
            # Log path comparison for first few tracks or tracks with issues
            if total_tracks < 5 or track_efficiency < 0.5 or track_efficiency > 1.0 or len(truth_path_layers) == 0:
                print(f"\n{'='*60}")
                print(f"Track {total_tracks} Path Analysis:")
                print(f"Truth-level path (all layers in current_track, post-seed): {truth_path_layers if len(truth_path_layers) > 0 else '[] (empty - track may only have seed hits)'}")
                print(f"Expected path (path_plan, post-seed): {expected_path_layers}")
                print(f"Actual layers where hits were selected: {actual_path_layers}")
                print(f"Layers visited but no hits found: {skipped_layers}")
                print(f"Termination reason: {termination_reason}")
                print(f"Truth layers: {len(truth_path_layers)}, Expected layers: {len(expected_path_layers)}, "
                      f"Layers with hits: {len(actual_path_layers)}, Layers without hits: {len(skipped_layers)}")
                print(f"Expected layers: {track_expected_hits}, Predicted hits: {track_predicted_hits}, Correct hits: {track_correct_hits}")
                print(f"Layers with correct hit: {len(layers_with_correct_hit)}")
                print(f"Efficiency: {track_efficiency:.3f}, Purity: {track_purity:.3f}")
                print(f"Note: Efficiency = layers with correct hit / total layers with truth hits (one hit per layer).")
                if len(truth_path_layers) == 0:
                    print(f"WARNING: Truth path is empty! Track may only have seed hits (nhits_min={particle_filters.get('nhits_min', 'unknown')})")
                    print(f"  Total hits in current_track: {len(test_env.current_track) if hasattr(test_env, 'current_track') and test_env.current_track is not None else 0}")
                    if hasattr(test_env, 'current_track') and test_env.current_track is not None:
                        all_layers = sorted(test_env.current_track['unique_layer_id'].unique())
                        print(f"  All layers in current_track: {all_layers}")
                        print(f"  Initial start layer (final seed): {initial_start_layer}")
                if track_efficiency > 1.0:
                    print(f"WARNING: Efficiency > 1.0! This suggests correct_hits > expected_hits")
                    print(f"  Expected hits from path_plan layers: {len(correct_hit_ids_in_path - seed_hit_ids)}")
                    print(f"  Truth hits in path_plan layers: {len([layer for layer in truth_path_layers if layer in expected_path_layers])}")
                print(f"{'='*60}\n")
            
            # Also track why the episode ended (already captured in loop above)
            # termination_reason is already set
            final_hit_number = getattr(test_env, 'hit_number', 0)
            track_length = len(test_env.current_track) if hasattr(test_env, 'current_track') and test_env.current_track is not None else 0
            
            # Use the termination reason from the loop (already set)
            term_reason = termination_reason
            
            track_results.append({
                'track_id': total_tracks,
                'expected_hits': track_expected_hits,
                'predicted_hits': track_predicted_hits,
                'correct_hits': track_correct_hits,
                'accuracy': track_accuracy,
                'efficiency': track_efficiency,
                'purity': track_purity,
                'reward': track_reward,
                'steps': track_steps,
                'time': track_time,
                'pt': track_pt if track_pt is not None else 0.0,
                'eta': track_eta if track_eta is not None else 0.0,
                'final_hit_number': final_hit_number,
                'track_length': track_length,
                'completion_ratio': final_hit_number / track_length if track_length > 0 else 0.0,
                'termination_reason': term_reason,
                'expected_path_layers': expected_path_layers,
                'actual_path_layers': actual_path_layers,
                'skipped_layers': skipped_layers,
                'expected_path_length': len(expected_path_layers),
                'actual_path_length': len(actual_path_layers),
                'skipped_count': len(skipped_layers),
            })
            
            total_tracks += 1
            total_correct_hits += track_correct_hits
            total_predicted_hits += track_predicted_hits
            total_expected_hits += track_expected_hits
            
            # Print progress
            if total_tracks % 10 == 0:
                avg_efficiency = total_correct_hits / total_expected_hits if total_expected_hits > 0 else 0.0
                avg_purity = total_correct_hits / total_predicted_hits if total_predicted_hits > 0 else 0.0
                print(f"Track {total_tracks}: Efficiency={avg_efficiency:.4f}, Purity={avg_purity:.4f}, "
                      f"Avg Reward={sum([r['reward'] for r in track_results[-10:]])/min(10, len(track_results)):.4f}")
    
    except StopIteration:
        print("Reached end of test dataset")
    
    # Calculate final metrics
    if total_tracks == 0:
        print("No tracks evaluated!")
        return
    
    overall_efficiency = total_correct_hits / total_expected_hits if total_expected_hits > 0 else 0.0
    overall_purity = total_correct_hits / total_predicted_hits if total_predicted_hits > 0 else 0.0
    overall_accuracy = total_correct_hits / total_predicted_hits if total_predicted_hits > 0 else 0.0
    avg_track_reward = sum([r['reward'] for r in track_results]) / len(track_results) if track_results else 0.0
    avg_track_time = total_time / total_tracks if total_tracks > 0 else 0.0
    avg_track_steps = sum([r['steps'] for r in track_results]) / len(track_results) if track_results else 0.0
    
    # Efficiency distribution
    efficiencies = [r['efficiency'] for r in track_results]
    purities = [r['purity'] for r in track_results]
    
    # Track completion statistics
    completion_ratios = [r['completion_ratio'] for r in track_results]
    avg_completion = np.mean(completion_ratios) if completion_ratios else 0.0
    avg_track_length = np.mean([r['track_length'] for r in track_results]) if track_results else 0.0
    avg_final_hit_number = np.mean([r['final_hit_number'] for r in track_results]) if track_results else 0.0
    
    # Path analysis statistics
    avg_expected_path_length = np.mean([r.get('expected_path_length', 0) for r in track_results if 'expected_path_length' in r]) if track_results else 0.0
    avg_actual_path_length = np.mean([r.get('actual_path_length', 0) for r in track_results if 'actual_path_length' in r]) if track_results else 0.0
    avg_skipped_count = np.mean([r.get('skipped_count', 0) for r in track_results if 'skipped_count' in r]) if track_results else 0.0
    
    # Count termination reasons
    termination_reasons = {}
    for r in track_results:
        reason = r.get('termination_reason', 'unknown')
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
    
    # Print results
    print(f"\n{'='*60}")
    print(f"TRACK-BY-TRACK EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Tracks Evaluated: {total_tracks}")
    print(f"Total Expected Hits (excluding seed): {total_expected_hits}")
    print(f"Total Predicted Hits (excluding seed): {total_predicted_hits}")
    print(f"Total Correct Hits (excluding seed): {total_correct_hits}")
    print(f"\nTracking Efficiency Metrics (seed hits excluded):")
    print(f"  Overall Efficiency: {overall_efficiency:.4f} ({total_correct_hits}/{total_expected_hits})")
    print(f"  Overall Purity: {overall_purity:.4f} ({total_correct_hits}/{total_predicted_hits})")
    print(f"  Overall Accuracy: {overall_accuracy:.4f}")
    print(f"  Mean Efficiency: {np.mean(efficiencies):.4f} ± {np.std(efficiencies):.4f}")
    print(f"  Mean Purity: {np.mean(purities):.4f} ± {np.std(purities):.4f}")
    print(f"  Efficiency Range: [{np.min(efficiencies):.4f}, {np.max(efficiencies):.4f}]")
    print(f"  Purity Range: [{np.min(purities):.4f}, {np.max(purities):.4f}]")
    print(f"\nTrack Completion Statistics:")
    print(f"  Average Track Length: {avg_track_length:.2f} hits")
    print(f"  Average Final Hit Number: {avg_final_hit_number:.2f}")
    print(f"  Average Completion Ratio: {avg_completion:.4f} ({avg_completion*100:.2f}%)")
    print(f"  Completion Range: [{np.min(completion_ratios):.4f}, {np.max(completion_ratios):.4f}]")
    print(f"  Average Predicted Hits per Track: {total_predicted_hits/total_tracks:.2f}")
    print(f"  Average Expected Hits per Track: {total_expected_hits/total_tracks:.2f}")
    print(f"\nPath Analysis:")
    print(f"  Average Expected Path Length: {avg_expected_path_length:.2f} layers")
    print(f"  Average Actual Path Length (with hits): {avg_actual_path_length:.2f} layers")
    print(f"  Average Skipped Layers: {avg_skipped_count:.2f} layers")
    print(f"  Path Coverage: {avg_actual_path_length/avg_expected_path_length*100:.1f}%" if avg_expected_path_length > 0 else "  Path Coverage: N/A")
    print(f"\nTermination Reasons:")
    for reason, count in termination_reasons.items():
        print(f"  {reason}: {count} ({count/total_tracks*100:.1f}%)")
    print(f"\nPerformance Metrics:")
    print(f"  Average Track Reward: {avg_track_reward:.4f}")
    print(f"  Average Track Steps: {avg_track_steps:.2f}")
    print(f"  Average Track Time: {avg_track_time*1000:.2f} ms")
    print(f"  Total Evaluation Time: {total_time:.2f} seconds")
    print(f"  Tracks per Second: {total_tracks/total_time:.2f}" if total_time > 0 else "  Tracks per Second: N/A")
    print(f"{'='*60}\n")
    
    # Create plots
    plot_dir = Path(model_path).parent
    plot_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    track_times = [r['time'] for r in track_results]
    track_pts = [r['pt'] for r in track_results if r['pt'] > 0]
    track_etas = [r['eta'] for r in track_results if r['eta'] != 0]
    efficiencies_by_pt = [r['efficiency'] for r in track_results if r['pt'] > 0]
    efficiencies_by_eta = [r['efficiency'] for r in track_results if r['eta'] != 0]
    pts_for_plot = [r['pt'] for r in track_results if r['pt'] > 0]
    etas_for_plot = [r['eta'] for r in track_results if r['eta'] != 0]
    
    # Plot 1: Timing per track
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(track_times)), [t*1000 for t in track_times], 'b-', alpha=0.6)
    plt.xlabel('Track Number')
    plt.ylabel('Time per Track (ms)')
    plt.title('Timing per Track')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=np.mean(track_times)*1000, color='r', linestyle='--', label=f'Mean: {np.mean(track_times)*1000:.2f} ms')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / 'timing_per_track.png', dpi=150)
    plt.close()
    print(f"Saved timing plot to {plot_dir / 'timing_per_track.png'}")
    
    # Plot 2: Efficiency vs pt
    if len(track_pts) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(pts_for_plot, efficiencies_by_pt, alpha=0.5, s=20)
        plt.xlabel('Track pT (GeV/c)')
        plt.ylabel('Tracking Efficiency')
        plt.title('Tracking Efficiency vs Track pT')
        plt.grid(True, alpha=0.3)
        
        # Bin by pt for better visualization
        if len(pts_for_plot) > 10:
            pt_bins = np.linspace(min(pts_for_plot), max(pts_for_plot), 10)
            bin_means = []
            bin_centers = []
            for i in range(len(pt_bins)-1):
                mask = (np.array(pts_for_plot) >= pt_bins[i]) & (np.array(pts_for_plot) < pt_bins[i+1])
                if mask.sum() > 0:
                    bin_means.append(np.mean(np.array(efficiencies_by_pt)[mask]))
                    bin_centers.append((pt_bins[i] + pt_bins[i+1]) / 2)
            if len(bin_means) > 0:
                plt.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Binned Average')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'efficiency_vs_pt.png', dpi=150)
        plt.close()
        print(f"Saved efficiency vs pt plot to {plot_dir / 'efficiency_vs_pt.png'}")
    
    # Plot 3: Efficiency vs eta
    if len(track_etas) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(etas_for_plot, efficiencies_by_eta, alpha=0.5, s=20)
        plt.xlabel('Track η')
        plt.ylabel('Tracking Efficiency')
        plt.title('Tracking Efficiency vs Track η')
        plt.grid(True, alpha=0.3)
        
        # Bin by eta for better visualization
        if len(etas_for_plot) > 10:
            eta_bins = np.linspace(min(etas_for_plot), max(etas_for_plot), 10)
            bin_means = []
            bin_centers = []
            for i in range(len(eta_bins)-1):
                mask = (np.array(etas_for_plot) >= eta_bins[i]) & (np.array(etas_for_plot) < eta_bins[i+1])
                if mask.sum() > 0:
                    bin_means.append(np.mean(np.array(efficiencies_by_eta)[mask]))
                    bin_centers.append((eta_bins[i] + eta_bins[i+1]) / 2)
            if len(bin_means) > 0:
                plt.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Binned Average')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'efficiency_vs_eta.png', dpi=150)
        plt.close()
        print(f"Saved efficiency vs eta plot to {plot_dir / 'efficiency_vs_eta.png'}")
    
    # Save results to file
    results_file = Path(model_path).parent / "evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("TRACK-BY-TRACK EVALUATION RESULTS (Test Set Only)\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Test Data: {test_data_dir}\n")
        f.write(f"Particle Filters: {particle_filters}\n")
        f.write(f"Hit Filters: {hit_filters}\n")
        f.write(f"\nMetrics (seed hits excluded from efficiency):\n")
        f.write(f"  Total Tracks: {total_tracks}\n")
        f.write(f"  Total Expected Hits (excluding seed): {total_expected_hits}\n")
        f.write(f"  Total Predicted Hits (excluding seed): {total_predicted_hits}\n")
        f.write(f"  Total Correct Hits (excluding seed): {total_correct_hits}\n")
        f.write(f"  Overall Efficiency: {overall_efficiency:.4f}\n")
        f.write(f"  Overall Purity: {overall_purity:.4f}\n")
        f.write(f"  Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"  Mean Efficiency: {np.mean(efficiencies):.4f} ± {np.std(efficiencies):.4f}\n")
        f.write(f"  Mean Purity: {np.mean(purities):.4f} ± {np.std(purities):.4f}\n")
        f.write(f"  Average Track Time: {avg_track_time*1000:.2f} ms\n")
        f.write(f"  Tracks per Second: {total_tracks/total_time:.2f}\n" if total_time > 0 else "  Tracks per Second: N/A\n")
    
    print(f"Results saved to {results_file}")
    print(f"Plots saved to {plot_dir}")
    
    return {
        'efficiency': overall_efficiency,
        'purity': overall_purity,
        'accuracy': overall_accuracy,
        'tracks_per_second': total_tracks/total_time if total_time > 0 else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained DQN model on test set (track-by-track)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to training configuration YAML file. If not provided, will look for config in checkpoint directory.'
    )
    parser.add_argument(
        '--test_data_dir',
        type=str,
        default=None,
        help='Optional path to test data directory (if different from config)'
    )
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.config, args.test_data_dir)
