"""Evaluation script for trained DQN model on test set - track-by-track evaluation."""
from pathlib import Path
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rl_tracking.lightning_modules.dqn import DQNLightning
from rl_tracking.preprocessing.hit_candidates import EventProcessor
from rl_tracking.utils.stream_loading import TrackingDataModule
from rl_tracking.utils.config_loader import load_config
from rl_tracking.utils.data_paths import resolve_data_directories
from rl_tracking.environment.tracking_env import TrackingEnv
from rl_tracking.environment.agent import Agent
from rl_tracking.replay.buffer import ReplayBuffer
import pandas as pd

DEFAULT_TRACKML_DIR = Path("/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/codalab-data/part_1")

def evaluate_model(model_path: str, config_path: str = None, test_data_dir: str = None, output_dir: str = None):
    """Evaluate a trained model on the test set track-by-track.
    
    Args:
        model_path: Path to the trained model checkpoint (.ckpt file)
        config_path: Optional path to the training configuration YAML file.
                     If None, will try to load from checkpoint directory.
        test_data_dir: Optional path to test data directory (if different from config)
        output_dir: Optional directory to store evaluation artifacts (plots, logs).
                    Defaults to <checkpoint_dir>/evaluation/<checkpoint_name>
    """
    checkpoint_path = Path(model_path).expanduser()
    if checkpoint_path.is_dir():
        potential_ckpts = sorted(checkpoint_path.glob("*.ckpt"))
        if len(potential_ckpts) == 1:
            checkpoint_path = potential_ckpts[0]
        else:
            raise ValueError(
                f"Checkpoint path '{checkpoint_path}' is a directory. "
                "Please provide the full path to a .ckpt file."
            )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint_path = checkpoint_path.resolve()
    model_path = str(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    checkpoint_stem = checkpoint_path.stem

    # If config_path not provided, try to find it in checkpoint directory
    if config_path is None:
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
    data_config_override = dict(data_config) if data_config else {}
    if test_data_dir is not None:
        data_config_override['data_dir'] = test_data_dir

    base_dir, test_data_directories = resolve_data_directories(
        data_config_override,
        default_dir=DEFAULT_TRACKML_DIR,
    )

    print(f"Using TrackML data directories for evaluation:")
    for directory in test_data_directories:
        print(f"  - {directory}")
    
    batch_size = data_config.get('batch_size', 32)
    val_split = data_config.get('val_split', 0.2)
    test_split = data_config.get('test_split', 0.1)
    num_workers = data_config.get('num_workers', 0)
    random_seed = data_config.get('random_seed', 42)
    event_processor_config = data_config.get('event_processor', {})
    n_neighbors = event_processor_config.get('n_neighbors', 20)
    
    # Initialize dataset
    ep = EventProcessor(base_dir, n_neighbors)
    dm = TrackingDataModule(
        file_paths=test_data_directories,
        batch_size=batch_size,
        event_processor=ep,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed,
    )
    
    # Extract model and environment configuration
    model_config = config.get('model', {})
    environment_config = config.get('environment', {})
    particle_filters = environment_config.get('particle_filters', {})
    hit_filters = environment_config.get('hit_filters', {})
    use_distance_reward = environment_config.get('use_distance_reward', False)
    use_truth_path_plan = environment_config.get('use_truth_path_plan', False)
    hit_feature_mode = environment_config.get('hit_feature_mode', 'absolute')
    state_feature_mode = environment_config.get('state_feature_mode', 'full')
    
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
            print(f"\n{'='*60}")
            print(f"Found hyperparameters in checkpoint:")
            print(f"{'='*60}")
            checkpoint_pf = checkpoint_hparams.get('particle_filters', 'N/A')
            print(f"  particle_filters: {checkpoint_pf}")
            if isinstance(checkpoint_pf, dict):
                print(f"    -> pt: {checkpoint_pf.get('pt', 'NOT SET')}")
                print(f"    -> nhits_min: {checkpoint_pf.get('nhits_min', 'NOT SET')}")
                print(f"    -> nhits_max: {checkpoint_pf.get('nhits_max', 'NOT SET')}")
            print(f"  hit_filters: {checkpoint_hparams.get('hit_filters', 'N/A')}")
            print(f"  use_distance_reward: {checkpoint_hparams.get('use_distance_reward', 'N/A')}")
            print(f"  use_truth_path_plan: {checkpoint_hparams.get('use_truth_path_plan', 'N/A')}")
            print(f"{'='*60}\n")
            
            # Compare config file vs checkpoint hyperparameters
            config_pf = particle_filters
            checkpoint_pf = checkpoint_hparams.get('particle_filters', {})
            if isinstance(checkpoint_pf, dict) and isinstance(config_pf, dict):
                if checkpoint_pf.get('pt') != config_pf.get('pt'):
                    print(f"⚠️  WARNING: pt filter mismatch!")
                    print(f"   Config file: pt={config_pf.get('pt', 'NOT SET')}")
                    print(f"   Checkpoint: pt={checkpoint_pf.get('pt', 'NOT SET')}")
                    print(f"   Using checkpoint value (this is what the model was trained with)\n")
            
            print(f"Using hyperparameters from checkpoint (most accurate - matches training)\n")
            # Use checkpoint hyperparameters if available
            particle_filters = checkpoint_hparams.get('particle_filters', particle_filters)
            hit_filters = checkpoint_hparams.get('hit_filters', hit_filters)
            use_distance_reward = checkpoint_hparams.get('use_distance_reward', use_distance_reward)
            use_truth_path_plan = checkpoint_hparams.get('use_truth_path_plan', use_truth_path_plan)
            hit_feature_mode = checkpoint_hparams.get('hit_feature_mode', hit_feature_mode)
            state_feature_mode = checkpoint_hparams.get('state_feature_mode', state_feature_mode)
        else:
            print(f"⚠️  No hyperparameters found in checkpoint. Using config file values.")
            print(f"   This may cause mismatches if config differs from training config!\n")
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
    print(f"  -> pt filter: {particle_filters.get('pt', 'NOT SET')} (tracks with pt > this value will be included)")
    print(f"  -> nhits_min: {particle_filters.get('nhits_min', 'NOT SET')}")
    print(f"  -> nhits_max: {particle_filters.get('nhits_max', 'NOT SET')}")
    print(f"Hit filters (used for test): {hit_filters}")
    print(f"Use distance reward: {use_distance_reward}")
    print(f"Use truth path plan: {use_truth_path_plan}")
    print(f"Hit feature mode: {hit_feature_mode}")
    print(f"State feature mode: {state_feature_mode}")
    print(f"{'='*60}\n")
    
    # Verify that pt filter is set correctly
    pt_filter = particle_filters.get('pt', None)
    if pt_filter is None:
        print(f"⚠️  WARNING: pt filter is not set! All tracks will be included regardless of pt.")
        print(f"   This may cause evaluation on tracks with pt < 1 if that's not desired.")
    else:
        print(f"ℹ️  pt filter is set to {pt_filter}")
        print(f"   Filter logic: tracks with pt > {pt_filter} will be included")
        print(f"   Note: Tracks with pt exactly equal to {pt_filter} will be EXCLUDED (strict >)")
        if pt_filter < 1.0:
            print(f"⚠️  WARNING: pt filter is set to {pt_filter}, which is < 1.0.")
            print(f"   This will include tracks with pt > {pt_filter}, including some with pt < 1.0")
        elif pt_filter == 1.0:
            print(f"ℹ️  pt filter is exactly 1.0 - tracks with pt > 1.0 will be included")
            print(f"   Tracks with pt <= 1.0 will be excluded")
    
    # Now setup for test to get test dataset
    dm.setup(stage="test")
    
    # Set model to evaluation mode
    model.eval()
    model.net.eval()
    model.target_net.eval()
    
    # Create test environment
    # CRITICAL: For deterministic evaluation, ensure no shuffling
    # Use worker_init_fn to set random seed if needed, but IterableDataset shouldn't shuffle
    test_env_dataloader = DataLoader(
        dm.test_dataset,
        batch_size=1,
        num_workers=0,  # Use 0 workers for deterministic ordering
        pin_memory=False,
        drop_last=False,
        shuffle=False  # Ensure no shuffling (shouldn't matter for IterableDataset, but explicit)
    )
    test_env = TrackingEnv(
        test_env_dataloader,
        particle_filters=particle_filters,
        hit_filters=hit_filters,
        use_distance_reward=use_distance_reward,
        use_truth_path_plan=use_truth_path_plan,
        deterministic=True,  # CRITICAL: Use deterministic ordering for evaluation reproducibility
        hit_feature_mode=hit_feature_mode,
        state_feature_mode=state_feature_mode,
    )
    if hasattr(test_env, "rank_selection_counts"):
        test_env.rank_selection_counts.clear()
    if hasattr(test_env, "rank_correct_counts"):
        test_env.rank_correct_counts.clear()
    
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
    print(f"Deterministic evaluation: particle order is sorted (not shuffled)")
    print(f"{'='*60}\n")
    
    # Set random seeds for reproducibility (if any randomness is used)
    import random
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    
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
    num_actions = getattr(getattr(test_env, "action_space", None), "n", 0)
    action_counts = np.zeros(num_actions, dtype=int) if num_actions else None
    invalid_action_count = 0
    
    max_attempts = max_tracks * 5  # Allow retries for skipped/invalid tracks
    progress_bar = tqdm(total=max_tracks, desc="Evaluating tracks", unit="track")

    confusion_pairs = []

    try:
        tracks_skipped = 0
        consecutive_none = 0
        attempt_count = 0
        while total_tracks < max_tracks:
            attempt_count += 1
            if attempt_count > max_attempts:
                print(
                    f"\nReached maximum attempts ({max_attempts}) with only {total_tracks} tracks evaluated."
                    " Stopping early to avoid infinite loops on invalid tracks."
                )
                break
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
            test_agent.state = obs
            test_agent.info = info or {}
            
            # Check if track has valid data
            if not hasattr(test_env, 'current_track') or test_env.current_track is None or len(test_env.current_track) == 0:
                tracks_skipped += 1
                progress_bar.set_postfix(skipped=tracks_skipped)
                if tracks_skipped <= 5:
                    print(f"WARNING: Track {total_tracks + 1} has no data, skipping...")
                continue
            
            # Debug candidate info for first few tracks
            if total_tracks < 3:
                mask_initial = info.get("hit_mask") if isinstance(info, dict) else None
                mask_count = int(mask_initial.sum()) if mask_initial is not None else -1
                print(f"DEBUG Track {total_tracks}: initial candidate count = {mask_count}")
            
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
            helix_available = hasattr(test_env, 'helix') and test_env.helix is not None
            seed_layer_ids = set()
            initial_start_layer = None
            path_plan_layers = []
            path_plan_hits_after_seed = set()
            correct_hit_ids_in_path = set()
            truth_path_layers = []
            
            # Verify this track should be included based on filters
            # (This is a sanity check - filters should have been applied already)
            track_pt_for_check = None
            if hasattr(test_env, 'current_track') and test_env.current_track is not None:
                # Get pt from track (should be the same for all hits in track)
                track_pt_for_check = test_env.current_track['pt'].iloc[0] if len(test_env.current_track) > 0 else None
                if track_pt_for_check is not None and particle_filters.get('pt', None) is not None:
                    if track_pt_for_check <= particle_filters['pt']:
                        print(f"⚠️  WARNING: Track {total_tracks} has pt={track_pt_for_check:.3f} which is <= filter threshold {particle_filters['pt']}")
                        print(f"   This track should have been filtered out! This suggests a filter application issue.")
            
            # Get track properties and seed hits
            track_pt = None
            track_eta = None
            if hasattr(test_env, 'current_track') and test_env.current_track is not None:
                track_df = test_env.current_track
                if track_df.empty:
                    continue

                correct_hit_ids = set(track_df['hit_id'].astype(int).values)

                if 'pt' in track_df.columns:
                    track_pt = float(track_df['pt'].iloc[0])

                if 'theta' in track_df.columns:
                    theta_val = track_df['theta'].iloc[0]
                    if theta_val is not None and 0 < theta_val < np.pi:
                        track_eta = -np.log(np.tan(theta_val / 2.0))

                # Determine seed hits/layers using environment seed length
                seed_len = getattr(test_env, 'SEED_LENGTH', 3)
                track_sorted = track_df.sort_values('unique_layer_id')
                seed_subset = track_sorted.head(seed_len)
                seed_hit_ids = set(seed_subset['hit_id'].astype(int).values)
                seed_layer_ids = set(seed_subset['unique_layer_id'].astype(int).values)

                # Fallback to helix-provided seeds if available
                if helix_available:
                    seed_df = getattr(test_env.helix, 'seed_hits', None)
                    if isinstance(seed_df, pd.DataFrame) and not seed_df.empty:
                        if 'hit_id' in seed_df.columns:
                            seed_hit_ids = set(seed_df['hit_id'].astype(int).values)
                        if 'unique_layer_id' in seed_df.columns:
                            seed_layer_ids = set(seed_df['unique_layer_id'].astype(int).values)
                    elif seed_df is not None:
                        try:
                            seed_hit_ids = set(int(hit) for hit in seed_df)
                        except TypeError:
                            pass
                    if track_pt is None and hasattr(test_env.helix, 'pt'):
                        track_pt = float(test_env.helix.pt)
                    theta_val = getattr(test_env.helix, 'theta', None)
                    if theta_val is not None and 0 < theta_val < np.pi:
                        track_eta = -np.log(np.tan(theta_val / 2.0))

                # Identify post-seed truth hits
                post_seed_df = track_df[~track_df['hit_id'].isin(seed_hit_ids)].copy()
                correct_hit_ids_in_path = set(post_seed_df['hit_id'].astype(int).values)

                # Expected layers correspond to post-seed truth layers
                truth_path_layers = [float(layer) for layer in sorted(post_seed_df['unique_layer_id'].unique())]
                track_expected_hits = len(truth_path_layers)

                # Keep reference layer for comparisons (max seed layer)
                initial_start_layer = float(max(seed_layer_ids)) if seed_layer_ids else None

                # If helix/path_plan available, capture but don't require
                if helix_available:
                    path_plan_source = getattr(test_env.helix, 'path_plan', None)
                    if path_plan_source is not None:
                        try:
                            iterable_layers = list(path_plan_source)
                        except TypeError:
                            iterable_layers = [path_plan_source]
                        path_plan_layers = [
                            float(layer)
                            for layer in iterable_layers
                            if initial_start_layer is None or float(layer) > float(initial_start_layer)
                        ]
                path_plan_hits_after_seed = correct_hit_ids_in_path if not path_plan_layers else set()
                if path_plan_layers:
                    path_plan_layer_set = set(path_plan_layers)
                    path_plan_hits_after_seed = set(
                        post_seed_df[post_seed_df['unique_layer_id'].isin(path_plan_layer_set)]['hit_id'].astype(int).values
                    )

                expected_hits_source = "all truth hits after seed"
                if total_tracks < 3:
                    print(
                        f"DEBUG Track {total_tracks}: initial_start_layer={initial_start_layer}, "
                        f"total_hits_in_track={len(track_df)}, "
                        f"post_seed_hits={len(post_seed_df)}, "
                        f"path_plan_hits_after_seed={len(path_plan_hits_after_seed)}, "
                        f"using: {expected_hits_source}, "
                        f"seed_hit_ids={len(seed_hit_ids)}"
                    )
                
                # Debug: if truth_path_layers is empty, it might mean the track only has seed hits
                # This can happen if nhits_min is 3 and all hits are in seed layers
            
            # Track the actual path taken vs expected path
            expected_path_layers = list(path_plan_layers) if path_plan_layers else list(truth_path_layers)
            actual_path_layers = []  # Layers where we actually selected hits
            skipped_layers = []  # Layers we visited but found no hits
            # truth_path_layers already computed above (before the loop)
            termination_reason = 'unknown'  # Why the track ended
            
            # Reconstruct track step by step
            done = False
            while not done:
                info_before = test_agent.info or {}
                candidate_ids = info_before.get("candidate_hit_ids", [])
                candidate_mask = info_before.get("hit_mask", np.zeros(num_actions, dtype=bool))
                correct_indices = [
                    idx
                    for idx, hit_id in enumerate(candidate_ids)
                    if idx < len(candidate_mask) and candidate_mask[idx] and hit_id in correct_hit_ids
                ]
                true_action_index = correct_indices[0] if correct_indices else None

                prev_hit_number = getattr(test_env, 'hit_number', 0)
                reward, done = test_agent.play_step(model.net, epsilon=epsilon, device=device)

                # Update action counts (Agent stores last action in replay buffer entry)
                if hasattr(test_env, 'last_selected_hit_id') and action_counts is not None:
                    last_action = getattr(test_env, 'last_selected_action_index', None)
                    if last_action is not None and 0 <= last_action < len(action_counts):
                        action_counts[last_action] += 1
                    elif last_action is not None:
                        invalid_action_count += 1
                predicted_action_index = getattr(test_env, 'last_selected_action_index', None)
                if (
                    predicted_action_index is not None
                    and true_action_index is not None
                    and 0 <= predicted_action_index < num_actions
                    and 0 <= true_action_index < num_actions
                ):
                    confusion_pairs.append((predicted_action_index, true_action_index))

                # Track why episode ended - distinguish between path complete vs no hits found
                if done:
                    # Check if path_plan is exhausted (end of expected path)
                    path_exhausted = False
                    if helix_available and hasattr(test_env.helix, 'path_plan') and test_env.helix.path_plan is not None:
                        unexplored_layers = test_env.helix.path_plan[test_env.helix.path_plan > test_env.helix.current_layer]
                        path_exhausted = len(unexplored_layers) == 0
                    elif expected_path_layers:
                        path_exhausted = len(actual_path_layers) >= len(expected_path_layers)
                    
                    # Check if we ran out of hits vs propagation failed
                    current_hit_number = getattr(test_env, 'hit_number', 0)
                    track_length = len(test_env.current_track) if hasattr(test_env, 'current_track') and test_env.current_track is not None else 0
                    
                    if path_exhausted:
                        termination_reason = 'path_complete'  # Reached end of path_plan
                    elif current_hit_number >= track_length:
                        termination_reason = 'completed_all_hits'  # All truth hits visited
                    elif test_agent.state is None:
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
                    
                    # Determine layer of selected hit using pre-step target layer
                    selected_hit_layer = target_layer_before_step
                    if selected_hit_layer is None and hasattr(test_env, 'last_selected_hit') and test_env.last_selected_hit is not None:
                        selected_hit = test_env.last_selected_hit
                        if isinstance(selected_hit, pd.Series) and 'unique_layer_id' in selected_hit.index:
                            selected_hit_layer = float(selected_hit['unique_layer_id'])
                        elif hasattr(selected_hit, 'unique_layer_id'):
                            selected_hit_layer = float(selected_hit.unique_layer_id)
                    if selected_hit_layer is None and hasattr(test_env, 'current_target_unique_layer'):
                        selected_hit_layer = float(test_env.current_target_unique_layer)
                    if total_tracks < 3:
                        info_debug = test_agent.info or {}
                        raw_scores = info_debug.get("raw_scores")
                        masked_scores = info_debug.get("masked_scores")
                        chosen_action = info_debug.get("chosen_action")
                        print(f"    DEBUG Track {total_tracks} Step {track_steps}: selected_hit_id={selected_hit_id}, "
                              f"layer={selected_hit_layer}, is_seed={selected_hit_id in seed_hit_ids}, "
                              f"is_correct={selected_hit_id in correct_hit_ids_in_path}, "
                              f"chosen_action={chosen_action}")
                        if raw_scores is not None:
                            print(f"    DEBUG Track {total_tracks} Step {track_steps}: raw_scores={raw_scores}")
                        if masked_scores is not None:
                            print(f"    DEBUG Track {total_tracks} Step {track_steps}: masked_scores={masked_scores}")
                    
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
                            
                            # Determine correctness using truth hit IDs
                            if selected_hit_id in correct_hit_ids_in_path:
                                track_correct_hits += 1
                                # Track that we found at least one correct hit in this layer
                                if selected_hit_layer is not None and initial_start_layer is not None:
                                    if selected_hit_layer > initial_start_layer:
                                        layers_with_correct_hit.add(selected_hit_layer)
                            
                            if total_tracks < 5:
                                layer_info = f"layer={selected_hit_layer}" if selected_hit_layer is not None else "layer=unknown"
                                correct_info = "CORRECT" if selected_hit_id in correct_hit_ids_in_path else "WRONG"
                                print(f"  Track {total_tracks}, Step {track_steps}: Selected hit_id={selected_hit_id} in {layer_info}, {correct_info}")
                else:
                    # No hit selected - track which layer we skipped (only post-seed layers)
                    # Only count as skipped if we're in a layer that's in the expected path
                    # and we haven't already selected a hit from this layer
                    current_layer = target_layer_before_step
                    if current_layer is None:
                        if helix_available and hasattr(test_env, 'helix') and test_env.helix is not None:
                            current_layer = getattr(test_env.helix, 'current_layer', None)
                        else:
                            current_layer = getattr(test_env, 'current_target_unique_layer', None)
                    if current_layer is not None:
                        current_layer = float(current_layer)
                    if current_layer is not None and initial_start_layer is not None and expected_path_layers:
                        if current_layer > initial_start_layer and current_layer in expected_path_layers:
                            if current_layer not in actual_path_layers and current_layer not in skipped_layers:
                                skipped_layers.append(current_layer)
                
                # Loop continues with agent state/info already updated inside play_step
                
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
                'completion_ratio': len(actual_path_layers) / track_expected_hits if track_expected_hits > 0 else 0.0,
                'termination_reason': term_reason,
                'expected_path_layers': expected_path_layers,
                'actual_path_layers': actual_path_layers,
                'skipped_layers': skipped_layers,
                'expected_path_length': len(expected_path_layers),
                'actual_path_length': len(actual_path_layers),
                'skipped_count': len(skipped_layers),
                'layer_correct_count': len(layers_with_correct_hit),
                'layer_total_count': track_expected_hits,
            })
            
            total_tracks += 1
            total_correct_hits += track_correct_hits
            total_predicted_hits += track_predicted_hits
            total_expected_hits += track_expected_hits
            progress_bar.update(1)
            
            # Print progress
            if total_tracks % 10 == 0:
                avg_efficiency = total_correct_hits / total_expected_hits if total_expected_hits > 0 else 0.0
                avg_purity = total_correct_hits / total_predicted_hits if total_predicted_hits > 0 else 0.0
                print(f"Track {total_tracks}: Efficiency={avg_efficiency:.4f}, Purity={avg_purity:.4f}, "
                      f"Avg Reward={sum([r['reward'] for r in track_results[-10:]])/min(10, len(track_results)):.4f}")
    
    except StopIteration:
        print("Reached end of test dataset")
    finally:
        progress_bar.close()
    
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
    avg_expected_path_length = np.mean([r.get('expected_path_length', 0) for r in track_results]) if track_results else 0.0
    avg_actual_path_length = np.mean([r.get('actual_path_length', 0) for r in track_results]) if track_results else 0.0
    avg_skipped_count = np.mean([r.get('skipped_count', 0) for r in track_results if 'skipped_count' in r]) if track_results else 0.0
    
    # Count termination reasons
    termination_reasons = {}
    for r in track_results:
        reason = r.get('termination_reason', 'unknown')
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

    total_layer_correct = sum(r.get('layer_correct_count', 0) for r in track_results)
    total_layer_total = sum(r.get('layer_total_count', 0) for r in track_results)
    overall_path_coverage = total_layer_correct / total_layer_total if total_layer_total > 0 else 0.0

    rank_metrics = {}
    if hasattr(test_env, "get_rank_selection_metrics"):
        rank_metrics = test_env.get_rank_selection_metrics()
    
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
    print(f"  Path Coverage: {overall_path_coverage*100:.1f}%" if total_layer_total > 0 else "  Path Coverage: N/A")
    print(f"\nTermination Reasons:")
    for reason, count in termination_reasons.items():
        print(f"  {reason}: {count} ({count/total_tracks*100:.1f}%)")
    if rank_metrics:
        print(f"\nRank Selection Metrics:")
        for rank in sorted(rank_metrics):
            stats = rank_metrics[rank]
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            accuracy = stats.get("accuracy", 0.0)
            if total > 0:
                print(f"  Rank {rank}: accuracy={accuracy:.4f} ({int(correct)}/{int(total)})")
    print(f"\nPerformance Metrics:")
    print(f"  Average Track Reward: {avg_track_reward:.4f}")
    print(f"  Average Track Steps: {avg_track_steps:.2f}")
    print(f"  Average Track Time: {avg_track_time*1000:.2f} ms")
    print(f"  Total Evaluation Time: {total_time:.2f} seconds")
    print(f"  Tracks per Second: {total_tracks/total_time:.2f}" if total_time > 0 else "  Tracks per Second: N/A")
    if action_counts is not None:
        total_actions = action_counts.sum()
        print(f"\nAction Usage (total selections: {total_actions}):")
        if total_actions > 0:
            for action_idx, count in enumerate(action_counts):
                if count > 0:
                    pct = count / total_actions * 100.0
                    print(f"  Action {action_idx:02d}: {count} selections ({pct:.2f}%)")
        if invalid_action_count:
            print(f"  Invalid actions encountered: {invalid_action_count}")
    print(f"{'='*60}\n")
    
    # Create plots
    if output_dir is not None:
        plot_dir = Path(output_dir).expanduser()
        plot_dir.mkdir(parents=True, exist_ok=True)
    else:
        evaluation_root = checkpoint_dir / "evaluation"
        plot_dir = evaluation_root / checkpoint_stem
        plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to: {plot_dir}")
    
    # Set CMS style for all plots
    hep.style.use("CMS")
    
    # Extract data for plotting
    track_times = [r['time'] for r in track_results]
    
    # CRITICAL: Filter tracks based on the pt filter threshold for plotting
    # Only include tracks that actually pass the filter (pt > filter_threshold)
    pt_filter_threshold = particle_filters.get('pt', None)
    if pt_filter_threshold is not None:
        # Only include tracks with pt > filter_threshold
        track_pts = [r['pt'] for r in track_results if r['pt'] > pt_filter_threshold]
        efficiencies_by_pt = [r['efficiency'] for r in track_results if r['pt'] > pt_filter_threshold]
        pts_for_plot = [r['pt'] for r in track_results if r['pt'] > pt_filter_threshold]
        
        # Count how many tracks were filtered out
        filtered_out_count = sum(1 for r in track_results if r['pt'] > 0 and r['pt'] <= pt_filter_threshold)
        if filtered_out_count > 0:
            print(f"⚠️  Filtered out {filtered_out_count} tracks with pt <= {pt_filter_threshold} from efficiency vs pt plot")
    else:
        # No pt filter, include all tracks with pt > 0
        track_pts = [r['pt'] for r in track_results if r['pt'] > 0]
        efficiencies_by_pt = [r['efficiency'] for r in track_results if r['pt'] > 0]
        pts_for_plot = [r['pt'] for r in track_results if r['pt'] > 0]
    
    track_etas = [r['eta'] for r in track_results if r['eta'] != 0]
    efficiencies_by_eta = [r['efficiency'] for r in track_results if r['eta'] != 0]
    etas_for_plot = [r['eta'] for r in track_results if r['eta'] != 0]
    
    # Plot 1: Timing per track
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(track_times)), [t*1000 for t in track_times], 'b-', alpha=0.6)
    ax.set_xlabel('Track Number', fontsize=12)
    ax.set_ylabel('Time per Track (ms)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.mean(track_times)*1000, color='r', linestyle='--', label=f'Mean: {np.mean(track_times)*1000:.2f} ms')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_dir / 'timing_per_track.png', dpi=150)
    plt.close()
    print(f"Saved timing plot to {plot_dir / 'timing_per_track.png'}")
    
    # Plot 2: Efficiency vs pt
    if len(track_pts) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pts_array = np.array(pts_for_plot)
        efficiencies_array = np.array(efficiencies_by_pt)
        
        # Use logarithmic scale for pt
        # Filter out negative or zero values for log scale
        valid_mask = pts_array > 0
        pts_valid = pts_array[valid_mask]
        efficiencies_valid = efficiencies_array[valid_mask]
        
        ax.scatter(pts_valid, efficiencies_valid, alpha=0.5, s=20, label='Individual Tracks')
        ax.set_xlabel('Track p$_T$ (GeV/c)', fontsize=12)
        ax.set_ylabel('Tracking Efficiency', fontsize=12)
        ax.set_xscale('log')  # Logarithmic x-axis
        ax.grid(True, alpha=0.3, which='both')  # Show grid for both major and minor ticks
        
        # Add average efficiency text box
        avg_eff_pt = np.mean(efficiencies_valid) if len(efficiencies_valid) > 0 else 0.0
        ax.text(0.02, 0.98, f'Average Efficiency: {avg_eff_pt:.3f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Bin by pt for better visualization using logarithmic bins
        if len(pts_valid) > 10:
            # Use logarithmic bins for better distribution
            n_bins = 20  # More bins
            pt_min = np.min(pts_valid)
            pt_max = np.max(pts_valid)
            
            # Create logarithmic bins
            log_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), n_bins + 1)
            
            bin_means = []
            bin_centers = []
            bin_counts = []
            for i in range(len(log_bins)-1):
                mask = (pts_valid >= log_bins[i]) & (pts_valid < log_bins[i+1])
                if mask.sum() > 0:
                    bin_means.append(np.mean(efficiencies_valid[mask]))
                    # Geometric mean for bin center (appropriate for log scale)
                    bin_centers.append(np.sqrt(log_bins[i] * log_bins[i+1]))
                    bin_counts.append(mask.sum())
            
            if len(bin_means) > 0:
                ax.plot(bin_centers, bin_means, 'r-', linewidth=2, marker='o', markersize=6, label='Binned Average')
                ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_dir / 'efficiency_vs_pt.png', dpi=150)
        plt.close()
        print(f"Saved efficiency vs pt plot to {plot_dir / 'efficiency_vs_pt.png'}")
    
    # Plot 3: Efficiency vs eta
    if len(track_etas) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        etas_array = np.array(etas_for_plot)
        efficiencies_array = np.array(efficiencies_by_eta)
        
        ax.scatter(etas_array, efficiencies_array, alpha=0.5, s=20, label='Individual Tracks')
        ax.set_xlabel('Track η', fontsize=12)
        ax.set_ylabel('Tracking Efficiency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add average efficiency text box
        avg_eff_eta = np.mean(efficiencies_array) if len(efficiencies_array) > 0 else 0.0
        ax.text(0.02, 0.98, f'Average Efficiency: {avg_eff_eta:.3f}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Bin by eta for better visualization
        if len(etas_for_plot) > 10:
            n_bins = 15  # More bins
            eta_bins = np.linspace(min(etas_array), max(etas_array), n_bins + 1)
            bin_means = []
            bin_centers = []
            bin_counts = []
            for i in range(len(eta_bins)-1):
                mask = (etas_array >= eta_bins[i]) & (etas_array < eta_bins[i+1])
                if mask.sum() > 0:
                    bin_means.append(np.mean(efficiencies_array[mask]))
                    bin_centers.append((eta_bins[i] + eta_bins[i+1]) / 2)
                    bin_counts.append(mask.sum())
            if len(bin_means) > 0:
                # Show data points on binned average (like pt plot)
                ax.plot(bin_centers, bin_means, 'r-', linewidth=2, marker='o', markersize=6, label='Binned Average')
                ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_dir / 'efficiency_vs_eta.png', dpi=150)
        plt.close()
        print(f"Saved efficiency vs eta plot to {plot_dir / 'efficiency_vs_eta.png'}")
    
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
    })

    # Plot 4: Efficiency vs track length (using expected hits post-seed)
    track_lengths = [r['expected_hits'] for r in track_results if r['expected_hits'] > 0]
    efficiencies_by_length = [r['efficiency'] for r in track_results if r['expected_hits'] > 0]
    if len(track_lengths) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        lengths_array = np.array(track_lengths)
        efficiencies_length_array = np.array(efficiencies_by_length)
        ax.set_xlabel('Track Length (truth hits after seed)', fontsize=12)
        ax.set_ylabel('Tracking Efficiency', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Average efficiency per unique track length
        unique_lengths = np.unique(lengths_array)
        if len(unique_lengths) > 0:
            averages = []
            for length in unique_lengths:
                mask = lengths_array == length
                if np.any(mask):
                    averages.append((length, np.mean(efficiencies_length_array[mask])))
            if averages:
                avg_lengths, avg_efficiencies = zip(*averages)
                ax.plot(avg_lengths, avg_efficiencies, 'r-', linewidth=2, marker='o', markersize=6, label='Average by track length')
                ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(plot_dir / 'efficiency_vs_track_length.png', dpi=150)
        plt.close()
        print(f"Saved efficiency vs track length plot to {plot_dir / 'efficiency_vs_track_length.png'}")
 
    # Plot 5: Action selection histogram
    if action_counts is not None and action_counts.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        actions = np.arange(len(action_counts))
        ax.bar(actions, action_counts, color='tab:blue', alpha=0.85)
        ax.set_xlabel('Action index', fontsize=12)
        ax.set_ylabel('Selections', fontsize=12)
        ax.set_title('Action Selection Frequency')
        ax.set_xticks(actions)
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / 'action_histogram.png', dpi=150)
        plt.close()
        print(f"Saved action histogram to {plot_dir / 'action_histogram.png'}")

    if confusion_pairs:
        fig, ax = plt.subplots(figsize=(10, 8))
        confusion_matrix = np.zeros((num_actions, num_actions), dtype=int)
        for pred, truth in confusion_pairs:
            confusion_matrix[pred, truth] += 1
        im = ax.imshow(confusion_matrix, cmap='viridis')
        ax.set_xlabel('True action index', fontsize=12)
        ax.set_ylabel('Predicted action index', fontsize=12)
        ax.set_title('Action Confusion Matrix')
        ax.set_xticks(np.arange(num_actions))
        ax.set_yticks(np.arange(num_actions))
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(plot_dir / 'action_confusion_matrix.png', dpi=150)
        plt.close()
        print(f"Saved action confusion matrix to {plot_dir / 'action_confusion_matrix.png'}")
    
    # Save results to file
    results_file = plot_dir / "evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("TRACK-BY-TRACK EVALUATION RESULTS (Test Set Only)\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write("Test Data directories:\n")
        for directory in test_data_directories:
            f.write(f"  - {directory}\n")
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
        if rank_metrics:
            f.write("\nRank Selection Metrics:\n")
            for rank in sorted(rank_metrics):
                stats = rank_metrics[rank]
                total = stats.get("total", 0)
                correct = stats.get("correct", 0)
                accuracy = stats.get("accuracy", 0.0)
                if total > 0:
                    f.write(
                        f"  Rank {rank}: accuracy={accuracy:.4f} ({int(correct)}/{int(total)})\n"
                    )
        if action_counts is not None:
            total_actions = action_counts.sum()
            f.write("\nAction Usage:\n")
            f.write(f"  Total actions observed: {total_actions}\n")
            if total_actions > 0:
                for action_idx, count in enumerate(action_counts):
                    if count > 0:
                        pct = count / total_actions * 100.0
                        f.write(f"  Action {action_idx:02d}: {count} selections ({pct:.2f}%)\n")
            if invalid_action_count:
                f.write(f"  Invalid actions encountered: {invalid_action_count}\n")
    
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
        help='Optional path to test data directory (overrides config; can point to a specific part directory)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to store evaluation artifacts (plots/results). Defaults to <checkpoint_dir>/evaluation/<checkpoint_name>'
    )
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.config, args.test_data_dir, args.output_dir)
