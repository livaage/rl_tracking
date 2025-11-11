"""Training script for DQN model with configuration file support."""
from pathlib import Path
import argparse
import shutil
import secrets
import re
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from rl_tracking.lightning_modules.dqn import DQNLightning
from rl_tracking.preprocessing.hit_candidates import EventProcessor
from rl_tracking.utils.stream_loading import TrackingDataModule
from rl_tracking.utils.config_loader import load_config
from rl_tracking.utils.data_paths import resolve_data_directories

DEFAULT_TRACKML_DIR = Path("/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/codalab-data/part_1")
import inspect
from rl_tracking.environment import agent as agent_module
print("Using rl_tracking from:", agent_module.__file__)
print("Agent class defined at:", inspect.getfile(agent_module.Agent))

def main(config_path: str | Path):
    """Main training function.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Resolve configuration path relative to common search locations
    config_path = _resolve_config_path(config_path)

    # Load configuration
    config = load_config(config_path)
    
    # Extract data configuration
    data_config = config.get('data', {})
    base_dir, data_directories = resolve_data_directories(
        data_config,
        default_dir=DEFAULT_TRACKML_DIR,
    )
    batch_size = data_config.get('batch_size', 32)
    val_split = data_config.get('val_split', 0.2)
    test_split = data_config.get('test_split', 0.1)
    num_workers = data_config.get('num_workers', 0)
    random_seed = data_config.get('random_seed', 42)
    event_processor_config = data_config.get('event_processor', {})
    n_neighbors = event_processor_config.get('n_neighbors', 20)
    
    print(f"Using TrackML data directories:")
    for directory in data_directories:
        print(f"  - {directory}")
    
    # Initialize dataset and dataloader
    ep = EventProcessor(base_dir, n_neighbors)
    dm = TrackingDataModule(
        file_paths=data_directories,
        batch_size=batch_size,
        event_processor=ep,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed,
    )
    
    dm.setup(stage="fit")
    
    # Extract model configuration
    model_config = config.get('model', {})
    environment_config = config.get('environment', {})
    particle_filters = environment_config.get('particle_filters', {})
    hit_filters = environment_config.get('hit_filters', {})
    use_distance_reward = environment_config.get('use_distance_reward', False)
    use_truth_path_plan = environment_config.get('use_truth_path_plan', False)
    hit_feature_mode = environment_config.get('hit_feature_mode', 'absolute')
    state_feature_mode = environment_config.get('state_feature_mode', 'full')
    state_feature_set = environment_config.get('state_feature_set', 'unspecified')
    
    # Initialize model with configuration
    model = DQNLightning(
        dm,
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
        hit_feature_mode=hit_feature_mode,
        state_feature_mode=state_feature_mode,
        state_feature_set=state_feature_set,
    )
    
    # Extract wandb configuration (needed for checkpoint directory name)
    wandb_config = config.get('wandb', {})
    wandb_project = wandb_config.get('project', 'RLTracking')
    base_name = wandb_config.get('name', 'initial_run')
    
    # Incorporate pt cut into run name if it exists
    # Remove any existing ptcut prefix first, then add the current one
    pt_cut = particle_filters.get('pt', None)
    if pt_cut is not None:
        # Remove existing ptcut prefix if present (e.g., "ptcut1_initial_run" -> "initial_run")
        base_name = re.sub(r'^ptcut\d+\.?\d*_', '', base_name)
        # Add current pt cut prefix
        base_name = f"ptcut{pt_cut}_{base_name}"
    
    # Generate unique run name by appending random string and timestamp
    # This ensures each run gets a unique directory even with the same config
    random_suffix = secrets.token_hex(4)  # 8-character random hex string
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb_name = f"{base_name}_{timestamp}_{random_suffix}"
    
    # Create checkpoint directory with unique experiment name
    checkpoint_dir = Path('checkpoints') / wandb_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting training run: {wandb_name}")
    print(f"Checkpoint directory: {checkpoint_dir.absolute()}")
    print(f"Config file: {config_path}")
    print(f"\nConfiguration Summary:")
    print(f"  Particle filters: {particle_filters}")
    print(f"  Hit filters: {hit_filters}")
    print(f"  Use distance reward: {use_distance_reward}")
    print(f"  Use truth path plan: {use_truth_path_plan}")
    print(f"  Hit feature mode: {hit_feature_mode}")
    print(f"  State feature mode: {state_feature_mode}")
    print(f"  State feature set: {state_feature_set}")
    print(f"  Learning rate: {model_config.get('lr', 0.01)}")
    print(f"  Batch size: {model_config.get('batch_size', 32)}")
    print(f"{'='*60}\n")
    
    # Extract trainer configuration
    trainer_config = config.get('trainer', {})
    checkpoint_config = trainer_config.get('callbacks', {}).get('checkpoint', {})
    early_stopping_config = trainer_config.get('callbacks', {}).get('early_stopping', {})
    
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config.get('monitor', 'val_hit_accuracy'),
        save_top_k=checkpoint_config.get('save_top_k', 1),
        mode=checkpoint_config.get('mode', 'max'),
        filename=checkpoint_config.get('filename', 'best-{epoch:02d}-{val_hit_accuracy:.4f}'),
        save_last=checkpoint_config.get('save_last', True),  # Also save last checkpoint
        dirpath=str(checkpoint_dir),  # Save checkpoints in experiment-specific directory
    )
    
    # Save config file to checkpoint directory for easy loading during evaluation
    config_dest = checkpoint_dir / Path(config_path).name
    shutil.copy2(config_path, config_dest)
    print(f"Saved config to {config_dest}")
    
    # Early stopping callback - stop if validation accuracy doesn't improve
    # Use validation accuracy since it's evaluated with epsilon=0 (pure exploitation)
    # This tracks actual policy quality, not just reduced exploration
    early_stopping_callback = EarlyStopping(
        monitor=early_stopping_config.get('monitor', 'val_hit_accuracy'),
        mode=early_stopping_config.get('mode', 'max'),  # Stop when efficiency stops increasing
        patience=early_stopping_config.get('patience', 50),  # Wait 50 validation checks
        min_delta=early_stopping_config.get('min_delta', 0.001),  # Minimum change to qualify as improvement
        verbose=True,
    )
    
    wandb_offline = wandb_config.get('offline', True)
    wandb_log_model = wandb_config.get('log_model', True)
    if wandb_offline and wandb_log_model:
        print("WandB offline mode enabled; disabling model artifact uploads.")
        wandb_log_model = False

    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        log_model=wandb_log_model,
        offline=wandb_offline,
    )
    
    # Initialize trainer
    trainer_kwargs = {
        "max_epochs": trainer_config.get('max_epochs', 5000),
        "callbacks": [checkpoint_callback, early_stopping_callback],
        "accelerator": trainer_config.get('accelerator', 'gpu'),
        "log_every_n_steps": trainer_config.get('log_every_n_steps', 10),
        "gradient_clip_val": trainer_config.get('gradient_clip_val', None),
        "logger": wandb_logger,
        "check_val_every_n_epoch": early_stopping_config.get('check_val_every_n_epoch', 1),
    }

    devices = trainer_config.get('devices')
    if devices is None and trainer_kwargs["accelerator"] in {'gpu', 'cuda'}:
        devices = 1
    if devices is not None:
        trainer_kwargs["devices"] = devices

    precision = trainer_config.get('precision')
    if precision is not None:
        trainer_kwargs["precision"] = precision

    strategy = trainer_config.get('strategy')
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy

    trainer = Trainer(**trainer_kwargs)
    
    # Train the model
    trainer.fit(model)
    
    _print_split_summary(dm.get_split_summary())


def _print_split_summary(summary: dict) -> None:
    """Print number of events drawn from each dataset part for every split."""
    if not summary:
        return
    print(f"\n{'='*60}")
    print("DATASET SPLIT SUMMARY")
    print(f"{'='*60}")
    for split_name in ['train', 'val', 'test']:
        counts = summary.get(split_name, {})
        total = sum(counts.values())
        print(f"\n{split_name.upper()} ({total} events)")
        if not counts:
            print("  (no events)")
            continue
        for part, count in sorted(counts.items()):
            print(f"  {part}: {count}")
    print(f"{'='*60}\n")


def _resolve_config_path(config_path: str | Path) -> Path:
    """
    Resolve a configuration path, supporting execution from arbitrary working directories.

    Args:
        config_path: User-provided config path (absolute or relative).

    Returns:
        Resolved Path object pointing to an existing file.

    Raises:
        FileNotFoundError: If the configuration file cannot be located.
    """
    path = Path(config_path).expanduser()

    search_candidates = []

    if path.is_absolute():
        search_candidates.append(path)
    else:
        # Working directory
        search_candidates.append((Path.cwd() / path).resolve())

        # Relative to training script directory and its parent (project src root)
        script_dir = Path(__file__).resolve().parent
        search_candidates.append((script_dir / path).resolve())
        search_candidates.append((script_dir.parent / path).resolve())

    for candidate in search_candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in search_candidates)
    raise FileNotFoundError(
        f"Config file not found: {config_path}. Locations checked: {searched}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN model with configuration file')
    parser.add_argument(
        '--config',
        type=str,
        default='../config/train_config.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()
    
    main(args.config)
