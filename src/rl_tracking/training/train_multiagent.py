"""Training script for Multi-Agent DQN model."""
from pathlib import Path
import argparse
import shutil
import secrets
import re
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from rl_tracking.lightning_modules.multiagent_dqn import MultiAgentDQNLightning
from rl_tracking.preprocessing.hit_candidates import EventProcessor
from rl_tracking.utils.stream_loading import TrackingDataModule
from rl_tracking.utils.config_loader import load_config


def main(config_path: str | Path):
    """Main training function for multi-agent DQN.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract data configuration
    data_config = config.get('data', {})
    data_dir = Path(data_config.get('data_dir', '/Users/liv/trackML/train_1/'))
    batch_size = data_config.get('batch_size', 32)
    val_split = data_config.get('val_split', 0.2)
    test_split = data_config.get('test_split', 0.1)
    num_workers = data_config.get('num_workers', 0)
    event_processor_config = data_config.get('event_processor', {})
    n_neighbors = event_processor_config.get('n_neighbors', 20)
    
    # Initialize dataset and dataloader
    ep = EventProcessor(data_dir, n_neighbors)
    dm = TrackingDataModule(
        file_paths=data_dir,
        batch_size=batch_size,
        event_processor=ep,
        num_workers=num_workers,
        val_split=val_split,
        test_split=test_split
    )
    
    dm.setup(stage="fit")
    
    # Extract model configuration
    model_config = config.get('model', {})
    environment_config = config.get('environment', {})
    particle_filters = environment_config.get('particle_filters', {})
    hit_filters = environment_config.get('hit_filters', {})
    use_distance_reward = environment_config.get('use_distance_reward', False)
    use_truth_path_plan = environment_config.get('use_truth_path_plan', False)
    proximity_threshold = environment_config.get('proximity_threshold', 50.0)
    
    # Initialize multi-agent model
    model = MultiAgentDQNLightning(
        dm,
        num_agents=model_config.get('num_agents', 3),
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
        proximity_threshold=proximity_threshold,
    )
    
    # Extract wandb configuration
    wandb_config = config.get('wandb', {})
    wandb_project = wandb_config.get('project', 'RLTracking')
    base_name = wandb_config.get('name', 'multiagent_initial_run')
    
    # Incorporate pt cut and num_agents into run name
    pt_cut = particle_filters.get('pt', None)
    num_agents = model_config.get('num_agents', 3)
    if pt_cut is not None:
        base_name = re.sub(r'^ptcut\d+\.?\d*_', '', base_name)
        base_name = f"ptcut{pt_cut}_{base_name}"
    base_name = f"ma{num_agents}_{base_name}"  # ma = multi-agent
    
    # Generate unique run name
    random_suffix = secrets.token_hex(4)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb_name = f"{base_name}_{timestamp}_{random_suffix}"
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints') / wandb_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting Multi-Agent training run: {wandb_name}")
    print(f"Checkpoint directory: {checkpoint_dir.absolute()}")
    print(f"Config file: {config_path}")
    print(f"\nConfiguration Summary:")
    print(f"  Number of agents: {num_agents}")
    print(f"  Proximity threshold: {proximity_threshold} cm")
    print(f"  Particle filters: {particle_filters}")
    print(f"  Hit filters: {hit_filters}")
    print(f"  Use distance reward: {use_distance_reward}")
    print(f"  Use truth path plan: {use_truth_path_plan}")
    print(f"  Learning rate: {model_config.get('lr', 0.01)}")
    print(f"  Batch size: {model_config.get('batch_size', 32)}")
    print(f"{'='*60}\n")
    
    # Extract trainer configuration
    trainer_config = config.get('trainer', {})
    checkpoint_config = trainer_config.get('callbacks', {}).get('checkpoint', {})
    early_stopping_config = trainer_config.get('callbacks', {}).get('early_stopping', {})
    
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        save_top_k=checkpoint_config.get('save_top_k', 1),
        mode=checkpoint_config.get('mode', 'min'),
        filename=checkpoint_config.get('filename', 'best-{epoch:02d}-{val_loss:.4f}'),
        save_last=checkpoint_config.get('save_last', True),
        dirpath=str(checkpoint_dir),
    )
    
    # Save config file
    config_dest = checkpoint_dir / Path(config_path).name
    shutil.copy2(config_path, config_dest)
    print(f"Saved config to {config_dest}")
    
    early_stopping_callback = EarlyStopping(
        monitor=early_stopping_config.get('monitor', 'val_hit_accuracy'),
        mode=early_stopping_config.get('mode', 'max'),
        patience=early_stopping_config.get('patience', 50),
        min_delta=early_stopping_config.get('min_delta', 0.001),
        verbose=True,
    )
    
    wandb_logger = WandbLogger(
        project=wandb_project,
        name=wandb_name,
        log_model=wandb_config.get('log_model', True),
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=trainer_config.get('max_epochs', 5000),
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator=trainer_config.get('accelerator', 'auto'),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 10),
        gradient_clip_val=trainer_config.get('gradient_clip_val', None),
        logger=wandb_logger,
        check_val_every_n_epoch=early_stopping_config.get('check_val_every_n_epoch', 1),
    )
    
    # Train the model
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Multi-Agent DQN model')
    parser.add_argument(
        '--config',
        type=str,
        default='../config/multiagent_train_config.yaml',
        help='Path to configuration YAML file'
    )
    args = parser.parse_args()
    
    main(args.config)


