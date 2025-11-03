"""Training script for DQN model with configuration file support."""
from pathlib import Path
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from rl_tracking.lightning_modules.dqn import DQNLightning
from rl_tracking.preprocessing.hit_candidates import EventProcessor
from rl_tracking.utils.stream_loading import TrackingDataModule
from rl_tracking.utils.config_loader import load_config


def main(config_path: str | Path):
    """Main training function.
    
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
        val_split=val_split
    )
    
    dm.setup(stage="fit")
    
    # Extract model configuration
    model_config = config.get('model', {})
    environment_config = config.get('environment', {})
    particle_filters = environment_config.get('particle_filters', {})
    hit_filters = environment_config.get('hit_filters', {})
    use_distance_reward = environment_config.get('use_distance_reward', False)
    
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
    )
    
    # Extract trainer configuration
    trainer_config = config.get('trainer', {})
    checkpoint_config = trainer_config.get('callbacks', {}).get('checkpoint', {})
    
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        save_top_k=checkpoint_config.get('save_top_k', 1),
        mode=checkpoint_config.get('mode', 'min')
    )
    
    # Extract wandb configuration
    wandb_config = config.get('wandb', {})
    wandb_logger = WandbLogger(
        project=wandb_config.get('project', 'RLTracking'),
        name=wandb_config.get('name', 'ptcut2_initial_run'),
        log_model=wandb_config.get('log_model', True),
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=trainer_config.get('max_epochs', 5000),
        callbacks=[checkpoint_callback],
        accelerator=trainer_config.get('accelerator', 'auto'),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 10),
        gradient_clip_val=trainer_config.get('gradient_clip_val', None),  # Get from config, default None
        logger=wandb_logger,
    )
    
    # Train the model
    trainer.fit(model)


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
