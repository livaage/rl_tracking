# RL Tracking

A Reinforcement Learning framework for particle tracking in particle physics experiments. This project uses Deep Q-Networks (DQN) implemented with PyTorch Lightning to learn optimal tracking strategies.

## Overview

This project implements a DQN-based agent for particle tracking reconstruction in high-energy physics detectors. The agent learns to propagate tracks through detector layers, selecting the best hit candidates at each step to reconstruct particle trajectories.

## Project Structure

```
rl_tracking/
├── src/rl_tracking/
│   ├── analysis/          # Analysis notebooks
│   ├── environment/       # RL environment and agent
│   │   ├── agent.py      # DQN agent implementation
│   │   └── tracking_env.py  # Gym-like tracking environment
│   ├── lightning_modules/  # PyTorch Lightning modules
│   │   └── dqn.py         # DQN Lightning module
│   ├── models/            # Neural network models
│   │   └── mlp.py         # DQN network architecture
│   ├── physics/           # Physics simulation and tracking logic
│   │   ├── track.py       # Helix track model
│   │   ├── propagate.py   # Track propagation
│   │   └── seed.py        # Track seeding
│   ├── preprocessing/     # Data preprocessing
│   │   └── KT_barrel_prep.py  # Hit candidates preparation
│   ├── replay/            # Experience replay
│   │   └── buffer.py      # Replay buffer
│   ├── training/          # Training scripts
│   │   └── train.py       # Main training script
│   └── utils/             # Utility functions
│       ├── loading.py     # Data loading utilities
│       └── experience_loader.py  # Experience dataset
└── README.md
```

## Installation

### Using Poetry (Recommended)

This project uses Poetry for dependency management:

```bash
# Navigate to the project root directory
cd /Users/liv/rl_tracking

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
# Navigate to the project root directory (NOT the training subdirectory)
cd /Users/liv/rl_tracking

# Install in development mode
pip install -e .

# Or install for current user only
pip install --user -e .
```

**Important:** Make sure you're in the `/Users/liv/rl_tracking` directory (where `pyproject.toml` and `setup.py` are located), NOT in the `training/` subdirectory.

#### Troubleshooting

**Error: "file does not appear to be a Python project"**
- You're in the wrong directory! The project root should contain both `pyproject.toml` and `setup.py`
- Navigate to the project root: `cd /Users/liv/rl_tracking`
- Then run `pip install -e .`

**Error: "trackml" module not found**
- Install TrackML from source: `pip install git+https://github.com/LAL/trackml-library`
- Or use Poetry which will handle the git dependency automatically

## Dependencies

- Python 3.12+
- PyTorch 2.4.1+
- PyTorch Lightning 2.4.0+
- Gymnasium 1.0.0+
- Pandas 2.2.2+
- scikit-learn 1.5.2+
- TrackML library
- Wandb 0.19.3+
- Numba 0.61.0+

## Usage

### Training

Start training by running the main training script:

```bash
cd /Users/liv/rl_tracking/src/rl_tracking/training
python train.py
```

The training script will:
1. Initialize the event processor and data module
2. Create a DQN Lightning model
3. Set up Weights & Biases logging
4. Train for 200 epochs with automatic checkpointing

### Configuration

You can modify training parameters in `training/train.py`:
- `batch_size`: Batch size for training
- `lr`: Learning rate
- `gamma`: Discount factor for RL
- `replay_size`: Size of experience replay buffer
- `max_epochs`: Number of training epochs
- Data directory path for TrackML data

## Features

- **DQN Agent**: Deep Q-Network implementation for discrete action selection
- **Tracking Environment**: Custom Gymnasium environment for particle tracking
- **Helix Propagation**: Physics-based track propagation through detector layers
- **Experience Replay**: Efficient experience storage and sampling
- **Automatic Checkpointing**: Model checkpoints saved automatically
- **Wandb Integration**: Experiment tracking and visualization

## Data Files

The project includes several data files needed for track propagation:
- `src/rl_tracking/tml_layer_remap.csv`: Layer ID remapping for detector geometry
- `src/rl_tracking/physics/allowed_layer_connections.csv`: Valid layer transitions
- `src/rl_tracking/physics/layer_info.csv`: Detector layer information
- `src/rl_tracking/physics/theta_path_plan.pkl`: Track path planning data
- `src/rl_tracking/physics/momentum_theta_path_plan_100.json`: Momentum-based path plans

These files are automatically loaded relative to their respective modules.

## Data Format

The system expects TrackML-compatible data with the following columns:
- `hit_id`, `x`, `y`, `z`: Hit position information
- `volume_id`, `layer_id`, `module_id`: Detector geometry
- `particle_id`: Ground truth particle ID
- `pt`: Transverse momentum
- `nhits`: Number of hits per track

## Environment Details

The tracking environment uses:
- **Observation Space**: Box(0, 120, (3, 4)) - 3x4 feature matrix
- **Action Space**: Discrete(3) - 3 discrete actions
- **Episode Termination**: When track propagation completes or no valid hits found

## Experimental Tracking

Training runs are automatically logged to Weights & Biases. View experiments at:
- Project: RLTracking
- Run metrics include: reward, epsilon, train_loss, episode_reward

## Contributing

1. Make sure dependencies are installed using `poetry install`
2. Follow the existing code structure
3. Update documentation as needed

## Author

liv <liv.helen.vage@cern.ch>

## License

[Add your license here]
