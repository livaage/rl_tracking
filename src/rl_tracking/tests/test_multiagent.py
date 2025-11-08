"""Test script for multi-agent tracking system."""
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_tracking.environment.multiagent_tracking_env import MultiAgentTrackingEnv
from rl_tracking.environment.multiagent import MultiAgent
from rl_tracking.models.pointer_network import PointerNetwork
from rl_tracking.replay.buffer import ReplayBuffer
from rl_tracking.preprocessing.hit_candidates import EventProcessor
from torch.utils.data import DataLoader


def test_multiagent_environment():
    """Test that the multi-agent environment initializes and runs."""
    print("Testing Multi-Agent Environment...")
    
    # Create a simple data loader (using a small subset for testing)
    data_dir = Path("/Users/liv/trackML/train_1/")
    if not data_dir.exists():
        print(f"Warning: Data directory {data_dir} does not exist. Skipping test.")
        return
    
    # Create event processor
    ep = EventProcessor(data_dir, n_neighbors=20)
    
    # Get file list
    file_list = list(data_dir.glob("event*-hits.csv"))
    if len(file_list) == 0:
        print(f"Warning: No event files found in {data_dir}. Skipping test.")
        return
    
    # Use just first file for testing
    from rl_tracking.utils.stream_loading import RLKernelIterableDataset
    dataset = RLKernelIterableDataset(file_list[:1], ep, seed=42)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    # Create environment
    num_agents = 3
    env = MultiAgentTrackingEnv(
        dataloader,
        num_agents=num_agents,
        particle_filters={'pt': 0.5, 'nhits_min': 4, 'nhits_max': 20},
        use_truth_path_plan=True,
        proximity_threshold=50.0
    )
    
    print(f"✓ Environment created with {num_agents} agents")
    
    # Test reset
    observations, info = env.reset()
    
    if observations is None:
        print("Warning: No observations returned from reset. May need more data.")
        return
    
    print(f"✓ Environment reset successful")
    print(f"  - Observation shape: {len(observations)} agents")
    print(f"  - Each observation dim: {observations[0].shape if observations else 'N/A'}")
    
    # Check observations have correct dimension
    expected_obs_dim = 4 + 4 * (num_agents - 1)  # Own state + other agents
    for i, obs in enumerate(observations):
        assert obs.shape[0] == expected_obs_dim, f"Agent {i} obs shape mismatch: {obs.shape[0]} != {expected_obs_dim}"
    print(f"✓ Observations have correct dimension: {expected_obs_dim}")
    
    # Test step
    actions = [0] * num_agents  # Each agent selects first hit
    next_obs, rewards, dones, next_info = env.step(actions)
    
    print(f"✓ Environment step successful")
    print(f"  - Rewards: {rewards}")
    print(f"  - Dones: {dones}")
    print(f"  - Any agent still active: {not all(dones)}")
    
    # Test a few more steps
    for step in range(5):
        if next_obs is None or all(dones):
            print(f"  - Episode ended after {step} additional steps")
            break
        
        # Random actions
        actions = [np.random.randint(0, 20) for _ in range(num_agents)]
        next_obs, rewards, dones, next_info = env.step(actions)
    
    print(f"✓ Multiple steps completed successfully")
    
    # Test accuracy tracking
    for i in range(num_agents):
        acc = env.get_accuracy(i)
        print(f"  - Agent {i} accuracy: {acc:.3f}")
    
    print("\n✅ Environment test passed!\n")


def test_multiagent_coordinator():
    """Test the multi-agent coordinator."""
    print("Testing Multi-Agent Coordinator...")
    
    # Create a simple data loader
    data_dir = Path("/Users/liv/trackML/train_1/")
    if not data_dir.exists():
        print(f"Warning: Data directory {data_dir} does not exist. Skipping test.")
        return
    
    ep = EventProcessor(data_dir, n_neighbors=20)
    file_list = list(data_dir.glob("event*-hits.csv"))
    if len(file_list) == 0:
        print(f"Warning: No event files found in {data_dir}. Skipping test.")
        return
    
    from rl_tracking.utils.stream_loading import RLKernelIterableDataset
    dataset = RLKernelIterableDataset(file_list[:1], ep, seed=42)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    # Create environment and agent
    num_agents = 3
    env = MultiAgentTrackingEnv(
        dataloader,
        num_agents=num_agents,
        particle_filters={'pt': 0.5, 'nhits_min': 4, 'nhits_max': 20},
        use_truth_path_plan=True,
    )
    
    replay_buffer = ReplayBuffer(capacity=1000)
    agent = MultiAgent(env, replay_buffer)
    
    print(f"✓ Multi-agent coordinator created")
    
    # Create a simple network
    state_dim = 4 + 4 * (num_agents - 1)
    hit_dim = 4
    hidden_dim = 128
    net = PointerNetwork(state_dim=state_dim, hit_dim=hit_dim, hidden_dim=hidden_dim)
    
    print(f"✓ Pointer network created (state_dim={state_dim}, hit_dim={hit_dim})")
    
    # Test getting actions
    actions = agent.get_actions(net, epsilon=0.5, device="cpu")
    print(f"✓ Actions generated: {actions}")
    assert len(actions) == num_agents, f"Expected {num_agents} actions, got {len(actions)}"
    
    # Test play_step
    rewards, done = agent.play_step(net, epsilon=0.5, device="cpu")
    print(f"✓ Play step completed")
    print(f"  - Rewards: {rewards}")
    print(f"  - Done: {done}")
    
    # Test replay buffer
    print(f"  - Replay buffer size: {len(replay_buffer)}")
    assert len(replay_buffer) > 0, "Replay buffer should have experiences"
    
    # Run a few more steps
    for step in range(10):
        rewards, done = agent.play_step(net, epsilon=0.3, device="cpu")
        if done:
            print(f"  - Episode completed after {step+1} steps")
            break
    
    print(f"✓ Multiple play steps completed")
    print(f"  - Final buffer size: {len(replay_buffer)}")
    
    print("\n✅ Coordinator test passed!\n")


def test_network_forward():
    """Test that the network handles multi-agent observations."""
    print("Testing Network Forward Pass...")
    
    num_agents = 3
    state_dim = 4 + 4 * (num_agents - 1)
    hit_dim = 4
    max_hits = 20
    batch_size = 8
    
    # Create network
    net = PointerNetwork(state_dim=state_dim, hit_dim=hit_dim, hidden_dim=128)
    
    # Create dummy batch
    states = torch.randn(batch_size, state_dim)
    hit_features = torch.randn(batch_size, max_hits, hit_dim)
    hit_mask = torch.ones(batch_size, max_hits, dtype=torch.bool)
    # Mask out some hits
    hit_mask[:, 10:] = False
    
    # Forward pass
    scores = net(states, hit_features, mask=hit_mask)
    
    print(f"✓ Forward pass successful")
    print(f"  - Input: states={states.shape}, hit_features={hit_features.shape}")
    print(f"  - Output: scores={scores.shape}")
    
    assert scores.shape == (batch_size, max_hits), f"Expected shape ({batch_size}, {max_hits}), got {scores.shape}"
    
    # Check that masked hits have very negative scores
    masked_scores = scores[:, 10:]
    assert (masked_scores < -50).all(), "Masked hits should have very negative scores"
    
    print(f"✓ Masking works correctly")
    
    # Test gradient flow
    loss = scores.mean()
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    assert has_grad, "Network should have gradients"
    
    print(f"✓ Gradients flow correctly")
    print("\n✅ Network test passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Multi-Agent Tracking System Tests")
    print("="*60 + "\n")
    
    try:
        test_multiagent_environment()
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_multiagent_coordinator()
    except Exception as e:
        print(f"❌ Coordinator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_network_forward()
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*60)
    print("Tests completed!")
    print("="*60)


