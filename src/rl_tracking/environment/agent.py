from rl_tracking.replay.buffer import ReplayBuffer
from torch import nn
import torch
import numpy as np
from typing import Tuple
from rl_tracking.environment.tracking_env import TrackingEnv

# Experience is now defined in replay.buffer to include hit_features
from rl_tracking.replay.buffer import Experience


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: TrackingEnv, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.state, self.info = self.env.reset()

    def reset(self) -> None:
        """Resets the environment and updates the state."""
        self.state, self.info = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the pointer network, score candidate hits and select the best one.

        Args:
            net: Pointer network that scores hits based on state
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action: Index of selected hit
        """
        # Handle case where state might be None
        if self.state is None:
            return 0  # Default action

        # Get hit features and mask from info
        info = getattr(self, "info", {})
        hit_features = info.get("hit_features", None)
        hit_mask = info.get("hit_mask", None)

        if hit_features is None or hit_mask is None:
            if np.random.random() < epsilon:
                action_space = getattr(self.env, "action_space", None)
                if action_space is not None:
                    return int(action_space.sample())
                return 0
            return 0

        # Ensure hit_features and hit_mask have consistent shapes
        hit_features = np.asarray(hit_features, dtype=np.float32)
        hit_mask = np.asarray(hit_mask, dtype=bool)

        # Ensure consistent max_hits (20)
        max_hits = 20
        hit_dim = 4

        # Truncate or pad hit_features to exactly [max_hits, hit_dim]
        if hit_features.shape[0] > max_hits:
            hit_features = hit_features[:max_hits]
            hit_mask = hit_mask[:max_hits]
        elif hit_features.shape[0] < max_hits:
            # Pad hit_features
            padding_needed = max_hits - hit_features.shape[0]
            if len(hit_features.shape) == 2:
                padding = np.zeros((padding_needed, hit_dim), dtype=np.float32)
                hit_features = np.vstack([hit_features, padding])
            else:
                # Reshape if needed
                hit_features = hit_features.reshape(-1, hit_dim)
                padding = np.zeros((padding_needed, hit_dim), dtype=np.float32)
                hit_features = np.vstack([hit_features, padding])

            # Pad mask with False
            hit_mask = np.pad(hit_mask, (0, padding_needed), constant_values=False)

        # Ensure shapes are correct
        if hit_features.shape != (max_hits, hit_dim):
            hit_features = hit_features.reshape(max_hits, hit_dim)[:max_hits]
        if len(hit_mask) != max_hits:
            if len(hit_mask) > max_hits:
                hit_mask = hit_mask[:max_hits]
            else:
                hit_mask = np.pad(
                    hit_mask, (0, max_hits - len(hit_mask)), constant_values=False
                )

        # Epsilon-greedy: random action
        if np.random.random() < epsilon:
            # Sample from valid hits only
            valid_indices = np.where(hit_mask)[0]
            if len(valid_indices) > 0:
                return int(np.random.choice(valid_indices))
            else:
                return 0

        # Use pointer network to score hits
        state_tensor = torch.tensor(self.state, dtype=torch.float32)
        hit_features_tensor = torch.tensor(hit_features, dtype=torch.float32)
        hit_mask_tensor = torch.tensor(hit_mask, dtype=torch.bool)

        if device not in ["cpu"]:
            state_tensor = state_tensor.to(device)
            hit_features_tensor = hit_features_tensor.to(device)
            hit_mask_tensor = hit_mask_tensor.to(device)

        # Get scores for all hits
        scores = net(
            state_tensor, hit_features_tensor, mask=hit_mask_tensor
        )  # [num_hits]

        # Select hit with highest score (that's valid)
        # Apply mask: set invalid hits to very negative score
        scores_masked = scores.clone()
        scores_masked[~hit_mask_tensor] = (
            -100.0
        )  # Match the range used in pointer network
        action = int(torch.argmax(scores_masked).item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """
        action = self.get_action(net, epsilon, device)
        state = self.state
        state_info = self.info  # Store info for replay buffer

        if state is not None:
            arr = np.asarray(state, dtype=np.float32)
            if arr.size not in (0, 15):
                print(f"[Agent] Current state size: {arr.size}, shape: {arr.shape}")
        else:
            print("[Agent] Current state is None")

        # Extract hit features and mask from current state info
        hit_features = state_info.get(
            "hit_features", np.zeros((20, 4), dtype=np.float32)
        )
        hit_mask = state_info.get("hit_mask", np.zeros(20, dtype=bool))

        # do step in the environment
        next_state, reward, done, _, step_info = self.env.step(action)
        if next_state is not None:
            next_arr = np.asarray(next_state, dtype=np.float32)
            if next_arr.size not in (0, 15):
                print(f"[Agent] Next state size: {next_arr.size}, shape: {next_arr.shape}, type: {type(next_state)}")
        else:
            print("[Agent] Next state is None")
        next_info = step_info or {}

        # Extract hit features and mask from next state info
        next_hit_features = next_info.get(
            "hit_features", np.zeros((20, 4), dtype=np.float32)
        )
        next_hit_mask = next_info.get("hit_mask", np.zeros(20, dtype=bool))

        # Store experience with hit features
        self.replay_buffer.append(
            Experience(
                state,
                action,
                next_state,
                reward,
                done,
                hit_features,
                next_hit_features,
                hit_mask,
                next_hit_mask,
            )
        )

        self.state = next_state
        self.info = next_info
        return reward, done
