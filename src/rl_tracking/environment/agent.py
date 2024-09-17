from collections import namedtuple
from replay.buffer import ReplayBuffer
from torch import nn
import torch
import numpy as np
from typing import Tuple

Experience = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward', 'done'])


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state, info = self.env.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state, info = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()

        else:
            state = torch.tensor(self.state)
            # state = self.state
            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=0)
            action = int(action.item())

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
        state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        state = self.state
        # print("state is", state)
        # do step in the environment
        next_state, reward, done, truncated, info = self.env.step(action)

        self.replay_buffer.append(Experience(state, action, next_state, reward, done))

        self.state = next_state
        if done:
            self.reset()
        return reward, done


