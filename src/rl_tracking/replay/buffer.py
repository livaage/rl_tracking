import numpy as np
from torch import nn
from typing import Tuple
from collections import deque, namedtuple
import random

Experience = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward', 'done'])


class ReplayBuffer(object):
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, next_states, rewards, dones = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),

        )

    def __len__(self):
        return len(self.memory)