from torch.utils.data import IterableDataset
from typing import Iterator, Tuple
from rl_tracking.replay.buffer import ReplayBuffer
import numpy as np

class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time

    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        # Only yield if buffer has enough samples
        if len(self.buffer) == 0:
            return
        
        # Sample batch_size or available samples, whichever is smaller
        sample_count = min(self.sample_size, len(self.buffer))
        if sample_count == 0:
            return
            
        batch = self.buffer.sample(sample_count)
        # Handle both old format (5 items) and new format (9 items with hit_features)
        if len(batch) == 5:
            states, actions, rewards, dones, new_states = batch
            for i in range(len(dones)):
                state = np.asarray(states[i], dtype=np.float32)
                action = np.asarray(actions[i], dtype=np.int64)
                reward = np.asarray(rewards[i], dtype=np.float32)
                done = np.asarray(dones[i], dtype=bool)
                new_state = np.asarray(new_states[i], dtype=np.float32)
                yield state, action, reward, done, new_state
        else:
            # New format with hit_features
            states, actions, rewards, dones, new_states, hit_features, next_hit_features, hit_masks, next_hit_masks = batch
            for i in range(len(dones)):
                state = np.asarray(states[i], dtype=np.float32)
                action = np.asarray(actions[i], dtype=np.int64)
                reward = np.asarray(rewards[i], dtype=np.float32)
                done = np.asarray(dones[i], dtype=bool)
                new_state = np.asarray(new_states[i], dtype=np.float32)
                hit_feat = np.asarray(hit_features[i], dtype=np.float32)
                next_hit_feat = np.asarray(next_hit_features[i], dtype=np.float32)
                hit_mask = np.asarray(hit_masks[i], dtype=bool)
                next_hit_mask = np.asarray(next_hit_masks[i], dtype=bool)
                yield state, action, reward, done, new_state, hit_feat, next_hit_feat, hit_mask, next_hit_mask
