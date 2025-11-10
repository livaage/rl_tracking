import numpy as np
from torch import nn
from typing import Tuple
from collections import deque, namedtuple
import random

Experience = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward', 'done', 
                         'hit_features', 'next_hit_features', 'hit_mask', 'next_hit_mask'])


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
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # If buffer is smaller than batch_size, sample with replacement
        actual_batch_size = min(batch_size, len(self.buffer))
        replace = batch_size > len(self.buffer)
        indices = np.random.choice(len(self.buffer), actual_batch_size, replace=replace)
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, next_states, rewards, dones, hit_features, next_hit_features, hit_masks, next_hit_masks = zip(*experiences)
        
        # Ensure all states have consistent shape - convert to numpy arrays and ensure same shape
        # Determine state dimensionality from available samples
        state_dim = 0
        for candidate in list(states) + list(next_states):
            if candidate is not None:
                candidate_len = len(np.asarray(candidate, dtype=np.float32).flatten())
                if candidate_len > state_dim:
                    state_dim = candidate_len
        if state_dim == 0:
            state_dim = 4  # Fallback if all states are None

        states_list = []
        next_states_list = []

        for state, next_state in zip(states, next_states):
            # Handle current state
            if state is None:
                state_arr = np.zeros(state_dim, dtype=np.float32)
            else:
                state_arr = np.asarray(state, dtype=np.float32).flatten()
                if len(state_arr) < state_dim:
                    state_arr = np.pad(state_arr, (0, state_dim - len(state_arr)), mode='constant')
                elif len(state_arr) > state_dim:
                    state_arr = state_arr[:state_dim]

            # Handle next state
            if next_state is None:
                next_state_arr = np.zeros(state_dim, dtype=np.float32)
            else:
                next_state_arr = np.asarray(next_state, dtype=np.float32).flatten()
                if len(next_state_arr) < state_dim:
                    next_state_arr = np.pad(next_state_arr, (0, state_dim - len(next_state_arr)), mode='constant')
                elif len(next_state_arr) > state_dim:
                    next_state_arr = next_state_arr[:state_dim]

            states_list.append(state_arr)
            next_states_list.append(next_state_arr)
        
        # Process hit_features: ensure they're numpy arrays
        hit_features_list = []
        next_hit_features_list = []
        hit_masks_list = []
        next_hit_masks_list = []
        
        max_hits = 20
        hit_dim = 4
        
        for hf, nhf, hm, nhm in zip(hit_features, next_hit_features, hit_masks, next_hit_masks):
            hf_arr = np.asarray(hf, dtype=np.float32) if hf is not None else np.zeros((max_hits, hit_dim), dtype=np.float32)
            nhf_arr = np.asarray(nhf, dtype=np.float32) if nhf is not None else np.zeros((max_hits, hit_dim), dtype=np.float32)
            hm_arr = np.asarray(hm, dtype=bool) if hm is not None else np.zeros(max_hits, dtype=bool)
            nhm_arr = np.asarray(nhm, dtype=bool) if nhm is not None else np.zeros(max_hits, dtype=bool)
            
            # Normalize shapes: truncate if > max_hits, pad if < max_hits
            # Handle hit_features
            if len(hf_arr.shape) == 2:
                if hf_arr.shape[0] > max_hits:
                    hf_arr = hf_arr[:max_hits]
                    hm_arr = hm_arr[:max_hits] if len(hm_arr) > max_hits else hm_arr
                elif hf_arr.shape[0] < max_hits:
                    padding = np.zeros((max_hits - hf_arr.shape[0], hit_dim), dtype=np.float32)
                    hf_arr = np.vstack([hf_arr, padding])
                    hm_arr = np.pad(hm_arr, (0, max_hits - len(hm_arr)), constant_values=False)
                if hf_arr.shape[1] != hit_dim:
                    # Fix feature dimension
                    hf_arr = hf_arr[:, :hit_dim] if hf_arr.shape[1] > hit_dim else np.pad(hf_arr, ((0, 0), (0, hit_dim - hf_arr.shape[1])), mode='constant')
            else:
                hf_arr = np.zeros((max_hits, hit_dim), dtype=np.float32)
            
            # Handle next_hit_features
            if len(nhf_arr.shape) == 2:
                if nhf_arr.shape[0] > max_hits:
                    nhf_arr = nhf_arr[:max_hits]
                    nhm_arr = nhm_arr[:max_hits] if len(nhm_arr) > max_hits else nhm_arr
                elif nhf_arr.shape[0] < max_hits:
                    padding = np.zeros((max_hits - nhf_arr.shape[0], hit_dim), dtype=np.float32)
                    nhf_arr = np.vstack([nhf_arr, padding])
                    nhm_arr = np.pad(nhm_arr, (0, max_hits - len(nhm_arr)), constant_values=False)
                if nhf_arr.shape[1] != hit_dim:
                    nhf_arr = nhf_arr[:, :hit_dim] if nhf_arr.shape[1] > hit_dim else np.pad(nhf_arr, ((0, 0), (0, hit_dim - nhf_arr.shape[1])), mode='constant')
            else:
                nhf_arr = np.zeros((max_hits, hit_dim), dtype=np.float32)
            
            # Ensure masks are exactly max_hits
            if len(hm_arr) > max_hits:
                hm_arr = hm_arr[:max_hits]
            elif len(hm_arr) < max_hits:
                hm_arr = np.pad(hm_arr, (0, max_hits - len(hm_arr)), constant_values=False)
            
            if len(nhm_arr) > max_hits:
                nhm_arr = nhm_arr[:max_hits]
            elif len(nhm_arr) < max_hits:
                nhm_arr = np.pad(nhm_arr, (0, max_hits - len(nhm_arr)), constant_values=False)
            
            # Final shape validation
            assert hf_arr.shape == (max_hits, hit_dim), f"hf_arr shape {hf_arr.shape} != ({max_hits}, {hit_dim})"
            assert nhf_arr.shape == (max_hits, hit_dim), f"nhf_arr shape {nhf_arr.shape} != ({max_hits}, {hit_dim})"
            assert len(hm_arr) == max_hits, f"hm_arr length {len(hm_arr)} != {max_hits}"
            assert len(nhm_arr) == max_hits, f"nhm_arr length {len(nhm_arr)} != {max_hits}"
            
            hit_features_list.append(hf_arr)
            next_hit_features_list.append(nhf_arr)
            hit_masks_list.append(hm_arr)
            next_hit_masks_list.append(nhm_arr)
        
        return (
            np.array(states_list, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(next_states_list, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(hit_features_list, dtype=np.float32),  # [batch_size, max_hits, hit_dim]
            np.array(next_hit_features_list, dtype=np.float32),  # [batch_size, max_hits, hit_dim]
            np.array(hit_masks_list, dtype=bool),  # [batch_size, max_hits]
            np.array(next_hit_masks_list, dtype=bool),  # [batch_size, max_hits]
        )

    def __len__(self):
        return len(self.buffer)

    # def clear(self) -> None:
    #     """Remove all stored experiences."""
    #     self.buffer.clear()