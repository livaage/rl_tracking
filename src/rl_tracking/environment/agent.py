from collections import namedtuple
from rl_tracking.replay.buffer import ReplayBuffer
from torch import nn
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from rl_tracking.environment.tracking_env import TrackingEnv
# Experience is now defined in replay.buffer to include hit_features
from rl_tracking.replay.buffer import Experience


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: TrackingEnv , replay_buffer: ReplayBuffer) -> None:
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
        info = getattr(self, 'info', {})
        hit_features = info.get('hit_features', None)
        hit_mask = info.get('hit_mask', None)
        
        if hit_features is None or hit_mask is None:
            # Fallback to random if info not available
            if np.random.random() < epsilon:
                return np.random.randint(0, 3)  # Default to 3 actions
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
                hit_mask = np.pad(hit_mask, (0, max_hits - len(hit_mask)), constant_values=False)
        
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
        scores = net(state_tensor, hit_features_tensor, mask=hit_mask_tensor)  # [num_hits]
        
        # Select hit with highest score (that's valid)
        # Apply mask: set invalid hits to very negative score
        scores_masked = scores.clone()
        scores_masked[~hit_mask_tensor] = -100.0  # Match the range used in pointer network
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
        
        # Extract hit features and mask from current state info
        hit_features = state_info.get('hit_features', np.zeros((20, 4), dtype=np.float32))
        hit_mask = state_info.get('hit_mask', np.zeros(20, dtype=bool))
        
        # do step in the environment
        next_state, reward, done = self.env.step(action)
        next_info = getattr(self.env, 'current_info', {})
        
        # Extract hit features and mask from next state info
        next_hit_features = next_info.get('hit_features', np.zeros((20, 4), dtype=np.float32))
        next_hit_mask = next_info.get('hit_mask', np.zeros(20, dtype=bool))
        
        # Store termination reason BEFORE reset (if done=True, reset will clear state)
        termination_info = None
        if done:
            env = self.env
            current_hit_number = getattr(env, 'hit_number', 0)
            track_length = len(env.current_track) if hasattr(env, 'current_track') and env.current_track is not None else 0
            
            # Store total_selections BEFORE reset (reset() clears it)
            total_selections = getattr(env, 'total_selections', 0)
            correct_selections = getattr(env, 'correct_selections', 0)
            
            # Determine termination reason before reset
            if track_length > 0 and current_hit_number >= track_length:
                termination_reason = 'completed_all_hits'
            elif track_length > 0 and current_hit_number < track_length:
                termination_reason = 'no_compatible_hits'
            else:
                termination_reason = 'unknown'
            
            # Also store track info for expected hits calculation
            # Get expected hits (excluding seed) BEFORE reset
            # IMPORTANT: Only count hits from layers in the path_plan, not all hits in current_track
            expected_hits = 0
            if env.current_track is not None and len(env.current_track) > 0:
                seed_hit_ids = set()
                if hasattr(env, 'helix') and env.helix is not None:
                    if hasattr(env.helix, 'seed_hits') and env.helix.seed_hits is not None:
                        seed_df = env.helix.seed_hits
                        if isinstance(seed_df, pd.DataFrame) and 'hit_id' in seed_df.columns:
                            seed_hit_ids = set(seed_df['hit_id'].values.astype(int))
                    
                    # Only count hits from layers in the path_plan that come AFTER the seed
                    # Use initial_start_layer (the final seed layer) not current_layer (which has advanced)
                    if hasattr(env.helix, 'path_plan') and env.helix.path_plan is not None:
                        # Get initial starting layer (final seed layer) - this doesn't change as we progress
                        initial_start_layer = getattr(env.helix, 'initial_start_layer', None)
                        if initial_start_layer is None:
                            # Fallback: try to get from seed_hits (last seed layer)
                            if hasattr(env.helix, 'seed_hits') and env.helix.seed_hits is not None:
                                seed_df = env.helix.seed_hits
                                if isinstance(seed_df, pd.DataFrame) and 'unique_layer_id' in seed_df.columns:
                                    seed_layers = seed_df['unique_layer_id'].values
                                    initial_start_layer = float(max(seed_layers)) if len(seed_layers) > 0 else None
                        
                        # Filter path_plan to only include post-seed layers
                        # Convert to list and handle numpy types properly
                        all_path_layers = [float(layer) for layer in env.helix.path_plan]
                        if initial_start_layer is not None:
                            initial_start_layer_float = float(initial_start_layer)
                            path_plan_layers = [layer for layer in all_path_layers if float(layer) > initial_start_layer_float]
                        else:
                            # Fallback: exclude seed layers by ID
                            path_plan_layers = [layer for layer in all_path_layers if layer not in seed_hit_ids]
                        path_plan_layers_set = set(path_plan_layers)
                        # Filter current_track to only include hits from post-seed path_plan layers
                        track_hits_in_path = env.current_track[
                            env.current_track['unique_layer_id'].isin(path_plan_layers_set)
                        ]
                        correct_hit_ids = set(track_hits_in_path['hit_id'].values)
                    else:
                        # Fallback: use all hits if path_plan not available
                        correct_hit_ids = set(env.current_track['hit_id'].values)
                else:
                    # Fallback: use all hits if helix not available
                    correct_hit_ids = set(env.current_track['hit_id'].values)
                
                expected_hits = len(correct_hit_ids) - len(seed_hit_ids)
                if expected_hits < 0:
                    expected_hits = 0
            
            termination_info = {
                'reason': termination_reason,
                'hit_number': current_hit_number,
                'track_length': track_length,
                'total_selections': total_selections,  # Store BEFORE reset
                'correct_selections': correct_selections,  # Store BEFORE reset
                'expected_hits': expected_hits,  # Store expected hits BEFORE reset
            }
            # Store in environment for later retrieval
            env._last_termination_info = termination_info

        # Store experience with hit features
        self.replay_buffer.append(Experience(
            state, action, next_state, reward, done,
            hit_features, next_hit_features, hit_mask, next_hit_mask
        ))

        self.state = next_state
        self.info = next_info
        if done:
            self.reset()
        return reward, done


