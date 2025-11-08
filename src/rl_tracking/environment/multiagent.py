"""Multi-agent coordinator for simultaneous track propagation."""
from collections import namedtuple
from rl_tracking.replay.buffer import ReplayBuffer
from torch import nn
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List
from rl_tracking.environment.multiagent_tracking_env import MultiAgentTrackingEnv
from rl_tracking.replay.buffer import Experience


class MultiAgent:
    """Multi-agent coordinator that manages multiple agents propagating tracks simultaneously.
    
    Each agent has its own state but shares the environment and hit pool.
    """

    def __init__(self, env: MultiAgentTrackingEnv, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: Multi-agent tracking environment
            replay_buffer: Shared replay buffer storing experiences from all agents
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.num_agents = env.num_agents
        
        # Reset environment and get initial states
        observations, info = self.env.reset()
        self.states = observations  # List of observations (one per agent)
        self.info = info  # Dict with per-agent info

    def reset(self) -> None:
        """Resets the environment and updates the states for all agents."""
        observations, info = self.env.reset()
        self.states = observations
        self.info = info

    def get_actions(self, net: nn.Module, epsilon: float, device: str) -> List[int]:
        """Get actions for all agents using the pointer network.

        Args:
            net: Pointer network that scores hits based on state
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            actions: List of actions (one per agent)
        """
        actions = []
        
        # Handle case where states might be None (episode ended)
        if self.states is None:
            return [0] * self.num_agents
        
        # Get action for each agent
        for agent_idx in range(self.num_agents):
            state = self.states[agent_idx]
            agent_info = self.info['agents'][agent_idx] if 'agents' in self.info else {}
            
            # Check if agent is active
            if not agent_info.get('active', False):
                actions.append(0)  # Dummy action for inactive agent
                continue
            
            # Get hit features and mask
            hit_features = agent_info.get('hit_features', None)
            hit_mask = agent_info.get('hit_mask', None)
            
            if hit_features is None or hit_mask is None:
                # Fallback to random action
                if np.random.random() < epsilon:
                    actions.append(np.random.randint(0, 20))
                else:
                    actions.append(0)
                continue
            
            # Ensure consistent shapes
            hit_features = np.asarray(hit_features, dtype=np.float32)
            hit_mask = np.asarray(hit_mask, dtype=bool)
            
            max_hits = 20
            hit_dim = 4
            
            # Normalize shapes
            if hit_features.shape[0] > max_hits:
                hit_features = hit_features[:max_hits]
                hit_mask = hit_mask[:max_hits]
            elif hit_features.shape[0] < max_hits:
                padding_needed = max_hits - hit_features.shape[0]
                if len(hit_features.shape) == 2:
                    padding = np.zeros((padding_needed, hit_dim), dtype=np.float32)
                    hit_features = np.vstack([hit_features, padding])
                else:
                    hit_features = hit_features.reshape(-1, hit_dim)
                    padding = np.zeros((padding_needed, hit_dim), dtype=np.float32)
                    hit_features = np.vstack([hit_features, padding])
                
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
                    action = int(np.random.choice(valid_indices))
                else:
                    action = 0
                actions.append(action)
                continue
            
            # Use pointer network to score hits
            state_tensor = torch.tensor(state, dtype=torch.float32)
            hit_features_tensor = torch.tensor(hit_features, dtype=torch.float32)
            hit_mask_tensor = torch.tensor(hit_mask, dtype=torch.bool)
            
            if device not in ["cpu"]:
                state_tensor = state_tensor.to(device)
                hit_features_tensor = hit_features_tensor.to(device)
                hit_mask_tensor = hit_mask_tensor.to(device)
            
            # Get scores for all hits
            # Note: The pointer network expects [batch_size, state_dim] and [batch_size, max_hits, hit_dim]
            # Since we have a single state, add batch dimension
            scores = net(state_tensor.unsqueeze(0), hit_features_tensor.unsqueeze(0), 
                        mask=hit_mask_tensor.unsqueeze(0))  # [1, max_hits]
            scores = scores.squeeze(0)  # [max_hits]
            
            # Select hit with highest score (that's valid)
            scores_masked = scores.clone()
            scores_masked[~hit_mask_tensor] = -100.0
            action = int(torch.argmax(scores_masked).item())
            
            actions.append(action)
        
        return actions

    @torch.no_grad()
    def play_step(
            self,
            net: nn.Module,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Tuple[List[float], bool]:
        """Carries out a single interaction step for all agents.

        Args:
            net: Pointer network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            rewards: List of rewards (one per agent)
            done: Boolean indicating if episode is complete
        """
        # Get actions for all agents
        actions = self.get_actions(net, epsilon, device)
        states = self.states
        info = self.info
        
        # Extract hit features for each agent
        hit_features_list = []
        hit_mask_list = []
        for agent_idx in range(self.num_agents):
            agent_info = info['agents'][agent_idx] if 'agents' in info else {}
            hit_features = agent_info.get('hit_features', np.zeros((20, 4), dtype=np.float32))
            hit_mask = agent_info.get('hit_mask', np.zeros(20, dtype=bool))
            hit_features_list.append(hit_features)
            hit_mask_list.append(hit_mask)
        
        # Step environment with all actions
        next_states, rewards, dones, next_info = self.env.step(actions)
        
        # Extract next hit features
        next_hit_features_list = []
        next_hit_mask_list = []
        if next_info and 'agents' in next_info:
            for agent_idx in range(self.num_agents):
                agent_info = next_info['agents'][agent_idx]
                next_hit_features = agent_info.get('hit_features', np.zeros((20, 4), dtype=np.float32))
                next_hit_mask = agent_info.get('hit_mask', np.zeros(20, dtype=bool))
                next_hit_features_list.append(next_hit_features)
                next_hit_mask_list.append(next_hit_mask)
        else:
            next_hit_features_list = [np.zeros((20, 4), dtype=np.float32)] * self.num_agents
            next_hit_mask_list = [np.zeros(20, dtype=bool)] * self.num_agents
        
        # Store experiences for all agents in replay buffer
        for agent_idx in range(self.num_agents):
            # Only store experience if agent was active
            agent_info = info['agents'][agent_idx] if 'agents' in info else {}
            if agent_info.get('active', False):
                state = states[agent_idx] if states is not None else np.zeros(self.env.observation_space.shape[0], dtype=np.float32)
                action = actions[agent_idx]
                next_state = next_states[agent_idx] if next_states is not None else np.zeros(self.env.observation_space.shape[0], dtype=np.float32)
                reward = rewards[agent_idx]
                done = dones[agent_idx]
                hit_features = hit_features_list[agent_idx]
                next_hit_features = next_hit_features_list[agent_idx]
                hit_mask = hit_mask_list[agent_idx]
                next_hit_mask = next_hit_mask_list[agent_idx]
                
                self.replay_buffer.append(Experience(
                    state, action, next_state, reward, done,
                    hit_features, next_hit_features, hit_mask, next_hit_mask
                ))
        
        # Update states
        self.states = next_states
        self.info = next_info
        
        # Check if episode is done (all agents done)
        episode_done = all(dones)
        
        if episode_done:
            self.reset()
        
        return rewards, episode_done


