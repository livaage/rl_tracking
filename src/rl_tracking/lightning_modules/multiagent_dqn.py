"""Multi-agent DQN Lightning module for simultaneous track propagation."""
from pytorch_lightning import LightningModule
from rl_tracking.models.pointer_network import PointerNetwork
from rl_tracking.environment.multiagent import MultiAgent
from rl_tracking.environment.multiagent_tracking_env import MultiAgentTrackingEnv
from rl_tracking.replay.buffer import ReplayBuffer
from torch import Tensor, nn
from typing import Tuple, List
from collections import OrderedDict, Counter
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
import torch.optim
from rl_tracking.utils.experience_loader import RLDataset


class MultiAgentDQNLightning(LightningModule):
    """Multi-agent DQN for simultaneous track propagation.
    
    Key features:
    - Multiple agents propagate tracks in lockstep
    - Shared network and replay buffer
    - Coordination through shared observation space and hit pool
    """
    
    def __init__(
        self,
        dm,
        num_agents: int = 3,
        batch_size: int = 16,
        lr: float = 1e-2,
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
        warm_start_steps: int = 1000,
        particle_filters: dict = None,
        hit_filters: dict = None,
        use_distance_reward: bool = False,
        use_truth_path_plan: bool = False,
        proximity_threshold: float = 50.0,
    ) -> None:
        """Initialize multi-agent DQN.

        Args:
            dm: Data module
            num_agents: Number of agents (tracks) to propagate simultaneously
            batch_size: size of the batches
            lr: learning rate
            gamma: discount factor
            sync_rate: how many frames to update target network
            replay_size: capacity of replay buffer
            warm_start_size: buffer fill size
            eps_last_frame: frame when epsilon stops decaying
            eps_start: starting epsilon
            eps_end: final epsilon
            episode_length: max episode length
            warm_start_steps: warmup steps
            particle_filters: Particle filtering criteria
            hit_filters: Hit filtering criteria
            use_distance_reward: Use distance-based reward
            use_truth_path_plan: Use truth-level path plan
            proximity_threshold: Group tracks within this distance
        """
        super().__init__()
        self.save_hyperparameters()

        # Use provided filters or defaults
        if particle_filters is None:
            particle_filters = {'pt': 1, 'nhits_min': 3, 'nhits_max': 20}
        if hit_filters is None:
            hit_filters = {}
        
        # Create multi-agent environment
        from torch.utils.data import DataLoader
        env_dataloader = DataLoader(
            dm.train_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        self.env = MultiAgentTrackingEnv(
            env_dataloader,
            num_agents=num_agents,
            particle_filters=particle_filters,
            hit_filters=hit_filters,
            use_distance_reward=use_distance_reward,
            use_truth_path_plan=use_truth_path_plan,
            proximity_threshold=proximity_threshold
        )
        
        # Network dimensions
        # State: own helix [z, r, x, y] + other agents [z, r, x, y] * (num_agents-1)
        state_dim = 4 + 4 * (num_agents - 1)
        hit_dim = 4
        hidden_dim = 256
        
        self.net = PointerNetwork(state_dim=state_dim, hit_dim=hit_dim, hidden_dim=hidden_dim)
        self.target_net = PointerNetwork(state_dim=state_dim, hit_dim=hit_dim, hidden_dim=hidden_dim)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.val_buffer = ReplayBuffer(self.hparams.replay_size // 4)
        
        # Create validation environment
        val_env_dataloader = DataLoader(
            dm.val_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        self.val_env = MultiAgentTrackingEnv(
            val_env_dataloader,
            num_agents=num_agents,
            particle_filters=particle_filters,
            hit_filters=hit_filters,
            use_distance_reward=use_distance_reward,
            use_truth_path_plan=use_truth_path_plan,
            proximity_threshold=proximity_threshold
        )
        self.val_agent = MultiAgent(self.val_env, self.val_buffer)
        
        self.agent = MultiAgent(self.env, self.buffer)
        
        self.total_reward = 0
        self.episode_reward = 0
        
        # Track episode statistics
        self.episode_termination_reasons = []
        self.episode_completion_ratios = []
        self.episode_expected_hits = []
        self.episode_predicted_hits = []
        self.episode_correct_hits = []
        
        self.populate(self.hparams.warm_start_steps)
        self._populate_val_buffer(max(100, self.hparams.batch_size * 2))

    def populate(self, steps: int = 1000) -> None:
        """Fill buffer with random experiences."""
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)
    
    def _populate_val_buffer(self, steps: int = 100) -> None:
        """Populate validation buffer."""
        device = "cpu"
        for _ in range(steps):
            self.val_agent.play_step(self.net, epsilon=1.0, device=device)

    def forward(self, x: Tensor, hit_features: Tensor, mask: Tensor = None) -> Tensor:
        """Forward pass through network."""
        return self.net(x, hit_features, mask)

    def dqn_mse_loss(self, batch: Tuple[Tensor, ...]) -> Tensor:
        """Calculate MSE loss for DQN."""
        # Unpack batch
        states, actions, rewards, dones, next_states, hit_features, next_hit_features, hit_masks, next_hit_masks = batch

        batch_size = states.shape[0]
        
        # Ensure rewards is 1D
        rewards = rewards.flatten()
        if rewards.shape[0] != batch_size:
            raise ValueError(f"Rewards batch size mismatch: {rewards.shape[0]} vs {batch_size}")
        
        # Ensure actions and dones are 1D
        actions = actions.flatten()
        dones = dones.flatten()
        
        # Convert to tensors if needed
        if isinstance(hit_features, np.ndarray):
            hit_features = torch.from_numpy(hit_features).float().to(states.device)
        if isinstance(hit_masks, np.ndarray):
            hit_masks = torch.from_numpy(hit_masks).bool().to(states.device)
        
        # Normalize shapes
        max_hits = 20
        hit_dim = 4
        
        # Ensure hit_features is [batch_size, max_hits, hit_dim]
        if hit_features.shape[1] > max_hits:
            hit_features = hit_features[:, :max_hits]
        elif hit_features.shape[1] < max_hits:
            padding = torch.zeros((batch_size, max_hits - hit_features.shape[1], hit_features.shape[2]),
                                 dtype=hit_features.dtype, device=hit_features.device)
            hit_features = torch.cat([hit_features, padding], dim=1)
        
        if hit_features.shape[2] != hit_dim:
            if hit_features.shape[2] > hit_dim:
                hit_features = hit_features[:, :, :hit_dim]
            else:
                padding = torch.zeros((batch_size, max_hits, hit_dim - hit_features.shape[2]),
                                     dtype=hit_features.dtype, device=hit_features.device)
                hit_features = torch.cat([hit_features, padding], dim=2)
        
        # Normalize hit_masks
        if hit_masks.shape[1] > max_hits:
            hit_masks = hit_masks[:, :max_hits]
        elif hit_masks.shape[1] < max_hits:
            padding = torch.zeros((batch_size, max_hits - hit_masks.shape[1]),
                                 dtype=hit_masks.dtype, device=hit_masks.device)
            hit_masks = torch.cat([hit_masks, padding], dim=1)
        
        # Clamp actions
        actions_clamped = actions.long().clamp(0, max_hits - 1)
        
        # Get Q-values
        scores_all = self.net(states, hit_features, mask=hit_masks)
        
        # Ensure selected actions are valid
        for i in range(batch_size):
            action_idx = actions_clamped[i].item()
            if not hit_masks[i, action_idx]:
                valid_indices = torch.where(hit_masks[i])[0]
                if len(valid_indices) > 0:
                    actions_clamped[i] = valid_indices[0]
                else:
                    actions_clamped[i] = 0
                    scores_all[i, 0] = 0.0
        
        state_action_values = scores_all.gather(1, actions_clamped.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            # Get next Q-values from target network
            if isinstance(next_hit_features, np.ndarray):
                next_hit_features = torch.from_numpy(next_hit_features).float().to(states.device)
            if isinstance(next_hit_masks, np.ndarray):
                next_hit_masks = torch.from_numpy(next_hit_masks).bool().to(states.device)
            
            # Normalize next hit features
            if next_hit_features.shape[1] != max_hits:
                if next_hit_features.shape[1] > max_hits:
                    next_hit_features = next_hit_features[:, :max_hits]
                    next_hit_masks = next_hit_masks[:, :max_hits] if next_hit_masks.shape[1] > max_hits else next_hit_masks
                else:
                    padding_needed = max_hits - next_hit_features.shape[1]
                    next_hf_padding = torch.zeros((batch_size, padding_needed, hit_dim),
                                                  dtype=next_hit_features.dtype, device=next_hit_features.device)
                    next_hit_features = torch.cat([next_hit_features, next_hf_padding], dim=1)
                    next_hm_padding = torch.zeros((batch_size, padding_needed),
                                                   dtype=next_hit_masks.dtype, device=next_hit_masks.device)
                    next_hit_masks = torch.cat([next_hit_masks, next_hm_padding], dim=1)
            
            if next_hit_features.shape[2] != hit_dim:
                if next_hit_features.shape[2] > hit_dim:
                    next_hit_features = next_hit_features[:, :, :hit_dim]
                else:
                    padding = torch.zeros((batch_size, max_hits, hit_dim - next_hit_features.shape[2]),
                                         dtype=next_hit_features.dtype, device=next_hit_features.device)
                    next_hit_features = torch.cat([next_hit_features, padding], dim=2)
            
            next_scores_all = self.target_net(next_states, next_hit_features, mask=next_hit_masks)
            next_state_values = next_scores_all.max(1)[0]
            next_state_values = torch.clamp(next_state_values, min=-50.0, max=200.0)
            next_state_values = torch.where(next_state_values < -50.0,
                                           torch.zeros_like(next_state_values),
                                           next_state_values)
            next_state_values = next_state_values.squeeze()
            if next_state_values.dim() == 0:
                next_state_values = next_state_values.unsqueeze(0)
            elif next_state_values.dim() > 1:
                next_state_values = next_state_values.flatten()
            
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards
        
        # Clip values
        state_action_values = torch.clamp(state_action_values, min=-50.0, max=200.0)
        expected_state_action_values = torch.clamp(expected_state_action_values, min=-50.0, max=200.0)

        # Huber loss
        loss_fn = nn.HuberLoss(delta=10.0)
        loss = loss_fn(state_action_values, expected_state_action_values)
        
        return loss

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Training step for multi-agent DQN."""
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame)
        self.log("epsilon", epsilon)

        # Step through environment with all agents
        rewards, done = self.agent.play_step(self.net, epsilon, device)
        
        # Average reward across agents
        avg_reward = np.mean([r for r in rewards])
        self.episode_reward += avg_reward
        
        # Track accuracy for each agent
        env = self.agent.env
        accuracies = []
        for agent_idx in range(self.agent.num_agents):
            acc = env.get_accuracy(agent_idx)
            if acc > 0:
                accuracies.append(acc)
        
        if len(accuracies) > 0:
            avg_accuracy = np.mean(accuracies)
            self.log("hit_accuracy", avg_accuracy, prog_bar=True)
        
        # Log coordination metrics periodically
        if self.global_step % 10 == 0:
            coord_metrics = env.get_coordination_metrics()
            self.log("coordination/overlap_ratio", coord_metrics['overlap_ratio'])
            self.log("coordination/avg_shared_hits", coord_metrics['avg_shared_hits'])
            self.log("coordination/conflict_rate", coord_metrics['conflict_rate'])
        
        # Calculate loss
        loss = self.dqn_mse_loss(batch)

        if done:
            # Log final episode coordination metrics
            coord_metrics = env.get_coordination_metrics()
            self.log("episode_coordination/overlap_ratio", coord_metrics['overlap_ratio'])
            self.log("episode_coordination/avg_shared_hits", coord_metrics['avg_shared_hits'])
            self.log("episode_coordination/conflict_rate", coord_metrics['conflict_rate'])
            
            # Reset for next episode
            env.reset_coordination_metrics()
            
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        tau = 0.05
        if self.global_step % self.hparams.sync_rate == 0:
            for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        self.log_dict({
            "reward": avg_reward,
            "train_loss": loss,
        })
        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> OrderedDict:
        """Validation step."""
        device = self.get_device(batch)
        
        # Run enough steps to complete multiple episodes
        val_episodes = 5
        val_losses = []
        val_accuracies = []
        val_rewards_list = []
        
        with torch.no_grad():
            val_env = self.val_agent.env
            val_agent = self.val_agent
            
            episodes_completed = 0
            max_steps_per_episode = 100  # Prevent infinite loops
            
            # Track coordination metrics across all validation episodes
            val_coordination_metrics = []
            
            # Track buffer size before and after
            buffer_size_before = len(self.val_buffer)
            
            while episodes_completed < val_episodes:
                # Reset accuracy counters for this episode
                for agent_idx in range(val_env.num_agents):
                    val_env.total_selections[agent_idx] = 0
                    val_env.correct_selections[agent_idx] = 0
                
                # Reset coordination metrics for this episode
                val_env.reset_coordination_metrics()
                
                val_episode_reward = 0
                steps_in_episode = 0
                
                while steps_in_episode < max_steps_per_episode:
                    rewards, done = val_agent.play_step(self.net, epsilon=0.0, device=device)
                    avg_reward = np.mean([r for r in rewards])
                    val_episode_reward += avg_reward
                    steps_in_episode += 1
                    
                    if done:
                        # Calculate accuracies for all agents (include 0.0)
                        accuracies = []
                        selections_per_agent = []
                        for agent_idx in range(val_env.num_agents):
                            acc = val_env.get_accuracy(agent_idx)
                            accuracies.append(acc)
                            selections_per_agent.append(val_env.total_selections[agent_idx])
                        
                        # Debug: log what we got
                        if episodes_completed == 0:  # Log first episode
                            print(f"First val episode: accuracies={accuracies}, selections={selections_per_agent}, reward={val_episode_reward:.2f}")
                        
                        if len(accuracies) > 0:
                            val_accuracies.append(np.mean(accuracies))
                        val_rewards_list.append(val_episode_reward)
                        
                        # Collect coordination metrics for this episode
                        coord_metrics = val_env.get_coordination_metrics()
                        val_coordination_metrics.append(coord_metrics)
                        
                        episodes_completed += 1
                        break
            
            # Log buffer statistics
            buffer_size_after = len(self.val_buffer)
            samples_added = buffer_size_after - buffer_size_before
            if self.global_step % 10 == 0:  # Log occasionally
                print(f"Validation: Completed {episodes_completed} episodes, added {samples_added} samples to buffer (total: {buffer_size_after})")
            
            # Compute loss on validation buffer if available
            # Check if buffer has enough samples (need at least batch_size)
            if len(self.val_buffer) >= self.hparams.batch_size:
                try:
                    val_batch_raw = self.val_buffer.sample(self.hparams.batch_size)
                    val_states_t = torch.tensor(val_batch_raw[0], dtype=torch.float32).to(device)
                    val_actions_t = torch.tensor(val_batch_raw[1], dtype=torch.int64).to(device)
                    val_next_states_t = torch.tensor(val_batch_raw[2], dtype=torch.float32).to(device)
                    val_rewards_t = torch.tensor(val_batch_raw[3], dtype=torch.float32).to(device)
                    val_dones_t = torch.tensor(val_batch_raw[4], dtype=torch.bool).to(device)
                    val_hf_t = torch.tensor(val_batch_raw[5], dtype=torch.float32).to(device)
                    val_nhf_t = torch.tensor(val_batch_raw[6], dtype=torch.float32).to(device)
                    val_hm_t = torch.tensor(val_batch_raw[7], dtype=torch.bool).to(device)
                    val_nhm_t = torch.tensor(val_batch_raw[8], dtype=torch.bool).to(device)
                    
                    val_batch = (val_states_t, val_actions_t, val_rewards_t, val_dones_t, val_next_states_t,
                                val_hf_t, val_nhf_t, val_hm_t, val_nhm_t)
                    val_loss = self.dqn_mse_loss(val_batch)
                    val_losses.append(val_loss.item())
                except Exception as e:
                    # Log the error so we can debug
                    print(f"Warning: Validation loss calculation failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Buffer doesn't have enough samples yet
                print(f"Info: Validation buffer has {len(self.val_buffer)} samples, need {self.hparams.batch_size} for loss calculation")
        
        # Log metrics
        if len(val_losses) > 0:
            avg_val_loss = sum(val_losses) / len(val_losses)
            self.log("val_loss", avg_val_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("val_loss", 0.0, on_step=False, on_epoch=True, prog_bar=True)
        
        # Always log val_hit_accuracy (even if 0.0) for early stopping callback
        if len(val_accuracies) > 0:
            avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
            self.log("val_hit_accuracy", avg_val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("val_hit_accuracy", 0.0, on_step=False, on_epoch=True, prog_bar=True)
        
        if len(val_rewards_list) > 0:
            avg_val_reward = sum(val_rewards_list) / len(val_rewards_list)
            self.log("val_episode_reward", avg_val_reward, on_step=False, on_epoch=True)
        
        # Log coordination metrics
        if len(val_coordination_metrics) > 0:
            avg_overlap = np.mean([m['overlap_ratio'] for m in val_coordination_metrics])
            avg_shared = np.mean([m['avg_shared_hits'] for m in val_coordination_metrics])
            avg_conflict = np.mean([m['conflict_rate'] for m in val_coordination_metrics])
            
            self.log("val_coordination/overlap_ratio", avg_overlap, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_coordination/avg_shared_hits", avg_shared, on_step=False, on_epoch=True)
            self.log("val_coordination/conflict_rate", avg_conflict, on_step=False, on_epoch=True)
        
        return torch.tensor(avg_val_loss if len(val_losses) > 0 else 0.0, device=device)

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer with learning rate scheduling."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=20,
                min_lr=1e-6,
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
        }
        
        return [optimizer], [scheduler]

    def _collate_fn(self, batch):
        """Custom collate function for multi-agent experiences."""
        if len(batch) == 0:
            raise ValueError("Cannot collate empty batch")
        
        first_item = batch[0]
        states, actions, rewards, dones, next_states, hit_features, next_hit_features, hit_masks, next_hit_masks = zip(*batch)
        
        # Convert states
        states_list = []
        next_states_list = []
        state_dim = self.env.observation_space.shape[0]
        
        for state, next_state in zip(states, next_states):
            state_arr = np.asarray(state, dtype=np.float32).flatten()
            next_state_arr = np.asarray(next_state, dtype=np.float32).flatten()
            
            # Ensure correct shape
            if state_arr.shape != (state_dim,):
                state_arr = np.pad(state_arr[:state_dim], (0, max(0, state_dim - len(state_arr))), mode='constant')[:state_dim]
            if next_state_arr.shape != (state_dim,):
                next_state_arr = np.pad(next_state_arr[:state_dim], (0, max(0, state_dim - len(next_state_arr))), mode='constant')[:state_dim]
            
            states_list.append(state_arr)
            next_states_list.append(next_state_arr)
        
        states_array = np.array(states_list, dtype=np.float32)
        next_states_array = np.array(next_states_list, dtype=np.float32)
        
        # Convert rewards, actions, dones
        batch_size = len(states_list)
        rewards_list = []
        for r in rewards:
            if isinstance(r, (list, tuple)):
                r = r[0] if len(r) > 0 else 0.0
            elif isinstance(r, np.ndarray):
                r = r.item() if r.size == 1 else float(r.flat[0])
            else:
                r = float(r)
            rewards_list.append(r)
        
        actions_list = []
        for a in actions:
            if isinstance(a, (list, tuple)):
                a = a[0] if len(a) > 0 else 0
            elif isinstance(a, np.ndarray):
                a = a.item() if a.size == 1 else int(a.flat[0])
            else:
                a = int(a)
            actions_list.append(a)
        
        dones_list = []
        for d in dones:
            if isinstance(d, (list, tuple)):
                d = d[0] if len(d) > 0 else False
            elif isinstance(d, np.ndarray):
                d = d.item() if d.size == 1 else bool(d.flat[0])
            else:
                d = bool(d)
            dones_list.append(d)
        
        rewards_array = np.array(rewards_list, dtype=np.float32)
        actions_array = np.array(actions_list, dtype=np.int64)
        dones_array = np.array(dones_list, dtype=bool)
        
        # Process hit features (same as single-agent)
        max_hits = 20
        hit_dim = 4
        hit_features_array = np.zeros((batch_size, max_hits, hit_dim), dtype=np.float32)
        next_hit_features_array = np.zeros((batch_size, max_hits, hit_dim), dtype=np.float32)
        hit_masks_array = np.zeros((batch_size, max_hits), dtype=bool)
        next_hit_masks_array = np.zeros((batch_size, max_hits), dtype=bool)
        
        for i, (hf, nhf, hm, nhm) in enumerate(zip(hit_features, next_hit_features, hit_masks, next_hit_masks)):
            hf_arr = np.asarray(hf, dtype=np.float32)
            nhf_arr = np.asarray(nhf, dtype=np.float32)
            hm_arr = np.asarray(hm, dtype=bool)
            nhm_arr = np.asarray(nhm, dtype=bool)
            
            if len(hf_arr.shape) == 2:
                if hf_arr.shape[0] > max_hits:
                    hf_arr = hf_arr[:max_hits]
                elif hf_arr.shape[0] < max_hits:
                    padding = np.zeros((max_hits - hf_arr.shape[0], hf_arr.shape[1]), dtype=np.float32)
                    hf_arr = np.vstack([hf_arr, padding])
                
                if hf_arr.shape[1] != hit_dim:
                    if hf_arr.shape[1] > hit_dim:
                        hf_arr = hf_arr[:, :hit_dim]
                    else:
                        hf_arr = np.pad(hf_arr, ((0, 0), (0, hit_dim - hf_arr.shape[1])), mode='constant')
                
                hit_features_array[i] = hf_arr
            
            if len(nhf_arr.shape) == 2:
                if nhf_arr.shape[0] > max_hits:
                    nhf_arr = nhf_arr[:max_hits]
                elif nhf_arr.shape[0] < max_hits:
                    padding = np.zeros((max_hits - nhf_arr.shape[0], nhf_arr.shape[1]), dtype=np.float32)
                    nhf_arr = np.vstack([nhf_arr, padding])
                
                if nhf_arr.shape[1] != hit_dim:
                    if nhf_arr.shape[1] > hit_dim:
                        nhf_arr = nhf_arr[:, :hit_dim]
                    else:
                        nhf_arr = np.pad(nhf_arr, ((0, 0), (0, hit_dim - nhf_arr.shape[1])), mode='constant')
                
                next_hit_features_array[i] = nhf_arr
            
            if len(hm_arr.shape) == 1:
                if len(hm_arr) > max_hits:
                    hm_arr = hm_arr[:max_hits]
                elif len(hm_arr) < max_hits:
                    hm_arr = np.pad(hm_arr, (0, max_hits - len(hm_arr)), constant_values=False)
                hit_masks_array[i] = hm_arr
            
            if len(nhm_arr.shape) == 1:
                if len(nhm_arr) > max_hits:
                    nhm_arr = nhm_arr[:max_hits]
                elif len(nhm_arr) < max_hits:
                    nhm_arr = np.pad(nhm_arr, (0, max_hits - len(nhm_arr)), constant_values=False)
                next_hit_masks_array[i] = nhm_arr
        
        # Convert to tensors
        states_tensor = torch.tensor(states_array, dtype=torch.float32)
        actions_tensor = torch.tensor(actions_array, dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards_array, dtype=torch.float32)
        dones_tensor = torch.tensor(dones_array, dtype=torch.bool)
        next_states_tensor = torch.tensor(next_states_array, dtype=torch.float32)
        hit_features_tensor = torch.tensor(hit_features_array, dtype=torch.float32)
        next_hit_features_tensor = torch.tensor(next_hit_features_array, dtype=torch.float32)
        hit_masks_tensor = torch.tensor(hit_masks_array, dtype=torch.bool)
        next_hit_masks_tensor = torch.tensor(next_hit_masks_array, dtype=torch.bool)
        
        return (states_tensor, actions_tensor, rewards_tensor, dones_tensor, next_states_tensor,
               hit_features_tensor, next_hit_features_tensor, hit_masks_tensor, next_hit_masks_tensor)

    def __dataloader(self, buffer: ReplayBuffer) -> DataLoader:
        """Initialize replay buffer dataloader."""
        if len(buffer) == 0:
            raise ValueError(f"Cannot create DataLoader with empty buffer")
        
        dataset = RLDataset(buffer, self.hparams.episode_length)
        effective_batch_size = min(self.hparams.batch_size, max(1, len(buffer)))
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=effective_batch_size,
            collate_fn=self._collate_fn,
            drop_last=False,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader(self.buffer)
    
    def val_dataloader(self) -> DataLoader:
        """Get validation loader."""
        return self.__dataloader(self.val_buffer)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer batch to device."""
        states, actions, rewards, dones, next_states, hit_features, next_hit_features, hit_masks, next_hit_masks = batch
        
        states = states.float().to(device) if states.is_floating_point() and states.dtype != torch.float32 else states.to(device)
        actions = actions.to(device)
        rewards = rewards.float().to(device) if rewards.is_floating_point() and rewards.dtype != torch.float32 else rewards.to(device)
        dones = dones.to(device)
        next_states = next_states.float().to(device) if next_states.is_floating_point() and next_states.dtype != torch.float32 else next_states.to(device)
        hit_features = hit_features.float().to(device) if hit_features.is_floating_point() and hit_features.dtype != torch.float32 else hit_features.to(device)
        next_hit_features = next_hit_features.float().to(device) if next_hit_features.is_floating_point() and next_hit_features.dtype != torch.float32 else next_hit_features.to(device)
        hit_masks = hit_masks.to(device)
        next_hit_masks = next_hit_masks.to(device)
        
        return states, actions, rewards, dones, next_states, hit_features, next_hit_features, hit_masks, next_hit_masks

    def get_device(self, batch) -> str:
        """Get current device."""
        if hasattr(batch[0], 'device'):
            device = batch[0].device
            if device.type == 'mps':
                return 'mps'
            elif device.type == 'cuda':
                return f'cuda:{device.index}' if device.index is not None else 'cuda'
        return "cpu"

