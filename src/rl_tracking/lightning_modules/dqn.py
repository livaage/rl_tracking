from pytorch_lightning import LightningModule
from rl_tracking.models.pointer_network import PointerNetwork
from rl_tracking.environment.agent import Agent
from rl_tracking.environment.tracking_env import TrackingEnv
from rl_tracking.replay.buffer import ReplayBuffer
from torch import Tensor, nn
from typing import Tuple, List
from collections import OrderedDict
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
import torch.optim
from rl_tracking.utils.logger import get_logger
from rl_tracking.utils.experience_loader import RLDataset

logger = get_logger()

class DQNLightning(LightningModule):
    def __init__(
        self,
        dm,
        batch_size: int = 16,
        lr: float = 1e-2,
        env: str = "CartPole-v0",
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
    ) -> None:
        """Basic DQN Model.

        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment

        """
        super().__init__()
        self.save_hyperparameters()

        # Use provided filters or defaults: no hit filtering, particles with pt>1 and 3-20 hits
        if particle_filters is None:
            particle_filters = {'pt': 1, 'nhits_min': 3, 'nhits_max': 20}
        if hit_filters is None:
            hit_filters = {}  # No hit filtering by default
        
        # TrackingEnv needs individual events, not batched data
        # Create a DataLoader with batch_size=1 for the environment to iterate over individual events
        from torch.utils.data import DataLoader
        env_dataloader = DataLoader(
            dm.train_dataset,
            batch_size=1,  # Environment needs individual events, not batches
            num_workers=0,  # Use 0 workers to avoid multiprocessing issues
            pin_memory=False,
            drop_last=False
        )
        hit_feature_mode = environment_config.get('hit_feature_mode', 'absolute')
        state_feature_mode = environment_config.get('state_feature_mode', 'full')
        
        self.env = TrackingEnv(
            env_dataloader, 
            particle_filters=particle_filters, 
            hit_filters=hit_filters,
            use_distance_reward=use_distance_reward,
            deterministic=False,
            hit_feature_mode=hit_feature_mode,
            state_feature_mode=state_feature_mode,
        )
        setattr(self.env, "context_tag", "train")

        # Use PointerNetwork instead of DQN for feature-based hit selection
        # State dim comes from enriched observation (track context features)
        # Hit dim: hit features [x, y, z, r] = 4
        state_dim = self.env.observation_space.shape[0]
        hit_dim = 4
        hidden_dim = 256
        
        self.net = PointerNetwork(state_dim=state_dim, hit_dim=hit_dim, hidden_dim=hidden_dim)
        self.target_net = PointerNetwork(state_dim=state_dim, hit_dim=hit_dim, hidden_dim=hidden_dim)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.val_buffer = ReplayBuffer(self.hparams.replay_size // 4)  # Smaller validation buffer
        
        # Create a separate validation agent that uses the validation buffer
        # CRITICAL: Use val_dataset for validation to ensure proper train/val split
        # CRITICAL: NEVER use test_dataset during training or validation - it's only for final evaluation
        val_env_dataloader = DataLoader(
            dm.val_dataset,  # Use validation dataset - separate events from training
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        self.val_env = TrackingEnv(
            val_env_dataloader,
            particle_filters=particle_filters,
            hit_filters=hit_filters,
            use_distance_reward=use_distance_reward,
            deterministic=True,
            hit_feature_mode=hit_feature_mode,
            state_feature_mode=state_feature_mode,
        )
        setattr(self.val_env, "context_tag", "val")
        self.val_agent = Agent(self.val_env, self.val_buffer)
        
        self.agent = Agent(self.env, self.buffer)
        
        # CRITICAL SAFEGUARD: Verify that test_dataset is NOT being used during training
        # The test dataset should only be used in evaluate.py for final evaluation
        # Note: This check is performed in stream_loading.py during setup(),
        # but we add a reminder here that test_dataset exists but should not be accessed
        if hasattr(dm, 'test_dataset'):
            # The overlap check is already done in stream_loading.py setup()
            # This is just a reminder that test_dataset exists but should never be accessed
            # during training or validation
            pass  # Dataset separation is verified in stream_loading.py
        self.total_reward = 0
        self.episode_reward = 0
        self.val_episode_reward = 0
        self.populate(self.hparams.warm_start_steps)
        # Populate validation buffer with enough samples (at least batch_size * 2 to ensure batching works)
        self._populate_val_buffer(max(100, self.hparams.batch_size * 2))

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)
    
    def _populate_val_buffer(self, steps: int = 100) -> None:
        """Populate validation buffer with random experiences."""
        # Use validation agent to populate validation buffer
        device = "cpu"  # Use CPU for buffer population
        for _ in range(steps):
            self.val_agent.play_step(self.net, epsilon=1.0, device=device)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values

        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data including hit_features

        Returns:
            loss

        """
        # Unpack batch - now includes hit_features
        if len(batch) == 5:
            # Backward compatibility: old format without hit_features
            states, actions, rewards, dones, next_states = batch
            hit_features = None
            next_hit_features = None
            hit_masks = None
            next_hit_masks = None
        else:
            states, actions, rewards, dones, next_states, hit_features, next_hit_features, hit_masks, next_hit_masks = batch

        # Ensure all tensors have consistent batch dimension
        batch_size = states.shape[0]
        
        # Ensure rewards is 1D with shape (batch_size,)
        # Flatten any extra dimensions and ensure correct shape
        rewards = rewards.view(batch_size, -1)[:, 0].float()

        # Ensure actions is 1D
        actions = actions.view(batch_size, -1)[:, 0].long()

        # Ensure dones is 1D
        dones = dones.view(batch_size, -1)[:, 0].bool()
        
        # Get scores for current states and hit features using pointer network
        if hit_features is not None:
            # Convert to tensors if numpy arrays
            if isinstance(hit_features, np.ndarray):
                hit_features = torch.from_numpy(hit_features).float().to(states.device)
            if isinstance(hit_masks, np.ndarray):
                hit_masks = torch.from_numpy(hit_masks).bool().to(states.device)
            
            # CRITICAL: Ensure consistent shapes - always truncate/pad to exactly max_hits
            max_hits = 20
            hit_dim = 4
            
            # Normalize hit_features to exactly [batch_size, max_hits, hit_dim]
            if hit_features.shape[1] > max_hits:
                # Truncate if too many hits
                hit_features = hit_features[:, :max_hits]
            elif hit_features.shape[1] < max_hits:
                # Pad if too few hits
                padding = torch.zeros((batch_size, max_hits - hit_features.shape[1], hit_features.shape[2]),
                                     dtype=hit_features.dtype, device=hit_features.device)
                hit_features = torch.cat([hit_features, padding], dim=1)
            
            # Normalize hit_dim dimension
            if hit_features.shape[2] != hit_dim:
                if hit_features.shape[2] > hit_dim:
                    hit_features = hit_features[:, :, :hit_dim]
                else:
                    padding = torch.zeros((batch_size, max_hits, hit_dim - hit_features.shape[2]),
                                         dtype=hit_features.dtype, device=hit_features.device)
                    hit_features = torch.cat([hit_features, padding], dim=2)
            
            # Normalize hit_masks to exactly [batch_size, max_hits]
            if hit_masks.shape[1] > max_hits:
                hit_masks = hit_masks[:, :max_hits]
            elif hit_masks.shape[1] < max_hits:
                padding = torch.zeros((batch_size, max_hits - hit_masks.shape[1]),
                                     dtype=hit_masks.dtype, device=hit_masks.device)
                hit_masks = torch.cat([hit_masks, padding], dim=1)
            
            # Clamp actions to valid range
            actions_clamped = actions.long().clamp(0, max_hits - 1)
            
            # Get scores for all hits: [batch_size, max_hits]
            scores_all = self.net(states, hit_features, mask=hit_masks)  # [batch_size, max_hits]
            
            # Check for NaN/inf in scores
            if torch.isnan(scores_all).any():
                nan_count = torch.isnan(scores_all).sum().item()
                raise ValueError(f"NaN found in scores_all: {nan_count} NaN values out of {scores_all.numel()}")
            
            # Log inf statistics for diagnostics
            if torch.isinf(scores_all).any():
                inf_mask = torch.isinf(scores_all)
                neg_inf_mask = (scores_all == float('-inf'))
                pos_inf_mask = (scores_all == float('inf'))
                
                total_inf = inf_mask.sum().item()
                neg_inf_count = neg_inf_mask.sum().item()
                pos_inf_count = pos_inf_mask.sum().item()
                valid_scores_count = (~inf_mask).sum().item()
                
                # Log detailed statistics
                if self.global_step % 100 == 0:
                    self.log("inf_scores_total", float(total_inf))
                    self.log("inf_scores_neg_inf", float(neg_inf_count))
                    self.log("inf_scores_pos_inf", float(pos_inf_count))
                    self.log("valid_scores_count", float(valid_scores_count))
                    
                    # Check if all inf are -inf (expected from masking)
                    if pos_inf_count > 0:
                        # This is a problem - positive inf from computation
                        self.log("warning_pos_inf_scores", float(pos_inf_count))
                        # Replace positive inf with a large finite value
                        scores_all = torch.where(pos_inf_mask, torch.tensor(1e4, device=scores_all.device), scores_all)
            
            # Ensure selected actions are within valid (non-masked) hits
            # For each sample, check if the action selects a valid hit
            for i in range(batch_size):
                action_idx = actions_clamped[i].item()
                if not hit_masks[i, action_idx]:
                    # Action selected a masked hit - find first valid hit instead
                    valid_indices = torch.where(hit_masks[i])[0]
                    if len(valid_indices) > 0:
                        actions_clamped[i] = valid_indices[0]
                    else:
                        # No valid hits - use action 0 and set score to 0
                        actions_clamped[i] = 0
                        scores_all[i, 0] = 0.0
            
            # Extract score for selected action (hit index)
            # CRITICAL: actions_clamped are the actions that were actually taken (from replay buffer)
            # These might not be what the current policy would select!
            # This is why loss can stay high even when accuracy improves - we're evaluating
            # old actions with the new policy
            state_action_values = scores_all.gather(1, actions_clamped.unsqueeze(-1)).squeeze(-1)  # [batch_size]
            
            # Log what the current policy would have selected vs what was actually taken
            if self.global_step % 100 == 0:
                with torch.no_grad():
                    # What would current policy select? (max score for valid hits)
                    current_policy_actions = []
                    for i in range(batch_size):
                        # Get valid hits
                        valid_mask = hit_masks[i]
                        if valid_mask.any():
                            valid_scores = scores_all[i][valid_mask]
                            if len(valid_scores) > 0:
                                # Get index of best valid hit
                                best_valid_idx = torch.where(valid_mask)[0][torch.argmax(valid_scores)]
                                current_policy_actions.append(best_valid_idx.item())
                            else:
                                current_policy_actions.append(actions_clamped[i].item())
                        else:
                            current_policy_actions.append(actions_clamped[i].item())
                    
                    # Compare: did current policy agree with historical action?
                    policy_agreement = sum(1 for i in range(batch_size) 
                                         if current_policy_actions[i] == actions_clamped[i].item()) / batch_size
                    self.log("policy_agreement_with_buffer", float(policy_agreement))
                    
                    # What Q-value would current policy get?
                    current_policy_values = scores_all[torch.arange(batch_size), 
                                                      torch.tensor(current_policy_actions, device=scores_all.device)]
                    self.log("current_policy_q_mean", float(current_policy_values.mean().item()))
                    self.log("historical_action_q_mean", float(state_action_values.mean().item()))
            
            # Check for NaN in extracted values
            if torch.isnan(state_action_values).any():
                raise ValueError(f"NaN in state_action_values after gather")
        else:
            # Fallback: use old DQN format (shouldn't happen with new architecture)
            scores_all = self.net(states)
            state_action_values = scores_all.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        
        # Log score statistics for all hits (not just selected)
        if self.global_step % 100 == 0:
            with torch.no_grad():
                self.log("scores_all_mean", float(scores_all.mean().item()))
                self.log("scores_all_std", float(scores_all.std().item()))
                self.log("scores_all_max", float(scores_all.max().item()))
                self.log("scores_all_min", float(scores_all.min().item()))
                # Log which actions are being selected (hit indices)
                selected_actions = actions.cpu().numpy()
                unique_actions, counts = np.unique(selected_actions, return_counts=True)
                for action_val, count in zip(unique_actions[:10], counts[:10]):  # Top 10 actions
                    self.log(f"action_{int(action_val)}_count", int(count))

        with torch.no_grad():
            # Get scores for next states using target network
            if next_hit_features is not None:
                if isinstance(next_hit_features, np.ndarray):
                    next_hit_features = torch.from_numpy(next_hit_features).float().to(states.device)
                if isinstance(next_hit_masks, np.ndarray):
                    next_hit_masks = torch.from_numpy(next_hit_masks).bool().to(states.device)
                
                # Normalize next_hit_features and next_hit_masks to consistent shapes
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
                
                next_scores_all = self.target_net(next_states, next_hit_features, mask=next_hit_masks)  # [batch_size, max_hits]
                
                # Get max score for next state (best hit)
                # Since we use -1e6 instead of -inf for masked hits, we can safely take max
                next_state_values = next_scores_all.max(1)[0]  # [batch_size]
                
                # Clamp next_state_values to reasonable range
                # If all hits were masked, next_state_values will be around -100
                # Set those to 0 since there's no valid next state
                next_state_values = torch.clamp(next_state_values, min=-50.0, max=200.0)
                # If all were masked (very negative), set to 0
                next_state_values = torch.where(next_state_values < -50.0,
                                               torch.zeros_like(next_state_values),
                                               next_state_values)
            else:
                # Fallback
                next_scores_all = self.target_net(next_states)
                next_state_values = next_scores_all.max(1)[0]
            # Ensure next_state_values is 1D
            next_state_values = next_state_values.squeeze()
            if next_state_values.dim() == 0:
                next_state_values = next_state_values.unsqueeze(0)
            elif next_state_values.dim() > 1:
                next_state_values = next_state_values.flatten()
            
            # Ensure shape matches batch_size
            if next_state_values.shape[0] != batch_size:
                raise ValueError(
                    f"Next state values shape {next_state_values.shape} doesn't match batch_size {batch_size}"
                )
            
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        # Ensure rewards and next_state_values have matching shapes for broadcasting
        if rewards.shape != next_state_values.shape:
            raise ValueError(
                f"Shape mismatch: rewards {rewards.shape} vs next_state_values {next_state_values.shape}, "
                f"batch_size={batch_size}, states={states.shape}, actions={actions.shape}, dones={dones.shape}"
            )

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards
        
        # Log scale diagnostics to understand the mismatch
        if self.global_step % 100 == 0:
            with torch.no_grad():
                self.log("scale_state_action_mean", float(state_action_values.mean().item()))
                self.log("scale_expected_mean", float(expected_state_action_values.mean().item()))
                self.log("scale_next_state_mean", float(next_state_values.mean().item()))
                self.log("scale_rewards_mean", float(rewards.mean().item()))
                self.log("scale_abs_diff_mean", float(torch.abs(state_action_values - expected_state_action_values).mean().item()))
        
        # Clip values to reasonable range based on reward scale
        # Rewards are [0, 1] or [0, 10], so Q-values should be in roughly [0, 100] range
        # Allow some negative for exploration but not extreme
        state_action_values = torch.clamp(state_action_values, min=-50.0, max=200.0)
        expected_state_action_values = torch.clamp(expected_state_action_values, min=-50.0, max=200.0)

        # Use Huber loss instead of MSE for better robustness to outliers
        # Huber loss is less sensitive to large errors, which helps with scale mismatches
        # Delta controls the transition point from quadratic to linear
        # Use a larger delta to make it more like MSE for small errors, L1 for large errors
        loss_fn = nn.HuberLoss(delta=10.0)  # Larger delta = more like MSE, less robust to outliers
        loss = loss_fn(state_action_values, expected_state_action_values)
        
        # Also log the raw MSE to see if it's improving
        mse_loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        if self.global_step % 100 == 0:
            self.log("mse_loss_raw", float(mse_loss.item()))
        
        # Check for NaN loss and log diagnostics
        if torch.isnan(loss) or torch.isinf(loss):
            self.log("error_nan_loss", 1.0)
            self.log("debug_state_action_mean", float(state_action_values.mean().item()))
            self.log("debug_state_action_std", float(state_action_values.std().item()))
            self.log("debug_expected_mean", float(expected_state_action_values.mean().item()))
            self.log("debug_expected_std", float(expected_state_action_values.std().item()))
            self.log("debug_rewards_mean", float(rewards.mean().item()))
            self.log("debug_rewards_std", float(rewards.std().item()))
            # Replace NaN/inf loss with a large but finite value to allow training to continue
            loss = torch.tensor(1e6, dtype=loss.dtype, device=loss.device)
            self.log("warning_loss_replaced", 1.0)
        
        # Log Q-value statistics for diagnostics
        if self.global_step % 100 == 0:
            with torch.no_grad():
                q_diff = expected_state_action_values - state_action_values
                self.log("q_value_mean", float(state_action_values.mean().item()))
                self.log("q_value_std", float(state_action_values.std().item()))
                self.log("target_q_mean", float(expected_state_action_values.mean().item()))
                self.log("target_q_std", float(expected_state_action_values.std().item()))
                self.log("q_value_diff_mean", float(q_diff.mean().item()))  # How much target is ahead
                self.log("reward_mean_in_batch", float(rewards.mean().item()))
                self.log("reward_std_in_batch", float(rewards.std().item()))
                self.log("reward_max_in_batch", float(rewards.max().item()))
                self.log("reward_min_in_batch", float(rewards.min().item()))
                # Check how often rewards are positive (for binary rewards, this is hit accuracy)
                positive_rewards = (rewards > 0).float().mean()
                self.log("reward_positive_ratio", float(positive_rewards.item()))
                
                # Track learning progress indicators
                # Q-value improvement (independent of exploration)
                q_improvement = q_diff.mean().item()
                self.log("learning_q_improvement", float(q_improvement))
                
                # Loss trend (smaller is better, tracks actual learning)
                self.log("learning_loss_value", float(loss.item()))
        
        return loss

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> Tensor:
        """Advance the environment, update replay buffer, and compute loss."""

        device = self.get_device(batch)
        epsilon = self.get_epsilon(
            self.hparams.eps_start,
            self.hparams.eps_end,
            self.hparams.eps_last_frame,
        )
        self.log("epsilon", epsilon)

        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        env = self.agent.env
        if getattr(env, "total_selections", 0) > 0:
            accuracy = env.correct_selections / max(env.total_selections, 1e-9)
            self.log("hit_accuracy", accuracy, prog_bar=True)

        self.log("step_reward", reward, prog_bar=False)
        self.log("episode reward", self.episode_reward)

        if done:
            stats = getattr(env, "last_episode_stats", None)
            if stats:
                expected = float(stats.get("expected_hits", 0.0))
                predicted = float(stats.get("total_selections", 0.0))
                correct = float(stats.get("correct_selections", 0.0))
                reason = stats.get("reason", "unknown")
                if expected > 0:
                    self.log("episode_hit_efficiency", correct / max(expected, 1e-9))
                if predicted > 0:
                    self.log("episode_hit_purity", correct / max(predicted, 1e-9))
                self.log("episode_expected_hits", expected)
                self.log("episode_selected_hits", predicted)
                self.log("episode_correct_hits", correct)
                reason_id = {
                    "finished_track": 0.0,
                    "no_candidates": 1.0,
                    "unknown": 2.0,
                }.get(reason, 2.0)
                self.log("episode_termination_reason", reason_id)
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        if not hasattr(self, "reward_history"):
            self.reward_history = []
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
        if self.global_step % 100 == 0 and self.reward_history:
            reward_array = np.array(self.reward_history)
            self.log("reward_mean", float(np.mean(reward_array)))
            self.log("reward_std", float(np.std(reward_array)))
            self.log("reward_min", float(np.min(reward_array)))
            self.log("reward_max", float(np.max(reward_array)))

        loss = self.dqn_mse_loss(batch)

        tau = 0.05
        if self.global_step % self.hparams.sync_rate == 0:
            for target_param, param in zip(
                self.target_net.parameters(), self.net.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

        self.log_dict({"reward": reward, "train_loss": loss})
        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> OrderedDict:
        """Run a handful of greedy validation episodes and log summary metrics."""

        device = self.get_device(batch)
        val_env = self.val_agent.env

        # Ensure validation samples reflect the current policy by clearing stale entries first
        if hasattr(self.val_buffer, "clear"):
            self.val_buffer.clear()

        val_rewards: List[float] = []
        efficiencies: List[float] = []
        purities: List[float] = []
        termination_reasons: List[str] = []
        accuracies: List[float] = []

        num_episodes = 10

        with torch.no_grad():
            for _ in range(num_episodes):
                self.val_agent.reset()
                done = False
                episode_reward = 0.0
                while not done:
                    reward, done = self.val_agent.play_step(
                        self.net, epsilon=0.0, device=device
                    )
                    episode_reward += reward
                val_rewards.append(episode_reward)

                stats = getattr(val_env, "last_episode_stats", None)
                if not stats:
                    continue

                expected = float(stats.get("expected_hits", 0.0))
                predicted = float(stats.get("total_selections", 0.0))
                correct = float(stats.get("correct_selections", 0.0))
                reason = stats.get("reason", "unknown")

                if expected > 0:
                    efficiencies.append(correct / max(expected, 1e-9))
                if predicted > 0:
                    purities.append(correct / max(predicted, 1e-9))
                if predicted > 0:
                    accuracies.append(correct / max(predicted, 1e-9))
                termination_reasons.append(reason)

        if val_rewards:
            self.log(
                "val_episode_reward",
                float(np.mean(val_rewards)),
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        if efficiencies:
            self.log(
                "val_hit_efficiency",
                float(np.mean(efficiencies)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
        else:
            self.log(
                "val_hit_efficiency",
                0.0,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
        if accuracies:
            self.log(
                "val_hit_accuracy",
                float(np.mean(accuracies)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
        else:
            self.log(
                "val_hit_accuracy",
                0.0,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
        if purities:
            self.log(
                "val_hit_purity",
                float(np.mean(purities)),
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        if termination_reasons:
            reason_counts = {reason: termination_reasons.count(reason) for reason in set(termination_reasons)}
            total = sum(reason_counts.values())
            for reason, count in reason_counts.items():
                self.log(
                    f"val_termination_{reason}",
                    float(count) / max(total, 1),
                    on_step=False,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )

        val_loss = None
        val_loss_value = None
        if len(self.val_buffer) > 0:
            try:
                sample = self.val_buffer.sample(self.hparams.batch_size)
                states_np, actions_np, next_states_np, rewards_np, dones_np, hf_np, nhf_np, hm_np, nhm_np = sample

                batch_tensors = (
                    torch.as_tensor(states_np, dtype=torch.float32, device=device),
                    torch.as_tensor(actions_np, dtype=torch.int64, device=device),
                    torch.as_tensor(rewards_np, dtype=torch.float32, device=device),
                    torch.as_tensor(dones_np, dtype=torch.bool, device=device),
                    torch.as_tensor(next_states_np, dtype=torch.float32, device=device),
                    torch.as_tensor(hf_np, dtype=torch.float32, device=device),
                    torch.as_tensor(nhf_np, dtype=torch.float32, device=device),
                    torch.as_tensor(hm_np, dtype=torch.bool, device=device),
                    torch.as_tensor(nhm_np, dtype=torch.bool, device=device),
                )
                val_loss = self.dqn_mse_loss(batch_tensors)
                val_loss_value = float(val_loss.item())
            except Exception as exc:
                logger.warning("Validation loss computation failed: %s", exc)

        if val_loss_value is None:
            val_loss_value = 1e6

        self.log(
            "val_loss",
            val_loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        if efficiencies:
            loss_tensor = (
                val_loss
                if val_loss is not None
                else torch.tensor(val_loss_value, device=device)
            )
            return OrderedDict(val_loss=loss_tensor)
        return OrderedDict()

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer with learning rate scheduling."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        
        # Add learning rate scheduler to reduce LR when validation plateaus
        # This helps with fine-tuning when learning slows down
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',  # train_loss should decrease
                factor=0.5,
                patience=20,
                min_lr=1e-6,
            ),
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1,
        }
        
        return [optimizer], [scheduler]

    def _collate_fn(self, batch):
        """Custom collate function to ensure float32 dtype for MPS compatibility."""
        # Handle empty batch
        if len(batch) == 0:
            raise ValueError("Cannot collate empty batch - buffer may be empty")
        
        # Check if batch items are new format (with hit_features) or old format
        first_item = batch[0]
        if len(first_item) == 5:
            # Old format: state, action, reward, done, next_state
            states, actions, rewards, dones, next_states = zip(*batch)
            hit_features = None
            next_hit_features = None
            hit_masks = None
            next_hit_masks = None
        else:
            # New format: includes hit_features
            states, actions, rewards, dones, next_states, hit_features, next_hit_features, hit_masks, next_hit_masks = zip(*batch)
        
        # Determine expected state dimension dynamically from the environment
        state_dim = getattr(getattr(self, "agent", None), "env", None)
        if state_dim is not None and getattr(state_dim, "observation_space", None) is not None:
            state_dim = int(state_dim.observation_space.shape[0])
        else:
            # Fallback: infer from the first state in the batch
            sample_state = first_item[0]
            state_dim = len(np.asarray(sample_state, dtype=np.float32).flatten())
            if state_dim == 0:
                state_dim = 4
        states_list = []
        next_states_list = []
        for state, next_state in zip(states, next_states):
            if state is None:
                state_arr = np.zeros(state_dim, dtype=np.float32)
            else:
                state_arr = np.asarray(state, dtype=np.float32).flatten()
                if state_arr.shape[0] != state_dim:
                    print(
                        f"[Collate] Adjusting state dim from {state_arr.shape} to {state_dim}"
                    )
                if state_arr.shape[0] < state_dim:
                    state_arr = np.pad(state_arr, (0, state_dim - state_arr.shape[0]), mode='constant')
                elif state_arr.shape[0] > state_dim:
                    state_arr = state_arr[:state_dim]
            if next_state is None:
                next_state_arr = np.zeros(state_dim, dtype=np.float32)
            else:
                next_state_arr = np.asarray(next_state, dtype=np.float32).flatten()
                if next_state_arr.shape[0] != state_dim:
                    print(
                        f"[Collate] Adjusting next_state dim from {next_state_arr.shape} to {state_dim}"
                    )
                if next_state_arr.shape[0] < state_dim:
                    next_state_arr = np.pad(next_state_arr, (0, state_dim - next_state_arr.shape[0]), mode='constant')
                elif next_state_arr.shape[0] > state_dim:
                    next_state_arr = next_state_arr[:state_dim]
            states_list.append(state_arr.astype(np.float32))
            next_states_list.append(next_state_arr.astype(np.float32))
        states_array = np.array(states_list, dtype=np.float32)
        next_states_array = np.array(next_states_list, dtype=np.float32)
        
        # Ensure rewards and actions are 1D arrays with same batch size
        # Convert each element to a scalar, handling various input types
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
        
        # Ensure all arrays have matching batch size
        if len(rewards_array) != batch_size:
            raise ValueError(f"Batch size mismatch: states={batch_size}, rewards={len(rewards_array)}, rewards_shape={rewards_array.shape}")
        if len(actions_array) != batch_size:
            raise ValueError(f"Batch size mismatch: states={batch_size}, actions={len(actions_array)}")
        if len(dones_array) != batch_size:
            raise ValueError(f"Batch size mismatch: states={batch_size}, dones={len(dones_array)}")
        
        # Final validation before tensor conversion
        assert states_array.shape == (batch_size, state_dim), f"States shape {states_array.shape} != ({batch_size}, {state_dim})"
        assert next_states_array.shape == (batch_size, state_dim), f"Next states shape {next_states_array.shape} != ({batch_size}, {state_dim})"
        assert rewards_array.shape == (batch_size,), f"Rewards shape {rewards_array.shape} != ({batch_size},)"
        assert actions_array.shape == (batch_size,), f"Actions shape {actions_array.shape} != ({batch_size},)"
        assert dones_array.shape == (batch_size,), f"Dones shape {dones_array.shape} != ({batch_size},)"
        
        # Convert to tensors with explicit float32 dtype
        states_tensor = torch.tensor(states_array, dtype=torch.float32)
        actions_tensor = torch.tensor(actions_array, dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards_array, dtype=torch.float32)
        dones_tensor = torch.tensor(dones_array, dtype=torch.bool)
        next_states_tensor = torch.tensor(next_states_array, dtype=torch.float32)
        
        # Final tensor shape validation
        if rewards_tensor.shape != (batch_size,):
            raise ValueError(f"Rewards tensor shape {rewards_tensor.shape} != ({batch_size},) after conversion")
        
        if hit_features is not None:
            # Process hit_features
            hit_features_list = list(hit_features)
            next_hit_features_list = list(next_hit_features)
            hit_masks_list = list(hit_masks)
            next_hit_masks_list = list(next_hit_masks)
            
            # Convert to numpy arrays and ensure consistent shapes (always max_hits=20, hit_dim=4)
            max_hits = 20
            hit_dim = 4
            hit_features_array = np.zeros((batch_size, max_hits, hit_dim), dtype=np.float32)
            next_hit_features_array = np.zeros((batch_size, max_hits, hit_dim), dtype=np.float32)
            hit_masks_array = np.zeros((batch_size, max_hits), dtype=bool)
            next_hit_masks_array = np.zeros((batch_size, max_hits), dtype=bool)
            
            for i, (hf, nhf, hm, nhm) in enumerate(zip(hit_features_list, next_hit_features_list, hit_masks_list, next_hit_masks_list)):
                hf_arr = np.asarray(hf, dtype=np.float32)
                nhf_arr = np.asarray(nhf, dtype=np.float32)
                hm_arr = np.asarray(hm, dtype=bool)
                nhm_arr = np.asarray(nhm, dtype=bool)
                
                # Normalize hf_arr to exactly (max_hits, hit_dim)
                if len(hf_arr.shape) == 2:
                    # Truncate if too large, pad if too small
                    if hf_arr.shape[0] > max_hits:
                        hf_arr = hf_arr[:max_hits]
                    elif hf_arr.shape[0] < max_hits:
                        padding = np.zeros((max_hits - hf_arr.shape[0], hf_arr.shape[1]), dtype=np.float32)
                        hf_arr = np.vstack([hf_arr, padding])
                    
                    # Ensure hit_dim is correct
                    if hf_arr.shape[1] != hit_dim:
                        if hf_arr.shape[1] > hit_dim:
                            hf_arr = hf_arr[:, :hit_dim]
                        else:
                            hf_arr = np.pad(hf_arr, ((0, 0), (0, hit_dim - hf_arr.shape[1])), mode='constant')
                    
                    hit_features_array[i] = hf_arr
                else:
                    # Invalid shape, use zeros
                    pass  # Already zeros from initialization
                
                # Same for nhf_arr
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
                
                # Normalize masks to exactly max_hits
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
            
            hit_features_tensor = torch.tensor(hit_features_array, dtype=torch.float32)
            next_hit_features_tensor = torch.tensor(next_hit_features_array, dtype=torch.float32)
            hit_masks_tensor = torch.tensor(hit_masks_array, dtype=torch.bool)
            next_hit_masks_tensor = torch.tensor(next_hit_masks_array, dtype=torch.bool)
            
            return (states_tensor, actions_tensor, rewards_tensor, dones_tensor, next_states_tensor,
                   hit_features_tensor, next_hit_features_tensor, hit_masks_tensor, next_hit_masks_tensor)
        
        return states_tensor, actions_tensor, rewards_tensor, dones_tensor, next_states_tensor

    def __dataloader(self, buffer: ReplayBuffer) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        # Don't create dataloader if buffer is empty
        if len(buffer) == 0:
            raise ValueError(f"Cannot create DataLoader with empty buffer. Buffer has {len(buffer)} samples.")
        
        dataset = RLDataset(buffer, self.hparams.episode_length)
        # Use actual batch_size, but ensure we have at least batch_size samples
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
        """Override to ensure all tensors are float32 when transferring to device.
        
        Our _collate_fn always returns a tuple of 5 tensors: (states, actions, rewards, dones, next_states)
        However, if we get a single tensor (which shouldn't happen), it means collate wasn't called properly.
        """
        # Handle case where batch is a single tensor (shouldn't happen - means collate wasn't called)
        if isinstance(batch, torch.Tensor):
            raise ValueError(f"Received single tensor instead of 5-tuple. This means collate_fn wasn't called. "
                           f"Check DataLoader configuration. Tensor shape: {batch.shape}")
        
        # Handle both old format (5 items) and new format (9 items with hit_features)
        if len(batch) == 5:
            states, actions, rewards, dones, next_states = batch
            # Convert float64/other floating types to float32, then move to device
            states = states.float().to(device) if states.is_floating_point() and states.dtype != torch.float32 else states.to(device)
            actions = actions.to(device)
            rewards = rewards.float().to(device) if rewards.is_floating_point() and rewards.dtype != torch.float32 else rewards.to(device)
            dones = dones.to(device)
            next_states = next_states.float().to(device) if next_states.is_floating_point() and next_states.dtype != torch.float32 else next_states.to(device)
            return states, actions, rewards, dones, next_states
        else:
            # New format with hit_features
            states, actions, rewards, dones, next_states, hit_features, next_hit_features, hit_masks, next_hit_masks = batch
            # Convert float64/other floating types to float32, then move to device
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
        """Retrieve device currently being used by minibatch."""
        if hasattr(batch[0], 'device'):
            device = batch[0].device
            if device.type == 'mps':
                return 'mps'
            elif device.type == 'cuda':
                return f'cuda:{device.index}' if device.index is not None else 'cuda'
        return "cpu"