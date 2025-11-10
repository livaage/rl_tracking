from torch import nn
import torch
import torch.nn.functional as F


class PointerNetwork(nn.Module):
    """
    Pointer network for scoring candidate hits based on current state.
    
    Architecture:
    - State encoder: Encodes current helix state (z, r, x, y, phi, etc.)
    - Hit encoder: Encodes each candidate hit's features (x, y, z, r, etc.)
    - Scorer: Computes compatibility score between state and each hit
    - Returns scores for each hit (higher = better match)
    """
    
    def __init__(self, state_dim=4, hit_dim=4, hidden_dim=256):
        """
        Args:
            state_dim: Dimension of state vector (helix position info)
            hit_dim: Dimension of hit feature vector (x, y, z, r, etc.)
            hidden_dim: Hidden dimension for encoders
        """
        super(PointerNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hit_dim = hit_dim
        self.hidden_dim = hidden_dim
        
        # State encoder: encodes current helix state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Query vector
        )
        
        # Hit encoder: encodes each candidate hit's features
        self.hit_encoder = self._build_hit_encoder(hit_dim)
        
        # Compatibility scorer: computes score between state and hit
        # Uses dot product attention
        self.score_projection = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to produce Q-values in the reward scale range."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use small initialization to keep outputs close to reward scale
                # Rewards are 0-1 (binary) or 0-10 (distance), so Q-values should start small
                torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain for smaller outputs
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)  # Start from zero
        
        # Scale final layer even smaller to match reward scale
        if hasattr(self, 'score_projection'):
            pass

    def _build_hit_encoder(self, hit_dim: int) -> nn.Sequential:
        encoder = nn.Sequential(
            nn.Linear(hit_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        for module in encoder.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
        return encoder
    
    def forward(self, state, hit_features, mask=None):
        """
        Forward pass: score each candidate hit given the current state.
        
        Args:
            state: Current state tensor [batch_size, state_dim] or [state_dim]
            hit_features: Candidate hit features [batch_size, num_hits, hit_dim] or [num_hits, hit_dim]
            mask: Optional mask for valid hits [batch_size, num_hits] or [num_hits]
                 True/1 for valid hits, False/0 for padding
        
        Returns:
            scores: Compatibility scores for each hit [batch_size, num_hits] or [num_hits]
        """
        # Handle single sample (no batch dimension)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            hit_features = hit_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = state.shape[0]
        num_hits = hit_features.shape[1]
        
        # Ensure num_hits matches expected max_hits (20) - truncate if necessary
        # This prevents shape mismatches if more hits are provided
        max_expected_hits = 20
        if num_hits > max_expected_hits:
            # Truncate hit_features to max_expected_hits
            hit_features = hit_features[:, :max_expected_hits]
            num_hits = max_expected_hits
            # Also truncate mask if provided
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask[:, :max_expected_hits]
                elif mask.dim() == 1:
                    mask = mask[:max_expected_hits]
        
        if state.shape[-1] != self.state_dim:
            print(
                f"[PointerNetwork] state dim mismatch: expected {self.state_dim}, got {state.shape[-1]}"
            )
            print(f"  state tensor shape: {state.shape}")
            raise RuntimeError("State dimension mismatch")

        # Encode state to query vector: [batch_size, hidden_dim]
        state_query = self.state_encoder(state)  # [batch_size, hidden_dim]
        
        # Check for inf/nan in state query before using
        if torch.isnan(state_query).any() or torch.isinf(state_query).any():
            state_query = torch.where(torch.isnan(state_query) | torch.isinf(state_query),
                                     torch.zeros_like(state_query), state_query)
        
        if hit_features.shape[-1] != self.hit_dim:
            print(
                f"[PointerNetwork] rebuilding hit encoder: prev hit_dim={self.hit_dim}, new hit_dim={hit_features.shape[-1]}"
            )
            self.hit_dim = hit_features.shape[-1]
            new_encoder = self._build_hit_encoder(self.hit_dim).to(hit_features.device)
            self.hit_encoder = new_encoder

        # Encode hit features to key vectors: [batch_size, num_hits, hidden_dim]
        hit_keys = self.hit_encoder(hit_features.view(-1, self.hit_dim))  # [batch_size * num_hits, hidden_dim]
        hit_keys = hit_keys.view(batch_size, num_hits, self.hidden_dim)  # [batch_size, num_hits, hidden_dim]
        
        # Check for inf/nan in hit keys
        if torch.isnan(hit_keys).any() or torch.isinf(hit_keys).any():
            hit_keys = torch.where(torch.isnan(hit_keys) | torch.isinf(hit_keys),
                                  torch.zeros_like(hit_keys), hit_keys)
        
        # Compute compatibility scores using attention mechanism
        # Expand state_query for broadcasting: [batch_size, 1, hidden_dim]
        state_query_expanded = state_query.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Compute dot product between query and keys, then project
        # Option 1: Dot product attention
        # scores = torch.sum(state_query_expanded * hit_keys, dim=-1)  # [batch_size, num_hits]
        
        # Option 2: Additive attention (more expressive)
        # Concatenate query with each key and project
        # query_key_concat = torch.cat([
        #     state_query_expanded.expand(-1, num_hits, -1),  # [batch_size, num_hits, hidden_dim]
        #     hit_keys  # [batch_size, num_hits, hidden_dim]
        # ], dim=-1)  # [batch_size, num_hits, 2*hidden_dim]
        # scores = self.score_projection(query_key_concat).squeeze(-1)  # [batch_size, num_hits]
        
        # Option 3: Scaled dot product (simpler, often works well)
        # Dot product with scaling
        scores = torch.sum(state_query_expanded * hit_keys, dim=-1) / (self.hidden_dim ** 0.5)
        
        # Clip scores to match reward scale
        # Rewards are in [0, 1] (binary) or [0, 10] (distance)
        # With gamma=0.99, max Q-value should be around reward_max / (1 - gamma) ≈ 10 / 0.01 = 1000
        # But we want to keep Q-values reasonable, so clamp to reward scale + some margin
        # For binary rewards [0,1]: max Q ≈ 100. For distance rewards [0,10]: max Q ≈ 1000
        # Use a conservative range that works for both: [-10, 50] initially
        scores = torch.clamp(scores, min=-10.0, max=50.0)
        
        # Replace any remaining NaN/inf with small finite values (not 0, as 0 might be a valid score)
        scores = torch.where(torch.isnan(scores), torch.zeros_like(scores), scores)
        # For inf, replace negative inf with very negative value, positive inf with very positive value
        scores = torch.where(scores == float('-inf'), torch.full_like(scores, -100.0), scores)
        scores = torch.where(scores == float('inf'), torch.full_like(scores, 100.0), scores)
        
        # Apply mask if provided (set invalid hits to very negative score)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            
            # Ensure mask matches scores shape (handle size mismatches)
            if mask.shape[1] != scores.shape[1]:
                # Truncate or pad mask to match scores shape
                if mask.shape[1] < scores.shape[1]:
                    # Pad mask with False (invalid hits)
                    padding = torch.zeros((mask.shape[0], scores.shape[1] - mask.shape[1]), 
                                         dtype=mask.dtype, device=mask.device)
                    mask = torch.cat([mask, padding], dim=1)
                else:
                    # Truncate mask
                    mask = mask[:, :scores.shape[1]]
            
            # Set invalid (masked) hits to a very negative score
            # Use a value much lower than valid scores but still reasonable
            # Since valid scores are in [-10, 50], use -100 for masked hits
            scores = scores.masked_fill(~mask, -100.0)
        
        # Remove batch dimension if input was single sample
        if squeeze_output:
            scores = scores.squeeze(0)
        
        return scores

