from torch import nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, n_actions)
        
        # Initialize weights with better scaling to prevent Q-value collapse
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights to prevent Q-value collapse."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use xavier_uniform with gain > 1 to initialize Q-values higher
                torch.nn.init.xavier_uniform_(m.weight, gain=2.0)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.1)  # Small positive bias

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)