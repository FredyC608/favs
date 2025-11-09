# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandalonePpoPolicy(nn.Module):
    """
    This is the standalone Actor-Critic network.
    It combines the feature extractor and the actor/critic heads.
    This is the *only* model class you'll need.
    """
    def __init__(self, input_shape=(5, 18, 20), features_dim=256, num_actions=6):
        super().__init__()
        
        in_channels = input_shape[0]
        
        # 1. Feature Extractor (CNN Backbone)
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            n_flatten = self.extractor(dummy_input).shape[1]
            
        self.shared_linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        
        # 2. Actor Head (outputs action logits)
        self.actor_head = nn.Linear(features_dim, num_actions)
        
        # 3. Critic Head (outputs a state value)
        self.critic_head = nn.Linear(features_dim, 1)

    def forward(self, x):
        """
        :param x: (N, 4, 18, 20) state tensor
        :return: (action_logits, state_value)
        """
        features = self.extractor(x)
        shared_features = self.shared_linear(features)
        
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        
        return action_logits, state_value