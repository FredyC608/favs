"""
Fast CPU inference module for deployment
Optimized to run under 4-second time limit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from case_closed_game import Direction


class ResidualBlock(nn.Module):
    """ResNet-style residual block"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class CaseClosedNetwork(nn.Module):
    """Policy-Value network for Case Closed (inference-optimized)"""

    def __init__(self, input_channels=8, num_filters=128, num_blocks=6):
        super().__init__()

        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])

        # Policy head (direction)
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 18 * 20, 4)

        # Boost head
        self.boost_conv = nn.Conv2d(num_filters, 16, kernel_size=1, bias=False)
        self.boost_bn = nn.BatchNorm2d(16)
        self.boost_fc = nn.Linear(16 * 18 * 20, 2)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 18 * 20, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            x = block(x)

        # Policy head (direction)
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        # Boost head
        boost = F.relu(self.boost_bn(self.boost_conv(x)))
        boost = boost.view(boost.size(0), -1)
        boost_logits = self.boost_fc(boost)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_logits, boost_logits, value


class FastInferenceAgent:
    """Fast inference wrapper for deployment"""

    def __init__(self, model_path):
        """Load trained model for inference

        Args:
            model_path: Path to deployment_model.pt file
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Create network
        config = checkpoint['config']
        self.network = CaseClosedNetwork(
            input_channels=config['input_channels'],
            num_filters=config['num_filters'],
            num_blocks=config['num_blocks']
        )

        # Load weights
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()

        # Set to inference mode
        torch.set_grad_enabled(False)

        print(f"[OK] Model loaded from {model_path}")
        print(f"  Parameters: {sum(p.numel() for p in self.network.parameters()):,}")

    def state_to_tensor(self, game_state, player_number):
        """Convert game state dict to neural network input

        Args:
            game_state: Dict from /send-state endpoint
            player_number: 1 or 2

        Returns:
            torch.Tensor of shape (1, 8, 18, 20)
        """
        state = np.zeros((8, 18, 20), dtype=np.float32)

        # Get board and trails
        board = game_state.get('board', [[]])
        agent1_trail = game_state.get('agent1_trail', [])
        agent2_trail = game_state.get('agent2_trail', [])

        # Determine "my" perspective vs "opponent"
        if player_number == 1:
            my_trail = agent1_trail
            opp_trail = agent2_trail
            my_boosts = game_state.get('agent1_boosts', 0)
            opp_boosts = game_state.get('agent2_boosts', 0)
        else:
            my_trail = agent2_trail
            opp_trail = agent1_trail
            my_boosts = game_state.get('agent2_boosts', 0)
            opp_boosts = game_state.get('agent1_boosts', 0)

        # Channel 0: My trail
        for pos in my_trail:
            x, y = pos
            if 0 <= x < 20 and 0 <= y < 18:
                state[0, y, x] = 1.0

        # Channel 1: Opponent trail
        for pos in opp_trail:
            x, y = pos
            if 0 <= x < 20 and 0 <= y < 18:
                state[1, y, x] = 1.0

        # Channel 2: My head (last position in trail)
        if my_trail:
            x, y = my_trail[-1]
            if 0 <= x < 20 and 0 <= y < 18:
                state[2, y, x] = 1.0

        # Channel 3: Opponent head
        if opp_trail:
            x, y = opp_trail[-1]
            if 0 <= x < 20 and 0 <= y < 18:
                state[3, y, x] = 1.0

        # Channels 4-5: Direction (simplified - filled uniformly for now)
        # In real implementation, track direction from trail history
        state[4, :, :] = 0.5  # Placeholder
        state[5, :, :] = 0.5  # Placeholder

        # Channel 6-7: Boosts
        state[6, :, :] = my_boosts / 3.0
        state[7, :, :] = opp_boosts / 3.0

        # Convert to tensor (add batch dimension)
        tensor = torch.from_numpy(state).unsqueeze(0)  # (1, 8, 18, 20)
        return tensor

    def get_move(self, game_state, player_number, current_direction=None):
        """Get best move from current game state

        Args:
            game_state: Dict from /send-state endpoint
            player_number: 1 or 2
            current_direction: Current direction (for masking opposite)

        Returns:
            str: One of "UP", "DOWN", "LEFT", "RIGHT"
        """
        # Convert state to tensor
        state_tensor = self.state_to_tensor(game_state, player_number)

        # Forward pass
        with torch.no_grad():
            policy_logits, boost_logits, value = self.network(state_tensor)

        # Apply action masking if we know current direction
        if current_direction is not None:
            mask = self._get_action_mask(current_direction)
            policy_logits = policy_logits.masked_fill(~mask, -1e8)

        # Get direction action
        probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        action_idx = torch.argmax(probs).item()
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        move = directions[action_idx]

        # Get boost decision (for now, don't use boost - can enable later)
        # boost_probs = F.softmax(boost_logits, dim=-1).squeeze(0)
        # use_boost = torch.argmax(boost_probs).item() == 1
        # if use_boost and game_state.get(f'agent{player_number}_boosts', 0) > 0:
        #     move = move + ":BOOST"

        return move

    def _get_action_mask(self, current_direction):
        """Get boolean mask for valid actions (can't go opposite direction)

        Args:
            current_direction: Current direction enum or string

        Returns:
            torch.Tensor of shape (1, 4) with True for valid actions
        """
        # Map opposite directions
        opposites = {
            "UP": 1,      # DOWN is index 1
            "DOWN": 0,    # UP is index 0
            "LEFT": 3,    # RIGHT is index 3
            "RIGHT": 2,   # LEFT is index 2
        }

        # If current_direction is a Direction enum, convert to string
        if isinstance(current_direction, Direction):
            dir_map = {
                Direction.UP: "UP",
                Direction.DOWN: "DOWN",
                Direction.LEFT: "LEFT",
                Direction.RIGHT: "RIGHT",
            }
            current_direction = dir_map[current_direction]

        mask = torch.ones(1, 4, dtype=torch.bool)

        if current_direction in opposites:
            opposite_idx = opposites[current_direction]
            mask[0, opposite_idx] = False

        return mask

    def get_move_with_confidence(self, game_state, player_number):
        """Get move along with confidence score

        Returns:
            tuple: (move_str, confidence_float)
        """
        state_tensor = self.state_to_tensor(game_state, player_number)

        with torch.no_grad():
            policy_logits, boost_logits, value = self.network(state_tensor)

        probs = F.softmax(policy_logits, dim=-1).squeeze(0)
        action_idx = torch.argmax(probs).item()
        confidence = probs[action_idx].item()

        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        return directions[action_idx], confidence
