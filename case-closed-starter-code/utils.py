# utils.py
import torch

# Constants
BOARD_HEIGHT = 18
BOARD_WIDTH = 20

# We create a map for: (dx, dy) -> index
# (0,-1) UP -> 0
# (0, 1) DOWN -> 1
# (-1,0) LEFT -> 2
# (1, 0) RIGHT -> 3
DIR_TO_IDX = {
    (0, -1): 0,
    (0, 1): 1,
    (-1, 0): 2,
    (1, 0): 3,
}
IDX_TO_DIR = {v: k for k, v in DIR_TO_IDX.items()}

def _build_relative_map():
    """
    Builds a (4, 3, 2) tensor for fast action translation.
    map[dir_idx, rel_action_idx] -> (dx, dy)
    """
    # [Forward, Left, Right]
    # Current Dir: UP (0, -1)
    up_moves = torch.tensor([[0, -1], [-1, 0], [1, 0]], dtype=torch.int8)
    # Current Dir: DOWN (0, 1)
    down_moves = torch.tensor([[0, 1], [1, 0], [-1, 0]], dtype=torch.int8)
    # Current Dir: LEFT (-1, 0)
    left_moves = torch.tensor([[-1, 0], [0, 1], [0, -1]], dtype=torch.int8)
    # Current Dir: RIGHT (1, 0)
    right_moves = torch.tensor([[1, 0], [0, -1], [0, 1]], dtype=torch.int8)
    
    return torch.stack([up_moves, down_moves, left_moves, right_moves])

# Pre-build the maps (they will be moved to device in the env)
RELATIVE_MAP = _build_relative_map()
DIR_TO_IDX_TORCH = torch.tensor([
    [0, 0, 0], # y=0
    [0, 0, 0], # y=1
    [0, 0, 0]  # y=2
], dtype=torch.long)
DIR_TO_IDX_TORCH[1, 0] = 2 # (-1, 0) -> LEFT
DIR_TO_IDX_TORCH[1, 2] = 3 # (1, 0) -> RIGHT
DIR_TO_IDX_TORCH[0, 1] = 0 # (0, -1) -> UP
DIR_TO_IDX_TORCH[2, 1] = 1 # (0, 1) -> DOWN


def vectorized_translate(current_dirs_xy, rel_actions, relative_map, dir_to_idx_torch):
    """
    Fast, vectorized translation of relative actions.
    
    :param current_dirs_xy: (N_ENVS, 2) tensor of (dx, dy)
    :param rel_actions: (N_ENVS,) tensor of [0, 1, 2]
    :param relative_map: The (4, 3, 2) map (on device)
    :param dir_to_idx_torch: The (3, 3) map (on device)
    :return: (N_ENVS, 2) tensor of new (dx, dy)
    """
    # 1. Convert (dx, dy) to dir_idx [0-3]
    # (dx+1, dy+1) maps (-1, -1) -> (0, 0)
    indices_x = (current_dirs_xy[:, 0] + 1).long()
    indices_y = (current_dirs_xy[:, 1] + 1).long()
    
    current_dir_indices = dir_to_idx_torch[indices_y, indices_x]
    
    # 2. Use advanced indexing to get all new moves at once
    new_dirs = relative_map[current_dir_indices].gather(
        1, rel_actions.view(-1, 1, 1).expand(-1, 1, 2)
    ).squeeze(1)
    
    return new_dirs.to(torch.int8)