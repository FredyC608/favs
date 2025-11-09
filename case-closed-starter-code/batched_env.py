# batched_env.py
import torch
import torch.nn.functional as F 
from model import StandalonePpoPolicy
from utils import (
    BOARD_HEIGHT, BOARD_WIDTH, 
    vectorized_translate, RELATIVE_MAP, DIR_TO_IDX_TORCH
)

# Constants
MAX_TURNS = 200
AGENT_1_START = (1, 2)
AGENT_2_START = (17, 15)
AGENT_1_START_DIR = (1, 0)  # RIGHT
AGENT_2_START_DIR = (-1, 0) # LEFT
# NUM_OBSTACLES = 5  # --- REMOVED: This is now dynamic ---
OBSTACLE_ID = 3     
VORONOI_REWARD_COEF = 0.0001 # Coefficient for the shaping reward
TIME_PENALTY = -0.001        # Small penalty for every step

class BatchedEnv:
    """
    A fully vectorized, GPU-based Tron environment.
    It manages `n_envs` games in parallel.
    """
    def __init__(self, n_envs, device):
        self.n_envs = n_envs
        self.device = device
        self.dims = torch.tensor([BOARD_WIDTH, BOARD_HEIGHT], 
                                 device=self.device, dtype=torch.long)
                                 
        # --- NEW: Curriculum Learning Variables ---
        self.num_obstacles = 0 # Start with 0 obstacles
        self.max_obstacles = 20 # Cap at 20 obstacles
        # --- END NEW ---

        # Game State Tensors
        self.grids = torch.zeros((n_envs, BOARD_HEIGHT, BOARD_WIDTH), 
                                 dtype=torch.int8, device=self.device)
        self.agent1_heads = torch.zeros((n_envs, 2), dtype=torch.long, device=self.device)
        self.agent2_heads = torch.zeros((n_envs, 2), dtype=torch.long, device=self.device)
        self.agent1_dirs = torch.zeros((n_envs, 2), dtype=torch.int8, device=self.device)
        self.agent2_dirs = torch.zeros((n_envs, 2), dtype=torch.int8, device=self.device)
        self.agent1_alive = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        self.agent2_alive = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        self.turn_counts = torch.zeros(n_envs, dtype=torch.int16, device=self.device)
        
        self.agent1_boosts = torch.full((n_envs,), 3, dtype=torch.int8, device=self.device)
        self.agent2_boosts = torch.full((n_envs,), 3, dtype=torch.int8, device=self.device)
        
        # State for alternating start positions
        self.p1_is_at_start_1 = torch.ones(n_envs, dtype=torch.bool, device=self.device)
        
        self.relative_map = RELATIVE_MAP.to(self.device)
        self.dir_to_idx_torch = DIR_TO_IDX_TORCH.to(self.device)

        # Opponent Policy
        self.opponent_policy = StandalonePpoPolicy().to(self.device)
        self.opponent_policy.eval() # Always in eval mode
        self.set_opponent_weights(self.opponent_policy.state_dict()) 
        
        self.env_indices = torch.arange(self.n_envs, device=self.device)
        
        # --- Voronoi Calculation Tensors ---
        self.last_voronoi_score = torch.zeros(n_envs, device=self.device, dtype=torch.float32)
        kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 
                              dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.flood_fill_kernel = kernel.repeat(1, 1, 1, 1)

    def set_opponent_weights(self, state_dict):
        """Loads new weights into the opponent policy."""
        self.opponent_policy.load_state_dict(state_dict)

    # --- NEW: Getter/Setter for Curriculum ---
    def get_num_obstacles(self):
        """Returns the current number of obstacles."""
        return self.num_obstacles
        
    def set_num_obstacles(self, new_count):
        """Sets the new obstacle count, respecting the max cap."""
        self.num_obstacles = min(new_count, self.max_obstacles) 
    # --- END NEW ---

    def _calculate_voronoi_scores(self):
        """
        Calculates the Voronoi (territory) score for all 256 envs in parallel.
        """
        territory = torch.zeros((self.n_envs, 2, BOARD_HEIGHT, BOARD_WIDTH), device=self.device, dtype=torch.float32)
        walls = (self.grids != 0).unsqueeze(1).float()
        visited = walls.clone()

        p1_heads = self.agent1_heads
        p2_heads = self.agent2_heads
        p1_head_flat = p1_heads[:, 1] * BOARD_WIDTH + p1_heads[:, 0]
        p2_head_flat = p2_heads[:, 1] * BOARD_WIDTH + p2_heads[:, 0]
        
        territory[:, 0].view(self.n_envs, -1).scatter_(1, p1_head_flat.unsqueeze(1), 1.0)
        territory[:, 1].view(self.n_envs, -1).scatter_(1, p2_head_flat.unsqueeze(1), 1.0)
        visited.view(self.n_envs, -1).scatter_(1, p1_head_flat.unsqueeze(1), 1.0)
        visited.view(self.n_envs, -1).scatter_(1, p2_head_flat.unsqueeze(1), 1.0)
        
        p1_alive_mask = self.agent1_alive.view(-1, 1, 1, 1)
        p2_alive_mask = self.agent2_alive.view(-1, 1, 1, 1)
        
        territory[:, 0:1] *= p1_alive_mask
        territory[:, 1:2] *= p2_alive_mask

        p1_terr = territory[:, 0:1] # (N, 1, H, W)
        p2_terr = territory[:, 1:2] # (N, 1, H, W)

        for _ in range(max(BOARD_HEIGHT, BOARD_WIDTH)):
            p1_growth = F.conv2d(p1_terr, self.flood_fill_kernel, padding=1)
            p2_growth = F.conv2d(p2_terr, self.flood_fill_kernel, padding=1)
            p1_growth = (p1_growth > 0).float()
            p2_growth = (p2_growth > 0).float()
            
            new_p1_cells = (p1_growth > visited) 
            new_p2_cells = (p2_growth > visited) & ~new_p1_cells 
            
            p1_terr[new_p1_cells] = 1.0
            p2_terr[new_p2_cells] = 1.0
            
            newly_visited = (new_p1_cells | new_p2_cells)
            visited += newly_visited.float() 
            
            if not newly_visited.any():
                break

        p1_score = p1_terr.sum(dim=(1, 2, 3))
        p2_score = p2_terr.sum(dim=(1, 2, 3))
        
        return p1_score - p2_score 

    def reset(self, dones=None):
        """Resets all environments specified by the `dones` mask."""
        if dones is None:
            dones = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        
        num_to_reset = dones.sum().item()
        if num_to_reset == 0:
            return self._get_obs() 

        # --- Position Alternating Logic ---
        self.p1_is_at_start_1[dones] = ~self.p1_is_at_start_1[dones]
        p1_at_start_1_mask = self.p1_is_at_start_1 & dones
        p1_at_start_2_mask = ~self.p1_is_at_start_1 & dones
        
        # Reset grids
        self.grids[dones] = 0
        
        # --- MODIFIED: Dynamic Obstacle Loop ---
        # Now loops 0 times at the start, and increases as agent learns
        for _ in range(self.num_obstacles):
            obs_x = torch.randint(0, BOARD_WIDTH, (num_to_reset,), device=self.device, dtype=torch.long)
            obs_y = torch.randint(0, BOARD_HEIGHT, (num_to_reset,), device=self.device, dtype=torch.long)
            self.grids[dones, obs_y, obs_x] = OBSTACLE_ID
        # --- END MODIFICATION ---
        
        # Pre-create tensors for start positions/directions
        agent1_start_pos = torch.tensor(AGENT_1_START, device=self.device, dtype=torch.long)
        agent2_start_pos = torch.tensor(AGENT_2_START, device=self.device, dtype=torch.long)
        agent1_start_dir = torch.tensor(AGENT_1_START_DIR, device=self.device, dtype=torch.int8)
        agent2_start_dir = torch.tensor(AGENT_2_START_DIR, device=self.device, dtype=torch.int8)

        # Reset agent states (Group 1: P1 starts at pos 1)
        self.agent1_heads[p1_at_start_1_mask] = agent1_start_pos
        self.agent1_dirs[p1_at_start_1_mask] = agent1_start_dir
        self.agent2_heads[p1_at_start_1_mask] = agent2_start_pos
        self.agent2_dirs[p1_at_start_1_mask] = agent2_start_dir
        
        # Reset agent states (Group 2: P1 starts at pos 2 - SWAPPED)
        self.agent1_heads[p1_at_start_2_mask] = agent2_start_pos 
        self.agent1_dirs[p1_at_start_2_mask] = agent2_start_dir 
        self.agent2_heads[p1_at_start_2_mask] = agent1_start_pos 
        self.agent2_dirs[p1_at_start_2_mask] = agent1_start_dir 

        # Reset common states for all 'dones' envs
        self.agent1_alive[dones] = True
        self.agent2_alive[dones] = True
        self.turn_counts[dones] = 0
        self.agent1_boosts[dones] = 3
        self.agent2_boosts[dones] = 3
        
        # Set initial trail positions on the grid
        self.grids[p1_at_start_1_mask, AGENT_1_START[1], AGENT_1_START[0]] = 1
        self.grids[p1_at_start_1_mask, AGENT_2_START[1], AGENT_2_START[0]] = 2
        self.grids[p1_at_start_2_mask, AGENT_2_START[1], AGENT_2_START[0]] = 1 
        self.grids[p1_at_start_2_mask, AGENT_1_START[1], AGENT_1_START[0]] = 2 
        
        # --- Calculate and store initial Voronoi score ---
        initial_scores = self._calculate_voronoi_scores()
        self.last_voronoi_score[dones] = initial_scores[dones]
        
        return self._get_obs()

    def _get_obs(self, for_opponent=False):
        """Builds the (n_envs, 5, H, W) state tensor."""
        obs = torch.zeros((self.n_envs, 5, BOARD_HEIGHT, BOARD_WIDTH), 
                          dtype=torch.float32, device=self.device)
        
        obstacles = (self.grids == OBSTACLE_ID)
        p1_trail = (self.grids == 1) | obstacles
        p2_trail = (self.grids == 2) | obstacles
        
        if for_opponent:
            obs[:, 0] = p2_trail 
            obs[:, 1] = p1_trail 
            my_boost_plane = (self.agent2_boosts.float() / 3.0).view(-1, 1, 1).expand(-1, BOARD_HEIGHT, BOARD_WIDTH)
        else:
            obs[:, 0] = p1_trail 
            obs[:, 1] = p2_trail 
            my_boost_plane = (self.agent1_boosts.float() / 3.0).view(-1, 1, 1).expand(-1, BOARD_HEIGHT, BOARD_WIDTH)

        # Scatter '1's at head locations
        p1_head_idx_flat = self.agent1_heads[:, 1] * BOARD_WIDTH + self.agent1_heads[:, 0]
        p2_head_idx_flat = self.agent2_heads[:, 1] * BOARD_WIDTH + self.agent2_heads[:, 0]
        
        obs[:, 4] = my_boost_plane
        
        p1_alive_flat_mask = self.agent1_alive.unsqueeze(1)
        p2_alive_flat_mask = self.agent2_alive.unsqueeze(1)

        if for_opponent:
            obs[:, 2].view(self.n_envs, -1).scatter_(1, p2_head_idx_flat.unsqueeze(1), p2_alive_flat_mask.float()) 
            obs[:, 3].view(self.n_envs, -1).scatter_(1, p1_head_idx_flat.unsqueeze(1), p1_alive_flat_mask.float()) 
        else:
            obs[:, 2].view(self.n_envs, -1).scatter_(1, p1_head_idx_flat.unsqueeze(1), p1_alive_flat_mask.float()) 
            obs[:, 3].view(self.n_envs, -1).scatter_(1, p2_head_idx_flat.unsqueeze(1), p2_alive_flat_mask.float()) 
        
        return obs

    def step(self, p1_rel_actions): 
        """Performs one step for all `n_envs` in parallel."""
        
        # 1. Get Opponent (P2) Actions (0-5)
        with torch.no_grad():
            opp_obs = self._get_obs(for_opponent=True)
            p2_logits, _ = self.opponent_policy(opp_obs)
            p2_dist = torch.distributions.Categorical(logits=p2_logits)
            p2_rel_actions = p2_dist.sample() 

        # 2. Parse Actions into Directions and Boosts
        p1_rel_dirs = p1_rel_actions % 3  
        p2_rel_dirs = p2_rel_actions % 3  
        p1_boost_requests = (p1_rel_actions >= 3)
        p2_boost_requests = (p2_rel_actions >= 3)
        p1_can_boost = p1_boost_requests & (self.agent1_boosts > 0)
        p2_can_boost = p2_boost_requests & (self.agent2_boosts > 0)
        self.agent1_boosts[p1_can_boost] -= 1
        self.agent2_boosts[p2_can_boost] -= 1
        p1_num_moves = torch.where(p1_can_boost, 2, 1)
        p2_num_moves = torch.where(p2_can_boost, 2, 1)

        # 3. Translate Relative to Absolute (Vectorized)
        p1_abs_moves = vectorized_translate(self.agent1_dirs, p1_rel_dirs, 
                                            self.relative_map, self.dir_to_idx_torch)
        p2_abs_moves = vectorized_translate(self.agent2_dirs, p2_rel_dirs,
                                            self.relative_map, self.dir_to_idx_torch)
        self.agent1_dirs = p1_abs_moves
        self.agent2_dirs = p2_abs_moves

        # --- 4. NEW MULTI-TICK MOVE LOOP ---
        for move_tick in range(1, 3): 
            if move_tick == 1:
                agents_to_move_p1 = self.agent1_alive
                agents_to_move_p2 = self.agent2_alive
            else: 
                agents_to_move_p1 = self.agent1_alive & (p1_num_moves == 2)
                agents_to_move_p2 = self.agent2_alive & (p2_num_moves == 2)
                
            if not agents_to_move_p1.any() and not agents_to_move_p2.any():
                break

            # 4a. Update Heads
            self.agent1_heads[agents_to_move_p1] = (self.agent1_heads[agents_to_move_p1] + p1_abs_moves[agents_to_move_p1]) % self.dims
            self.agent2_heads[agents_to_move_p2] = (self.agent2_heads[agents_to_move_p2] + p2_abs_moves[agents_to_move_p2]) % self.dims

            # 4b. Check Collisions
            p1_next_vals = self.grids[self.env_indices, self.agent1_heads[:, 1], self.agent1_heads[:, 0]]
            p2_next_vals = self.grids[self.env_indices, self.agent2_heads[:, 1], self.agent2_heads[:, 0]]
            
            p1_crashes = (p1_next_vals != 0) & agents_to_move_p1
            p2_crashes = (p2_next_vals != 0) & agents_to_move_p2
            both_moved = agents_to_move_p1 & agents_to_move_p2
            head_on = (self.agent1_heads == self.agent2_heads).all(dim=1) & both_moved
            p1_dead = p1_crashes | head_on
            p2_dead = p2_crashes | head_on

            # 4c. Update Grid and Alive Status
            alive_and_moved_p1 = agents_to_move_p1 & ~p1_dead
            alive_and_moved_p2 = agents_to_move_p2 & ~p2_dead

            self.grids[alive_and_moved_p1, self.agent1_heads[alive_and_moved_p1, 1], self.agent1_heads[alive_and_moved_p1, 0]] = 1
            self.grids[alive_and_moved_p2, self.agent2_heads[alive_and_moved_p2, 1], self.agent2_heads[alive_and_moved_p2, 0]] = 2
            
            self.agent1_alive &= ~p1_dead
            self.agent2_alive &= ~p2_dead

        # --- 5. END MULTI-TICK LOOP ---

        # 6. Calculate Rewards & Dones (This logic is now at the end)
        self.turn_counts += 1
        
        # --- Voronoi Reward Shaping ---
        current_voronoi_score = self._calculate_voronoi_scores()
        voronoi_reward = (current_voronoi_score - self.last_voronoi_score) * VORONOI_REWARD_COEF
        self.last_voronoi_score = current_voronoi_score
        
        # --- MODIFIED: Initialize rewards with BOTH shaping terms ---
        rewards = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        # Apply shaping rewards *only* to living agents
        rewards[self.agent1_alive] = voronoi_reward[self.agent1_alive] + TIME_PENALTY
        # --- END MODIFICATION ---
        
        # Check for crash-based dones
        crash_dones = ~self.agent1_alive | ~self.agent2_alive
        
        # Check for timeout-based dones
        timeout_dones = (self.turn_counts >= MAX_TURNS) & ~crash_dones 
        
        if timeout_dones.any():
            p1_length = (self.grids == 1).sum(dim=(1, 2)).float()
            p2_length = (self.grids == 2).sum(dim=(1, 2)).float()
            
            p1_wins_by_length = (p1_length > p2_length) & timeout_dones
            p1_loses_by_length = (p2_length > p1_length) & timeout_dones

            # Terminal rewards *overwrite* the shaping reward
            rewards[p1_wins_by_length] = 1.0
            rewards[p1_loses_by_length] = -1.0
        
        # Terminal rewards for crashes *overwrite* the shaping reward
        rewards[~self.agent1_alive & self.agent2_alive] = -1.0 # P1 Loss
        rewards[self.agent1_alive & ~self.agent2_alive] = 1.0  # P1 Win
        
        # Final done mask
        dones = crash_dones | timeout_dones
        
        # 7. Reset finished envs
        if dones.any():
            self.reset(dones=dones)
            
        return self._get_obs(), rewards, dones, {} # obs, rew, done, info