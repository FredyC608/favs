# batched_env.py
import torch
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
        
        self.relative_map = RELATIVE_MAP.to(self.device)
        self.dir_to_idx_torch = DIR_TO_IDX_TORCH.to(self.device)

        # Opponent Policy
        self.opponent_policy = StandalonePpoPolicy().to(self.device)
        self.opponent_policy.eval() # Always in eval mode
        self.set_opponent_weights(self.opponent_policy.state_dict()) 
        
        self.env_indices = torch.arange(self.n_envs, device=self.device)

    def set_opponent_weights(self, state_dict):
        """Loads new weights into the opponent policy."""
        self.opponent_policy.load_state_dict(state_dict)

    def reset(self, dones=None):
        """Resets all environments specified by the `dones` mask."""
        if dones is None:
            dones = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        
        num_to_reset = dones.sum().item()
        if num_to_reset == 0:
            return self._get_obs() # No reset needed

        # Reset grids
        self.grids[dones] = 0
        
        # Reset agent states
        self.agent1_heads[dones] = torch.tensor(AGENT_1_START, device=self.device, dtype=torch.long)
        self.agent2_heads[dones] = torch.tensor(AGENT_2_START, device=self.device, dtype=torch.long)
        
        self.agent1_dirs[dones] = torch.tensor(AGENT_1_START_DIR, device=self.device, dtype=torch.int8)
        self.agent2_dirs[dones] = torch.tensor(AGENT_2_START_DIR, device=self.device, dtype=torch.int8)
        
        self.agent1_alive[dones] = True
        self.agent2_alive[dones] = True
        self.turn_counts[dones] = 0
        
        self.agent1_boosts[dones] = 3
        self.agent2_boosts[dones] = 3
        
        # Set initial trail positions on the grid
        self.grids[dones, AGENT_1_START[1], AGENT_1_START[0]] = 1
        self.grids[dones, AGENT_2_START[1], AGENT_2_START[0]] = 2
        
        return self._get_obs()

    def _get_obs(self, for_opponent=False):
        """Builds the (n_envs, 5, H, W) state tensor."""
        obs = torch.zeros((self.n_envs, 5, BOARD_HEIGHT, BOARD_WIDTH), 
                          dtype=torch.float32, device=self.device)
        
        p1_trail = (self.grids == 1)
        p2_trail = (self.grids == 2)
        
        if for_opponent:
            obs[:, 0] = p2_trail # Opponent's trail
            obs[:, 1] = p1_trail # "My" (P1's) trail
            my_boost_plane = (self.agent2_boosts.float() / 3.0).view(-1, 1, 1).expand(-1, BOARD_HEIGHT, BOARD_WIDTH)
        else:
            obs[:, 0] = p1_trail # My trail
            obs[:, 1] = p2_trail # Opponent's trail
            my_boost_plane = (self.agent1_boosts.float() / 3.0).view(-1, 1, 1).expand(-1, BOARD_HEIGHT, BOARD_WIDTH)

        # Scatter '1's at head locations
        p1_head_idx_flat = self.agent1_heads[:, 1] * BOARD_WIDTH + self.agent1_heads[:, 0]
        p2_head_idx_flat = self.agent2_heads[:, 1] * BOARD_WIDTH + self.agent2_heads[:, 0]
        
        obs[:, 4] = my_boost_plane
        
        # Create masks for alive agents
        p1_alive_flat_mask = self.agent1_alive.unsqueeze(1)
        p2_alive_flat_mask = self.agent2_alive.unsqueeze(1)

        if for_opponent:
            obs[:, 2].view(self.n_envs, -1).scatter_(1, p2_head_idx_flat.unsqueeze(1), p2_alive_flat_mask.float()) # My (P2) head
            obs[:, 3].view(self.n_envs, -1).scatter_(1, p1_head_idx_flat.unsqueeze(1), p1_alive_flat_mask.float()) # Opp's (P1) head
        else:
            obs[:, 2].view(self.n_envs, -1).scatter_(1, p1_head_idx_flat.unsqueeze(1), p1_alive_flat_mask.float()) # My (P1) head
            obs[:, 3].view(self.n_envs, -1).scatter_(1, p2_head_idx_flat.unsqueeze(1), p2_alive_flat_mask.float()) # Opp's (P2) head
        
        return obs

    def step(self, p1_rel_actions): # p1_rel_actions is now (N_ENVS,) with values 0-5
        """Performs one step for all `n_envs` in parallel."""
        
        # 1. Get Opponent (P2) Actions (0-5)
        with torch.no_grad():
            opp_obs = self._get_obs(for_opponent=True)
            p2_logits, _ = self.opponent_policy(opp_obs)
            p2_dist = torch.distributions.Categorical(logits=p2_logits)
            p2_rel_actions = p2_dist.sample() # (n_envs,)

        # 2. Parse Actions into Directions and Boosts
        p1_rel_dirs = p1_rel_actions % 3  # (0, 1, 2)
        p2_rel_dirs = p2_rel_actions % 3  # (0, 1, 2)
        
        p1_boost_requests = (p1_rel_actions >= 3)
        p2_boost_requests = (p2_rel_actions >= 3)
        
        # Check who can *actually* boost
        p1_can_boost = p1_boost_requests & (self.agent1_boosts > 0)
        p2_can_boost = p2_boost_requests & (self.agent2_boosts > 0)
        
        # Decrement boost counts
        self.agent1_boosts[p1_can_boost] -= 1
        self.agent2_boosts[p2_can_boost] -= 1
        
        # Determine number of moves for each agent
        p1_num_moves = torch.where(p1_can_boost, 2, 1)
        p2_num_moves = torch.where(p2_can_boost, 2, 1)

        # 3. Translate Relative to Absolute (Vectorized)
        p1_abs_moves = vectorized_translate(self.agent1_dirs, p1_rel_dirs, 
                                            self.relative_map, self.dir_to_idx_torch)
        p2_abs_moves = vectorized_translate(self.agent2_dirs, p2_rel_dirs,
                                            self.relative_map, self.dir_to_idx_torch)
        
        # Update directions (this happens regardless of move count)
        self.agent1_dirs = p1_abs_moves
        self.agent2_dirs = p2_abs_moves

        # --- 4. NEW MULTI-TICK MOVE LOOP ---
        # We must loop twice (max moves)
        for move_tick in range(1, 3): # Tick 1, then Tick 2
            
            # --- Tick 1 Logic (All living agents move) ---
            if move_tick == 1:
                agents_to_move_p1 = self.agent1_alive
                agents_to_move_p2 = self.agent2_alive
            
            # --- Tick 2 Logic (Only living, boosted agents move) ---
            else: 
                agents_to_move_p1 = self.agent1_alive & (p1_num_moves == 2)
                agents_to_move_p2 = self.agent2_alive & (p2_num_moves == 2)
                
            # If no one is moving this tick, break the loop
            if not agents_to_move_p1.any() and not agents_to_move_p2.any():
                break

            # 4a. Update Heads for agents moving this tick
            self.agent1_heads[agents_to_move_p1] = (self.agent1_heads[agents_to_move_p1] + p1_abs_moves[agents_to_move_p1]) % self.dims
            self.agent2_heads[agents_to_move_p2] = (self.agent2_heads[agents_to_move_p2] + p2_abs_moves[agents_to_move_p2]) % self.dims

            # 4b. Check Collisions (This logic needs to be applied carefully)
            # This is complex: you must check P1[move] vs P2[move] (head-on),
            # then P1[move] vs Grid, and P2[move] vs Grid.
            
            # Simplified collision check (you'll need to expand this)
            p1_next_vals = self.grids[self.env_indices, self.agent1_heads[:, 1], self.agent1_heads[:, 0]]
            p2_next_vals = self.grids[self.env_indices, self.agent2_heads[:, 1], self.agent2_heads[:, 0]]

            # Check only for agents that just moved
            p1_crashes = (p1_next_vals != 0) & agents_to_move_p1
            p2_crashes = (p2_next_vals != 0) & agents_to_move_p2
            
            # Head-on check: only if both moved *this tick*
            both_moved = agents_to_move_p1 & agents_to_move_p2
            head_on = (self.agent1_heads == self.agent2_heads).all(dim=1) & both_moved
            
            p1_dead = p1_crashes | head_on
            p2_dead = p2_crashes | head_on

            # 4c. Update Grid and Alive Status
            # Agents that just moved and *didn't* die
            alive_and_moved_p1 = agents_to_move_p1 & ~p1_dead
            alive_and_moved_p2 = agents_to_move_p2 & ~p2_dead

            # Draw new trails
            self.grids[alive_and_moved_p1, self.agent1_heads[alive_and_moved_p1, 1], self.agent1_heads[alive_and_moved_p1, 0]] = 1
            self.grids[alive_and_moved_p2, self.agent2_heads[alive_and_moved_p2, 1], self.agent2_heads[alive_and_moved_p2, 0]] = 2
            
            # Update alive status
            self.agent1_alive &= ~p1_dead
            self.agent2_alive &= ~p2_dead

        # --- 5. END MULTI-TICK LOOP ---

        # 6. Calculate Rewards & Dones (This logic is now at the end)
        self.turn_counts += 1
        
        rewards = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        rewards[self.agent1_alive] = 0.000 # Living reward
        rewards[~self.agent1_alive & self.agent2_alive] = -1.0 # P1 Loss
        rewards[self.agent1_alive & ~self.agent2_alive] = 1.0  # P1 Win
        
        # Check for crash-based dones
        crash_dones = ~self.agent1_alive | ~self.agent2_alive
        
        # Check for timeout-based dones
        timeout_dones = (self.turn_counts >= MAX_TURNS) & ~crash_dones # Only timeout if not already crashed
        
        if timeout_dones.any():
            # Calculate trail lengths ONLY for envs that timed out
            p1_length = (self.grids == 1).sum(dim=(1, 2)).float()
            p2_length = (self.grids == 2).sum(dim=(1, 2)).float()
            
            p1_wins_by_length = (p1_length > p2_length) & timeout_dones
            p1_loses_by_length = (p2_length > p1_length) & timeout_dones
            # If lengths are equal, it's a draw (reward remains 0.001)

            # Assign win/loss rewards for timeout
            rewards[p1_wins_by_length] = 1.0
            rewards[p1_loses_by_length] = -1.0
        
        # Final done mask
        dones = crash_dones | timeout_dones
        
        # 7. Reset finished envs
        if dones.any():
            self.reset(dones=dones)
            
        return self._get_obs(), rewards, dones, {} # obs, rew, done, info