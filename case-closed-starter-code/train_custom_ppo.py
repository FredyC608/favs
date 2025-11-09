# train_custom_ppo.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import numpy as np
import random  
from collections import deque 

# Imports from our project files
from batched_env import BatchedEnv
from model import StandalonePpoPolicy

# --- Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ENVS = 256              # Parallel environments
N_STEPS = 1024            # Steps per env to collect for each update
BATCH_SIZE = 512          # Mini-batch size for learning
N_EPOCHS = 10             # Epochs per update
GAMMA = 0.99              # Discount factor
GAE_LAMBDA = 0.95         # GAE lambda
CLIP_EPS = 0.2            # PPO clip range
ENT_COEF = 0.02           # Entropy coefficient
LR = 3e-4
TOTAL_TIMESTEPS = 100_000_000 # Back to a longer run
SELF_PLAY_FREQ = 10       # Update opponent every 10 training updates
MODEL_SAVE_PATH = 'ppo_policy_weights.pth' # Final output

# --- NEW: Curriculum Learning ---
CURRICULUM_WIN_RATE_THRESHOLD = 0.65 # Win rate to trigger difficulty increase
# --- END NEW ---

# --- Checkpoint Resuming ---
RESUME_CHECKPOINT = "./checkpoints/ppo_policy_update_0.pth"
START_UPDATE = 0 


# --- Rollout Buffer Class ---
class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_shape, device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.obs_shape = obs_shape

        self.obs = torch.zeros((n_steps, n_envs, *obs_shape), device=device)
        self.actions = torch.zeros((n_steps, n_envs), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((n_steps, n_envs), device=device)
        self.rewards = torch.zeros((n_steps, n_envs), device=device)
        self.dones = torch.zeros((n_steps, n_envs), device=device)
        self.values = torch.zeros((n_steps, n_envs), device=device)
        self.step_ptr = 0

    def store(self, obs, action, log_prob, reward, done, value):
        self.obs[self.step_ptr] = obs
        self.actions[self.step_ptr] = action
        self.log_probs[self.step_ptr] = log_prob
        self.rewards[self.step_ptr] = reward
        self.dones[self.step_ptr] = done
        self.values[self.step_ptr] = value
        self.step_ptr += 1

    def reset_ptr(self):
        self.step_ptr = 0

    def compute_gae(self, last_value, gamma, gae_lambda):
        advantages = torch.zeros_like(self.rewards)
        last_adv = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t] 
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1] 
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
            advantages[t] = last_adv

        returns = advantages + self.values
        return advantages, returns

    def get_batches(self, advantages, returns, batch_size):
        total_samples = self.n_steps * self.n_envs
        indices = torch.randperm(total_samples, device=self.device)

        # Flatten all data
        flat_obs = self.obs.view(-1, *self.obs_shape)
        flat_actions = self.actions.view(-1)
        flat_log_probs = self.log_probs.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)

        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                flat_obs[batch_indices],
                flat_actions[batch_indices],
                flat_log_probs[batch_indices],
                flat_advantages[batch_indices],
                flat_returns[batch_indices],
            )

# --- Main Training Function ---
def run_training():
    print(f"--- Starting Training on {DEVICE} ---")
    start_time = time.time()

    # Create a directory for checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    env = BatchedEnv(n_envs=N_ENVS, device=DEVICE)
    policy = StandalonePpoPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    # --- Add Cosine Annealing Scheduler ---
    num_updates = TOTAL_TIMESTEPS // (N_ENVS * N_STEPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_updates, eta_min=0)
    
    # --- Opponent Pool ---
    opponent_pool = deque(maxlen=20) 
    
    local_start_update = 0 
    
    if RESUME_CHECKPOINT is not None:
        if os.path.exists(RESUME_CHECKPOINT):
            print(f"Loading weights from {RESUME_CHECKPOINT}...")
            policy.load_state_dict(torch.load(RESUME_CHECKPOINT, map_location=DEVICE))
            
            local_start_update = START_UPDATE 
            print(f"Resuming training from update {local_start_update + 1}")
            
            print("Fast-forwarding LR scheduler...")
            for _ in range(local_start_update):
                scheduler.step()
            
        else:
            print(f"WARNING: Checkpoint file not found: {RESUME_CHECKPOINT}. Starting new training.")
    else:
        print("Starting new training run.")
    
    # Add the initial policy to the pool and sync the opponent
    initial_weights = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
    opponent_pool.append(initial_weights)
    env.set_opponent_weights(policy.state_dict()) 
    
    # --- NEW: Set initial obstacle count (not needed, env defaults to 0) ---
    # print(f"Starting with {env.get_num_obstacles()} obstacles.")
    # --- END NEW ---

    obs_shape = (5, 18, 20)
    buffer = RolloutBuffer(N_STEPS, N_ENVS, obs_shape, DEVICE)

    # Add metrics tracking
    ep_reward_queue = deque(maxlen=100)
    ep_length_queue = deque(maxlen=100)
    current_ep_rewards = torch.zeros(N_ENVS, device=DEVICE)
    current_ep_lengths = torch.zeros(N_ENVS, device=DEVICE)

    current_obs = env.reset()
    # num_updates = TOTAL_TIMESTEPS // (N_ENVS * N_STEPS) # Moved this up

    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Total updates: {num_updates}")

    # --- Start loop from local_start_update ---
    for update in range(local_start_update + 1, num_updates + 1):
        update_start_time = time.time()

        # --- 1. Rollout Phase ---
        policy.eval()
        for step in range(N_STEPS):
            with torch.no_grad():
                logits, value = policy(current_obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, done, _ = env.step(action)
            buffer.store(current_obs, action, log_prob, reward, done, value.squeeze())
            current_obs = next_obs

            # Update metrics trackers
            current_ep_rewards += reward
            current_ep_lengths += 1
            finished_envs_mask = (done == 1.0)
            if finished_envs_mask.any():
                ep_reward_queue.extend(current_ep_rewards[finished_envs_mask].cpu().numpy())
                ep_length_queue.extend(current_ep_lengths[finished_envs_mask].cpu().numpy())
                current_ep_rewards[finished_envs_mask] = 0
                current_ep_lengths[finished_envs_mask] = 0

        buffer.reset_ptr()

        # --- 2. Compute GAE & Returns ---
        with torch.no_grad():
            _, last_value = policy(current_obs)
        advantages, returns = buffer.compute_gae(last_value.squeeze(), GAMMA, GAE_LAMBDA)

        # --- 3. Learning Phase ---
        policy.train()
        for _ in range(N_EPOCHS):
            for batch in buffer.get_batches(advantages, returns, BATCH_SIZE):
                b_obs, b_actions, b_log_probs, b_advs, b_returns = batch

                logits, new_value = policy(b_obs)
                new_dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = new_dist.log_prob(b_actions)
                entropy = new_dist.entropy().mean()

                ratio = (new_log_prob - b_log_probs).exp()
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_advs
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(new_value.squeeze(), b_returns)

                loss = actor_loss + 0.5 * critic_loss - ENT_COEF * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        # --- 4. Self-Play Update & Logging ---
        update_end_time = time.time()
        sps = int((N_ENVS * N_STEPS) / (update_end_time - update_start_time))
        
        # --- MODIFIED: Added Obstacle count to log ---
        current_lr = optimizer.param_groups[0]['lr']
        num_obstacles = env.get_num_obstacles()
        
        if len(ep_reward_queue) > 0:
            mean_ep_rew = np.mean(ep_reward_queue)
            mean_ep_len = np.mean(ep_length_queue)
            win_rate = np.mean([1.0 if r == 1.0 else 0.0 for r in ep_reward_queue])
            
            print(f"Update {update}/{num_updates} | SPS: {sps} | LR: {current_lr:.1e} | Obst: {num_obstacles} | EpRew: {mean_ep_rew:.3f} | EpLen: {mean_ep_len:.1f} | WinRate: {win_rate:.2f}")
            
            # --- NEW: Curriculum Learning Logic ---
            if win_rate > CURRICULUM_WIN_RATE_THRESHOLD:
                new_obstacles = num_obstacles + 1
                env.set_num_obstacles(new_obstacles)
                print(f"--- CURRICULUM UPDATE: Win rate {win_rate:.2f} > {CURRICULUM_WIN_RATE_THRESHOLD}. Increasing obstacles to {new_obstacles}. ---")
            # --- END NEW ---
                
        else:
            print(f"Update {update}/{num_updates} | SPS: {sps} | LR: {current_lr:.1e} | Obst: {num_obstacles} | (Collecting episode stats...)")
        # --- END MODIFICATION ---

        # --- Opponent Pool Update ---
        if update % SELF_PLAY_FREQ == 0:
            print("Saving checkpoint and adding to opponent pool...")
            
            current_weights = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
            opponent_pool.append(current_weights)

            random_opponent_weights = random.choice(opponent_pool)
            random_opponent_weights_on_device = {k: v.to(DEVICE) for k, v in random_opponent_weights.items()}
            env.set_opponent_weights(random_opponent_weights_on_device)
            print("Opponent policy updated from a RANDOM past checkpoint.")

            checkpoint_path = os.path.join("checkpoints", f"ppo_policy_update_{update}.pth")
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        # --- END Opponent Pool Update ---

        # --- Step the LR scheduler ---
        scheduler.step()
        # --- END ---


    print(f"--- Training Complete. Total time: {(time.time() - start_time)/60:.2f} mins ---")

    torch.save(policy.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    return MODEL_SAVE_PATH

# --- This will be executed when you run `python train_custom_ppo.py` ---
if __name__ == "__main__":
    trained_model_path = run_training()
    print(f"Training finished! Model saved to {trained_model_path}")