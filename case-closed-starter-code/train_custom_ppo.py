# train_custom_ppo.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import numpy as np

# Imports from our project files
from batched_env import BatchedEnv
from model import StandalonePpoPolicy

# --- Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ENVS = 256              # Parallel environments
N_STEPS = 512             # Steps per env to collect for each update
BATCH_SIZE = 512          # Mini-batch size for learning
N_EPOCHS = 10             # Epochs per update
GAMMA = 0.99              # Discount factor
GAE_LAMBDA = 0.95         # GAE lambda
CLIP_EPS = 0.2            # PPO clip range
ENT_COEF = 0.02           # Entropy coefficient
LR = 3e-4
TOTAL_TIMESTEPS = 5_000_000
SELF_PLAY_FREQ = 10       # Update opponent every 10 training updates
MODEL_SAVE_PATH = 'ppo_policy_weights.pth' # Final output

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
                next_non_terminal = 1.0 - self.dones[t] # Use current done for the last step
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1] # Use next done for other steps
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
    
    env = BatchedEnv(n_envs=N_ENVS, device=DEVICE)
    policy = StandalonePpoPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    obs_shape = (5, 18, 20)
    buffer = RolloutBuffer(N_STEPS, N_ENVS, obs_shape, DEVICE)
    
    current_obs = env.reset()
    num_updates = TOTAL_TIMESTEPS // (N_ENVS * N_STEPS)
    
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Number of updates: {num_updates}")

    for update in range(1, num_updates + 1):
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
                
                # PPO Clip Loss
                ratio = (new_log_prob - b_log_probs).exp()
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_advs
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic Loss
                critic_loss = F.mse_loss(new_value.squeeze(), b_returns)
                
                loss = actor_loss + 0.5 * critic_loss - ENT_COEF * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        # --- 4. Self-Play Update & Logging ---
        if update % SELF_PLAY_FREQ == 0:
            print(f"Update {update}/{num_updates}: Updating opponent policy.")
            env.set_opponent_weights(policy.state_dict())
            torch.save(policy.state_dict(), MODEL_SAVE_PATH)
            
        update_end_time = time.time()
        sps = int((N_ENVS * N_STEPS) / (update_end_time - update_start_time))
        print(f"Update {update}/{num_updates} | SPS: {sps} | Mean Reward: {buffer.rewards.mean().item():.3f}")

    print(f"--- Training Complete. Total time: {(time.time() - start_time)/60:.2f} mins ---")
    torch.save(policy.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")
    
    return MODEL_SAVE_PATH

# --- This will be executed when you run `python train_custom_ppo.py` ---
if __name__ == "__main__":
    trained_model_path = run_training()
    print(f"Training finished! Model saved to {trained_model_path}")