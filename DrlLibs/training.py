import torch
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import time
import os
import numpy as np
from collections import deque

from DrlLibs import DRLResourceSchedulingEnv
from DrlLibs.DRL_config import (
    get_algorithm_config, 
    get_training_config,
    print_algorithm_info
)

class TrainingCallback(BaseCallback):
    """Simple callback to track reward progress during training (no early stopping)."""
    
    def __init__(self, algorithm_name: str, log_interval: int = 16, 
                 moving_avg_window: int = 100, verbose=0):
        super().__init__(verbose)
        self.algorithm_name = algorithm_name
        self.log_interval = log_interval
        self.moving_avg_window = moving_avg_window
        
        self.timesteps_log = []
        self.rewards_log = []
        self.cumulative_rewards = []
        self.total_reward = 0.0
        
        self.reward_buffer = deque(maxlen=moving_avg_window)
        self.moving_avg_rewards = []
        
        self.episode_loss_rates = []
        self.episode_timesteps = []
        self.episodes_seen = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [])
        if len(rewards) > 0:
            reward = rewards[0]
            self.total_reward += reward
            self.reward_buffer.append(reward)
            if self.num_timesteps % self.log_interval == 0:
                self.timesteps_log.append(self.num_timesteps)
                self.rewards_log.append(reward)
                self.cumulative_rewards.append(self.total_reward)
                if len(self.reward_buffer) >= min(self.moving_avg_window // 2, 10):
                    current_moving_avg = np.mean(list(self.reward_buffer))
                    self.moving_avg_rewards.append(current_moving_avg)
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        if len(infos) > 0 and len(dones) > 0:
            info = infos[0]
            done = dones[0]
            if done and info:
                self.episodes_seen += 1
                if 'total_packet_loss_rate' in info:
                    self.episode_loss_rates.append(info['total_packet_loss_rate'])
                    self.episode_timesteps.append(self.num_timesteps)
        return True  # Always continue training


def create_environment(simParams, simEnv, obvMode="perfect", num_episodes=5000):
    """Create and return the resource scheduling environment."""
     
    env = DRLResourceSchedulingEnv(
        simParams,
        simEnv,
        obvMode,
        num_episodes
    )
    return Monitor(env)

def make_env(simParams, simEnv, obvMode, timesteps_per_episode, rank=0):
    """Create a single environment instance."""
    def _init():
        env = DRLResourceSchedulingEnv(
            simParams,
            simEnv,
            obvMode,
            timesteps_per_episode
        )
        return Monitor(env)
    return _init

def create_parallel_environment(simParams, simEnv, obvMode="perfect", 
                              timesteps_per_episode=5000, n_envs=4):
    """Create parallel environments."""
    if n_envs == 1:
        # Single environment
        return create_environment(simParams, simEnv, obvMode, timesteps_per_episode)
    else:
        # Multiple parallel environments
        env_fns = [make_env(simParams, simEnv, obvMode, timesteps_per_episode, i) 
                   for i in range(n_envs)]
        return SubprocVecEnv(env_fns)

def train_drl_agent(algorithm_name: str, env, total_timesteps, save_path, agentName,
                   moving_avg_window: int = 100):
    """Train a DRL agent using the specified algorithm with early stopping based on moving average."""    
    print(f"\n{'='*60}")
    print(f"Training {algorithm_name} as {agentName} Agent")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Moving average window: {moving_avg_window} data points")
    
    # Handle both single and parallel environments
    try:
        if hasattr(env, 'unwrapped'):
            # Single environment
            underlying_env = env.unwrapped
            print(f"Environment: {underlying_env.n_users} users, {underlying_env.bandwidth} bandwidth")
        elif hasattr(env, 'num_envs'):
            # Parallel environments (VecEnv)
            print(f"Environment: {env.num_envs} parallel environments")
        else:
            print("Environment: Unknown environment type")
    except AttributeError:
        # Fallback if any attribute access fails
        print("Environment: Parallel environments (details not accessible)")
    
    print(f"Save path: {save_path}.zip")
    
    # Get algorithm configuration
    config = get_algorithm_config(algorithm_name, env)
    algorithm_class = config["class"]
    params = config["params"]
    
    # Create callback to track performance with early stopping
    callback = TrainingCallback(algorithm_name, log_interval=16, 
                               moving_avg_window=moving_avg_window,
                               verbose=1)
    
    # Create model
    model = algorithm_class(
        "MlpPolicy", 
        env, 
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **params
    )
    
    # Train the model
    start_time = time.time()
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_time = time.time() - start_time
    
    # Save the model
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/{agentName}.zip")
    
    # Print training summary
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"Training completed normally")
    print(f"Total timesteps: {callback.num_timesteps}")
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Model saved to: {save_path}/{agentName}.zip")
    
    return model, callback, training_time