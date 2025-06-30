from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import os

from DrlLibs import DRLResourceSchedulingEnv
from DrlLibs.DRL_config import (
    get_algorithm_config, 
    get_training_config,
    print_algorithm_info
)

class TrainingCallback(BaseCallback):
    """Custom callback to track detailed training performance."""
    
    def __init__(self, algorithm_name: str, verbose=0):
        super().__init__(verbose)
        self.algorithm_name = algorithm_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_alphas = []
        self.episode_loss_rates = []
        self.timesteps = []
        self.episodes = 0
    
    def _on_step(self) -> bool:
        # Track per-step metrics
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Check if episode ended
            if 'episode' in info or self.locals.get('dones', [False])[0]:
                self.episodes += 1
                
                # Log episode metrics
                if 'total_packet_loss_rate' in info:
                    self.episode_loss_rates.append(info['total_packet_loss_rate'])
                    self.timesteps.append(self.num_timesteps)
        
        return True
    

def create_environment(simParams, simEnv):
    """Create and return the resource scheduling environment."""
     
    env = DRLResourceSchedulingEnv(
        simParams,
        simEnv
    )
    return Monitor(env)


def train_drl_agent(algorithm_name: str, env, total_timesteps, save_path, agentName):
    """Train a DRL agent using the specified algorithm."""    
    underlying_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    print(f"\n{'='*60}")
    print(f"Training {algorithm_name} Agent")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Environment: {underlying_env.n_users} users, {underlying_env.bandwidth} bandwidth")
    print(f"Save path: {save_path}.zip")
    # Get algorithm configuration
    config = get_algorithm_config(algorithm_name, env)
    algorithm_class = config["class"]
    params = config["params"]
    # Create callback to track performance
    callback = TrainingCallback(algorithm_name)
    # Create model
    model = algorithm_class(
        "MlpPolicy", 
        env, 
        verbose=1,
        device='cpu',
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
    
    print(f"{algorithm_name} training completed in {training_time:.2f} seconds")
    print(f"Model saved to: {save_path}/{agentName}.zip")
    
    return model, callback, training_time