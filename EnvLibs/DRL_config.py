"""
Configuration file for DRL algorithms used in resource scheduling.
Contains algorithm-specific hyperparameters and settings.
"""

import numpy as np
from stable_baselines3 import SAC, PPO, A2C, TD3, DQN
from stable_baselines3.common.noise import NormalActionNoise


def get_algorithm_config(algorithm_name: str, env):
    """
    Get algorithm-specific configuration including class, hyperparameters, and action space type.
    
    Args:
        algorithm_name: Name of the algorithm ("SAC", "PPO", "A2C", "TD3", "DQN")
        env: The environment (needed for action space dimensions)
    
    Returns:
        Dictionary containing algorithm class, parameters, and action space type
    """
    
    configs = {
        "SAC": {
            "class": SAC,
            "params": {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
            },
            "action_space": "continuous",
            "description": "Soft Actor-Critic - off-policy algorithm for continuous control"
        },
        
        "PPO": {
            "class": PPO,
            "params": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
            },
            "action_space": "continuous",
            "description": "Proximal Policy Optimization - on-policy algorithm"
        },
        
        "A2C": {
            "class": A2C,
            "params": {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.01,
                "vf_coef": 0.25,
            },
            "action_space": "continuous",
            "description": "Advantage Actor-Critic - on-policy algorithm"
        },
        
        "TD3": {
            "class": TD3,
            "params": {
                "learning_rate": 1e-3,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "action_noise": NormalActionNoise(
                    mean=np.zeros(env.action_space.shape[-1]), 
                    sigma=0.1 * np.ones(env.action_space.shape[-1])
                ),
            },
            "action_space": "continuous",
            "description": "Twin Delayed Deep Deterministic Policy Gradient - off-policy algorithm"
        },
        
        "DQN": {
            "class": DQN,
            "params": {
                "learning_rate": 3e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 1000,
                "exploration_fraction": 0.1,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
            "action_space": "discrete",
            "description": "Deep Q-Network - off-policy algorithm for discrete actions"
        }
    }
    
    if algorithm_name not in configs:
        available_algorithms = list(configs.keys())
        raise ValueError(f"Algorithm {algorithm_name} not supported. Available algorithms: {available_algorithms}")
    
    return configs[algorithm_name]


def get_environment_config():
    """
    Get default environment configuration for resource scheduling.
    Uses the same parameters as the MDP solver demos for fair comparison.
    
    Returns:
        Dictionary containing environment parameters
    """
    return {
        "n_users": 4,           # Same as demo N_user
        "len_window": 10,       # Same as demo LEN_window
        "r_bar": 4,             
        "bandwidth": 30,        
        "max_episode_steps": 500,
        "reward_mode": "packet_loss",
        "action_modes": {
            "full_action": "Direct control over w, r, M, alpha parameters",
            "alpha_only": "Only control alpha parameter",
            "discrete_alpha": "Discrete alpha selection"
        }
    }


def get_training_config():
    """
    Get default training configuration.
    
    Returns:
        Dictionary containing training parameters
    """
    return {
        "total_timesteps": 200000,
        "eval_episodes": 20,
        "eval_seed": 42,
        "save_models": True,
        "save_plots": True,
        "models_dir": "models",
        "plots_dir": "plots"
    }


def print_algorithm_info():
    """Print information about all available algorithms."""
    
    print("Available DRL Algorithms:")
    print("=" * 60)
    
    # Create a dummy environment for getting configs
    from EnvLibs.DRL_EnvSim import DRLResourceSchedulingEnv
    dummy_env = DRLResourceSchedulingEnv(n_users=4, action_mode="full_action")
    
    algorithms = ["SAC", "PPO", "A2C", "TD3", "DQN"]
    
    for alg in algorithms:
        config = get_algorithm_config(alg, dummy_env)
        print(f"\n{alg}:")
        print(f"  Description: {config['description']}")
        print(f"  Action Space: {config['action_space']}")
        print(f"  Key Parameters:")
        for param, value in config['params'].items():
            if param != 'action_noise':  # Skip action_noise for cleaner output
                print(f"    - {param}: {value}")
    
    dummy_env.close()


def print_environment_info():
    """Print information about the environment configuration."""
    
    env_config = get_environment_config()
    
    print("\nEnvironment Configuration:")
    print("=" * 40)
    print(f"Number of users: {env_config['n_users']}")
    print(f"Window length: {env_config['len_window']}")
    print(f"Base rate (r_bar): {env_config['r_bar']}")
    print(f"Total bandwidth: {env_config['bandwidth']}")
    print(f"Max episode steps: {env_config['max_episode_steps']}")
    print(f"Reward mode: {env_config['reward_mode']}")
    
    print("\nAvailable Action Modes:")
    for mode, description in env_config['action_modes'].items():
        print(f"  - {mode}: {description}")


def print_training_info():
    """Print information about the training configuration."""
    
    training_config = get_training_config()
    
    print("\nTraining Configuration:")
    print("=" * 40)
    for param, value in training_config.items():
        print(f"{param}: {value}")


if __name__ == "__main__":
    # Print all configuration information
    print_algorithm_info()
    print_environment_info()
    print_training_info()