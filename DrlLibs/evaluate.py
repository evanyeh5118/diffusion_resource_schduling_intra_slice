import numpy as np
import matplotlib.pyplot as plt
import os

from DrlLibs.DRL_config import (
    get_algorithm_config, 
    get_training_config,
)
from DrlLibs.training import create_environment

def evaluate_drl_agent(model, env, algorithm_name: str, eval_seed=None):
    """Evaluate a trained DRL agent."""
    
    training_config = get_training_config()
    if eval_seed is None:
        eval_seed = training_config["eval_seed"]
    
    print(f"\n{'='*60}")
    print(f"Evaluating {algorithm_name} Agent")
    print(f"{'='*60}")
    
    episode_rewards = []
    episode_loss_rates = []
    episode_alphas = []
    episode_actions = []
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards.append(reward)
        episode_alphas.append(info['alpha'])
        episode_actions.append(action.copy() if hasattr(action, 'copy') else action)
        episode_loss_rates.append(info['total_packet_loss_rate'])

        if terminated or truncated:
            break
    
    # Calculate statistics
    results = {
        'algorithm': algorithm_name,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_loss_rate': np.mean(episode_loss_rates),
        'std_loss_rate': np.std(episode_loss_rates),
        'avg_alpha': np.mean(episode_alphas),
        'episode_rewards': episode_rewards,
        'episode_loss_rates': episode_loss_rates,
        'episode_alphas': episode_alphas,
        'episode_actions': episode_actions
    }
    
    print(f"Evaluation Results:")
    print(f"  Average Reward: {results['avg_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"  Average Loss Rate: {results['avg_loss_rate']:.4f} ± {results['std_loss_rate']:.4f}")
    print(f"  Average Alpha: {results['avg_alpha']:.4f}")
    
    return results


def load_and_evaluate(simParams, simEnv, load_path, algorithm_name, obvMode="perfect", episode_timesteps=1000):
    """Load a trained model and evaluate it."""
    
    print(f"Loading {algorithm_name} model from {load_path}")
    
    # Create environment
    env = create_environment(simParams, simEnv, obvMode, episode_timesteps)
    
    # Get algorithm class
    config = get_algorithm_config(algorithm_name, env)
    algorithm_class = config["class"]
    
    # Load model
    model = algorithm_class.load(load_path)
    
    # Evaluate
    eval_results = evaluate_drl_agent(model, env, algorithm_name)
    
    env.close()
    return model, eval_results
