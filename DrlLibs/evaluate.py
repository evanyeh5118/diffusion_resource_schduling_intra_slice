import numpy as np
import matplotlib.pyplot as plt
import os

from DrlLibs.DRL_config import (
    get_algorithm_config, 
    get_training_config,
)
from DrlLibs.training import create_environment

def evaluate_drl_agent(model, env, algorithm_name: str, num_episodes=None, eval_seed=None):
    """Evaluate a trained DRL agent."""
    
    training_config = get_training_config()
    if num_episodes is None:
        num_episodes = training_config["eval_episodes"]
    if eval_seed is None:
        eval_seed = training_config["eval_seed"]
    
    print(f"\n{'='*60}")
    print(f"Evaluating {algorithm_name} Agent")
    print(f"{'='*60}")
    
    episode_rewards = []
    episode_loss_rates = []
    episode_alphas = []
    episode_lengths = []
    episode_actions = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=eval_seed + episode)
        episode_reward = 0
        episode_alpha_list = []
        episode_action_list = []
        step_count = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_alpha_list.append(info['alpha'])
            episode_action_list.append(action.copy() if hasattr(action, 'copy') else action)
            step_count += 1
            
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_loss_rates.append(info['total_packet_loss_rate'])
                episode_alphas.append(np.mean(episode_alpha_list))
                episode_lengths.append(step_count)
                episode_actions.append(episode_action_list)
                break
    
    # Calculate statistics
    results = {
        'algorithm': algorithm_name,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_loss_rate': np.mean(episode_loss_rates),
        'std_loss_rate': np.std(episode_loss_rates),
        'avg_alpha': np.mean(episode_alphas),
        'avg_episode_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_loss_rates': episode_loss_rates,
        'episode_alphas': episode_alphas,
        'episode_actions': episode_actions
    }
    
    print(f"Evaluation Results:")
    print(f"  Average Reward: {results['avg_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"  Average Loss Rate: {results['avg_loss_rate']:.4f} ± {results['std_loss_rate']:.4f}")
    print(f"  Average Alpha: {results['avg_alpha']:.4f}")
    print(f"  Average Episode Length: {results['avg_episode_length']:.2f}")
    
    return results


def load_and_evaluate(simParams, simEnv, load_path, algorithm_name, num_episodes=10):
    """Load a trained model and evaluate it."""
    
    print(f"Loading {algorithm_name} model from {load_path}")
    
    # Create environment
    env = create_environment(simParams, simEnv)
    
    # Get algorithm class
    config = get_algorithm_config(algorithm_name, env)
    algorithm_class = config["class"]
    
    # Load model
    model = algorithm_class.load(load_path)
    
    # Evaluate
    eval_results = evaluate_drl_agent(model, env, algorithm_name, num_episodes=num_episodes)
    
    env.close()
    return model, eval_results


'''
def evaluate_model(model, env, algorithm_name, num_episodes=10, eval_seed=42):
    """Evaluate a trained model and return performance metrics."""
    
    episode_rewards = []
    episode_loss_rates = []
    episode_alphas = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        # Use same seed for fair comparison across algorithms
        obs, _ = env.reset(seed=eval_seed + episode)  # ← FIXED SEED PER EPISODE
        episode_reward = 0
        episode_alpha_list = []
        step_count = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_alpha_list.append(info['alpha'])
            step_count += 1
            
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_loss_rates.append(info['total_packet_loss_rate'])
                episode_alphas.append(np.mean(episode_alpha_list))
                episode_lengths.append(step_count)
                break
    
    return {
        'algorithm': algorithm_name,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_loss_rate': np.mean(episode_loss_rates),
        'std_loss_rate': np.std(episode_loss_rates),
        'avg_alpha': np.mean(episode_alphas),
        'avg_episode_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_loss_rates': episode_loss_rates,
        'episode_alphas': episode_alphas
    }
'''