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
    
    # Handle different environment reset return formats
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        if len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result[0]  # Take first element as observation
    else:
        obs = reset_result  # Single observation returned
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        
        # Handle different step return formats
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")
        
        episode_rewards.append(reward)
        
        # Safely extract info values
        if isinstance(info, dict):
            if 'alpha' in info:
                episode_alphas.append(info['alpha'])
            else:
                episode_alphas.append(0.0)  # Default value
                
            if 'total_packet_loss_rate' in info:
                episode_loss_rates.append(info['total_packet_loss_rate'])
            else:
                episode_loss_rates.append(0.0)  # Default value
        else:
            episode_alphas.append(0.0)
            episode_loss_rates.append(0.0)
        
        episode_actions.append(action.copy() if hasattr(action, 'copy') else action)

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
