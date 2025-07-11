import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_results(callback, eval_results, algorithm_name: str, save_plots=True):
    """Plot training and evaluation progress with moving average rewards."""
    
    # Check if we have timestep-level reward data
    if (hasattr(callback, 'timesteps_log') and len(callback.timesteps_log) > 0 and
        hasattr(callback, 'rewards_log') and len(callback.rewards_log) > 0):
        
        print(f"Plotting training progress: {len(callback.timesteps_log)} data points over {callback.timesteps_log[-1]} timesteps")
        
        timesteps = np.array(callback.timesteps_log)
        rewards = np.array(callback.rewards_log)
        
        # Create a clean plot showing training and evaluation progress
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Moving average of rewards during training
        plt.subplot(2, 1, 1)
        
        # Calculate moving average with 1000-step window (since each episode is 1000 steps)
        window = min(1000 // callback.log_interval, len(rewards))  # Convert episode length to data points
        if window < 2:
            window = min(10, len(rewards) // 2)  # Fallback for very short training
            
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(timesteps[window-1:], moving_avg, 
                    label=f'{algorithm_name} Training Moving Avg ({window} data points ≈ 1 episode)', 
                    color='red', linewidth=2)
        else:
            # If not enough data for moving average, just plot the raw data
            plt.plot(timesteps, rewards, label=f'{algorithm_name} Training Reward', 
                    color='red', linewidth=2)
        
        plt.xlabel('Training Timesteps')
        plt.ylabel('Average Reward')
        plt.title(f'{algorithm_name} Training Progress: Moving Average Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Subplot 2: Moving average of rewards during evaluation (20-step window)
        plt.subplot(2, 1, 2)
        
        if eval_results and 'episode_rewards' in eval_results and len(eval_results['episode_rewards']) > 0:
            eval_rewards = np.array(eval_results['episode_rewards'])
            eval_steps = np.arange(1, len(eval_rewards) + 1)
            
            # Fixed 20-step moving average for evaluation
            eval_window = 20
            
            if len(eval_rewards) >= eval_window:
                eval_moving_avg = np.convolve(eval_rewards, np.ones(eval_window)/eval_window, mode='valid')
                plt.plot(eval_steps[eval_window-1:], eval_moving_avg, 
                        label=f'{algorithm_name} Evaluation Moving Avg ({eval_window} steps)', 
                        color='green', linewidth=2)
            else:
                # If not enough data for 20-step moving average, plot raw evaluation rewards
                plt.plot(eval_steps, eval_rewards, 
                        label=f'{algorithm_name} Evaluation Reward (raw)', 
                        color='green', linewidth=2)
            
            plt.xlabel('Evaluation Episode Steps')
            plt.ylabel('Average Reward')
            plt.title(f'{algorithm_name} Evaluation Performance: 20-Step Moving Average Reward')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add comparison statistics
            training_final_avg = np.mean(rewards[-min(window, len(rewards)):]) if len(rewards) > 0 else 0
            eval_avg = np.mean(eval_rewards)
            
            stats_text = f"""Training vs Evaluation:
Training Final Avg: {training_final_avg:.4f}
Evaluation Avg: {eval_avg:.4f}
Performance Gap: {eval_avg - training_final_avg:.4f}

Training Stats:
Total Timesteps: {timesteps[-1]:,}
Episodes Completed: {callback.episodes_seen}
Data Points: {len(timesteps)}"""
            
        else:
            plt.text(0.5, 0.5, 'No evaluation data available', 
                    transform=plt.gca().transAxes, ha='center', va='center', fontsize=14)
            plt.title(f'{algorithm_name} Evaluation: No Data Available')
            
            # Stats without evaluation comparison
            training_final_avg = np.mean(rewards[-min(window, len(rewards)):]) if len(rewards) > 0 else 0
            stats_text = f"""Training Stats:
Total Timesteps: {timesteps[-1]:,}
Episodes Completed: {callback.episodes_seen}
Data Points: {len(timesteps)}
Final Avg Reward: {training_final_avg:.4f}"""
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
    else:
        # Fallback: Show that no training data was captured
        print("No training reward data captured. Check callback configuration.")
        
        # Still show evaluation results if available
        if eval_results and 'episode_rewards' in eval_results:
            plt.figure(figsize=(10, 6))
            episode_rewards = eval_results['episode_rewards']
            plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, 
                    label='Evaluation Episode Rewards', color='green', linewidth=2)
            plt.xlabel('Step in Evaluation Episode')
            plt.ylabel('Reward')
            plt.title(f'{algorithm_name} Evaluation Results')
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            print("No evaluation data available either.")
            return
    
    if save_plots:
        filename = f'{algorithm_name.lower()}_training_progress.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    plt.close()  # <-- This ensures the figure is closed and not shown
