import matplotlib.pyplot as plt
import numpy as np
import os

from DrlLibs.DRL_config import get_training_config

def plot_training_results(callback, eval_results, algorithm_name: str, save_plots=True):
    """Plot training progress and evaluation results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{algorithm_name} Agent Training and Evaluation Results', fontsize=16)
    
    # 1. Training Loss Rates Over Time
    if callback.episode_loss_rates and callback.timesteps:
        ax1 = axes[0, 0]
        ax1.plot(callback.timesteps, callback.episode_loss_rates, alpha=0.7, color='blue')
        ax1.set_title('Training: Loss Rate Over Time')
        ax1.set_xlabel('Training Timesteps')
        ax1.set_ylabel('Packet Loss Rate')
        ax1.grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No training data available', 
                       transform=axes[0, 0].transAxes, ha='center', va='center')
        axes[0, 0].set_title('Training: Loss Rate Over Time')
    
    # 2. Evaluation Reward Distribution
    ax2 = axes[0, 1]
    ax2.hist(eval_results['episode_rewards'], bins=15, alpha=0.7, color='orange')
    ax2.axvline(eval_results['avg_reward'], color='red', linestyle='--', 
                label=f'Mean: {eval_results["avg_reward"]:.4f}')
    ax2.set_title('Evaluation: Reward Distribution')
    ax2.set_xlabel('Episode Reward')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Evaluation Loss Rate Distribution
    ax3 = axes[0, 2]
    ax3.hist(eval_results['episode_loss_rates'], bins=15, alpha=0.7, color='green')
    ax3.axvline(eval_results['avg_loss_rate'], color='red', linestyle='--',
                label=f'Mean: {eval_results["avg_loss_rate"]:.4f}')
    ax3.set_title('Evaluation: Loss Rate Distribution')
    ax3.set_xlabel('Packet Loss Rate')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Alpha Value Distribution
    ax4 = axes[1, 0]
    ax4.hist(eval_results['episode_alphas'], bins=15, alpha=0.7, color='purple')
    ax4.axvline(eval_results['avg_alpha'], color='red', linestyle='--',
                label=f'Mean: {eval_results["avg_alpha"]:.4f}')
    ax4.set_title('Evaluation: Alpha Value Distribution')
    ax4.set_xlabel('Alpha Value')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Action Space Analysis
    ax5 = axes[1, 1]
    if eval_results['episode_actions'] and len(eval_results['episode_actions']) > 0:
        try:
            first_action = eval_results['episode_actions'][0][0]
            if hasattr(first_action, '__len__') and len(first_action) > 1:
                # Continuous actions
                all_actions = []
                for ep_actions in eval_results['episode_actions']:
                    for action in ep_actions:
                        if hasattr(action, '__len__'):
                            all_actions.append(action)
                
                if all_actions:
                    all_actions = np.array(all_actions)
                    # Plot distribution of first few action dimensions
                    colors = ['blue', 'orange', 'green', 'red']
                    for i in range(min(4, all_actions.shape[1])):
                        ax5.hist(all_actions[:, i], bins=20, alpha=0.5, 
                                color=colors[i % len(colors)], label=f'Action {i}')
                    ax5.set_title('Action Distribution (First 4 Dimensions)')
                    ax5.set_xlabel('Action Value')
                    ax5.set_ylabel('Frequency')
                    ax5.legend()
            else:
                # Discrete actions
                all_actions = []
                for ep_actions in eval_results['episode_actions']:
                    all_actions.extend(ep_actions)
                ax5.hist(all_actions, bins=20, alpha=0.7, color='cyan')
                ax5.set_title('Action Distribution (Discrete)')
                ax5.set_xlabel('Action Value')
                ax5.set_ylabel('Frequency')
        except Exception as e:
            ax5.text(0.5, 0.5, f'Action analysis failed:\n{str(e)}', 
                    transform=ax5.transAxes, ha='center', va='center')
            ax5.set_title('Action Analysis (Error)')
    else:
        ax5.text(0.5, 0.5, 'No action data available', 
                transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('Action Distribution')
    
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Summary
    ax6 = axes[1, 2]
    metrics = ['Avg Reward', 'Avg Loss Rate', 'Avg Alpha']
    values = [eval_results['avg_reward'], eval_results['avg_loss_rate'], eval_results['avg_alpha']]
    errors = [eval_results['std_reward'], eval_results['std_loss_rate'], 0]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = ax6.bar(metrics, values, yerr=errors, capsize=5, alpha=0.7, color=colors)
    ax6.set_title('Performance Summary')
    ax6.set_ylabel('Value')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, height + (0.01 * height if height > 0 else -0.01 * abs(height)),
                f'{value:.4f}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    
    if save_plots:
        training_config = get_training_config()
        os.makedirs(training_config["plots_dir"], exist_ok=True)
        plot_path = f'{training_config["plots_dir"]}/{algorithm_name.lower()}_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
    
    plt.show()
