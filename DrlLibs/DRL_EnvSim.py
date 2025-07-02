import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

from EnvLibs import Environment, RewardKernel, TrafficGenerator


class DRLResourceSchedulingEnv(gym.Env):
    """
    Gym environment for the MDP-based resource scheduling problem.
    
    This environment wraps the MdpSchedule components and provides a standard
    Gym interface for RL agents to learn optimal resource allocation policies.
    """
    metadata = {"render_modes": ["human"]}
    def __init__(
        self,
        simParams,
        simEnv,
        obvMode="perfect",
        max_episode_steps: int = 5000,
    ):
        super().__init__()
        
        self.simParams = simParams
        self.simEnv = simEnv
        self.n_users = simParams['N_user']
        self.len_window = simParams['LEN_window']
        self.r_bar = simParams['r_bar']
        self.bandwidth = simParams['B']
        self.alpha_range = simParams['alpha_range']
        self.discrete_alpha_steps = simParams['discrete_alpha_steps']
        self.obvMode = obvMode
        self.max_episode_steps = max_episode_steps
                
        # Episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.episode_alpha_values = []
        
        # Define action space based on mode
        self._setup_action_space()
        
        # Define observation space based on mode
        self._setup_observation_space()
        
        # Reset to get initial state
        self.reset()
    
    def _setup_action_space(self):
        """Setup the action space based on action_mode."""
        # Symmetric normalized action space [-1, 1] for all actions
        action_dim = 2 * self.n_users + 2  # [w, r, M, alpha]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

    def _setup_observation_space(self):
        """Setup the observation space - simplified to only user traffic states."""
        # Simplified: Only user traffic states (normalized)
        obs_dim = self.n_users  # Just the current user states
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

    def _from_dl_action_to_env_action(self, action) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """Convert RL action to policy parameters (w, r, M, alpha)."""
        # Convert normalized [-1, 1] actions to actual values
        w_normalized = action[:self.n_users]
        r_normalized = action[self.n_users:2*self.n_users]
        M_normalized = action[2*self.n_users]
        alpha_normalized = action[2*self.n_users + 1]
        
        # Convert w from [-1, 1] to binary (threshold at 0)
        w = (w_normalized > 0.0).astype(int)
        
        # Convert r from [-1, 1] to [0, bandwidth]
        r_raw = (r_normalized + 1.0) / 2.0 * self.bandwidth  # [0, bandwidth]
        r = np.clip(r_raw, 0, self.bandwidth)
        
        # Convert M from [-1, 1] to [1, min(10, len_window)]
        M_raw = (M_normalized + 1.0) / 2.0 * (min(10, self.len_window) - 1) + 1
        M = int(np.clip(M_raw, 1, min(10, self.len_window)))
        
        # Convert alpha from [-1, 1] to [alpha_min, alpha_max]
        alpha_norm = (alpha_normalized + 1.0) / 2.0  # Convert to [0, 1]
        alpha = self.alpha_range[0] + alpha_norm * (self.alpha_range[1] - self.alpha_range[0])
        
        # Normalize r to respect total bandwidth constraint
        total_r = np.sum(r)
        if total_r > self.bandwidth:
            r = r * (self.bandwidth / total_r)  # Scale down proportionally
        
        return w, r, M, alpha
    
    def observe(self):
        # Get next observation 
        # normalized to [0, 1]
        self.u, self.u_predicted = self.simEnv.getStates()
        if self.obvMode == "perfect":
            obs = (self.u / self.len_window).astype(np.float32)
        elif self.obvMode == "predicted":
            obs = (self.u_predicted / self.len_window).astype(np.float32)
        else:
            raise ValueError(f"Invalid observation mode: {self.obvMode}")
        return obs
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment with realistic temporal constraints."""
        # ================== Update the Simulation Environment ==================
        obs = self.observe()
        # Convert action to policy parameters based on CURRENT state
        w, r, M, alpha = self._from_dl_action_to_env_action(action)
        # Apply action to environment
        reward = self.simEnv.applyActions(w, r, M, alpha)
        self.simEnv.updateStates()
        #==========================================================================
        # Track episode data
        self.episode_rewards.append(reward)
        self.episode_alpha_values.append(alpha)
        # Update step counter
        self.current_step += 1
        #==========================================================================
        # Check if episode is done
        terminated = self.current_step >= self.max_episode_steps
        truncated = False
        # Create info dict with temporal information
        info = {
            'alpha': alpha,
            'M': M,
            'w': w.copy(),
            'r': r.copy(),
            'num_type1_users': np.sum(w),
            'num_type2_users': np.sum(1 - w),
            'total_resource_allocation': np.sum(r),
            'episode_length': self.current_step,
            'total_packet_loss_rate': self.simEnv.getPacketLossRate(),
            'actual_current_states': self.u.copy(),
            'predicted_current_states': self.u_predicted.copy()
        }
        return obs, -reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        # Reset the Simulation Environment 
        self.simEnv.reset()
        self.simEnv.updateStates()
        # Reset episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.episode_alpha_values = []
        # Get initial observation
        obs = self.observe()
        info = {
            'episode': 0,
            'total_packet_loss_rate': 0.0,
        }
        
        return obs, info
    
    def render(self, mode: str = "human"):
        """Render the environment (optional)."""
        if mode == "human":
            if len(self.episode_rewards) > 0:
                print(f"Step: {self.current_step}")
                print(f"Last Reward: {self.episode_rewards[-1]:.4f}")
                print(f"Last Alpha: {self.episode_alpha_values[-1]:.3f}")
                print(f"Total Packet Loss Rate: {self.env.getPacketLossRate():.4f}")
                print(f"User States: {self.current_user_states}")
                print("-" * 50)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics for the completed episode."""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'episode_length': len(self.episode_rewards),
            'total_reward': sum(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'avg_alpha': np.mean(self.episode_alpha_values),
            'final_packet_loss_rate': self.env.getPacketLossRate(),
            'rewards': self.episode_rewards.copy(),
            'alphas': self.episode_alpha_values.copy()
        }
    