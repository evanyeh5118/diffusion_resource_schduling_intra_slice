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
        n_users: int = 4,
        len_window: int = 10,
        r_bar: int = 4,
        bandwidth: int = 40,
        max_episode_steps: int = 1000,
        action_mode: str = "full_action",  # "alpha_only", "full_action", or "discrete_alpha"
        observation_mode: str = "full",   # "full", "aggregated"
        alpha_range: Tuple[float, float] = (0.01, 1.0),
        discrete_alpha_steps: int = 20,
        random_seed: Optional[int] = None,
        reward_mode: str = "packet_loss",  # "packet_loss", "mdp_reward"
        traffic_data_path: Optional[str] = "Results/TrafficData/trafficData.pkl",  # NEW: Path to real traffic data
        use_real_traffic: bool = True,  # NEW: Whether to use real traffic data
        traffic_update_mode: str = "sequential"  # NEW: "markov" or "sequential"
    ):
        """
        Initialize the MDP Resource Scheduling Environment.
        
        Args:
            n_users: Number of users in the system (default: 4)
            len_window: Length of the observation window
            r_bar: Base data rate for Type2 users
            bandwidth: Total available bandwidth B
            max_episode_steps: Maximum steps per episode
            action_mode: Type of action space
                - "alpha_only": Only control alpha parameter
                - "full_action": Direct control over w, r, M, alpha (TRUE full control)
                - "discrete_alpha": Discrete alpha selection
            observation_mode: Type of observation space (kept for compatibility)
                - "full": Only user traffic states
                - "aggregated": Only user traffic states (same as full now)
            alpha_range: Valid range for alpha parameter
            discrete_alpha_steps: Number of discrete alpha values (for discrete_alpha mode)
            random_seed: Random seed for reproducibility
            reward_mode: Type of reward calculation
            traffic_data_path: Path to real traffic data pickle file (NEW)
            use_real_traffic: Whether to use real traffic data or synthetic (NEW)
            traffic_update_mode: How to update traffic - "markov" uses transition matrices, "sequential" reads data directly (NEW)
        """
        super().__init__()
        
        # Environment parameters
        self.params = {
            'N_user': n_users,
            'LEN_window': len_window,
            'r_bar': r_bar,
            'B': bandwidth,
            'randomSeed': random_seed
        }
        
        self.n_users = n_users
        self.len_window = len_window
        self.r_bar = r_bar
        self.bandwidth = bandwidth
        self.max_episode_steps = max_episode_steps
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        self.alpha_range = alpha_range
        self.discrete_alpha_steps = discrete_alpha_steps
        self.reward_mode = reward_mode
        self.traffic_data_path = traffic_data_path
        self.use_real_traffic = use_real_traffic
        self.traffic_update_mode = traffic_update_mode
        
        # Initialize traffic generator and MdpSchedule components
        self.traffic_generator = TrafficGenerator(self.params)
        
        # Load real traffic data if requested
        if self.use_real_traffic and self.traffic_data_path:
            self._load_traffic_data()
        
        self.env = Environment(self.params, self.traffic_generator)
        self.reward_kernel = RewardKernel(self.params)
        
        # Episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.episode_alpha_values = []
        
        # NEW: Track previous state for realistic observations
        self.previous_user_states = None
        
        # Define action space based on mode
        self._setup_action_space()
        
        # Define observation space based on mode
        self._setup_observation_space()
        
        # Reset to get initial state
        self.reset()
    
    def _setup_action_space(self):
        """Setup the action space based on action_mode."""
        if self.action_mode == "alpha_only":
            # Symmetric normalized action space [-1, 1] for alpha
            self.action_space = spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(1,), 
                dtype=np.float32
            )
        elif self.action_mode == "discrete_alpha":
            # Discrete alpha selection
            self.action_space = spaces.Discrete(self.discrete_alpha_steps)
            self.alpha_values = np.linspace(
                self.alpha_range[0], 
                self.alpha_range[1], 
                self.discrete_alpha_steps
            )
        elif self.action_mode == "full_action":
            # Symmetric normalized action space [-1, 1] for all actions
            action_dim = 2 * self.n_users + 2  # [w, r, M, alpha]
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_dim,),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")
    
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
    
    def _get_observation(self) -> np.ndarray:
        """Get observation - DRL agent can only observe PREVIOUS time window traffic."""
        if self.previous_user_states is None:
            # For the very first observation, use current state (we have to start somewhere)
            user_states = self.current_user_states / self.len_window
        else:
            # Agent observes the PREVIOUS traffic state (realistic constraint)
            user_states = self.previous_user_states / self.len_window
        
        return user_states.astype(np.float32)
    
    def _action_to_policy_params(self, action) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """Convert RL action to policy parameters (w, r, M, alpha)."""
        if self.action_mode == "alpha_only":
            # Convert [-1, 1] to [alpha_min, alpha_max]
            alpha_normalized = (action[0] + 1.0) / 2.0  # Convert to [0, 1]
            alpha = self.alpha_range[0] + alpha_normalized * (self.alpha_range[1] - self.alpha_range[0])
            
            # Use the typeAllocator logic from demo.py
            w = self._type_allocator(self.current_user_states)
            r = np.floor(alpha * self.bandwidth) / (np.sum(w) + 1e-10) * w
            M = 3  # Fixed M as in demo
            
        elif self.action_mode == "discrete_alpha":
            alpha = self.alpha_values[action]
            w = self._type_allocator(self.current_user_states)
            r = np.floor(alpha * self.bandwidth) / (np.sum(w) + 1e-10) * w
            M = 3
            
        elif self.action_mode == "full_action":
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
    
    def _type_allocator(self, u: np.ndarray) -> np.ndarray:
        """Type allocator from demo.py - assigns users to Type1 vs Type2."""
        w = (u > int(self.len_window * 0.5)).astype(int)
        return w
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment with realistic temporal constraints."""
        
        # Store current state as previous (for next observation)
        self.previous_user_states = self.current_user_states.copy()
        
        # Convert action to policy parameters based on CURRENT state
        # (This represents the decision delay - agent decided based on previous info)
        w, r, M, alpha = self._action_to_policy_params(action)
        
        # Apply action to environment using the CURRENT traffic state
        # (The action is applied to the actual current traffic, not what agent observed)
        if self.reward_mode == "packet_loss":
            reward = self.env.applyActions(w, r, M, alpha)
        else:  # mdp_reward
            reward = self.reward_kernel.getReward(self.current_user_states, w, r, M, alpha)
        
        # Track episode data
        self.episode_rewards.append(reward)
        self.episode_alpha_values.append(alpha)
        
        # Update step counter
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_episode_steps
        truncated = False
        
        # Update states for next step (this becomes the new "current" state)
        if not terminated:
            if self.use_real_traffic and self.traffic_update_mode == "sequential":
                self.current_user_states = self.traffic_generator.updateReadTraffic()
            else:
                self.current_user_states = self.env.updateStates()
        
        # Get next observation (will use the previous_user_states we just stored)
        obs = self._get_observation()
        
        # Create info dict with temporal information
        info = {
            'alpha': alpha,
            'M': M,
            'w': w.copy() if self.action_mode == "full_action" else None,
            'r': r.copy() if self.action_mode == "full_action" else None,
            'num_type1_users': np.sum(w),
            'num_type2_users': np.sum(1 - w),
            'total_resource_allocation': np.sum(r),
            'episode_length': self.current_step,
            'total_packet_loss_rate': self.env.getPacketLossRate(),
            'observed_previous_states': self.previous_user_states.copy() if self.previous_user_states is not None else None,
            'actual_current_states': self.current_user_states.copy()
        }
        
        return obs, -reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        if seed is not None:
            self.params['randomSeed'] = seed
            self.traffic_generator = TrafficGenerator(self.params)
            
            # Reload traffic data if using real traffic
            if self.use_real_traffic and self.traffic_data_path:
                self._load_traffic_data()
                
            self.env = Environment(self.params, self.traffic_generator)
            self.reward_kernel = RewardKernel(self.params)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.episode_alpha_values = []
        
        # Reset temporal state tracking
        self.previous_user_states = None
        
        # Reset environment
        self.env.reset()
        
        # Get initial user states
        if self.use_real_traffic and self.traffic_update_mode == "sequential":
            self.current_user_states = self.traffic_generator.updateReadTraffic()
        else:
            self.current_user_states = self.env.updateStates()
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'episode': 0,
            'total_packet_loss_rate': 0.0,
            'using_real_traffic': self.use_real_traffic,
            'traffic_update_mode': self.traffic_update_mode
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
    
    def _load_traffic_data(self):
        """Load real traffic data from pickle file."""
        import pickle
        import os
        
        try:
            if not os.path.exists(self.traffic_data_path):
                print(f"  Traffic data file not found: {self.traffic_data_path}")
                print("   Falling back to synthetic traffic generation.")
                self.use_real_traffic = False
                return
                
            with open(self.traffic_data_path, 'rb') as f:
                traffic_data = pickle.load(f)
            
            # Register the real traffic data
            self.traffic_generator.registerDataset(traffic_data['traffic'])
            print(f" Loaded real traffic data from: {self.traffic_data_path}")
            print(f"   Traffic data shape: {traffic_data['traffic'].shape}")
            
        except Exception as e:
            print(f" Error loading traffic data: {str(e)}")
            print("   Falling back to synthetic traffic generation.")
            self.use_real_traffic = False


# Example usage and testing
if __name__ == "__main__":
    # Test the environment with simplified state space and 4 users
    env = DRLResourceSchedulingEnv(
        action_mode="full_action",
        observation_mode="full",
        max_episode_steps=100,
        n_users=4,  # Default is now 4
        traffic_update_mode="sequential"  # Same as demo02
    )
    
    print("Action Space:", env.action_space)
    print("Action Space Shape:", env.action_space.shape)
    print("Observation Space:", env.observation_space)
    print("Observation Space Shape:", env.observation_space.shape)
    
    # Test random policy
    obs, info = env.reset()
    print("Initial observation shape:", obs.shape)
    print("Initial observation (user traffic states):", obs)
    
    total_reward = 0
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {step}:")
        print(f"  User traffic states: {obs}")
        print(f"  Action shape: {action.shape}")
        print(f"  w (user types): {info['w']}")
        print(f"  r (resources): {info['r']}")
        print(f"  M: {info['M']}, Alpha: {info['alpha']:.3f}")
        print(f"  Reward: {reward:.4f}")
        
        if terminated or truncated:
            break
    
    print(f"\nTotal Reward: {total_reward:.4f}")