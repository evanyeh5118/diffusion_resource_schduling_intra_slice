import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .DiffusionQ import DiffusionQLearner

class DiffusionPolicyInterface:
    def __init__(self, 
                 envParams,
                 max_episode_steps: int = 5000,
                 ):
        self.n_users = envParams['N_user']
        self.max_episode_steps = max_episode_steps
        self.alpha_range = envParams['alpha_range']
        self.bandwidth = envParams['B']
        self.len_window = envParams['LEN_window']
        self.diffusionQ = None
        self._init()

    def _init(self):
        self.state_dim = self.n_users # [u_i | i=0,1,...,N-1]
        self.action_dim = 2 * self.n_users + 2 # [w_i, r_i, M, alpha | i=0,1,...,N-1]
        self.diffusionQ = DiffusionQLearner(self.state_dim, self.action_dim)
    
    def _from_diffusionQ_action_to_env_action(self, action):
        """Convert RL action to policy parameters (w, r, M, alpha)."""
        """The action is normalized to [-1, 1]"""
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
    
    def _from_env_action_to_diffusionQ_action(self, action):
        """Convert policy parameters (w, r, M, alpha) to RL action."""
        """The action is normalized to [-1, 1]"""
        w, r, M, alpha = action
        # Convert all values to [-1, 1]
        w_normalized = w[w==0] = -1
        r_normalized = (2.0*r / self.bandwidth - 1.0)
        M_normalized = (M - 1) / (min(10, self.len_window) - 1)
        alpha_normalized = (2.0*alpha - self.alpha_range[0]) / (self.alpha_range[1] - self.alpha_range[0])
        return np.concatenate([w_normalized, r_normalized, M_normalized, alpha_normalized])
    
    def sample(self, state: np.ndarray):
        state = torch.as_tensor(state, dtype=torch.float32)
        action = self.diffusionQ.sample(state).cpu().detach().numpy()
        return self._from_diffusionQ_action_to_env_action(action)

    def train(self, data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
              batch_size: int = 1024,
              epochs: int = 10,
              steps_per_epoch: int = 1000):
        
        (S, A, R, S_next) = data
        # Build loader
        dataset = TensorDataset(
            torch.as_tensor(S), torch.as_tensor(A),
            torch.as_tensor(R), torch.as_tensor(S_next), 
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        it = iter(loader)
        for ep in tqdm(range(1, epochs + 1), desc="Training epochs"):
            for _ in range(steps_per_epoch):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                Ld, Lq, loss_critic = self.diffusionQ.update(batch)
            print(f"Epoch {ep:4d}/{epochs}  Ld={Ld:.4f}  Lq={Lq:.4f}  loss_critic={loss_critic:.4f}")

        info = {
            'Ld': Ld,
            'Lq': Lq,
            'loss_critic': loss_critic
        }
        model_state_dict = self.diffusionQ.state_dict()
        return model_state_dict, info



