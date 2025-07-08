import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .DiffusionPolicy import DiffusionSchedule, DiffusionPolicy
from .DiffusionPolicyEfficient import EfficientDiffusionPolicy
from .DoubleQ import QNetwork, EMATarget

class DiffusionQLearner(nn.Module):
    """Holds two Q-networks and their EMA targets, with update logic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.1,
        lr: float = 5e-2,
        eta: float = 1e-6,
        device: torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        # Diffusion policy
        self.sched = DiffusionSchedule().to(self.device)
        #self.diffusion_policy = DiffusionPolicy(state_dim, action_dim, self.sched).to(self.device)
        self.diffusion_policy = EfficientDiffusionPolicy(state_dim, action_dim, self.sched).to(self.device)
        # Critics
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        # EMA targets
        self.diffusion_policy_target = EMATarget(self.diffusion_policy, tau).to(self.device)
        self.q1_target = EMATarget(self.q1, tau).to(self.device)
        self.q2_target = EMATarget(self.q2, tau).to(self.device)
        # Optimizer for both critics
        self.optimizer_critic = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.optimizer_policy = torch.optim.Adam(
            list(self.diffusion_policy.parameters()), lr=lr
        )
        # Hyperparams
        self.gamma = gamma
        self.eta = eta

    def sample(self, s: torch.Tensor) -> torch.Tensor:
        return self.diffusion_policy.sample(s)

    def _update_critic(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        s, a, r, s_next = batch
        # Move to correct device
        s, a, s_next = s.float().to(self.device), a.float().to(self.device), s_next.float().to(self.device)
        r = r.float().to(self.device).squeeze(-1)
        # Compute target Q-value via EMA networks 
        with torch.no_grad():
            a_next = self.diffusion_policy_target.target.sample(s_next)     
            q1_next = self.q1_target(s_next, a_next)
            q2_next = self.q2_target(s_next, a_next)
            q_target = r + self.gamma * torch.min(q1_next, q2_next)
         # Current Q estimates
        q1_pred, q2_pred = self.q1(s, a), self.q2(s, a)
        loss = F.mse_loss(q1_pred, q_target) +  F.mse_loss(q2_pred, q_target)
        # Optimize critics
        self.optimizer_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0)
        self.optimizer_critic.step()

        return loss.item()
        
    def _update_policy(self, 
                       batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        s, a = batch[0], batch[1]
        s, a = s.float().to(self.device), a.float().to(self.device)
        # ==================================================
        a_est = self.diffusion_policy.approximate_action(s, a)
        q = self.q1(s, a_est)
        Lq = -q.mean()
        Ld = self.diffusion_policy.diffusion_loss(s, a)
        loss = Ld + self.eta * Lq
        # ==================================================
        self.optimizer_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion_policy.parameters(), 5.0)
        self.optimizer_policy.step()

        return Ld.cpu().detach().numpy(), Lq.cpu().detach().numpy()
        
    def _update_policy_online(self, 
                              batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        s = batch[0].float().to(self.device)
        a_est = self.diffusion_policy.sample(s)
        q = self.q1(s, a_est)
        loss = -self.eta*q.mean()
        
        self.optimizer_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion_policy.parameters(), 10.0)
        self.optimizer_policy.step()
        return loss.item()

    def update(self, 
               batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
               clonePolicy: bool = True) -> float:
        loss_critic = self._update_critic(batch)
        if clonePolicy:
            Ld, Lq = self._update_policy(batch)
        else:
            Ld, Lq = 0.0, self._update_policy_online(batch)

        # Soft-update EMA targets
        self.diffusion_policy_target.soft_update()
        self.q1_target.soft_update()
        self.q2_target.soft_update()

        return Ld, Lq, loss_critic

    
