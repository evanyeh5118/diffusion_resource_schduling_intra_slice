import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .DiffusionPolicy import DiffusionSchedule, DiffusionPolicy
from .DoubleQ import QNetwork, EMATarget

class DiffusionQLearner(nn.Module):
    """Holds two Q-networks and their EMA targets, with update logic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.001,
        lr: float = 3e-3,
        eta: float = 1e-6,
        dropout_p: float = 0.2,
        device: torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        # Diffusion policy
        self.sched = DiffusionSchedule().to(self.device)
        self.diffusion_policy = DiffusionPolicy(state_dim, action_dim, self.sched).to(self.device)
        # Critics
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        # EMA targets
        self.diffusion_policy_target = EMATarget(self.diffusion_policy, tau).to(self.device)
        self.q1_target = EMATarget(self.q1, tau).to(self.device)
        self.q2_target = EMATarget(self.q2, tau).to(self.device)
        #self.diffusion_policy_target = EMATarget(self.diffusion_policy, tau).to(self.device)

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
        self.dropout_p = dropout_p

    def sample(self, s: torch.Tensor) -> torch.Tensor:
        return self.diffusion_policy.sample(s)

    def update_critic(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        s, a, r, s_next = batch
        # Move to correct device
        s, a, s_next = s.float().to(self.device), a.float().to(self.device), s_next.float().to(self.device)
        r = r.float().to(self.device).squeeze(-1)

        a_next = self.diffusion_policy_target.target.sample(s_next)       
        # Compute target Q-value via EMA networks (clipped double-Q)
        with torch.no_grad():
            q1_next = self.q1_target(s_next, a_next)
            q2_next = self.q2_target(s_next, a_next)
            q_target = r + self.gamma * torch.min(q1_next, q2_next)
            # ========= clip q_target to [-1, 1] =========
            q_target = torch.clamp(q_target, min=-10, max=10)

         # Current Q estimates
        q1_pred, q2_pred = self.q1(s, a), self.q2(s, a)
        loss = F.mse_loss(q1_pred, q_target) +  F.mse_loss(q2_pred, q_target)

        # Optimize critics
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_critic.step()

        return loss.item()
        
    def update_policy(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        s, a = batch[0], batch[1]
        s, a = s.float().to(self.device), a.float().to(self.device)
        # ==================================================
        # ========= Compute Ld =============================
        # ==================================================
        Ld = self.diffusion_policy.diffusion_loss(s, a)
        # ==================================================
        # ========= Compute Lq =============================
        # ==================================================
        # ==== Dropout a_est with probability dropout_p ====
        if torch.rand(1).item() < self.dropout_p:
            a_est = self.diffusion_policy.sample(s)
        else:
            a_est = a
        
        q = self.q1(s, a_est)
        with torch.no_grad():
            scale = torch.mean(torch.abs(self.q1(s, a))) + 1e-6
        Lq = -self.eta / scale * q.mean()
        loss = Ld + Lq

        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        return Ld.detach(), Lq.detach()
        
        
    def update(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        loss_critic = self.update_critic(batch)
        Ld, Lq = self.update_policy(batch)

        # Soft-update EMA targets
        self.diffusion_policy_target.soft_update()
        self.q1_target.soft_update()
        self.q2_target.soft_update()

        return Ld, Lq, loss_critic