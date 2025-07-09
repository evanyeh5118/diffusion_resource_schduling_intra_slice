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
        N_action_candidates: int = 20,
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
        self.N_action_candidates = N_action_candidates
        self.a_dim = action_dim

    def sample(self, s: torch.Tensor, N: int = 10) -> torch.Tensor:
        B = s.size(0)
        # 1) replicate each state N times → shape (B*N, state_dim)
        s_rep = s.unsqueeze(1).expand(B, N, s.size(-1)).reshape(-1, s.size(-1))
        a_N_flat = self.diffusion_policy.sample(s_rep)
        a_N = a_N_flat.view(B, N, self.a_dim)
        a_N_mean = a_N.mean(dim=1)
        return a_N_mean
    
    def update(self, 
               batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
               clonePolicy: bool = True,
               updateCriticOnly: bool = False) -> float:
        loss_critic = self._update_critic(batch)
        Ld, Lq = 0.0, 0.0
        if updateCriticOnly is False:
            if clonePolicy:
                Ld, Lq = self._update_policy(batch)
            else:
                Ld, Lq = self._update_policy_Ql(batch)
            self.diffusion_policy_target.soft_update()
            
        # Soft-update EMA targets
        self.q1_target.soft_update()
        self.q2_target.soft_update()

        return Ld, Lq, loss_critic

    def _update_critic(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        s, a, r, s_next = batch
        # Move to correct device
        s, a, s_next = s.float().to(self.device), a.float().to(self.device), s_next.float().to(self.device)
        r = r.float().to(self.device).squeeze(-1)
        # Compute target Q-value via EMA networks 
        with torch.no_grad():
            a_next = self.diffusion_policy_target.target.sample(s_next)     
            noise  = (0.1*torch.randn_like(a_next)).clamp(-0.5,0.5)
            a_next = (a_next + noise).clamp(-1, 1)
            q1_next = self.q1_target(s_next, a_next)
            q2_next = self.q2_target(s_next, a_next)
            q_target = r + self.gamma * torch.min(q1_next, q2_next)
         # Current Q estimates
        q1_pred, q2_pred = self.q1(s, a), self.q2(s, a)
        loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        # Optimize critics
        self.optimizer_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 5.0)
        self.optimizer_critic.step()

        return loss.item()
    
    def _greedy_action_approximation(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B, D, A, N = s.shape[0], s.shape[1], a.shape[1], self.N_action_candidates
        s_rep = s.unsqueeze(1).expand(-1, N, -1).reshape(B * N, D)
        a_rep = a.unsqueeze(1).expand(-1, N, -1).reshape(B * N, A)
        a_cand = self.diffusion_policy.approximate_action(s_rep, a_rep).clamp(-1, 1)  # (B*N, action_dim)
        q_cand = self.q1(s_rep, a_cand).view(B, N)                # (B, N)
        best_idx = torch.argmax(q_cand, dim=1)                     # (B,)
        a_best = a_cand.view(B, N, -1)[torch.arange(B), best_idx]  # (B, action_dim)
        return a_best
        
    def _update_policy(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[float, float]:
        """
        Offline policy update with N-candidate re-ranking:
        - Ld: diffusion (BC) loss on dataset actions
        - Lq: Q-improvement loss using best of N samples
        Returns (Ld, Lq).
        """
        # Unpack and move to device
        s, a = batch[0].float().to(self.device), batch[1].float().to(self.device)
        # 1) Behavior cloning loss
        Ld = self.diffusion_policy.diffusion_loss(s, a)
        # 2) Greedy action approximation
        a_best = self._greedy_action_approximation(s, a)
        # 3) Q-improvement loss
        q_best = self.q1(s, a_best)                                # (B,)
        Lq = -q_best.mean()                                        # scalar
        # 4) Combined loss
        with torch.no_grad():
            norm_term = self.q1(s, a).mean()
        loss = Ld + (self.eta / (norm_term + 1e-8)) * Lq
        # 5) Backprop & clip
        self.optimizer_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion_policy.parameters(), 5.0)
        self.optimizer_policy.step()

        return Ld.item(), Lq.item()

    def _greedy_sample(self, s: torch.Tensor) -> torch.Tensor:
        B, D, N = s.shape[0], s.shape[1], self.N_action_candidates
        s_rep = s.unsqueeze(1).expand(-1, N, -1).reshape(B * N, D)
        a_cand = self.diffusion_policy.sample(s_rep)
        q_cand = self.q1(s_rep, a_cand).view(B, N)                # (B, N)
        best_idx = torch.argmax(q_cand, dim=1)     
        a_best = a_cand.view(B, N, -1)[torch.arange(B), best_idx]  # (B, action_dim)
        return a_best

    def _update_policy_Ql(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[float, float]:
        """
        Offline policy update with N-candidate re-ranking:
        - Ld: diffusion (BC) loss on dataset actions
        - Lq: Q-improvement loss using best of N samples
        Returns (Ld, Lq).
        """
        # Unpack and move to device
        s, a = batch[0].float().to(self.device), batch[1].float().to(self.device)
        # 2) Greedy action approximation
        a_best = self._greedy_action_approximation(s, a)
        # 3) Q-improvement loss
        q_best = self.q1(s, a_best)                                # (B,)
        Lq = -self.eta * q_best.mean()                                        # scalar
        # 4) Backprop & clip
        self.optimizer_policy.zero_grad()
        Lq.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion_policy.parameters(), 5.0)
        self.optimizer_policy.step()

        return 0.0, Lq.item()

    
