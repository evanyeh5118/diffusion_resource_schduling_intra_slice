import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .DiffusionPolicy import DiffusionSchedule, DiffusionPolicy
from .DiffusionPolicyEfficient import EfficientDiffusionPolicy
from .DoubleQ import DoubleQLearner, EMATarget

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
        iql_tau: float = 0.1,
        device: torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        # Diffusion policy
        self.sched = DiffusionSchedule().to(self.device)
        self.diffusion_policy = EfficientDiffusionPolicy(state_dim, action_dim, self.sched).to(self.device)
        self.diffusion_policy_target = EMATarget(self.diffusion_policy, tau).to(self.device)
        self.optimizer_policy = torch.optim.Adam(
            list(self.diffusion_policy.parameters()), lr=lr
        )
        # Double Q-learning
        self.double_q = DoubleQLearner(state_dim, action_dim, gamma, tau, lr, iql_tau, device).to(self.device)
        # Hyperparams
        self.gamma = gamma
        self.eta = eta
        self.N_action_candidates = N_action_candidates
        self.a_dim = action_dim
        self.iql_tau = iql_tau

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
               iql_flag: bool = False) -> float:
        s, a, r, s_next = batch
        s, a, s_next = s.float().to(self.device), a.float().to(self.device), s_next.float().to(self.device)
        r = r.float().to(self.device).squeeze(-1)
        a_next = self.diffusion_policy_target.target.sample(s_next)     
        if iql_flag == False:
            loss_critic = self.double_q.update((s, a, r, s_next, a_next))
            Ld, Lq = self._update_policy(s, a)
        else:
            loss_critic = self.double_q.update_iql((s, a, r, s_next))
            Ld, Lq = self._update_policy_iql(s, a)
        return Ld, Lq, loss_critic

    def _greedy_action_approximation(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B, D, A, N = s.shape[0], s.shape[1], a.shape[1], self.N_action_candidates
        s_rep = s.unsqueeze(1).expand(-1, N, -1).reshape(B * N, D)
        a_rep = a.unsqueeze(1).expand(-1, N, -1).reshape(B * N, A)
        a_cand = self.diffusion_policy.approximate_action(s_rep, a_rep).clamp(-1, 1)  # (B*N, action_dim)
        q_cand = self.double_q.q1(s_rep, a_cand).view(B, N)                # (B, N)
        best_idx = torch.argmax(q_cand, dim=1)                     # (B,)
        a_best = a_cand.view(B, N, -1)[torch.arange(B), best_idx]  # (B, action_dim)
        return a_best
        
    def _update_policy(self, s: torch.Tensor, a: torch.Tensor) -> tuple[float, float]:
        """
        Offline policy update with N-candidate re-ranking:
        - Ld: diffusion (BC) loss on dataset actions
        - Lq: Q-improvement loss using best of N samples
        Returns (Ld, Lq).
        """
        # 1) Behavior cloning loss
        Ld = self.diffusion_policy.diffusion_loss(s, a)
        # 2) Greedy action approximation
        a_best = self._greedy_action_approximation(s, a)
        # 3) Q-improvement loss
        q_best = self.double_q.q1(s, a_best)                                # (B,)
        Lq = -q_best.mean()                                        # scalar
        # 4) Combined loss
        with torch.no_grad():
            norm_term = self.double_q.q1(s, a).mean()
        loss = Ld + (self.eta / (norm_term + 1e-8)) * Lq
        # 5) Backprop & clip
        self.optimizer_policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion_policy.parameters(), 5.0)
        self.optimizer_policy.step()
        self.diffusion_policy_target.soft_update()

        return Ld.item(), Lq.item()

    def _update_policy_iql(self, s, a):
        # ---- Update policy via weighted regression ----
        # 1) behavior cloning term
        Ld = self.diffusion_policy.diffusion_loss(s, a)
        # 2) Greedy action approximation for policy-improvement
        a_hat = self._greedy_action_approximation(s, a)
        # weights w = exp[(Q(s,a) - V(s)) / tau]
        with torch.no_grad():
            q1_hat = self.double_q.q1(s, a_hat).squeeze(-1)
            q2_hat = self.double_q.q2(s, a_hat).squeeze(-1)
            q_hat = torch.min(q1_hat, q2_hat)
            v_s = self.double_q.v_net(s)
            w = torch.exp((q_hat - v_s) / self.iql_tau).clamp(max=100.0)  # avoid explosion

        # 3) weighted L2 regression:  E[w * ||a - a_hat||^2]
        Lq = (w * ((a_hat - a) ** 2).sum(dim=-1)).mean()
        # 4) Combined loss and backprop
        loss_pi = Ld + self.eta * Lq
        self.optimizer_policy.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.diffusion_policy.parameters(), 5.0)
        self.optimizer_policy.step()
        self.diffusion_policy_target.soft_update()

        return Ld.item(), Lq.item()