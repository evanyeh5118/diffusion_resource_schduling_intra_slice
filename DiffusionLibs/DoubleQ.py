# double_q.py
"""
Double Q-Learning with soft-updated target networks using EMA.

This minimal module defines:
  • QNetwork: a simple MLP critic Q(s,a).
  • EMATarget: wraps a source network, maintaining a frozen target copy and
    performing polyak updates via soft_update().
  • DoubleQLearner: holds two critics (q1, q2) and their EMA targets,
    with an `update()` method to perform a double-Q critic update on
    batches of transitions.

Usage:
  learner = DoubleQLearner(state_dim, action_dim)
  for batch in dataloader:
      loss = learner.update(batch)

Author: ChatGPT (OpenAI o4-mini)
Date:   2025-07-03
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
# ---------------------------------------------------------------------------
# Q-network
# --------------------------------------------------------------------------()
class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64, 64, 64, 64),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim + action_dim
        for h in hidden_sizes:
            layers += [nn.Linear(input_dim, h), nn.Mish()]
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for state-action pairs."""
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)
    
class VNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64, 64, 64, 64),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim 
        for h in hidden_sizes:
            layers += [nn.Linear(input_dim, h), nn.Mish()]
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Compute V-values for state-action pairs."""
        return self.net(s).squeeze(-1)


# ---------------------------------------------------------------------------
# EMA target wrapper
# ---------------------------------------------------------------------------
class EMATarget(nn.Module):
    """Wraps a network to maintain a frozen EMA-smoothed copy."""

    def __init__(self, source: nn.Module, tau: float = 0.005):
        """
        Args:
            source: the network to track
            tau: smoothing coefficient (0 < tau ≤ 1)
        """
        super().__init__()
        self.source = source
        self.tau = tau
        # Deep-copy registers `target` as a sub-module
        self.target = copy.deepcopy(source)
        self.freeze()

    def freeze(self) -> None:
        """Disable gradients for the target network."""
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def soft_update(self) -> None:
        """Perform in-place polyak averaging of target towards source."""
        for tp, sp in zip(self.target.parameters(), self.source.parameters()):
            tp.data.mul_(1 - self.tau)
            tp.data.add_(self.tau * sp.data)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        # Delegate forward to the target network
        return self.target(*args, **kwargs)

# ---------------------------------------------------------------------------
# Double Q-Learning module
# ---------------------------------------------------------------------------
class DoubleQLearner(nn.Module):
    """Holds two Q-networks and their EMA targets, with update logic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        iql_tau: float = 0.1,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        # Critics
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        # EMA targets
        self.q1_target = EMATarget(self.q1, tau).to(self.device)
        self.q2_target = EMATarget(self.q2, tau).to(self.device)
        # Hyperparams
        self.gamma = gamma
        # Optimizer for both critics
        self.optimizer_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.scheduler_q = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_q, gamma=0.99)
        # V-network
        self.v_net = VNetwork(state_dim).to(self.device)
        self.optimizer_v = torch.optim.Adam(self.v_net.parameters(), lr=lr)
        self.scheduler_v = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_v, gamma=0.99)
        self.iql_tau = iql_tau


    def update(self, input_tuple) -> float:
        s, a, r, s_next, a_next = input_tuple
        with torch.no_grad():
            noise  = (0.1*torch.randn_like(a_next)).clamp(-0.5,0.5)
            a_next = (a_next + noise).clamp(-0.5, 0.5)
            q1_next = self.q1_target(s_next, a_next)
            q2_next = self.q2_target(s_next, a_next)
            q_target = r + self.gamma * torch.min(q1_next, q2_next)
         # Current Q estimates
        q1_pred, q2_pred = self.q1(s, a), self.q2(s, a)
        loss_q = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        # Optimize critics
        self.optimizer_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 5.0)
        self.optimizer_q.step()
        self.q1_target.soft_update()
        self.q2_target.soft_update()

        return loss.item()

    def update_iql(self, input_tuple):
        """One training step of IQL: 
           - update V via expectile regression
           - update Q via Bellman using V for next-state target
           - update policy via weighted regression on action-approximation
        """
        s, a, r, s_next = input_tuple
        # ---- 1) Update V network via expectile regression  ----
        with torch.no_grad():
            q1, q2 = self.q1_target(s, a), self.q2_target(s, a)
            q_min = torch.min(q1, q2).squeeze(-1)
        v = self.v_net(s)
        diff = q_min - v
        loss_v = self._expectile_loss(diff, self.iql_tau)
        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()
        self.scheduler_v.step()

        # ---- 2) Update Q networks ----
        with torch.no_grad():
            # target uses V(s') instead of Q
            v_next = self.v_net(s_next)
            q_target = (r + self.gamma * v_next).detach()
        q1_pred, q2_pred = self.q1(s, a), self.q2(s, a)
        loss_q = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        self.optimizer_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 100.0)
        self.optimizer_q.step()

        # ---- 3) Update target networks ----
        self.q1_target.soft_update()
        self.q2_target.soft_update()
        self.scheduler_q.step()

        return loss_q.item()

    def _expectile_loss(self, diff, tau):
        # diff = Q(s,a) - V(s)
        weight = torch.where(diff > 0, tau, 1 - tau)
        return (weight * (diff ** 2)).mean()
    
    def estimate_reward(self, s, a):
        q = torch.min(self.q1(s, a), self.q2(s, a))
        return q.mean().item()
    