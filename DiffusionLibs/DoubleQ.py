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
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (16, 16, 16),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim + action_dim
        for h in hidden_sizes:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for state-action pairs."""
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)

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
        self.optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

    def update(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> float:
        """
        Perform a double-Q update step.

        Batch format: (s, a, r, s_next, a_next, done)
          - s, s_next: shape (B, state_dim)
          - a, a_next: shape (B, action_dim)
          - r, done: shape (B,) or (B,1)
        Returns:
          scalar total MSE loss over both critics
        """
        s, a, r, s_next, a_next, done = batch
        # Move to correct device
        s, a = s.to(self.device), a.to(self.device)
        r = r.to(self.device).squeeze(-1)
        s_next, a_next = s_next.to(self.device), a_next.to(self.device)
        done = done.to(self.device).squeeze(-1)

        # Compute target Q-value via EMA networks (clipped double-Q)
        with torch.no_grad():
            q1_next = self.q1_target(s_next, a_next)
            q2_next = self.q2_target(s_next, a_next)
            q_target = r + self.gamma * (1 - done) * torch.min(q1_next, q2_next)

        # Current Q estimates
        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        # MSE losses
        loss_q1 = F.mse_loss(q1_pred, q_target)
        loss_q2 = F.mse_loss(q2_pred, q_target)
        loss = loss_q1 + loss_q2

        # Optimize critics
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft-update EMA targets
        self.q1_target.soft_update()
        self.q2_target.soft_update()

        return loss.item()


# ---------------------------------------------------------------------------
# Offline training helper
# ---------------------------------------------------------------------------

def offline_dq(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    state_dim: int,
    action_dim: int,
    batch_size: int = 256,
    epochs: int = 1000,
    steps_per_epoch: int = 1000,
    device: str | torch.device = "cpu",
) -> DoubleQLearner:
    """
    Train DoubleQLearner on offline data.

    Args:
        data: (states, actions, rewards, next_states, dones)
        state_dim: dimension of state
        action_dim: dimension of action
    Returns:
        Trained DoubleQLearner
    """
    dev = torch.device(device)
    states, actions, rewards, next_states, a_next, dones = data
    learner = DoubleQLearner(state_dim, action_dim, device=dev)

    # Build loader
    dataset = TensorDataset(
        torch.as_tensor(states), torch.as_tensor(actions),
        torch.as_tensor(rewards), torch.as_tensor(next_states), 
        torch.as_tensor(a_next), torch.as_tensor(dones)
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
            loss = learner.update(batch)
        if ep % 50 == 0:
            print(f"Epoch {ep:4d}  Loss={loss:.4f}")
    return learner