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
        hidden_sizes: tuple[int, ...] = (8, 8, 8, 8, 4, 4),
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
        hidden_sizes: tuple[int, ...] = (8, 8, 8),
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



