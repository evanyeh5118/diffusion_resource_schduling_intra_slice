# diffusion_bc.py
"""Behavior-cloning-only *Diffusion Policy* (Section 3.1 of Wang et al., 2023).

This is a *minimal* reference implementation that:
  • defines a conditional denoising-diffusion model π_θ(a|s)
  • learns it with the pure reconstruction/denoising loss (no Q-learning)
  • demonstrates training on a toy 2-D bandit dataset and sampling for eval.

Author : ChatGPT (OpenAI o3)
Date    : 2025-07-03
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Utility layers / helpers
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int, *, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal time-step embedding (same as DDPM)."""
    half = dim // 2
    freq = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freq[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, dim % 2))  # zero-pad if odd


def mlp(inp: int, out: int, hidden: Sequence[int] = (256, 256, 256)):
    mods: list[nn.Module] = []
    prev = inp
    for h in hidden:
        mods += [nn.Linear(prev, h), nn.Mish()]
        prev = h
    mods.append(nn.Linear(prev, out))
    return nn.Sequential(*mods)

# ---------------------------------------------------------------------------
# Diffusion schedule (β₁…β_N) and helpers
# ---------------------------------------------------------------------------

@dataclass
class DiffusionSchedule:
    N: int = 30              # number of noise steps
    beta_min: float = 0.1
    beta_max: float = 10.0

    def __post_init__(self):
        i = torch.arange(1, self.N + 1)
        self.beta = 1.0 - torch.exp(
            -self.beta_min / self.N
            - 0.5 * (self.beta_max - self.beta_min) * (2 * i - 1) / self.N**2
        )
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device: torch.device) -> DiffusionSchedule:
        for name in ("beta", "alpha", "alpha_bar"):
            setattr(self, name, getattr(self, name).to(device))
        return self


# ---------------------------------------------------------------------------
# Conditional Diffusion Policy π_θ(a|s)
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """Implements Eq. (1) + the denoising (behavior-cloning) loss from Eq. (6)."""

    def __init__(self, state_dim: int, action_dim: int, schedule: DiffusionSchedule):
        super().__init__()
        self.s_dim, self.a_dim = state_dim, action_dim
        self.schedule = schedule
        self.eps_net = mlp(state_dim + action_dim + 128, action_dim)

    def forward(self, a_t: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        emb = timestep_embedding(t, 128)
        return self.eps_net(torch.cat([a_t, s, emb], dim=-1))
        #return self.eps_net(torch.cat([a_t, s, emb], dim=0))

    def diffusion_loss(self, s: torch.Tensor, a_0: torch.Tensor) -> torch.Tensor:
        """Compute L_d: MSE between true noise and model prediction."""
        B = a_0.size(0)
        t = torch.randint(1, self.schedule.N + 1, (B,), device=a_0.device)
        alpha_bar_t = self.schedule.alpha_bar[t - 1]
        noise = torch.randn_like(a_0)
        a_t = (torch.sqrt(alpha_bar_t)[:, None] * a_0
               + torch.sqrt(1 - alpha_bar_t)[:, None] * noise)
        eps_pred = self(a_t, s, t)
        return F.mse_loss(eps_pred, noise)

    def sample(self, s: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion to sample actions given states."""
        B, dev = s.size(0), s.device
        a = torch.randn(B, self.a_dim, device=dev)
        for i in reversed(range(1, self.schedule.N + 1)):
            b_i = self.schedule.beta[i - 1]
            a_i = self.schedule.alpha[i - 1]
            ab_i = self.schedule.alpha_bar[i - 1]
            eps = self(a, s, torch.full((B,), i, device=dev, dtype=torch.long))
            mean = (a - b_i / torch.sqrt(1 - ab_i) * eps) / torch.sqrt(a_i)
            noise = torch.randn_like(a) if i > 1 else 0.0
            a = mean + torch.sqrt(b_i) * noise
        return a
    


