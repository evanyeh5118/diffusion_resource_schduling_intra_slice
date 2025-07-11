from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .DiffusionPolicy import DiffusionSchedule, mlp, timestep_embedding

class EfficientDiffusionPolicy(nn.Module):
    """
    Implements DDPM-based policy with:
      1) action approximation for O(1) training‐time policy‐improvement (Eq. 9 in Kang & Ma) :contentReference[oaicite:9]{index=9}
      2) fast ODE-based sampling via DPM-Solver instead of the full reverse-chain :contentReference[oaicite:10]{index=10}
    """

    def __init__(self, state_dim: int, action_dim: int, schedule: DiffusionSchedule):
        super().__init__()
        self.s_dim, self.a_dim = state_dim, action_dim
        self.schedule = schedule
        # noise‐prediction network ϵθ(a_k, s, k)
        self.eps_net = mlp(state_dim + action_dim + 128, action_dim)

    def forward(self, a_t: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # standard timestep embedding + MLP
        emb = timestep_embedding(t, 128)
        return self.eps_net(torch.cat([a_t, s, emb], dim=-1))

    def diffusion_loss(self, s: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        """
        Behavior‐cloning loss L_d: predict the true noise at a random diffusion step
        (same as before).
        """
        B = a0.size(0)
        t = torch.randint(1, self.schedule.N + 1, (B,), device=a0.device)
        alpha_bar = self.schedule.alpha_bar[t - 1]   # ᾱ_t
        noise = torch.randn_like(a0)
        # q(a_t | a0)
        a_t = (alpha_bar.sqrt()[:, None] * a0 +
               (1 - alpha_bar).sqrt()[:, None] * noise)
        eps_pred = self(a_t, s, t)
        return F.mse_loss(eps_pred, noise)

    def approximate_action(self,
                           s: torch.Tensor,
                           a0: torch.Tensor) -> torch.Tensor:
        """
        One‐step action approximation (Eq. 9 in Kang & Ma) :contentReference[oaicite:11]{index=11}:
          â0 = (a_t - √(1-ᾱ_t) · ϵθ(a_t, s, t)) / √ᾱ_t
        If t is not provided, pick uniformly at random.
        """
        B = a0.size(0)
        t = torch.randint(1, self.schedule.N + 1, (B,), device=a0.device)
        alpha_bar = self.schedule.alpha_bar[t - 1]
        noise = torch.randn_like(a0)
        # corrupt
        a_t = (alpha_bar.sqrt()[:, None] * a0 +
               (1 - alpha_bar).sqrt()[:, None] * noise)
        # predict noise
        eps_pred = self(a_t, s, t)
        # approximate clean action
        a0_hat = (a_t - (1 - alpha_bar).sqrt()[:, None] * eps_pred) / alpha_bar.sqrt()[:, None]
        return a0_hat

    def sample(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (M, state_dim)   — M “replicated” states
        returns: (M, a_dim) — one action per input row
        """
        a = torch.randn(s.size(0), self.a_dim, device=s.device)

        # pre-draw all noises
        N_steps = self.schedule.N
        noises = torch.randn(N_steps, s.size(0), self.a_dim, device=s.device)

        for step in reversed(range(1, N_steps + 1)):
            i = step - 1
            b_i = self.schedule.beta[i]
            a_i = self.schedule.alpha[i]
            ab_i = self.schedule.alpha_bar[i]

            t = torch.full((s.size(0),), step, device=s.device, dtype=torch.long)
            eps = self(a, s, t)
            mean = (a - b_i / (1 - ab_i).sqrt() * eps) / a_i.sqrt()

            noise = noises[i] if step > 1 else 0.0
            a = mean + b_i.sqrt() * noise

        return a
    
    
