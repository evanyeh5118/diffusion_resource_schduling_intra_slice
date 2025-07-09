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
                           a0: torch.Tensor,
                           t: torch.Tensor = None) -> torch.Tensor:
        """
        One‐step action approximation (Eq. 9 in Kang & Ma) :contentReference[oaicite:11]{index=11}:
          â0 = (a_t - √(1-ᾱ_t) · ϵθ(a_t, s, t)) / √ᾱ_t
        If t is not provided, pick uniformly at random.
        """
        B = a0.size(0)
        if t is None:
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

    def sample(self, s: torch.Tensor, num_actions: int = 10) -> torch.Tensor:
        """
        s:            (B, state_dim)
        num_actions:  how many actions to sample per state
        returns:      (B, a_dim)  ← averaged over num_actions
        """
        B = s.size(0)
        device = s.device

        # replicate each state num_actions times → (B * num_actions, state_dim)
        s_rep = s.unsqueeze(1).expand(B, num_actions, -1).reshape(-1, s.size(-1))

        # initial noisy actions for all samples → (B * num_actions, a_dim)
        a = torch.randn(B * num_actions, self.a_dim, device=device)

        # pre-draw all per-step noises: shape = (N_steps, B * num_actions, a_dim)
        N_steps = self.schedule.N
        all_noises = torch.randn(
            N_steps, B * num_actions, self.a_dim, device=device
        )

        # reverse diffusion
        for step in reversed(range(1, N_steps + 1)):
            i = step - 1
            b_i = self.schedule.beta[i]
            a_i = self.schedule.alpha[i]
            ab_i = self.schedule.alpha_bar[i]

            t = torch.full(
                (B * num_actions,),
                step,
                device=device,
                dtype=torch.long
            )
            eps = self(a, s_rep, t)
            mean = (a - b_i / (1 - ab_i).sqrt() * eps) / a_i.sqrt()

            noise = all_noises[i] if step > 1 else 0.0
            a = mean + b_i.sqrt() * noise

        # reshape to (B, num_actions, a_dim)
        a = a.view(B, num_actions, self.a_dim)
        # average over the num_actions dimension → (B, a_dim)
        return a.mean(dim=1)
    

    '''
    def sample(self, s: torch.Tensor) -> torch.Tensor:
        a = torch.randn(s.size(0), self.a_dim, device=s.device)
        for i in reversed(range(1, self.schedule.N + 1)):
            b_i = self.schedule.beta[i - 1]
            a_i = self.schedule.alpha[i - 1]
            ab_i = self.schedule.alpha_bar[i - 1]
            eps = self(a, s, torch.full((s.size(0),), i, device=s.device, dtype=torch.long))
            mean = (a - b_i / (1 - ab_i).sqrt() * eps) / a_i.sqrt()
            noise = torch.randn_like(a) if i > 1 else 0.0
            a = mean + b_i.sqrt() * noise
        return a
    '''