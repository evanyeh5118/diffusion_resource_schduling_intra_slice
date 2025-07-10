# dpm_solver_torch.py
import math, torch
from torch import Tensor
from typing import Tuple, List, Dict

from .DiffusionPolicy import DiffusionSchedule
# -----------------------------------------------------------------------------#
# 1.  Noise schedule helper – adapts your DiffusionSchedule to continuous time #
# -----------------------------------------------------------------------------#

class VPSchedule:
    """
    Wraps the discrete β-schedule (β₁…β_N) you already computed in DiffusionSchedule
    and turns it into continuous-time primitives needed by DPM-Solver:
    -  ᾱ(t)  (cum-prod of 1-β)         -> marginal log-mean-coeff
    -  σ(t)  (sqrt(1-ᾱ(t))             -> marginal std
    -  λ(t)  = log α(t) – log σ(t)     -> half-logSNR  (monotonic ↔ time)
    """
    def __init__(self, ds: "DiffusionSchedule"):
        # store tensors as 1-D column for torch.take_along_dim
        self.t_array  = torch.linspace(0., 1., ds.N, device=ds.beta.device)
        self.log_a    = ds.alpha_bar.log().mul(0.5)          # log α_k
        self.total_N  = ds.N
        self.T, self.eps = 1., 1. / ds.N                     # [eps, 1]

    # discrete → continuous interpolation (piece-wise linear in λ-space)
    def _interp(self, t: Tensor, y: Tensor) -> Tensor:
        # clamp so that eps ≤ t ≤ 1
        t = t.clamp_(self.eps, self.T)
        idx = torch.floor(t * self.total_N).long() - 1
        idx = idx.clamp(0, self.total_N - 2)
        t0 = self.t_array[idx]; t1 = self.t_array[idx + 1]
        w  = (t - t0) / (t1 - t0)
        return (1 - w) * y[idx] + w * y[idx + 1]

    # α(t), σ(t), λ(t)
    def log_mean_coeff(self, t: Tensor) -> Tensor:
        return self._interp(t, self.log_a)

    def alpha(self, t: Tensor) -> Tensor:
        return self.log_mean_coeff(t).exp()

    def std(self, t: Tensor) -> Tensor:
        return torch.sqrt(1 - torch.exp(2 * self.log_mean_coeff(t)))

    def lambda_(self, t: Tensor) -> Tensor:
        loga = self.log_mean_coeff(t)
        return loga - torch.log( self.std(t) )

    # continuous λ → time  (monotone ⇒ Newton is OK for small N)
    def inv_lambda(self, lam: Tensor, iters: int = 4) -> Tensor:
        # init with linear map
        t = (lam - self.lambda_(torch.tensor(self.eps, device=lam.device))) / \
            (self.lambda_(torch.tensor(self.T, device=lam.device)) -
             self.lambda_(torch.tensor(self.eps, device=lam.device)))
        t = t.clamp(self.eps, self.T)
        for _ in range(iters):          # Newton iterate λ(t)=lam
            f = self.lambda_(t) - lam
            dldt = ( self.lambda_(t + 1e-4) - self.lambda_(t - 1e-4) ) / 2e-4
            t = (t - f / dldt).clamp(self.eps, self.T)
        return t
# -----------------------------------------------------------------------------#
# 2.  Single-step DPM-Solver (orders 1–3) for VP SDE                           #
# -----------------------------------------------------------------------------#

class DpmSolverVP:
    def __init__(self, model_eps, schedule: VPSchedule, predict_x0=False):
        """
        model_eps : callable(x,t)  – your εθ noise-predictor (expects t as int or float Tensor)
        predict_x0: if True, model returns x₀ directly (cf. DPM-Solver++)
        """
        self.m = model_eps
        self.ns = schedule
        self.predict_x0 = predict_x0

    # helpers ------------------------------------------------------------------
    def _phi(self, h: Tensor, k: int):
        """φ_k(h)   – expo helpers used in paper (k∈{1,2,3})"""
        if k == 1:
            return torch.expm1(h)              # eʰ-1
        if k == 2:
            return self._phi(h, 1) / h - 1
        if k == 3:
            return self._phi(h, 2) / h - 0.5
        raise ValueError

    # order-1 (DDIM) -----------------------------------------------------------
    def _update_1(self, x: Tensor, s: Tensor, t: Tensor,
                  eps_s: Tensor = None) -> Tuple[Tensor, Tensor]:
        ns = self.ns
        h  = ns.lambda_(t) - ns.lambda_(s)
        if eps_s is None:
            eps_s = self.m(x, s)
        if self.predict_x0:
            alpha_t = ns.alpha(t)
            phi1 = torch.expm1(-h)
            x_t = (ns.std(t)/ns.std(s)).unsqueeze(-1) * x - \
                  (alpha_t * phi1).unsqueeze(-1) * eps_s
        else:
            phi1 = self._phi(h, 1)
            x_t = torch.exp(ns.log_mean_coeff(t) - ns.log_mean_coeff(s)).unsqueeze(-1)*x - \
                  (ns.std(t)*phi1).unsqueeze(-1)*eps_s
        return x_t, eps_s

    # order-2 (singlestep) -----------------------------------------------------
    def _update_2(self, x: Tensor, s: Tensor, t: Tensor,
                  r1: float = 0.5) -> Tensor:
        ns = self.ns
        h   = ns.lambda_(t) - ns.lambda_(s)
        s1  = ns.inv_lambda(ns.lambda_(s) + r1 * h)
        x_s1, eps_s = self._update_1(x, s, s1)
        eps_s1 = self.m(x_s1, s1)

        if self.predict_x0:
            alpha_t = ns.alpha(t)
            phi1 = torch.expm1(-h)
            x_t = (ns.std(t)/ns.std(s)).unsqueeze(-1) * x - \
                  alpha_t.unsqueeze(-1)*(
                     phi1 * eps_s + 0.5/r1*phi1 * (eps_s1 - eps_s) )
        else:
            phi1 = self._phi(h,1); phi2 = self._phi(h,2)
            x_t = torch.exp(ns.log_mean_coeff(t)-ns.log_mean_coeff(s)).unsqueeze(-1)*x - \
                  ns.std(t).unsqueeze(-1)*(
                       phi1*eps_s + (phi2/r1)* (eps_s1 - eps_s) )
        return x_t

    # order-3 (singlestep) -----------------------------------------------------
    def _update_3(self, x: Tensor, s: Tensor, t: Tensor,
                  r1: float = 1/3, r2: float = 2/3) -> Tensor:
        ns = self.ns
        h   = ns.lambda_(t) - ns.lambda_(s)

        s1 = ns.inv_lambda(ns.lambda_(s) + r1 * h)
        x_s1, eps_s = self._update_1(x, s, s1)
        eps_s1 = self.m(x_s1, s1)

        s2 = ns.inv_lambda(ns.lambda_(s) + r2 * h)
        # second sub-step (order-2) to get x_s2
        phi11 = torch.expm1(-r2*h) if self.predict_x0 else self._phi(r2*h,1)
        if self.predict_x0:
            x_s2 = (ns.std(s2)/ns.std(s)).unsqueeze(-1)*x - \
                   (ns.alpha(s2)*phi11).unsqueeze(-1)*eps_s
        else:
            x_s2 = torch.exp(ns.log_mean_coeff(s2)-ns.log_mean_coeff(s)).unsqueeze(-1)*x - \
                   (ns.std(s2)*phi11).unsqueeze(-1)*eps_s
        eps_s2 = self.m(x_s2, s2)

        # finite-difference coefficients
        d1  = ( (eps_s1 - eps_s)/(r1*h).unsqueeze(-1) )
        d2  = ( (eps_s2 - eps_s)/(r2*h).unsqueeze(-1) - d1 ) / (r2 - r1)

        if self.predict_x0:
            alpha_t = ns.alpha(t)
            phi1 = torch.expm1(-h);   phi2 = self._phi(-h,2); phi3 = self._phi(-h,3)
            x_t = (ns.std(t)/ns.std(s)).unsqueeze(-1)*x - \
                  alpha_t.unsqueeze(-1)*( phi1*eps_s + phi2*h.unsqueeze(-1)*d1
                                        + 0.5*phi3*(h**2).unsqueeze(-1)*d2 )
        else:
            phi1 = self._phi(h,1); phi2=self._phi(h,2); phi3=self._phi(h,3)
            x_t = torch.exp(ns.log_mean_coeff(t)-ns.log_mean_coeff(s)).unsqueeze(-1)*x - \
                  ns.std(t).unsqueeze(-1)*( phi1*eps_s + phi2*h.unsqueeze(-1)*d1
                                           + 0.5*phi3*(h**2).unsqueeze(-1)*d2 )
        return x_t

    # public sampler -----------------------------------------------------------
    @torch.no_grad()
    def sample(self, s: Tensor, n_steps: int = 15, order: int = 3) -> Tensor:
        """
        s        : (B, state_dim)  conditioning states
        n_steps  : NFE budget (15 is default in EDP & Kang-Ma)
        order    : 1|2|3
        returns  : one action per row  (same shape as self.m’s first output)
        """
        B = s.size(0)
        device = s.device
        x = torch.randn(B, self.m.output_dim, device=device)

        # time grid (uniform in logSNR = λ) – works best for VP   :contentReference[oaicite:0]{index=0}
        t = torch.linspace(self.ns.T, self.ns.eps, n_steps + 1, device=device)
        for i in range(n_steps):
            s_time = t[i].expand(B)
            t_time = t[i + 1].expand(B)
            if order == 1:
                x, _ = self._update_1(x, s_time, t_time)
            elif order == 2 or (n_steps - i) == 1:
                x = self._update_2(x, s_time, t_time)
            else:
                x = self._update_3(x, s_time, t_time)
        # final prediction x₀
        if self.predict_x0:
            return x
        eps0 = self.m(x, torch.full((B,), self.ns.eps, device=device))
        alpha0 = self.ns.alpha(torch.tensor(self.eps, device=device))
        return (x - self.ns.std(torch.tensor(self.eps, device=device)).unsqueeze(-1)*eps0) / alpha0

'''
# -----------------------------------------------------------------------------#
# 3.  Plug-in to your EfficientDiffusionPolicy                                 #
# -----------------------------------------------------------------------------#

def build_solver(policy: EfficientDiffusionPolicy,
                 steps: int = 15,
                 order: int = 3,
                 predict_x0: bool = False):
    sched = VPSchedule(policy.schedule)           # convert discrete β’s
    model_eps = lambda a_t, t: policy(a_t, s=None, t=(t * policy.schedule.N).long())
    solver = DpmSolverVP(model_eps, sched, predict_x0=predict_x0)
    return solver
'''