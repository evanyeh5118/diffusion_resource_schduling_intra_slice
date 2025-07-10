import torch
from torch import Tensor

class NoiseScheduleVP:
    """
    Variance-Preserving (VP) SDE noise schedule for diffusion models.
    Supports continuous-time scheduling with linear or cosine functions.
    """
    def __init__(self, schedule_type: str = 'linear'):
        assert schedule_type in ('linear', 'cosine'), "schedule_type must be 'linear' or 'cosine'"
        self.schedule_type = schedule_type

    def beta_t(self, t: Tensor) -> Tensor:
        """Instantaneous noise rate beta(t)."""
        if self.schedule_type == 'linear':
            # Linear schedule from beta_min to beta_max
            beta_min, beta_max = 0.1, 20.0
            return beta_min + t * (beta_max - beta_min)
        else:
            # Cosine schedule as in Nichol & Dhariwal
            s = 0.008
            return (
                torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
                / torch.cos(s / (1 + s) * torch.pi / 2) ** 2
            )

    def marginal_log_mean_coeff(self, t: Tensor) -> Tensor:
        """Compute log(alpha_bar_t) where alpha_bar_t = exp(-∫_0^t beta(s) ds/2)."""
        if self.schedule_type == 'linear':
            beta_min, beta_max = 0.1, 20.0
            # ∫_0^t beta(s) ds = beta_min * t + 0.5 * (beta_max - beta_min) * t^2
            integral = beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2
            return -0.5 * integral
        else:
            # For cosine schedule, approximate via closed form
            s = 0.008
            f = torch.cos((t + s) / (1 + s) * torch.pi / 2) / torch.cos(s / (1 + s) * torch.pi / 2)
            return torch.log(f)

    def marginal_alpha(self, t: Tensor) -> Tensor:
        """Alpha coefficient: sqrt(alpha_bar_t)."""
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t: Tensor) -> Tensor:
        """Standard deviation sigma_t = sqrt(1 - alpha_bar_t)."""
        alpha_bar = torch.exp(2 * self.marginal_log_mean_coeff(t))
        return torch.sqrt(1.0 - alpha_bar)

    def marginal_lambda(self, t: Tensor) -> Tensor:
        """Half-log signal-to-noise ratio: log(alpha_t / sigma_t)."""
        log_mean = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2 * log_mean))
        return log_mean - log_std

    def inverse_lambda(self, lam: Tensor) -> Tensor:
        """Inverse of lambda mapping: find t such that marginal_lambda(t) = lam via Newton's method."""
        # Initialize t from sigmoid of lam
        t = torch.sigmoid(-lam)  # heuristic init
        for _ in range(10):
            f = self.marginal_lambda(t) - lam
            # derivative df/dt = beta(t)/2 * (1 + exp(-2*lambda(t)))
            beta = self.beta_t(t)
            alpha = torch.exp(self.marginal_log_mean_coeff(t))
            sigma = torch.sqrt(1 - alpha**2)
            df_dt = beta * (1 / (2 * sigma**2) + 1 / 2)
            t = t - f / (df_dt + 1e-5)
            t = t.clamp(0.0, 1.0)
        return t


class DPM_Solver:
    """
    A PyTorch-compatible implementation of the DPM-Solver for diffusion models.
    Supports 1st, 2nd, and 3rd-order ODE solvers for fast sampling.
    """
    def __init__(
        self,
        model_fn,
        noise_schedule,
        predict_x0: bool = False,
        thresholding: bool = False,
        max_val: float = 1.0,
        device: torch.device = None,
    ):
        self.model_fn = model_fn
        self.noise_schedule = noise_schedule
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def sample(
        self,
        x_T: Tensor,
        timesteps: torch.Tensor,
        orders: list,
    ) -> Tensor:
        """
        Run DPM-Solver sampling from noise x_T through the given timesteps.
        Args:
            x_T: initial noise tensor at time T
            timesteps: 1D tensor of times [t_0, t_1, ..., t_N]
            orders: list of solver orders for each interval
        Returns:
            x_0: generated sample
        """
        x = x_T.to(self.device)
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i+1]
            order = orders[i]
            if order == 1:
                x = self.dpm_solver_first_update(x, t_cur, t_next)
            elif order == 2:
                x = self.dpm_solver_second_order_update(x, t_cur, t_next)
            elif order == 3:
                x = self.dpm_solver_third_order_update(x, t_cur, t_next)
            else:
                raise ValueError(f"Unsupported order: {order}")
            if self.thresholding:
                x = x.clamp(-self.max_val, self.max_val)
        return x

    def _alpha(self, t: Tensor) -> Tensor:
        return self.noise_schedule.marginal_alpha(t)

    def _sigma(self, t: Tensor) -> Tensor:
        return self.noise_schedule.marginal_std(t)

    def _lambda(self, t: Tensor) -> Tensor:
        return self.noise_schedule.marginal_lambda(t)

    def _model_pred(self, x: Tensor, t: Tensor) -> Tensor:
        """Wrap model_fn to always predict noise epsilon"""
        eps = self.model_fn(x, t)
        return eps

    def dpm_solver_first_update(self, x: Tensor, t: Tensor, s: Tensor) -> Tensor:
        """
        First-order DPM-Solver (equivalent to DDIM)
        """
        lambda_t = self._lambda(t)
        lambda_s = self._lambda(s)
        h = lambda_s - lambda_t
        alpha_t = self._alpha(t)
        alpha_s = self._alpha(s)
        sigma_t = self._sigma(t)
        # predict noise at t
        e_t = self._model_pred(x, t)
        # update
        x_next = (alpha_s / alpha_t) * x - torch.exp(-lambda_s) * (torch.exp(h) - 1) * e_t
        return x_next

    def singlestep_dpm_solver_second_update(self, x: Tensor, t: Tensor, s: Tensor) -> Tensor:
        """
        Second-order single-step DPM-Solver.
        """
        lambda_t = self._lambda(t)
        lambda_s = self._lambda(s)
        h = lambda_s - lambda_t
        r = 0.5  # midpoint
        s1 = torch.scalar_tensor(float(lambda_t + r * h)).to(t)
        # convert back to time domain
        t1 = self.noise_schedule.inverse_lambda(s1)
        # evaluations
        e_t = self._model_pred(x, t)
        x_mid = (self._alpha(t1) / self._alpha(t)) * x - torch.exp(-s1) * (torch.exp(r * h) - 1) * e_t
        e_t1 = self._model_pred(x_mid, t1)
        # second-order update
        x_next = (
            (self._alpha(s) / self._alpha(t)) * x
            - torch.exp(-lambda_s) * (((1 - r) * torch.exp(h) + r) - 1) * e_t
            - torch.exp(-lambda_s) * (torch.exp(h) - 1) * r * e_t1
        )
        return x_next

    def singlestep_dpm_solver_third_update(self, x: Tensor, t: Tensor, s: Tensor) -> Tensor:
        """
        Third-order single-step DPM-Solver.
        """
        lambda_t = self._lambda(t)
        lambda_s = self._lambda(s)
        h = lambda_s - lambda_t
        # coefficients for 3rd-order
        r1 = (3 + torch.sqrt(torch.tensor(3.0))) / 6
        r2 = (3 - torch.sqrt(torch.tensor(3.0))) / 6
        s1 = torch.scalar_tensor(float(lambda_t + r1 * h)).to(t)
        s2 = torch.scalar_tensor(float(lambda_t + r2 * h)).to(t)
        t1 = self.noise_schedule.inverse_lambda(s1)
        t2 = self.noise_schedule.inverse_lambda(s2)
        e_t = self._model_pred(x, t)
        # first mid
        x1 = (self._alpha(t1) / self._alpha(t)) * x - torch.exp(-s1) * (torch.exp(r1 * h) - 1) * e_t
        e_t1 = self._model_pred(x1, t1)
        # second mid
        x2 = (self._alpha(t2) / self._alpha(t)) * x - torch.exp(-s2) * (torch.exp(r2 * h) - 1) * e_t
        e_t2 = self._model_pred(x2, t2)
        # third-order update
        x_next = (
            (self._alpha(s) / self._alpha(t)) * x
            - torch.exp(-lambda_s) * ((torch.exp(h) - 1) - ((1 - r1) * torch.exp(h) + r1 - 1) - ((1 - r2) * torch.exp(h) + r2 - 1)) * e_t
            - torch.exp(-lambda_s) * (
                ((1 - r1) * torch.exp(h) + r1 - 1) * e_t1 + ((1 - r2) * torch.exp(h) + r2 - 1) * e_t2
            )
        )
        return x_next

    def get_time_steps_and_orders(
        self,
        num_steps: int,
        order: int = 2,
        skip_type: str = 'logSNR',
    ):
        """
        Helper to generate timesteps and solver orders.
        Returns a tuple (timesteps, orders).
        """
        return self.noise_schedule.get_orders_and_timesteps_for_singlestep_solver(
            order=order, num_steps=num_steps, skip_type=skip_type
        )
