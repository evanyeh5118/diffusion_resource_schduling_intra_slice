import torch
from .dpm_solver_pytorch import NoiseScheduleVP, DPM_Solver

# -- tiny helper at end ------------------------------------------------------
def solve(model_fn, betas, x_T, steps=15, order=3, algorithm='dpmsolver++'):
    """
    Convenience function – one-liner to go from x_T ~ N(0,1) to x_0
    using the supplied epsilon-model `model_fn` (expects (x, t_float)).
    """
    ns = NoiseScheduleVP(schedule='discrete', betas=betas, dtype=x_T.dtype)
    solver = DPM_Solver(model_fn, ns, algorithm_type=algorithm)
    # sample with uniform time spacing (“time_uniform” in paper)
    ts = solver.get_time_steps('time_uniform', ns.T, 1.0 / ns.total_N,
                               steps, x_T.device)
    x = x_T
    for i in range(len(ts) - 1):
        s, t = ts[i].unsqueeze(0), ts[i + 1].unsqueeze(0)
        # third/second/first-order kernel chosen automatically
        if order == 3 and i <= len(ts) - 4:
            x = solver.singlestep_dpm_solver_third_update(x, s, t)
        elif order >= 2 and i <= len(ts) - 3:
            x = solver.singlestep_dpm_solver_second_update(x, s, t)
        else:
            x = solver.dpm_solver_first_update(x, s, t)
    return x

class DPMSolver:
    """
    Plug-and-play DPM-Solver sampler for diffusion_bc.DiffusionPolicy.
    policy : the trained DiffusionPolicy
    """
    def __init__(self, policy, steps=15, order=3, algorithm='dpmsolver++'):
        self.policy = policy
        self.steps = steps
        self.order = order
        self.algorithm = algorithm
        # 1. build a NoiseSchedule from the policy’s discrete betas
        betas = policy.schedule.beta.to(policy.schedule.beta.device)
        self.ns = NoiseScheduleVP(schedule='discrete', betas=betas)

        # 2. wrap the policy’s epsilon-network so it matches DPM-Solver’s (x,t)->eps
        def eps_model(x, t_cont):
            """
            x : (B, action_dim)  – current noisy action a_t
            t_cont : (B,) float in (0,1]
            """
            # Convert continuous time in (0,1] -> integer step 1…N
            N = policy.schedule.N
            # make sure 1 → N, 1/N → 1
            idx = (t_cont * N).clamp(min=1.0, max=float(N))  # float
            idx = idx.round().long()
            # we need the state condition; cache it via closure
            return eps_model.specific_state, idx  # placeholder (patched below)

        # we’ll replace eps_model each time we call sample()
        self._base_eps_model = eps_model

    @torch.no_grad()
    def sample(self, state, solver_steps=None):
        """
        state : (B, state_dim) torch tensor
        returns : (B, action_dim) sampled action
        """
        solver_steps = solver_steps or self.steps
        B, device = state.size(0), state.device
        action_dim = self.policy.a_dim

        # 1. define a state-specific epsilon model (closure captures `state`)
        def eps_model(x, t_cont):
            N = self.policy.schedule.N
            idx = (t_cont * N).clamp(min=1.0, max=float(N)).round().long()
            # broadcast state to batch of x
            s_rep = state if state.size(0) == x.size(0) else state.repeat(x.size(0) // B, 1)
            return self.policy(x, s_rep, idx)  # εθ(a_t , s , t)

        # 2. prepare x_T ~ N(0,1)
        x_T = torch.randn(B, action_dim, device=device, dtype=state.dtype)

        # 3. solve reverse ODE
        return solve(eps_model, self.policy.schedule.beta, x_T,
                     steps=solver_steps, order=self.order, algorithm=self.algorithm)