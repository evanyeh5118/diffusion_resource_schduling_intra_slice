import numpy as np
from tqdm import tqdm


def softmax(x, temperature: float = 1.0):
    # ---------- Numerically-stable soft-max ----------
    _EPS          = 1e-12          # small constant to avoid /0
    _MAX_EXP_ARG  = 700.0          # ~ np.log(np.finfo(float).max)   
    
    if temperature <= 0:
        raise ValueError("temperature must be strictly positive")

    # Log-sum-exp trick
    z = (x - np.nanmax(x)) / temperature        # shift for stability
    z = np.clip(z, -_MAX_EXP_ARG, 0.0)          # exp() domain safety
    exp_z = np.exp(z)
    denom = exp_z.sum()

    # If everything under-flowed, fall back to uniform
    if denom < _EPS or np.isnan(denom):
        return np.full_like(x, 1.0 / len(x), dtype=np.float64)

    return exp_z / (denom + _EPS)

def _evaluate_policy_exact(policy, mdpKernel, gamma: float = 0.99):
    """
    Vectorised policy evaluation with built-in NaN protection.
    """
    # ---------- Numerically-stable soft-max ----------
    _EPS          = 1e-12          # small constant to avoid /0
    _MAX_EXP_ARG  = 700.0          # ~ np.log(np.finfo(float).max)   
    S, A = mdpKernel.N_states, mdpKernel.N_actions

    # 1.  Ensure policy is well-formed (no NaN / rows sum to 1)
    policy = np.nan_to_num(policy, nan=1.0 / A, posinf=1.0 / A, neginf=1.0 / A)
    row_sums = policy.sum(axis=1, keepdims=True)
    policy  /= np.maximum(row_sums, _EPS)               # renormalise in place

    # 2.  Reward vector r_π
    r_pi = np.zeros(S, dtype=np.float64)
    for s in range(S):
        for a in range(A):
            p_sa = policy[s, a]
            if p_sa > 0.0:
                r = 1.0-np.clip(mdpKernel._getReward(s, a), 0, 1.0)
                r_pi[s] += p_sa * r

    # 3.  Transition matrix P_π
    P_pi = np.zeros((S, S), dtype=np.float64)
    for a in range(A):
        T_a = mdpKernel._getTransition(a)                    # shape (S, S)
        P_pi += (policy[:, a, None] * T_a)

    # 4.  Solve (I − γ P_π) v = r_π
    A_mat = np.eye(S) - gamma * P_pi
    try:
        v = np.linalg.solve(A_mat, r_pi)
    except np.linalg.LinAlgError:                       # singular → least-squares
        v, *_ = np.linalg.lstsq(A_mat, r_pi, rcond=None)

    return v

def _optimize_policy_gradient(mdpKernel,
                            lr: float = 1e-1,
                            gamma: float = 0.99,
                            temperature: float = 1.0,
                            max_iterations: int = 20,
                            theta: float = 1e-10):
    S, A = mdpKernel.N_states, mdpKernel.N_actions
    policy = np.full((S, A), 1.0 / A, dtype=np.float64)   # initial stochastic policy

    with tqdm(range(max_iterations), desc="Policy Optimization") as ite_bar:
        for _ in ite_bar:
            # 1. Evaluate current policy
            v = _evaluate_policy_exact(policy, mdpKernel, gamma=gamma)
            ite_bar.set_postfix({
                'V': f'{v.mean():.6f}'
            })

            # 2. Compute Q-values under current V
            Q = np.empty_like(policy)
            for a in range(A):
                r_sa = np.vectorize(mdpKernel._getReward)(np.arange(S), np.full(S, a))
                r_sa = 1.0 - np.clip(r_sa, 0, 1.0)
                T_a = mdpKernel._getTransition(a)
                Q[:, a] = r_sa + gamma * (T_a @ v)

            # 3. Policy gradient step (softmax-compatible)
            logits = np.log(policy + 1e-12) + lr * Q / temperature  # add scaled Q to logits
            new_policy = np.apply_along_axis(softmax, 1, logits, temperature)

            # 4. NaN protection
            nan_rows = np.isnan(new_policy).any(axis=1)
            if nan_rows.any():
                new_policy[nan_rows] = 1.0 / A

            # 5. Convergence check
            max_diff = np.abs(policy - new_policy).max()
            policy = new_policy
            if max_diff < theta:
                break

    return v, policy
