import numpy as np
from tqdm import tqdm
from .MdpBuilderHelpers import *

class MdpKernel:
    """Minimal Markov‑Decision‑Process (MDP) kernel with utility functions
    to *maximize* expected discounted rewards **and** to *minimize* expected
    discounted costs (or rewards, if you treat them as costs).

    Parameters
    ----------
    transitionTable : np.ndarray
        Shape ``(N_states, N_states, N_actions)`` where
        ``transitionTable[s, s', a] = P(S_{t+1}=s' | S_t=s, A_t=a)``.
    rewardTable : np.ndarray
        Shape ``(N_states, N_actions)`` giving the *immediate* reward/cost
        obtained by executing action *a* in state *s*.  In many control
        settings rewards are non‑positive costs, but here we support either
        convention.
    actionTable : list | np.ndarray
        Optional metadata for actions (e.g. strings).
    """

    # ------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------
    def __init__(self, params=None):
        # if params is None, then it is in load mode
        if params is not None:
            self.set_params(params)       
            self.rewardTable = params['rewardTable']
            self.transitionTable = params['transitionTable']     
        self.policy = None
        self.V = None

    def set_params(self, params):
        self.N_user = params['N_user']
        self.LEN_window = params['LEN_window']
        self.r_bar = params['r_bar']
        self.B = params['B']
        self.N_aggregation = params['N_aggregation']
        self.N_states = params['N_states']
        self.N_actions = params['N_actions']
        self.aggregationMap = params['aggregationMap']
        self.actionTable = params['actionTable'] 

    # ------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------
    def load_policy(self, mdpParams, policyMode='deterministic'):
        self.set_params(mdpParams)
        self.mode = policyMode
        self.V = mdpParams['V']
        self.policy = mdpParams['policy']

    def predict(self, uOrigin):
        sOrigin = tuple_to_index(uOrigin, self.LEN_window+1)
        sAggregated = self._from_origin_to_aggregated_state(sOrigin)
        return self._getAction(sAggregated)

    def optimize_policy(self, mode: str = "deterministic", gamma: float = 0.99, theta: float = 1e-6, max_iterations: int = 20, lr=1e-2, temperature=1.0):
        self.mode = mode
        if mode == "deterministic":
            return self._optimize_policy_deterministic(gamma, theta, max_iterations)
        elif mode == "stochastic":
            return self._optimize_policy_stochastic(
            gamma=gamma, theta=theta, max_iterations=max_iterations, temperature=temperature, lr=lr)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def _optimize_policy_stochastic(self, gamma: float = 0.99, theta: float = 1e-6, max_iterations: int = 20, temperature=1.0, lr=1e-2):
        self.policy, self.V = self._optimize_policy_gradient(temperature=temperature, gamma=gamma, theta=theta, max_iterations=max_iterations, lr=lr)
        return self.V, self.policy

    def _optimize_policy_deterministic(self, 
                                      gamma: float = 0.99, 
                                      theta: float = 1e-6, 
                                      max_iterations: int = 1_000,
                                      ):
        self.V = np.zeros(self.N_states)  # cost‑to‑go initialisation
        for _ in range(max_iterations):
            delta = 0.0
            for s in range(self.N_states):
                # Bellman backup for minimisation
                q = [self._getReward(s, a) + gamma * np.dot(self._getTransition(a)[s], self.V)
                     for a in range(self.N_actions)]
                best_q = min(q)
                delta = max(delta, abs(best_q - self.V[s]))
                self.V[s] = best_q
            if delta < theta:
                break

        # Greedy (actually *descent*) policy extraction
        self.policy = np.empty(self.N_states, dtype=int)
        for s in range(self.N_states):
            q = [self._getReward(s, a) + gamma * np.dot(self._getTransition(a)[s], self.V)
                 for a in range(self.N_actions)]
            self.policy[s] = int(np.argmin(q))
        return self.V.copy(), self.policy.copy()
    
    def _from_origin_to_aggregated_state(self, sOrigin):
        uOrigin = index_to_tuple(sOrigin, self.LEN_window+1, self.N_user)
        uAggregated = self.aggregationMap[uOrigin]
        sAggregated = tuple_to_index(uAggregated, self.N_aggregation)
        return sAggregated
    

    def _getAction(self, sAggregated):
        #--------------compute action--------------
        if self.mode == "deterministic":
            a = self.policy[sAggregated]
        elif self.mode == "stochastic":
            a = np.random.choice(self.N_actions, p=self.policy[sAggregated])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        (w, alpha, M) = self.actionTable[a]
        uAggregated = index_to_tuple(sAggregated, self.N_aggregation, self.N_user)
        r = self._getDependentAction(np.array(uAggregated), np.array(w), alpha, self.B)
        return (w, r, M, alpha)
    
    def _getDependentAction(self, u, w, alpha, B):
        r = np.floor(alpha*B)/(np.sum(w)+1e-10) * w
        return r
    
    def _getReward(self, s: int, a: int):
        """Return *expected* immediate reward ``R(s,a)``."""
        return self.rewardTable[s, a]

    def _getTransition(self, a: int):
        """Return state‑transition matrix ``P_a`` for action *a*."""
        return self.transitionTable[:, :, a]
    

   #=====================================================
   # ==== Stochastic solution ========================
   # ====================================================


    # ---------- Exact policy evaluation ----------
    def _evaluate_policy_exact(self, policy, gamma: float = 0.99):
        """
        Vectorised policy evaluation with built-in NaN protection.
        """
        # ---------- Numerically-stable soft-max ----------
        _EPS          = 1e-12          # small constant to avoid /0
        _MAX_EXP_ARG  = 700.0          # ~ np.log(np.finfo(float).max)   
        S, A = self.N_states, self.N_actions

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
                    r = 1.0-np.clip(self._getReward(s, a), 0, 1.0)
                    r_pi[s] += p_sa * r

        # 3.  Transition matrix P_π
        P_pi = np.zeros((S, S), dtype=np.float64)
        for a in range(A):
            T_a = self._getTransition(a)                    # shape (S, S)
            P_pi += (policy[:, a, None] * T_a)

        # 4.  Solve (I − γ P_π) v = r_π
        A_mat = np.eye(S) - gamma * P_pi
        try:
            v = np.linalg.solve(A_mat, r_pi)
        except np.linalg.LinAlgError:                       # singular → least-squares
            v, *_ = np.linalg.lstsq(A_mat, r_pi, rcond=None)

        return v


    # ---------- Soft policy-iteration ----------
    def _optimize_policy_gradient(self,
                                  lr: float = 1e-1,
                                  gamma: float = 0.99,
                                  temperature: float = 1.0,
                                  max_iterations: int = 20,
                                  theta: float = 1e-10):
        """
        Gradient-based policy optimization using model-based Bellman gradients.
        """
        S, A = self.N_states, self.N_actions
        policy = np.full((S, A), 1.0 / A, dtype=np.float64)   # initial stochastic policy

        with tqdm(range(max_iterations), desc="Policy Optimization") as ite_bar:
            for _ in ite_bar:
                # 1. Evaluate current policy
                v = self._evaluate_policy_exact(policy, gamma=gamma)
                ite_bar.set_postfix({
                    'V': f'{v.mean():.6f}'
                })

                # 2. Compute Q-values under current V
                Q = np.empty_like(policy)
                for a in range(A):
                    r_sa = np.vectorize(self._getReward)(np.arange(S), np.full(S, a))
                    r_sa = 1.0 - np.clip(r_sa, 0, 1.0)
                    T_a = self._getTransition(a)
                    Q[:, a] = r_sa + gamma * (T_a @ v)

                # 3. Policy gradient step (softmax-compatible)
                logits = np.log(policy + 1e-12) + lr * Q / temperature  # add scaled Q to logits
                new_policy = np.apply_along_axis(self.softmax, 1, logits, temperature)

                # 4. NaN protection
                nan_rows = np.isnan(new_policy).any(axis=1)
                if nan_rows.any():
                    new_policy[nan_rows] = 1.0 / A

                # 5. Convergence check
                max_diff = np.abs(policy - new_policy).max()
                policy = new_policy
                if max_diff < theta:
                    break

        return policy, v

    def softmax(self, x, temperature: float = 1.0):
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