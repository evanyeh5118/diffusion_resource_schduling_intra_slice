import numpy as np
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
    def load_policy(self, mdpParams):
        self.set_params(mdpParams)
        self.V = mdpParams['V']
        self.policy = mdpParams['policy']

    def predict(self, uOrigin):
        sOrigin = tuple_to_index(uOrigin, self.LEN_window+1)
        sAggregated = self._from_origin_to_aggregated_state(sOrigin)
        return self._getAction(sAggregated)

    def optimize_policy(self, mode: str = "deterministic", gamma: float = 0.99, theta: float = 1e-6, max_iterations: int = 1_000):
        if mode == "deterministic":
            return self._optimize_policy_deterministic(gamma, theta, max_iterations)
        elif mode == "stochastic":
            return self._optimize_policy_stochastic(temperature=1.0, gamma=gamma, theta=theta, max_iterations=max_iterations)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def _optimize_policy_stochastic(self, temperature: float = 1.0, gamma: float = 0.99, theta: float = 1e-6, max_iterations: int = 1_000):
        self.policy, self.V = self._policy_iteration(temperature=temperature, gamma=gamma, theta=theta, max_iterations=max_iterations)
        return self.policy, self.V

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
        a = self.policy[sAggregated]
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

    def _evaluate_policy_exact(self, policy):
        """
        policy: numpy array of shape (N_states, N_actions)
                policy[s, a] gives π(a|s)
        Returns: v, numpy array of shape (N_states,)
        """
        S, A = self.N_states, self.N_actions

        # Build reward vector r_pi
        r_pi = np.zeros(S)
        for s in range(S):
            for a in range(A):
                prob = policy[s, a]
                if prob:
                    r_pi[s] += prob * self._getReward(s, a)

        # Build transition matrix P_pi
        P_pi = np.zeros((S, S))
        for s in range(S):
            for a in range(A):
                prob = policy[s, a]
                if prob:
                    for s_next, p in self._getTransition(a)[s].items():
                        P_pi[s, s_next] += prob * p

        # Solve (I - gamma * P_pi) v = r_pi
        A_mat = np.eye(S) - self.gamma * P_pi
        v = np.linalg.solve(A_mat, r_pi)
        return v

    def _policy_iteration(self, temperature=1.0, gamma=0.99, theta=1e-6, max_iterations=1000):
        """
        Soft (stochastic) policy iteration with array-based policy.

        Returns:
          policy: numpy array (N_states, N_actions)
          value_function: numpy array (N_states,)
        """
        S, A = self.N_states, self.N_actions

        policy = np.ones((S, A)) / A

        for it in range(max_iterations):
            # Policy evaluation
            v = self._evaluate_policy_exact(policy)

            # Policy improvement (softmax)
            new_policy = np.zeros_like(policy)
            # Compute Q-table
            Q = np.zeros((S, A))
            for s in range(S):
                for a in range(A):
                    r_sa = self._getReward(s, a)
                    expected_v = sum(
                        p * v[s_next]
                        for s_next, p in self._getTransition(a)[s].items()
                    )
                    Q[s, a] = r_sa + gamma * expected_v

            # Softmax update per state
            max_diff = 0.0
            for s in range(S):
                q = Q[s]
                q_max = q.max()
                exp_q = np.exp((q - q_max) / temperature)
                probs = exp_q / exp_q.sum()
                max_diff = max(max_diff, np.max(np.abs(policy[s] - probs)))
                new_policy[s] = probs

            policy = new_policy
            if max_diff < theta:
                break

        return policy, v
