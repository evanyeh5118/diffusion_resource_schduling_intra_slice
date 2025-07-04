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
        if params is not None:
            self.set_params(params)
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
        self.rewardTable = params['rewardTable']
        self.transitionTable = params['transitionTable']
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

    def optimize_policy(self, gamma: float = 0.99, theta: float = 1e-6, max_iterations: int = 1_000):
        return self._optimize_policy_deterministic(gamma, theta, max_iterations)
    

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

    '''
    def optimize_stochastic_policy(self,
                                   gamma: float = 0.99,
                                   theta: float = 1e-6,
                                   max_iterations: int = 1_000,
                                   ):
        # 1) Initialize to a uniform random policy
        # policy[s,a] = probability of taking a in s
        self.policy = np.ones((self.N_states, self.N_actions)) / self.N_actions
        # 2) Zero initial value function
        self.V = np.zeros(self.N_states)

        for _ in range(max_iterations):
            # ----- Policy Evaluation -----
            while True:
                delta = 0.0
                for s in range(self.N_states):
                    v_old = self.V[s]
                    # V[s] = sum_a pi[s,a] * [ R(s,a) + γ Σ_s' P(s'|s,a) V[s'] ]
                    self.V[s] = sum(
                        self.policy[s, a] *
                        (self.getReward(s, a) +
                        gamma * np.dot(self.getTransition(a)[s], self.V))
                        for a in range(self.N_actions)
                    )
                    delta = max(delta, abs(v_old - self.V[s]))
                if delta < theta:
                    break

            # ----- Policy Improvement -----
            policy_stable = True
            for s in range(self.N_states):
                old_action_probs = self.policy[s].copy()
                # compute Q(s,a) for all a
                q = np.array([
                    self.getReward(s, a) +
                    gamma * np.dot(self.getTransition(a)[s], self.V)
                    for a in range(self.N_actions)
                ])
                # find the minimal cost
                best = q.min()
                # all actions achieving that cost
                best_actions = np.where(q == best)[0]
                # new policy: uniform over best actions
                new_probs = np.zeros_like(old_action_probs)
                new_probs[best_actions] = 1.0 / len(best_actions)
                self.policy[s] = new_probs

                if not np.allclose(old_action_probs, new_probs):
                    policy_stable = False

            if policy_stable:
                break

        return self.V, self.policy
'''

