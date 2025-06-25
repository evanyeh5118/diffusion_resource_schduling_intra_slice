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
    def __init__(self, params):
        self.N_user = params['N_user']
        self.LEN_window = params['LEN_window']
        self.r_bar = params['r_bar']
        self.B = params['B']
        #self.M_list = params['M_list']
        self.N_aggregation = params['N_aggregation']
        self.N_states = params['N_states']
        self.N_actions = params['N_actions']
        self.aggregationMap = params['aggregationMap']
        #------------------------------------------------------------
        self.rewardTable = params['rewardTable']
        self.transitionTable = params['transitionTable']
        self.actionTable = params['actionTable']
        self.policy = None
        self.V = None
        
    # ------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------
    def getReward(self, s: int, a: int):
        """Return *expected* immediate reward ``R(s,a)``."""
        return self.rewardTable[s, a]

    def getTransition(self, a: int):
        """Return state‑transition matrix ``P_a`` for action *a*."""
        return self.transitionTable[:, :, a]
   
    def getActionFromObervation(self, uOrigin):
        sOrigin = tuple_to_index(uOrigin, self.LEN_window+1)
        sAggregated = self.from_origin_to_aggregated_state(sOrigin)
        return self.getAction(sAggregated)

    def getAction(self, sAggregated):
        #--------------compute action--------------
        a = self.policy[sAggregated]
        (w, alpha, M) = self.actionTable[a]
        uAggregated = index_to_tuple(sAggregated, self.N_aggregation, self.N_user)
        r = self.getDependentAction(np.array(uAggregated), np.array(w), alpha, self.B)
        return (w, r, M, alpha)
    
    def getDependentAction(self, u, w, alpha, B):
        r = np.floor(alpha*B)/(np.sum(w)+1e-10) * w
        return r

    # ------------------------------------------------------------
    # 1) Reward‑maximising value iteration (default in RL texts)
    # ------------------------------------------------------------
    def optimize_policy(self, gamma: float = 0.99, theta: float = 1e-6, max_iterations: int = 1_000):
        """Compute an *optimal* **deterministic** policy that **maximises**
        expected discounted return via **value iteration**.

        Returns
        -------
        V : np.ndarray of shape (N_states,)
            Optimal state‑value function.
        policy : np.ndarray of ints, shape (N_states,)
            Action indices implementing the greedy policy.
        """
        V = np.zeros(self.N_states)  # initial guess

        for _ in range(max_iterations):
            delta = 0.0
            for s in range(self.N_states):
                # Bellman optimality backup for maximisation
                q = [self.getReward(s, a) + gamma * np.dot(self.getTransition(a)[s], V)
                     for a in range(self.N_actions)]
                best_q = min(q)
                delta = max(delta, abs(best_q - V[s]))
                V[s] = best_q
            if delta < theta:
                break

        # Greedy policy extraction
        policy = np.empty(self.N_states, dtype=int)
        for s in range(self.N_states):
            q = [self.getReward(s, a) + gamma * np.dot(self.getTransition(a)[s], V)
                 for a in range(self.N_actions)]
            policy[s] = int(np.argmax(q))
        return V, policy

    # ------------------------------------------------------------
    # 2) Cost‑minimising value iteration (new!)
    # ------------------------------------------------------------
    def minimize_policy(self, gamma: float = 0.99, theta: float = 1e-6, max_iterations: int = 1_000):
        """Compute an *optimal* deterministic policy that **minimises** the
        expected discounted sum of rewards (i.e. treats rewards as *costs*).

        This solves the Bellman *optimality* equation for **minimisation**:

            V*(s) = min_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s') ].

        Notes
        -----
        If your rewards already represent costs (e.g. non‑negative numbers in
        ``[0, 1]`` you wish to drive to *zero*), then this method directly
        optimises for the smallest cumulative cost.  It is equivalent to
        calling :py:meth:`optimize_policy` on the *negated* reward function.
        """
        self.V = np.zeros(self.N_states)  # cost‑to‑go initialisation

        for _ in range(max_iterations):
            delta = 0.0
            for s in range(self.N_states):
                # Bellman backup for minimisation
                q = [self.getReward(s, a) + gamma * np.dot(self.getTransition(a)[s], self.V)
                     for a in range(self.N_actions)]
                best_q = min(q)
                delta = max(delta, abs(best_q - self.V[s]))
                self.V[s] = best_q
            if delta < theta:
                break

        # Greedy (actually *descent*) policy extraction
        self.policy = np.empty(self.N_states, dtype=int)
        for s in range(self.N_states):
            q = [self.getReward(s, a) + gamma * np.dot(self.getTransition(a)[s], self.V)
                 for a in range(self.N_actions)]
            self.policy[s] = int(np.argmin(q))
        return self.V, self.policy

    def from_origin_to_aggregated_state(self, sOrigin):
        uOrigin = index_to_tuple(sOrigin, self.LEN_window+1, self.N_user)
        uAggregated = self.aggregationMap[uOrigin]
        sAggregated = tuple_to_index(uAggregated, self.N_aggregation)
        return sAggregated