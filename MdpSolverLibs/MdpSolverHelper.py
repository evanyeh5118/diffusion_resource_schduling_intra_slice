import numpy as np

#=====================================================
   # ==== Stochastic solution ========================
   # ====================================================

def evaluate_policy_exact(
        policy, N_states, N_actions, gamma):
    """
    policy: numpy array of shape (N_states, N_actions)
            policy[s, a] gives π(a|s)
    Returns: v, numpy array of shape (N_states,)
    """
    S, A = N_states, N_actions

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

def policy_iteration(self, init_policy=None, temperature=1.0, tol=1e-8, max_iters=1000):
    """
    Soft (stochastic) policy iteration with array-based policy.

    Returns:
        policy: numpy array (N_states, N_actions)
        value_function: numpy array (N_states,)
    """
    S, A = self.N_states, self.N_actions

    # Initialize policy
    if init_policy is None:
        policy = np.ones((S, A)) / A
    else:
        policy = init_policy.copy()

    for it in range(max_iters):
        # Policy evaluation
        v = self.evaluate_policy_exact(policy)

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
                Q[s, a] = r_sa + self.gamma * expected_v

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
        if max_diff < tol:
            break

    return policy, v