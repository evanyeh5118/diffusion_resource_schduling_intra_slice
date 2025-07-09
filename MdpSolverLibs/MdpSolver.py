import numpy as np
from tqdm import tqdm
from .MdpBuilderHelpers import *
from .MdpSolverHelper import  _optimize_policy_gradient

class MdpKernel:
    def __init__(self, params=None):
        if params is not None:
            self.set_params(params)       
            self.rewardTable = params['rewardTable']
            self.transitionTable = params['transitionTable']     
        self.policy_deter = None
        self.policy_stoch = None
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

    def load_policy(self, mdpParams, policyMode='deterministic', randomR=False):
        self.set_params(mdpParams)
        self.mode = policyMode
        self.randomR = randomR
        self.policy_deter = mdpParams['policy_deter']
        self.policy_stoch = mdpParams['policy_stoch']
        if policyMode == "deterministic" and self.policy_deter is None:
            raise ValueError("policy_deter is not provided")
        elif policyMode == "stochastic" and self.policy_stoch is None:
            raise ValueError("policy_stoch is not provided")

    def predict(self, uOrigin):
        sOrigin = tuple_to_index(uOrigin, self.LEN_window+1)
        sAggregated = self._from_origin_to_aggregated_state(sOrigin)
        return self._getAction(sAggregated)

    def optimize_policy(self, 
                        mode: str = "deterministic", 
                        gamma: float = 0.99, 
                        theta: float = 1e-6, 
                        max_iterations: int = 20, 
                        lr=1e-2, 
                        temperature=1.0):
        self.mode = mode
        if mode == "deterministic":
            return self._optimize_policy_deterministic(gamma, theta, max_iterations)
        elif mode == "stochastic":
            return self._optimize_policy_stochastic(
            gamma=gamma, theta=theta, max_iterations=max_iterations, temperature=temperature, lr=lr)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def _optimize_policy_stochastic(self, 
                                    gamma: float = 0.99, 
                                    theta: float = 1e-6, 
                                    max_iterations: int = 1_000, 
                                    temperature=1.0, 
                                    lr=1e-2):
        self.V, self.policy_stoch = _optimize_policy_gradient(
            self,
            temperature=temperature, gamma=gamma, theta=theta, max_iterations=max_iterations, lr=lr
        )
        return self.V.copy(), self.policy_stoch.copy()

    def _optimize_policy_deterministic(self, 
                                      gamma: float = 0.99, 
                                      theta: float = 1e-6, 
                                      max_iterations: int = 1_000,
                                      ):
        self.V = np.zeros(self.N_states)  # cost‑to‑go initialisation
        with tqdm(range(max_iterations), desc="Value Iteration") as ite_bar:
            for _ in ite_bar:
                delta = 0.0
                ite_bar.set_postfix({
                    'V': f'{self.V.mean():.6f}'
                })
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
        self.policy_deter = np.empty(self.N_states, dtype=int)
        for s in range(self.N_states):
            q = [self._getReward(s, a) + gamma * np.dot(self._getTransition(a)[s], self.V)
                 for a in range(self.N_actions)]
            self.policy_deter[s] = int(np.argmin(q))
        return self.V.copy(), self.policy_deter.copy()
    
    def _from_origin_to_aggregated_state(self, sOrigin):
        uOrigin = index_to_tuple(sOrigin, self.LEN_window+1, self.N_user)
        uAggregated = self.aggregationMap[uOrigin]
        sAggregated = tuple_to_index(uAggregated, self.N_aggregation)
        return sAggregated
    

    def _getAction(self, sAggregated):
        #--------------compute action--------------
        if self.mode == "deterministic":
            a = self.policy_deter[sAggregated]
        elif self.mode == "stochastic":
            a = np.random.choice(self.N_actions, p=self.policy_stoch[sAggregated])
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        (w, alpha, M) = self.actionTable[a]
        uAggregated = index_to_tuple(sAggregated, self.N_aggregation, self.N_user)
        if self.randomR == True:
            r = self._getRandomAction(w)
        else:
            r = self._getDependentAction(np.array(uAggregated), np.array(w), alpha, self.B)
        return (w, r, M, alpha)
    
    def _getDependentAction(self, u, w, alpha, B):
        r = np.floor(alpha*B)/(np.sum(w)+1e-10) * w
        return r
    
    def _getRandomAction(self, w):
        return (np.random.randint(0, self.B, self.N_user)*np.array(w)).astype(int)
    
    def _getReward(self, s: int, a: int):
        """Return *expected* immediate reward ``R(s,a)``."""
        return self.rewardTable[s, a]

    def _getTransition(self, a: int):
        """Return state‑transition matrix ``P_a`` for action *a*."""
        return self.transitionTable[:, :, a]
    

   #=====================================================
   # ==== Stochastic solution ========================
   # ====================================================


    