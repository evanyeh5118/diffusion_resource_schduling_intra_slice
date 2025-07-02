from EnvLibs import Environment, RewardKernel, TrafficGenerator
from EnvLibs.EnvConfigs import getEnvConfig
import pickle
import numpy as np
import math

def createEnv(envParams, trafficDataParentPath):    
    with open(f'{trafficDataParentPath}/trafficData_{envParams["dataflow"]}_LenWindow{envParams["LEN_window"]}.pkl', 'rb') as f:
        trafficData = pickle.load(f)
    trafficGenerator = TrafficGenerator(envParams)
    trafficGenerator.registerDataset(
        trafficData['trafficSource_train_actual'], trafficData['trafficSource_test_actual'],
        trafficData['trafficTarget_train_predicted'], trafficData['trafficTarget_test_predicted']
    )
    simEnv = Environment(envParams, trafficGenerator)
    simEnv.selectMode(mode="train", type="data")
    return simEnv

class PolicySimulator:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
    
    def runSimulation(self, policy, obvMode="perfect", num_epochs=1000, mode="test", type="data"):
        self.env.reset()
        self.env.selectMode(mode=mode, type=type)
        rewardRecord = []   
        alphaRecord = []
        for epoch in range(num_epochs):
            self.env.updateStates()
            u, u_predicted = self.env.getStates()
            if obvMode == "perfect":
                (w, r, M, alpha) = policy.predict(u)
            elif obvMode == "predicted":
                (w, r, M, alpha) = policy.predict(u_predicted)
            else:
                raise ValueError(f"Invalid observation mode: {obvMode}")
            reward = self.env.applyActions(np.array(w), np.array(r), M, alpha)
            rewardRecord.append(reward)
            alphaRecord.append(alpha)

        return rewardRecord

class PolicyDemoAdaptiveAlpha:
    def __init__(self, params):
        self.params = params
        self.rewardKernel = RewardKernel(params)
        self.M = 3
        self.alphaList = np.linspace(params['alpha_range'][0], params['alpha_range'][1], params['discrete_alpha_steps'])
    
    def predict(self, u):
        w = self.typeAllocator(u, self.params['LEN_window'])
        JmdpRecord = []
        for alpha in self.alphaList:
            r = np.floor(alpha*self.params['B'])/(np.sum(w)+1e-10) * w 
            Jmdp = self.rewardKernel.getReward(u, w, r, self.M, alpha)
            JmdpRecord.append(Jmdp)
        alpha = self.alphaList[np.argmin(JmdpRecord)]
        r = self.getDependentAction(u, w, alpha, self.params['B'])
        return w, r, self.M, alpha
    
    def typeAllocator(self, u, lEN_window):
        w = (u>int(lEN_window*0.1)).astype(int)
        return w
    
    def getDependentAction(self, u, w, alpha, B):
        r = np.floor(alpha*B)/(np.sum(w)+1e-10) * w
        return r
    
'''
def compute_stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution of a Markov transition matrix P.
    Solves for pi such that pi @ P = pi and sum(pi) = 1.
    """
    S = P.shape[0]
    A = np.vstack([P.T - np.eye(S), np.ones(S)])
    b = np.append(np.zeros(S), 1)
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi

    
def optimal_threshold_binning_uniform_arr(
    pmf: np.ndarray, K: int
    ) -> np.ndarray:
        def _entropy(p: float) -> float:
            """Shannon entropy of a single probability."""
            return -p * math.log2(p) if p > 0 else 0.0
        
        N = len(pmf)
        if not (1 <= K <= N):
            raise ValueError("K must satisfy 1 ≤ K ≤ N.")

        # Cumulative sums for O(1) range-probability queries.
        cumsum = np.cumsum(np.concatenate(([0.0], pmf)))  # length N+1

        def group_entropy(i: int, j: int) -> float:
            """Entropy of the mass contained in slice [i:j)."""
            return _entropy(cumsum[j] - cumsum[i])

        # DP tables
        dp = np.full((N + 1, K + 1), -np.inf)
        back = np.full((N + 1, K + 1), -1, dtype=int)
        dp[0, 0] = 0.0

        # Forward DP
        for i in range(1, N + 1):                      # prefix length
            for k in range(1, min(K, i) + 1):          # #bins used
                # try every previous cut-point j
                for j in range(k - 1, i):
                    cand = dp[j, k - 1] + group_entropy(j, i)
                    if cand > dp[i, k]:
                        dp[i, k] = cand
                        back[i, k] = j

        # Back-track optimal cuts
        cuts: List[int] = []
        i, k = N, K
        while k > 0:
            j = back[i, k]
            if j <= 0:        # j == 0 means the cut is at the very start;
                break         # we already add 0 below, so skip duplicates
            cuts.append(j)
            i, k = j, k - 1

        thresholds = np.array([0] + sorted(cuts) + [N], dtype=int)
        if thresholds.size != K + 1 or not np.all(np.diff(thresholds) > 0):
            raise RuntimeError("Failed to find a valid set of thresholds.")
        return thresholds
'''