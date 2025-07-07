from EnvLibs import RewardKernel
from tqdm import tqdm
import numpy as np

class PolicySimulator:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
    
    def runSimulation(self, policy, num_windows=1000, obvMode="perfect", mode="test", type="data"):
        self.env.reset()
        self.env.selectMode(mode=mode, type=type)
        rewardRecord = []   
        actionsRecord = []
        uRecord = []
        uNextRecord = []
        with tqdm(range(num_windows), desc="Simulation Progress") as window_bar:
            for window in window_bar:
                u, u_predicted = self.env.getStates()
                if obvMode == "perfect":
                    (w, r, M, alpha) = policy.predict(u)
                elif obvMode == "predicted":
                    (w, r, M, alpha) = policy.predict(u_predicted)
                else:
                    raise ValueError(f"Invalid observation mode: {obvMode}")
                reward = self.env.applyActions(np.array(w), np.array(r), M, alpha)
                self.env.updateStates()
                u_next, u_next_predicted = self.env.getStates()
                #============ Record Results ============
                rewardRecord.append(reward)
                actionsRecord.append((np.array(w), np.array(r), M, alpha))
                if obvMode == "perfect":
                    uRecord.append(u)
                    uNextRecord.append(u_next)
                elif obvMode == "predicted":
                    uRecord.append(u_predicted)
                    uNextRecord.append(u_next_predicted)
                window_bar.set_postfix({
                    'avg reward': f'{np.mean(rewardRecord):.6f}'
                })

        simResult = {
            "rewardRecord": rewardRecord,
            "actionsRecord": actionsRecord,
            "uRecord": uRecord,
            "uNextRecord": uNextRecord
        }
        return simResult
    
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
    

