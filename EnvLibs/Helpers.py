from EnvLibs import Environment, RewardKernel, TrafficGenerator
from EnvLibs.EnvConfigs import getEnvConfig
import pickle
import numpy as np

def createEnv(simParams):
    trafficGenerator = TrafficGenerator(simParams)
    for taskName in ["Task0", "Task1", "Task2"]:
        dataflow = simParams['dataflow']
        lenWindow = simParams['LEN_window']
        with open(f'Results/TrafficData/trafficData_{taskName}_{dataflow}_LenWindow{lenWindow}.pkl', 'rb') as f:
            trafficData = pickle.load(f)
        trafficGenerator.registerDataset(trafficData['traffic'], train_ratio=0.7)
    simEnv = Environment(simParams, trafficGenerator)
    simEnv.selectMode(mode="train", type="data")
    return simEnv

class PolicySimulator:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
    
    def runSimulation(self, policy, num_epochs=1000, mode="test", type="data"):
        self.env.reset()
        self.env.selectMode(mode=mode, type=type)
        rewardRecord = []   
        alphaRecord = []
        for epoch in range(num_epochs):
            u = self.env.getStates()
            (w, r, M, alpha) = policy.predict(u)
            reward = self.env.applyActions(np.array(w), np.array(r), M, alpha)
            self.env.updateStates()
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
        w = (u>int(lEN_window*0.5)).astype(int)
        return w
    
    def getDependentAction(self, u, w, alpha, B):
        r = np.floor(alpha*B)/(np.sum(w)+1e-10) * w
        return r