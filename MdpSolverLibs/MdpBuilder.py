from .MdpBuilderHelpers import *
from EnvLibs import *
from .MdpSolver import *

class MdpFormulator:
    def __init__(self, params, M_original):
        self.params = params
        self.N_user = params['N_user']
        self.LEN_window = params['LEN_window']
        self.r_bar = params['r_bar']
        self.B = params['B']
        #self.M = params['M']
        self.M_list = params['M_list']
        self.N_aggregation = params['N_aggregation']
        #------------------------------------------------------------
        self.M_original = M_original
        self.N_states_original = len(self.M_original) # (=self.LEN_window+1)
        self.rewardKernel = RewardKernel(params)
        self.alphaList = np.linspace(params['alpha_range'][0], params['alpha_range'][1], params['discrete_alpha_steps'])
        #------------------------------------------------------------
        (self.N_states, self.N_actions) = (None, None)
        self.actionSpace = None
        (self.M_aggregationSingle, self.p_aggregationSingle) = (None, None)
        (self.M_aggregation, self.p_aggregation) = (None, None)
        self.initialize()

    def initialize(self):
        #------------------------------------------------------------
        self.N_states = self.N_aggregation ** self.N_user
        self.N_actions = 2**self.N_user * len(self.alphaList) * len(self.M_list) #action = (w={0,1}, alpha, M)
        #------------------------------------------------------------
        self.buildActionSpace()

    def aggregateModel(self, approximate=False):
        self.buildAggregatedModel()
        if approximate:
            self.buildAggregatedRewardTableApproximate()
        else:
            self.buildAggregatedRewardTable()
        
    def buildAggregatedModel(self):
        #---------- get aggregation mapping ----------
        self.p_original = compute_stationary_distribution(self.M_original)
        self.thresholds = optimal_threshold_binning_uniform_arr(self.p_original, self.N_aggregation)[1:]
        self.aggregationMap = np.searchsorted(self.thresholds, np.arange(self.N_states_original))
        C = vector_to_mapping_matrix(self.aggregationMap)
        #---------- get aggregated transition matrix ----------
        self.M_aggregationSingle = compute_aggregated_transition_matrix(self.M_original, C)
        self.M_aggregation = compute_joint_transition_matrix(self.M_aggregationSingle, self.N_user)
        self.p_aggregation = compute_stationary_distribution(self.M_aggregation)
    

    def buildActionSpace(self):
        self.actionSpace = []
        for idx_w in range(2**self.N_user*len(self.M_list)):
            w = index_to_tuple(idx_w, 2, self.N_user)
            for alpha in self.alphaList:
                for M in self.M_list:
                    self.actionSpace.append((w, alpha, M))

    def buildAggregatedRewardTable(self):  
        N_stateOriginal = (self.LEN_window+1)**self.N_user
        N_stateAggregated = self.N_aggregation**self.N_user
        self.aggregatedRewardTable = np.zeros((N_stateAggregated, self.N_actions))
        for sOrigin in range(N_stateOriginal):
            uOrigin = index_to_tuple(sOrigin, self.LEN_window+1, self.N_user)
            sAggregated = self.from_origin_to_aggregated_state(sOrigin)
            for a in range(self.N_actions):
                reward, _ = self.getRewardHelper(uOrigin, sAggregated, a)
                self.aggregatedRewardTable[sAggregated, a] += (
                   reward * np.prod(self.p_original[uOrigin])
                )
    def buildAggregatedRewardTableApproximate(self):
        self.aggregatedRewardTable = np.zeros((self.N_states, self.N_actions))
        for s in range(self.N_states):
            uOriginExpectation = self.from_aggregated_to_origin_state_expectation(s)
            for a in range(self.N_actions):
                reward, _ = self.getRewardHelper(uOriginExpectation, s, a)
                self.aggregatedRewardTable[s, a] = reward

    def getRewardHelper(self, uOrigin, sAggregated,a):
        #--------------compute action--------------
        (w, alpha, M) = self.actionSpace[a]
        uAggregated = index_to_tuple(sAggregated, self.N_aggregation, self.N_user)
        r = self.getDependentAction(np.array(uAggregated), np.array(w), alpha, self.B)
        #--------------compute reward--------------
        reward = self.rewardKernel.getReward(np.array(uOrigin), np.array(w), np.array(r), M, alpha)
        return reward, (w, r, M, alpha)
    
    def getDependentAction(self,u, w, alpha, B):
        r = np.floor(alpha*B)/(np.sum(w)+1e-10) * w
        return r
    
    def getMdpKernel(self):
        params = self.params.copy()
        params['N_states'] = self.N_states
        params['N_actions'] = self.N_actions
        params['aggregationMap'] = self.aggregationMap
        params['N_aggregation'] = self.N_aggregation
        #------------------------------------------------
        transitionTable = np.zeros((self.N_states, self.N_states, self.N_actions))
        for i in range(self.N_actions): transitionTable[:,:,i] = self.M_aggregation
        actionTable = {i: self.actionSpace[i] for i in range(self.N_actions)}
        rewardTable = self.aggregatedRewardTable
        params['transitionTable'] = transitionTable
        params['rewardTable'] = rewardTable
        params['actionTable'] = actionTable
        return MdpKernel(params), params
        
    def from_origin_to_aggregated_state(self, sOrigin):
        uOrigin = index_to_tuple(sOrigin, self.LEN_window+1, self.N_user)
        uAggregated = self.aggregationMap[uOrigin]
        sAggregated = tuple_to_index(uAggregated, self.N_aggregation)
        return sAggregated
    

    def from_aggregated_to_origin_state_expectation(self, sAggregated):
        uAggregated = index_to_tuple(sAggregated, self.N_aggregation, self.N_user)
        uOrigiinApproximate = []
        for u in uAggregated:
            uOrigin_within_aggregation = np.where(self.aggregationMap == u)[0]
            uExpWeighted = (self.p_original[uOrigin_within_aggregation]/np.sum(self.p_original[uOrigin_within_aggregation])) * uOrigin_within_aggregation
            #print(uExpWeighted)
            uOrigiinApproximate.append(int(np.floor(np.sum(uExpWeighted))))
        return uOrigiinApproximate