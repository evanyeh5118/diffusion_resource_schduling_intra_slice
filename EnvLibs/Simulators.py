import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, random

from EnvLibs import TrafficGenerator
from EnvLibs import successfulPacketCDF


class SimulatorType1:
    def __init__(self, params):
        self.B = params['B']
        self.r_bar = params['r_bar']
        self.LEN_window = params['LEN_window']
        self.successTotal = 0
        self.activeTotal = 0

    def stepSimple(self, u, w, r):
        # Filter active users (w==1) and get their corresponding traffic and rates
        active_mask = w == 1
        traffics = u[active_mask]
        rb = r[active_mask]
        for i, traffic in enumerate(traffics):
            for _ in range(traffic):
                if random.random() <= successfulPacketCDF(rb[i]):
                    self.successTotal += 1
            self.activeTotal += traffic

    def step(self, u, w, r, alpha):
        active_mask = w == 1 # active user in Type I
        traffics = u[active_mask] # user's traffic in Type I
        rb = r[active_mask] # reserveed rb for each user in Type I
        N = np.sum(w) # N user in Type I

        userActiveMap = buildActivationMap(N, traffics, self.LEN_window)

        successWindow = 0
        activeWindow = 0
        for k in range(self.LEN_window):
            B_type1 = int(np.ceil(alpha*self.B))
            active_idx = [u for u in range(N) if userActiveMap[u, k] > 0]
            if not active_idx:
                continue
            for i in active_idx:
                if B_type1 > 0 and random.random() < successfulPacketCDF(min(rb[i], B_type1)):
                    successWindow += 1
                    B_type1 -= rb[i]
        
        activeWindow = np.sum(traffics)
        self.successTotal += successWindow
        self.activeTotal += activeWindow
        failuresWindow = activeWindow-successWindow
        return failuresWindow, activeWindow
            
    def getLossRate(self):
        return 1-self.successTotal/(self.activeTotal+1e-10)

    def reset(self):
        self.successTotal = 0
        self.activeTotal = 0

class SimulatorType2:
    def __init__(self, params):
        self.B = params['B']
        self.r_bar = params['r_bar']
        self.epsilon = 1-successfulPacketCDF(params['r_bar'])
        self.LEN_window = params['LEN_window']

        self.failuresTotal = 0
        self.activeTotal = 0

    def step(self, u, w, M, alpha):
        self.W = int(np.ceil((1-alpha)*self.B/self.r_bar))
        #M = min(self.W-1, M)
        failuresWindow, activeWindow = self.stepHelper(u, w, self.W, M, self.epsilon, self.LEN_window)
        self.failuresTotal += failuresWindow
        self.activeTotal += activeWindow
        return failuresWindow, activeWindow

    def getLossRate(self):
        return self.failuresTotal/self.activeTotal

    def reset(self):
        self.failuresTotal = 0
        self.activeTotal = 0        
    
    def stepHelper(self, u, w, W, M, epsilon, LEN_window, seed=None):
        if seed is not None:
            random.seed(seed)
    
        N = np.sum(1-w) # N user in Type II
        traffics = u[w==0] # traffic of users in Type II
        userActiveMap = buildActivationMap(N, traffics, LEN_window)

        failuresTotal = 0
        for k in range(LEN_window):
            active_idx = [u for u in range(N) if userActiveMap[u, k] > 0]
            if not active_idx:
                continue
            failures = collisionCheck(active_idx, W, M, epsilon)
            failuresTotal += failures
        
        activeTotal = np.sum(u*(1-w))

        return failuresTotal, activeTotal

def collisionCheck(active_idx, W, M, epsilon):
    # 2) Each active user chooses β distinct RBs
    failures = 0
    rb_usage = np.zeros((W, ))          # how many users pick each RB
    user_rbs = {}               # mapping user -> list of RBs
    for u in active_idx:
        rbs = random.sample(range(W), min(M, W))
        user_rbs[u] = rbs
        for rb in rbs:
            rb_usage[rb] += 1

    # 3) Check, for every active user, whether *any* replica succeeds
    for u in active_idx:
        success = False
        for rb in user_rbs[u]:
            if rb_usage[rb] == 1 and random.random() > epsilon:              # no collision
                success = True
                break
        if not success:
            failures += 1

    return failures


def buildActivationMap(N, traffics, LEN_window):
    userActive = np.zeros((N, LEN_window))
    for i, traffic in enumerate(traffics): # i:u_i in Type II
        activateTime = random_select_without_repetition(traffic, LEN_window)
        userActive[i, activateTime] = 1 # 1: active, 0: inactive
    return userActive

def random_select_without_repetition(N, L, seed=None):
    if N > L + 1:
        raise ValueError(f"Cannot select {N} unique numbers from range [0, {L}] (only {L+1} options available)")
    
    if seed is not None:
        np.random.seed(seed)
    (N,L) = (int(N), int(L))
    all_numbers = np.arange(L).astype(int)
    selected = np.random.choice(all_numbers, size=N, replace=False).astype(int)
    
    return selected