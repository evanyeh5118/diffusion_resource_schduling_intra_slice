import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from .WirelessModel import successfulPacketCDF


class RewardKernel:
    def __init__(self, params):
        self.Type1RewardKernel = Type1RewardKernel(params)
        self.Type1Constraint = Type1Constraint(params)
        self.Type2RewardKernel = Type2RewardKernel(params)

    def getReward(self, u, w, r, M, alpha):
        self.Jtype1 = self.Type1RewardKernel.getReward(u, w, r)
        self.Jc = self.Type1Constraint.getPenalty(w, r, alpha)
        self.Jtype2 = self.Type2RewardKernel.getReward(u, w, M, alpha)

        N_type1 = np.sum(w)
        N_type2 = np.sum(1-w)
        r = N_type1 / (N_type1 + N_type2)
        self.J = r*self.Jtype1 + (1-r)*self.Jtype2
        #self.J = (N_type1*self.Jtype1 + N_type2*self.Jtype2)
        return self.J + self.Jc

class Type1RewardKernel:
    def __init__(self, params):
        self.B = params['B']

    def getReward(self, u, w, r):
      J = 1-np.sum(w * u * successfulPacketCDF(r))  / (np.sum(w*u)+1e-10)
      return J
    
class Type1Constraint:
    def __init__(self, params):
       self.B = params['B']
    
    def getPenalty(self, w, r, alpha):
        cstCost = (alpha*self.B - np.sum(w*r))**2 if np.sum(w*r) - alpha*self.B > 0 else 0
        return cstCost

class Type2RewardKernel:
    def __init__(self, params):
        #self.p = params['p']
        self.LEN_window = params['LEN_window']
        self.r_bar = params['r_bar']
        self.B = params['B']
        self.epsilon = 1-successfulPacketCDF(self.r_bar)

    def getReward(self, u, w, M, alpha):
        N = np.sum(1-w)
        if N == 0:
            return 0
        W = np.ceil((1-alpha)*self.B/self.r_bar).astype(int)
        p = (np.sum(u*(1-w)) / (self.LEN_window * N + 1e-10))
        if W < M:
            return np.inf
            #raise ValueError(f"Insufficient Type II resource")
        e = packetLossWithTransmissionErrors(N, W, M, p, self.epsilon)
        return e

def binom(n, k):
    return math.comb(n, k)

def packetLossWithTransmissionErrors(N, W, M, p, e_p):
    total = 0.0
    for l in range(1, M + 1):
        sign = (-1) ** (l + 1)
        collision_free = (1 - p) + p * binom(W - l, M) / binom(W, M)
        total += sign * binom(M, l) * (1 - e_p) ** l * collision_free ** (N - 1)
    return 1.0 - total