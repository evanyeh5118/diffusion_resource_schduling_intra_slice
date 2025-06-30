import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class TrafficGenerator:
    def __init__(self, params):
        self.N_user = params['N_user']
        self.LEN_window = params['LEN_window']
        self.randomState = params['randomSeed']
        self.N_states = self.LEN_window+1
        #--------------------------------
        self.userStates = None
        self.userStatePivots = None
        self.trafficDataTrain = None
        self.trafficDataTest = None
        self.activeTrafficData = None
        self.M_train = None
        self.M_test = None
        self.M_active = None
        self.mode = None
        self.type = None

    def selectModeAndType(self, mode="train", type="markov"):
        # mode: train, test 
        # type: markov, data
        (self.mode, self.type) = (mode, type)
        if mode == "test":
            self.activeTrafficData = self.trafficDataTest  
            self.M_active = self.M_test
        else:
            self.activeTrafficData = self.trafficDataTrain
            self.M_active = self.M_train
        self.userStates = np.random.randint(0, self.N_states, (self.N_user, ))
        self.userStatePivots = np.random.randint(0, len(self.activeTrafficData)-1, (self.N_user, ))

    def updateTraffic(self):
        if self.type == "markov":
            for i in range(self.N_user):
                self.userStates[i] = generate_next_state(self.M_active, self.userStates[i]) 
        elif self.type == "data":
            for i in range(self.N_user):
                self.userStates[i] = self.activeTrafficData[self.userStatePivots[i]]
                self.userStatePivots[i] = (self.userStatePivots[i] + 1) % len(self.activeTrafficData)
        else:
            raise ValueError(f"Invalid mode or type: {self.mode}, {self.type}")
        return self.userStates

    def registerDataset(self, trafficData, train_ratio=0.7):
        if self.trafficDataTrain is None:
            self.trafficDataTrain = trafficData[0:int(len(trafficData)*train_ratio)].astype(int)
            self.trafficDataTest = trafficData[int(len(trafficData)*train_ratio):].astype(int)
        else:
            self.trafficDataTrain = np.concatenate((self.trafficDataTrain, trafficData[0:int(len(trafficData)*train_ratio)].astype(int)))
            self.trafficDataTest = np.concatenate((self.trafficDataTest, trafficData[int(len(trafficData)*train_ratio):].astype(int)))
        self.M_train = compute_markov_transition_matrix(self.trafficDataTrain, self.N_states)
        self.M_test = compute_markov_transition_matrix(self.trafficDataTest, self.N_states)
        self.selectModeAndType(mode="train", type="data")

    def getM(self, mode="train"):
        if mode == "train":
            return self.M_train
        elif mode == "test":
            return self.M_test
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    def reset(self):
        self.userStates = np.random.randint(0, self.N_states, (self.N_user, ))
        self.userStatePivots = np.random.randint(0, len(self.activeTrafficData)-1, (self.N_user, ))

def generate_random_transition_matrix(N, alpha=None, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    transition_matrix = np.zeros((N, N))

    if alpha is None:
        alpha = 1.0

    if np.isscalar(alpha):
        alpha = np.full(N, alpha)
    
    for i in range(N):
        transition_matrix[i, :] = np.random.dirichlet(alpha)
    
    return transition_matrix


def generate_next_state(transition_matrix, current_state, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Validate inputs
    N = transition_matrix.shape[0]
    if transition_matrix.shape != (N, N):
        raise ValueError("Transition matrix must be square")
    
    if current_state < 0 or current_state >= N:
        raise ValueError(f"Current state {current_state} is out of bounds [0, {N-1}]")
    
    # Get transition probabilities for current state
    transition_probs = transition_matrix[current_state, :]
    
    # Check if probabilities sum to 1 (with small tolerance for floating point errors)
    if not np.isclose(np.sum(transition_probs), 1.0, atol=1e-10):
        raise ValueError(f"Transition probabilities for state {current_state} do not sum to 1")
    
    # Generate next state using multinomial distribution
    next_state = np.random.choice(N, p=transition_probs)
    
    return next_state

def compute_markov_transition_matrix(dataset, N):
    """
    Compute Markov transition matrix from a dataset of states.
    
    Args:
        dataset: Array of state indices in [0, N-1]
        N: Number of states (states are 0, 1, ..., N-1)
        
    Returns:
        transition_matrix: NxN matrix where transition_matrix[i,j] = P(next_state=j | current_state=i)
    """
    # Initialize transition matrix
    transition_matrix = np.zeros((N, N))
    
    # Count transitions
    for i in range(len(dataset) - 1):
        current_state = dataset[i]
        next_state = dataset[i + 1]
        transition_matrix[current_state, next_state] += 1
    
    # Normalize rows to get probabilities
    row_sums = transition_matrix.sum(axis=1)
    # Avoid division by zero by adding small epsilon
    row_sums = np.where(row_sums == 0, 1, row_sums)
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]
    
    return transition_matrix
