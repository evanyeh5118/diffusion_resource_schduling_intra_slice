import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#from EnvLibs.Helpers import optimal_threshold_binning_uniform_arr, compute_stationary_distribution

class TrafficGenerator:
    def __init__(self, params):
        self.N_user = params['N_user']
        self.LEN_window = params['LEN_window']
        self.randomState = params['randomSeed']
        self.N_states = self.LEN_window+1
        #--------------------------------
        self.userStates = None
        self.userStatePredicted = None
        self.userStatePivots = None
        (self.trafficDataTrain_actual, self.trafficDataTest_actual) = (None, None)
        (self.trafficDataTrain_predicted, self.trafficDataTest_predicted) = (None, None)
        self.activeTrafficData_actual = None
        self.activeTrafficData_predicted = None
        self.M_train = None
        self.M_test = None
        self.M_active = None
        self.mode = None
        self.type = None
        #------------------------------------------
        self.N_aggregation = params['N_aggregation']
        self.thresholds = None
        self.aggregationMap = None

    def selectModeAndType(self, mode="train", type="markov"):
        # mode: train, test 
        # type: markov, data
        (self.mode, self.type) = (mode, type)
        if mode == "test":
            self.activeTrafficData_actual = self.trafficDataTest_actual.astype(int)
            self.activeTrafficData_predicted = self.trafficDataTest_predicted.astype(int)
            self.M_active = self.M_test
        else:
            self.activeTrafficData_actual = self.trafficDataTrain_actual.astype(int)
            self.activeTrafficData_predicted = self.trafficDataTrain_predicted.astype(int)
            self.M_active = self.M_train
        self.reset()

    def getUserStates(self):
        return self.userStates, self.userStatePredicted

    def updateTraffic(self):
        if self.type == "markov":
            for i in range(self.N_user):
                self.userStates[i] = generate_next_state(self.M_active, self.userStates[i]) 
                self.userStatePredicted[i] = self.userStates[i]
        elif self.type == "data":
            for i in range(self.N_user):
                self.userStatePivots[i] = (self.userStatePivots[i] + 1) % len(self.activeTrafficData_actual)
                self.userStates[i] = self.activeTrafficData_actual[self.userStatePivots[i]]
                self.userStatePredicted[i] = self.activeTrafficData_predicted[self.userStatePivots[i]]
        else:
            raise ValueError(f"Invalid mode or type: {self.mode}, {self.type}")
 
    def registerDataset(self, 
                        trafficDataTrain_actual, trafficDataTest_actual,
                        trafficDataTrain_predicted=None, trafficDataTest_predicted=None):
        if self.trafficDataTrain_actual is None:
            self.trafficDataTrain_actual = trafficDataTrain_actual.astype(int)
            self.trafficDataTest_actual = trafficDataTest_actual.astype(int)
        else:
            self.trafficDataTrain_actual = np.concatenate((self.trafficDataTrain_actual, trafficDataTrain_actual.astype(int)))
            self.trafficDataTest_actual = np.concatenate((self.trafficDataTest_actual, trafficDataTest_actual.astype(int)))
        if trafficDataTrain_predicted is not None and self.trafficDataTrain_predicted is None:
            self.trafficDataTrain_predicted = trafficDataTrain_predicted.astype(int)
            self.trafficDataTest_predicted = trafficDataTest_predicted.astype(int)
        else:
            self.trafficDataTrain_predicted = np.concatenate((self.trafficDataTrain_predicted, trafficDataTrain_predicted.astype(int)))
            self.trafficDataTest_predicted = np.concatenate((self.trafficDataTest_predicted, trafficDataTest_predicted.astype(int)))
        self.M_train = compute_markov_transition_matrix(self.trafficDataTrain_actual, self.N_states)
        self.M_test = compute_markov_transition_matrix(self.trafficDataTest_actual, self.N_states)
        self.selectModeAndType(mode="train", type="data")

    def getM(self, mode="train"):
        if mode == "train":
            return self.M_train
        elif mode == "test":
            return self.M_test
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    def reset(self):
        self.userStates = np.zeros((self.N_user, ))
        self.userStatePredicted = np.zeros((self.N_user, ))
        self.userStatePivots = np.random.randint(0, len(self.activeTrafficData_actual)-1, (self.N_user, ))

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
