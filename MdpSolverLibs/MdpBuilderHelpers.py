import numpy as np
from typing import Dict, List, Tuple, Union
import math

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

def compute_aggregated_transition_matrix(P: np.ndarray, C: np.ndarray) -> np.ndarray:
    # Ensure input arrays are NumPy arrays
    P = np.asarray(P)
    C = np.asarray(C)

    # Compute stationary distribution of P
    pi = compute_stationary_distribution(P)

    # Construct diagonal matrix Pi
    Pi = np.diag(pi)

    # Compute numerator and denominator
    numerator = C.T @ Pi @ P @ C
    denominator = C.T @ Pi @ C

    # Inverse of denominator
    denom_inv = np.linalg.inv(denominator)

    # Aggregated transition matrix
    P_hat = denom_inv @ numerator
    return P_hat

def vector_to_mapping_matrix(v):
    v = np.asarray(v)
    X_size = len(v)
    Y_size = np.max(v) + 1  # Assumes j ∈ {0, ..., max(v)}

    C = np.zeros((X_size, Y_size), dtype=int)
    C[np.arange(X_size), v] = 1
    return C

def compute_joint_transition_matrix(M: np.ndarray, N: int) -> np.ndarray:
    """
    Compute the joint transition matrix M_y for N independent Markov processes.
    
    Given a transition matrix M for a single process x, compute the transition matrix
    for the vector y = (x_1, ..., x_N) where each x_i follows the same Markov process M.
    
    For independent processes, the joint transition matrix is the N-fold Kronecker product
    of M with itself: M_y = M ⊗ M ⊗ ... ⊗ M (N times)
    
    Parameters:
    -----------
    M : np.ndarray
        Transition matrix for a single Markov process (S x S matrix)
    N : int
        Number of independent processes
        
    Returns:
    --------
    np.ndarray
        Joint transition matrix M_y of shape (S^N x S^N)
    """
    M = np.asarray(M)
    
    # Validate input
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    if N <= 0:
        raise ValueError("N must be a positive integer")
    
    S = M.shape[0]  # Number of states for single process
    
    # For N=1, return M directly
    if N == 1:
        return M
    
    # For N>1, compute N-fold Kronecker product
    M_y = M.copy()
    for _ in range(N - 1):
        M_y = np.kron(M_y, M)
    
    return M_y


def index_to_tuple(k, N, L):
    """
    Convert flat index k to tuple (x_0, ..., x_{L-1})
    where each x_i ∈ [0, N-1]
    """
    x = []
    for _ in range(L):
        x.append(k % N)
        k //= N
    return list(reversed(x))


def tuple_to_index(x, N):
    """
    Convert tuple x = (x_0, ..., x_{L-1}) to flat index
    assuming each x_i ∈ [0, N-1]
    """
    k = 0
    for xi in x:
        k = k * N + xi
    return k

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