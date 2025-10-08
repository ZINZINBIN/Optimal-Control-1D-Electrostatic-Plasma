import numpy as np
from typing import Union, Optional
from numba import jit

@jit(nopython=True)
def Gaussian_Elimination_TriDiagonal(A_origin:np.ndarray, B_origin:np.ndarray):
    A = A_origin.copy()
    B = B_origin.copy().reshape(-1)
    
    n = A.shape[0]
            
    X = np.zeros_like(B)
    
    # Forward Elimination
    for idx_j in range(1,n):
        A[idx_j, idx_j] = A[idx_j, idx_j] - A[idx_j, idx_j - 1] * A[idx_j-1, idx_j] / A[idx_j-1, idx_j-1]
        B[idx_j] = B[idx_j] - A[idx_j, idx_j - 1] * B[idx_j - 1] / A[idx_j-1, idx_j-1]
        A[idx_j, idx_j-1] = 0
        
    # Backward Substitution 
    X[-1] = B[-1] / A[-1,-1]
    for idx_i in range(n-2,-1,-1):
        X[idx_i] = (B[idx_i] - A[idx_i, idx_i +1] * X[idx_i + 1]) / A[idx_i, idx_i]

    return X

@jit(nopython=True)
def Gaussian_Elimination_Periodic(A: np.ndarray, B: np.ndarray, gamma:float = 5.0):
    
    N = A.shape[0]

    if A.shape[0] != A.shape[1] or B.shape[0] != N:
        raise ValueError("Matrix A must be square and B must have compatible dimensions.")

    A_new = A.copy()
    A_new[0,0] -= gamma
    A_new[-1,-1] -= A[0,-1] * A[-1,0] / gamma
    A_new[-1,0] = 0.0
    A_new[0,-1] = 0.0

    u = np.zeros(N, dtype = float)
    u[0] = gamma
    u[-1] = A[-1,0]

    v = np.zeros(N, dtype = float)
    v[0] = 1
    v[-1] = A[0,-1] / gamma

    x = Gaussian_Elimination_TriDiagonal(A_new, B)
    q = Gaussian_Elimination_TriDiagonal(A_new, u)
    
    x -= q * np.dot(v,x) / (1 + np.dot(v,q))
    return x