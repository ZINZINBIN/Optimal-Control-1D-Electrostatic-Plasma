import numpy as np
from numba import jit
from typing import Optional, Literal
from src.env.solve import Gaussian_Elimination_Periodic
from src.env.interpolate import CIC, TSC

@jit(nopython=True)
def generate_grad(L:float, N_mesh:int):
    
    grad = np.zeros((N_mesh, N_mesh))
    
    dx = L / N_mesh

    for idx_i in range(0, N_mesh):
        if idx_i > 0:
            grad[idx_i, idx_i - 1] = -1.0

        if idx_i < N_mesh - 1:
            grad[idx_i, idx_i + 1] = 1.0

    # periodic condition
    grad[0, N_mesh - 1] = -1.0
    grad[N_mesh - 1, 0] = 1.0
    grad /= 2 * dx
    
    return grad

@jit(nopython=True)
def generate_laplacian(L:float, N_mesh:int):
    dx = L / N_mesh
    laplacian = np.zeros((N_mesh, N_mesh))

    for idx_i in range(0, N_mesh):
        if idx_i > 0:
            laplacian[idx_i, idx_i - 1] = 1.0

        if idx_i < N_mesh - 1:
            laplacian[idx_i, idx_i + 1] = 1.0

        laplacian[idx_i, idx_i] = -2.0

    laplacian[0, N_mesh - 1] = 1.0
    laplacian[N_mesh - 1, 0] = 1.0
    laplacian /= dx**2

    return laplacian

def compute_n(u:np.ndarray, dx:float, N_mesh:int, n0:float, L:float, N:int, return_all:bool=False, interpol:Literal['CIC', 'TSC'] = "CIC"):

    # Periodicity
    u[:N] = np.mod(u[:N], L)

    # Extract generalized coordinate
    x = u[:N]

    if interpol == "CIC":
        n, idx_l, idx_r, w_l, w_r = CIC(x, n0, L, N, N_mesh, dx)

        if return_all:
            return n, idx_l, idx_r, w_l, w_r
        else:
            return n

    elif interpol == "TSC":
        n, idx_l, idx_m, idx_r, w_l, w_m, w_r = TSC(x, n0, L, N, N_mesh, dx)

        if return_all:
            return n, idx_l, idx_m, idx_r, w_l, w_m, w_r
        else:
            return n


def compute_E(
    u: np.ndarray,
    dx: float,
    N_mesh: int,
    n0: float,
    L: float,
    N: int,
    grad: Optional[np.ndarray] = None,
    laplacian: Optional[np.ndarray] = None,
    return_all: bool = False,
    interpol: Literal["CIC", "TSC"] = "CIC",
    E_external: Optional[np.ndarray] = None,
):

    if grad is None:
        grad = generate_grad(L, N_mesh)

    if laplacian is None:
        laplacian = generate_laplacian(L, N_mesh)

    if interpol == "CIC":
        n, idx_l, idx_r, w_l, w_r = compute_n(u, dx, N_mesh, n0, L, N, True, interpol)

    elif interpol == "TSC":
        n, idx_l, idx_m, idx_r, w_l, w_m, w_r = compute_n(u, dx, N_mesh, n0, L, N, True, interpol)

    phi_mesh = Gaussian_Elimination_Periodic(laplacian, n - n0, gamma = 5.0).reshape(-1, 1)
    E_mesh = (-1) * np.matmul(grad, phi_mesh)
    
    if E_external is not None:
        E_mesh = E_mesh + E_external

    if interpol == "CIC":
        E = w_l * E_mesh[idx_l[:,0]] + w_r * E_mesh[idx_r[:,0]]
        phi = w_l * phi_mesh[idx_l[:, 0]] + w_r * phi_mesh[idx_r[:, 0]]

    elif interpol == "TSC":
        E = w_l * E_mesh[idx_l[:,0]] + w_m * E_mesh[idx_m[:,0]] + w_r * E_mesh[idx_r[:,0]]
        phi = w_l * phi_mesh[idx_l[:, 0]] + w_m * phi_mesh[idx_m[:,0]] + w_r * phi_mesh[idx_r[:, 0]]

    if return_all:
        return E, phi, E_mesh, phi_mesh
    else:
        return E, E_mesh


def compute_electric_energy(
    x: np.array,
    dx: float,
    N: int,
    N_mesh: int,
    n0: float,
    L: float,
    interpol: Literal["CIC", "TSC"] = "CIC",
):
    _, E_mesh = compute_E(x, dx, N_mesh, n0, L, N, None, None, False, interpol)
    PE = 0.5 * np.sum(E_mesh * E_mesh) * dx
    PE *= N / L
    return PE

def compute_hamiltonian(
    x: np.array,
    v: np.array,
    dx: float,
    N:int,
    N_mesh:int,
    n0:float = 1.0,
    L:float = 50.0,
    interpol: Literal["CIC", "TSC"] = "CIC",
):

    KE = 0.5 * np.sum(v * v)
    PE = compute_electric_energy(x, dx, N, N_mesh, n0, L, interpol)    
    H = KE + PE
    return H
