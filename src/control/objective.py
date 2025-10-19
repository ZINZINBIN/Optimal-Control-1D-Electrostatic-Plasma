import numpy as np
from typing import Optional
from scipy.special import rel_entr
from src.env.util import compute_E

eps = 1e-12

def estimate_f(state:np.ndarray, N_mesh:int, L:float, vmin:float, vmax:float, n0:float):
    N = state.shape[0] // 2
    dx = L / N_mesh
    dv = (vmax-vmin) / N_mesh
    dist, _, _ = np.histogram2d(state[:N].ravel(), state[N:].ravel(), bins=[N_mesh, N_mesh], density=False, range = np.array([[0,L],[vmin,vmax]]))
    dist *= n0 / dx / dv / N
    return dist

def estimate_KL_divergence(f:np.ndarray, feq:np.ndarray, dx:float = 0.1, dv:float = 0.04):
    kl_div = np.sum(rel_entr(f, feq + eps)) * dx * dv
    return kl_div

def estimate_electric_energy(state:np.ndarray, E_external:Optional[np.ndarray], N_mesh:int, L:float, n0:float):
    N = state.shape[0] // 2
    x = state[:N]
    dx = L / N_mesh
    _, E_mesh = compute_E(x, dx, N_mesh, n0, L, N, None, None, False, "CIC")
    
    if E_external is not None:
        E_total = E_mesh + E_external
    else:
        E_total = E_mesh
    
    PE = 0.5 * np.sum(E_total * E_total) * dx
    
    # Rescale by number of particles to get total energy
    # PE *= N / L
    return PE