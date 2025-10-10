import numpy as np
from src.env.util import compute_E

def estimate_f(state:np.ndarray, N_mesh:int, L:float, vmin:float, vmax:float, n0:float):
    N = state.shape[0] // 2
    dx = L / N_mesh
    dv = (vmax-vmin) / N_mesh
    dist, _, _ = np.histogram2d(state[:N].ravel(), state[N:].ravel(), bins=[N_mesh, N_mesh], density=False, range = np.array([[0,L],[vmin,vmax]]))
    dist *= n0 / dx / dv / N
    return dist

def estimate_KL_divergence(f:np.ndarray, feq:np.ndarray, dx:float, dv:float):
    mask = (f != 0) & (feq != 0)
    kl_div = (f[mask] * np.log(f[mask] / feq[mask])).sum() * dx * dv
    return kl_div

def estimate_electric_energy(state:np.ndarray, E_external:np.ndarray, N_mesh:int, L:float, n0:float):
    N = state.shape[0] // 2
    x = state[:N]
    dx = L / N_mesh
    _, E_mesh = compute_E(x, dx, N_mesh, n0, L, N, None, None, False, "CIC")
    E_total = E_mesh + E_external
    
    PE = 0.5 * np.sum(E_total * E_total) * dx
    PE *= N / L
    return PE