import numpy as np
from typing import Callable
from src.util import check_invalid_value

# -----------------------------
# Baselines (explicit, non-symplectic)
# -----------------------------
def forward_euler(eta: np.ndarray, grad_func: Callable, dt: float):
    grad_eta = grad_func(eta)
    return eta + dt * grad_eta

def explicit_midpoint(eta: np.ndarray, grad_func: Callable, dt: float):
    grad_eta = grad_func(eta)
    grad_eta_half = grad_func(eta + 0.5 * dt * grad_eta)
    eta_f = eta + dt * grad_eta_half
    return eta_f

# -----------------------------
# Symplectic (general vector field f = (dq, dp))
# eta = [q, p] with len(q)=len(p)=n; f returns [dq, dp]
# -----------------------------
def _symplectic_forward_step(eta: np.ndarray, grad_func: Callable, dt: float, c:float, d:float) -> np.ndarray:
    n = len(eta) // 2

    q = eta[:n,:].copy()
    p = eta[n:,:].copy()

    eta_f = np.empty_like(eta)

    # update p
    if d != 0.0:
        eta_f[n:,:] = p + d * grad_func(eta)[n:,:] * dt
    else:
        eta_f[n:,:] = p

    if c != 0.0:
        # eta(t + 1/2)
        eta_m = eta_f.copy()
        eta_m[:n,:] = q

        # update q
        eta_f[:n,:] = q + c * grad_func(eta_m)[:n,:] * dt
        
    else:
        eta_f[:n,:] = q

    return eta_f

# Symplectic integration: 1st order
def symplectic_euler(eta: np.ndarray, grad_func: Callable, dt: float):
    return _symplectic_forward_step(eta, grad_func, dt, c=1.0, d=1.0)

# Symplectic integration: 2nd order (Stromer-Verlet)
def verlet(eta: np.ndarray, grad_func: Callable, dt: float):    
    eta = _symplectic_forward_step(eta, grad_func, dt, 1.0, 0.5)
    eta = _symplectic_forward_step(eta, grad_func, dt, 0.0, 0.5)
    return eta

# Ruth's 4th order symplectic integration
def symplectic_4th_order(eta: np.ndarray, grad_func: Callable, dt: float):
    
    phi = 2 ** (1 / 3)
    c1 = c4 = 1 / (2 * (2 - phi))
    c2 = c3 = (1 - phi) / (2 * (2 - phi))
    d1 = d3 = 1 / (2 - phi)
    d2 = (-1) * phi / (2 - phi)
    d4 = 0

    eta = _symplectic_forward_step(eta, grad_func, dt, c1, d1)
    eta = _symplectic_forward_step(eta, grad_func, dt, c2, d2)
    eta = _symplectic_forward_step(eta, grad_func, dt, c3, d3)
    eta = _symplectic_forward_step(eta, grad_func, dt, c4, d4)
    return eta

# Symplectic integration: Implicit 2nd order method
def implicit_midpoint(eta: np.ndarray, grad_func: Callable, dt: float, n_epochs:int = 100, eps:float = 1e-12, alpha:float=0.5):

    dx = verlet(eta, grad_func, dt) - eta

    def _g(dx:np.ndarray):
        return grad_func(0.5 * dx + 1.0 * eta) * dt

    is_converged = False

    for _ in range(n_epochs):
        dx_prev = dx.copy()
        dx = (1-alpha) * dx + alpha * _g(dx)

        if np.linalg.norm(dx - dx_prev) < eps:
            is_converged = True
            break

        if check_invalid_value(dx):
            break

    if is_converged:
        eta_next = eta + dx
    else:
        eta_next = verlet(eta, grad_func, dt)

    return eta_next