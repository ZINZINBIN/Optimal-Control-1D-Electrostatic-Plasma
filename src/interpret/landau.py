import numpy as np
from src.env.util import compute_E, generate_grad, generate_laplacian
from sklearn.linear_model import LinearRegression

def compute_bounce_time(perturbed_amplitude:float):
    '''
        Bounce time (in normalized unit)
        Tb = 2 * pi / sqrt(perturbed_amplitude)
        
        If the simulation time is less than Tb, the linear Landau damping is valid.
        If the simulation time is greater than Tb, the nonlinear effect (particle trapping) becomes significant.
        The problem is inherently nonlinear, irrespective of the initial perturbation amplitude and wavenumber.
    '''
    return 1.0 / np.sqrt(perturbed_amplitude)

def compute_numerical_entropy(n0:float, L: float, dx: float, N_mesh: float, vmin:float, vmax:float, dv:float, snapshot:np.ndarray):
    
    N = snapshot.shape[0] // 2
    Nv_mesh = int(vmax - vmin / dv)
    
    hist, _, _ = np.histogram2d(snapshot[:N].ravel(), snapshot[N:].ravel(), bins = [N_mesh, Nv_mesh], range = np.array([[0,L],[vmin,vmax]]))
    hist *= n0 / dx / dv / N
    
    mask = hist != 0
    S = (-1) * (hist[mask] * np.log(hist[mask])).sum() * dx * dv
    return S

def compute_linear_damping_rate_analytic(k:float, v_th:float, n0:float):
    """
    Linear Landau damping rate for Langmuir wave (valid only for k * lamda_de << 1)
    Parameters:
            k: wave number
            v_th: thermal velocity
            n0: electron density
    Output:
        gamma: linear damping rate        
    """
    w_pe = np.sqrt(4 * np.pi * n0)
    lamda_de = v_th / w_pe

    gamma = np.exp((-1) * 1 / (2 * (k * lamda_de)**2)) / ((k * lamda_de)**3) * np.sqrt(np.pi / 8) * w_pe
    return gamma

def compute_linear_damping_rate(tmax: float, n0:float, L: float, dx: float, N_mesh: float, snapshot: np.ndarray):
    """
    Linear Landau damping rate with given snapshot data by log(E^2) = 2 * gamma * t + C
    Parameters:
            tmax: timelength
            n0: electron density
            L: physical length scale
            dx: mesh size
            N_mesh: number of mesh
            snapshot: snapshot data (2N x Nt)
    Output:
        gamma: linear damping rate (numerical)
    """

    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    ts = np.linspace(0, tmax, Nt)

    # Gradient and Laplacian
    G = generate_grad(L, N_mesh)
    Lap = generate_laplacian(L, N_mesh)

    E_mesh_t = [compute_E(snapshot[:, i].reshape(-1, 1), dx, N_mesh, n0, L, N, G, Lap)[1] for i in range(Nt)]
    E2_t = np.array([np.sum(E_mesh_t[i].ravel() ** 2) * dx for i in range(Nt)])
    log_E2_t = np.log(E2_t)

    lr = LinearRegression()
    lr.fit(ts.reshape(-1,1), log_E2_t.reshape(-1,1))

    gamma = 0.5 * lr.coef_[0]
    return gamma.item()