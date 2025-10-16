import numpy as np
from src.env.util import compute_E, generate_grad, generate_laplacian

def compute_E_k_spectrum(n0:float, L: float, dx: float, N_mesh: float, snapshot: np.ndarray, return_abs:bool = True):

    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    # Gradient and Laplacian
    G = generate_grad(L, N_mesh)
    Lap = generate_laplacian(L, N_mesh)

    E_mesh_t = [compute_E(snapshot[:, i].reshape(-1, 1), dx, N_mesh, n0, L, N, G, Lap)[1] for i in range(Nt)]
    E_mesh_t_arr = np.concatenate(E_mesh_t, axis=1) # (N_mesh, Nt)

    Ek_t = np.fft.fft(E_mesh_t_arr, axis=0) / N_mesh * 2.0
    ks = np.fft.fftfreq(N_mesh, d=dx) * 2.0 * np.pi
    
    if return_abs:
        Ek_t_spectrum = np.abs(Ek_t)
    else:
        Ek_t_spectrum = Ek_t

    # masking for positive value
    mask = ks >= 0
    Ek_t_spectrum = Ek_t_spectrum[mask, :]
    ks = ks[mask]
    return ks, Ek_t_spectrum