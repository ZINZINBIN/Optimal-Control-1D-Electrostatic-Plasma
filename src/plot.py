import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional
from scipy.stats import gaussian_kde
from src.env.util import compute_E, generate_grad, generate_laplacian
from src.interpret.spectrum import compute_E_k_spectrum

def plot_x_dist_snapshot(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    N_mesh:Optional[int] = 500,
):

    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)

    else:
        filepath = None

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), facecolor="white", dpi=120)

    # snapshot info
    N = snapshot.shape[0] // 2

    # velocity array for plot
    x = np.linspace(xmin, xmax, N_mesh)
    density_od = gaussian_kde(snapshot[:N])

    ax.cla()
    ax.plot(x, density_od(x))
    ax.set_xlabel("x")
    ax.set_ylabel(r"$f(x,\cdot)$")
    ax.set_xlim([xmin, xmax])
    ax.legend(loc="upper right")
    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, ax

def plot_v_dist_snapshot(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
    N_mesh:Optional[int] = 500,
):

    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)

    else:
        filepath = None

    fig, ax = plt.subplots(1, 1, figsize=(5, 4), facecolor="white", dpi=120)

    # snapshot info
    N = snapshot.shape[0] // 2

    # velocity array for plot
    v = np.linspace(vmin, vmax, N_mesh)
    density_od = gaussian_kde(snapshot[N:])

    ax.cla()
    ax.plot(v, density_od(v))
    ax.set_xlabel("v")
    ax.set_ylabel(r"$f(\cdot,v)$")
    ax.set_xlim([vmin, vmax])
    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)
        
    return fig, ax

def plot_dist_snapshot(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
    N_mesh:Optional[int] = 500,
):

    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)

    else:
        filepath = None

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), facecolor="white", dpi=120)

    # snapshot info
    N = snapshot.shape[0] // 2

    # Scope
    extent = [xmin, xmax, vmin, vmax]
    dist, _, _ = np.histogram2d(snapshot[:N].ravel(), snapshot[N:].ravel(), bins=N_mesh, density=False)
    im = ax.imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_title(r"$f(x,v)$")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, ax

def plot_two_stream_snapshot(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
):
    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)
    else:
        filepath = None

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nh = N // 2

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), facecolor="white", dpi=120)
    
    ax.cla()
    ax.scatter(snapshot[0:Nh], snapshot[N:N+Nh], s=0.3, color="blue", alpha=0.5)
    ax.scatter(snapshot[Nh:N], snapshot[N+Nh:], s=0.3, color="red", alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.axis([xmin, xmax, vmin, vmax])
    ax.set_title("Phase space")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, ax

def plot_bump_on_tail_snapshot(
    snapshot: np.ndarray,
    save_dir: Optional[str],
    filename: Optional[str],
    xmin: Optional[float] = 0.0,
    xmax: Optional[float] = 50.0,
    vmin: Optional[float] = -10.0,
    vmax: Optional[float] = 10.0,
    high_electron_indice: Optional[np.ndarray] = None,
):
    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)
    else:
        filepath = None

    # Snapshot info
    N = snapshot.shape[0] // 2
    
    if high_electron_indice is not None:
        low_electron_indice = np.array([i for i in range(0,N) if i not in high_electron_indice])
    else:
        low_electron_indice = np.arange(0,N)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), facecolor="white", dpi=120)

    ax.cla()
    ax.scatter(snapshot[low_electron_indice], snapshot[low_electron_indice + N], s=0.3, color="blue", alpha=0.5)
    
    if high_electron_indice is not None:
        ax.scatter(snapshot[high_electron_indice], snapshot[high_electron_indice+N], s=0.3, color="red", alpha=0.5)
        
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.axis([xmin, xmax, vmin, vmax])
    ax.set_title("Phase space")
    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, ax


def plot_x_dist_evolution(
    snapshot: np.ndarray,
    save_dir: Optional[str],
    filename: Optional[str],
    xmin: Optional[float] = 0.0,
    xmax: Optional[float] = 50.0,
    N_mesh: Optional[int] = 500,
):

    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)

    else:
        filepath = None

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), facecolor="white", dpi=120, sharey=True)
    axes = axes.ravel()

    # snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    # x array for plot
    x = np.linspace(xmin, xmax, N_mesh)

    # t = 0
    density_od = gaussian_kde(snapshot[:N,0])

    axes[0].cla()
    axes[0].plot(x, density_od(x))
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$f(x,\cdot)$")
    axes[0].set_xlim([xmin, xmax])
    axes[0].set_title(r"$t=0$")

    # t = tmax/2
    density_od = gaussian_kde(snapshot[:N, Nt//2])

    axes[1].cla()
    axes[1].plot(x, density_od(x))
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$f(x,\cdot)$")
    axes[1].set_xlim([xmin, xmax])
    axes[1].set_title(r"$t=t_{max}/2$")

    # t = tmax
    density_od = gaussian_kde(snapshot[:N,-1])

    axes[2].cla()
    axes[2].plot(x, density_od(x))
    axes[2].set_xlabel("x")
    axes[2].set_ylabel(r"$f(x,\cdot)$")
    axes[2].set_xlim([xmin, xmax])
    axes[2].set_title(r"$t=t_{max}$")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, axes


def plot_v_dist_evolution(
    snapshot: np.ndarray,
    save_dir: Optional[str],
    filename: Optional[str],
    vmin: Optional[float] = -10.0,
    vmax: Optional[float] = 10.0,
    N_mesh: Optional[int] = 500,
):

    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)

    else:
        filepath = None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=120, sharey = True)
    axes = axes.ravel()

    # snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    # velocity array for plot
    v = np.linspace(vmin, vmax, N_mesh)

    # t = 0
    density_od = gaussian_kde(snapshot[N:,0])

    axes[0].cla()
    axes[0].plot(v, density_od(v))
    axes[0].set_xlabel("v")
    axes[0].set_ylabel(r"$f(\cdot,v)$")
    axes[0].set_xlim([vmin, vmax])
    axes[0].set_title(r"$t=0$")

    # t = tmax/2
    density_od = gaussian_kde(snapshot[N:, Nt//2])

    axes[1].cla()
    axes[1].plot(v, density_od(v))
    axes[1].set_xlabel("v")
    axes[1].set_ylabel(r"$f(\cdot,v)$")
    axes[1].set_xlim([vmin, vmax])
    axes[1].set_title(r"$t=t_{max}/2$")

    # t = tmax
    density_od = gaussian_kde(snapshot[N:,-1])

    axes[2].cla()
    axes[2].plot(v, density_od(v))
    axes[2].set_xlabel("v")
    axes[2].set_ylabel(r"$f(\cdot,v)$")
    axes[2].set_xlim([vmin, vmax])
    axes[2].set_title(r"$t=t_{max}$")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, axes


def plot_dist_evolution(
    snapshot: np.ndarray,
    save_dir: Optional[str],
    filename: Optional[str],
    xmin: Optional[float] = 0.0,
    xmax: Optional[float] = 50.0,
    vmin: Optional[float] = -10.0,
    vmax: Optional[float] = 10.0,
    N_mesh: Optional[int] = 500,
):
    # check directory
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, filename)
    else:
        filepath = None

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    # Scope
    extent = [xmin, xmax, vmin, vmax]

    fig, axes = plt.subplots(1, 3, figsize=(15, 3), facecolor="white", dpi=120, constrained_layout=True)
    axes = axes.ravel()

    axes[0].cla()
    dist, _, _ = np.histogram2d(snapshot[:N,0].ravel(), snapshot[N:,0].ravel(), bins=N_mesh, density=False)
    axes[0].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("v")
    axes[0].axis([xmin, xmax, vmin, vmax])
    axes[0].set_title(r"$t=0$")

    axes[1].cla()
    dist, _, _ = np.histogram2d(snapshot[:N,Nt//2].ravel(), snapshot[N:,Nt//2].ravel(), bins=N_mesh, density=False)
    axes[1].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].axis([xmin, xmax, vmin, vmax])
    axes[1].set_title(r"$t=t_{max}/2$")

    axes[2].cla()
    dist, _, _ = np.histogram2d(snapshot[:N,-1].ravel(), snapshot[N:,-1].ravel(), bins=N_mesh, density=False)
    im = axes[2].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")

    axes[2].set_xlabel("x")
    axes[2].set_ylabel("v")
    axes[2].axis([xmin, xmax, vmin, vmax])
    axes[2].set_title(r"$t=t_{max}$")

    fig.colorbar(im, ax=axes.ravel().tolist())

    if filepath is not None:
        plt.savefig(filepath, dpi=120)
        
    return fig, axes


def plot_two_stream_evolution(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
):
    # check directory
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, filename)
    else:
        filepath = None

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nh = N // 2
    Nt = snapshot.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), facecolor="white", dpi=120)

    axes = axes.ravel()
    
    axes[0].cla()
    axes[0].scatter(snapshot[0:Nh,0], snapshot[N:N+Nh,0], s=0.3, color="blue", alpha=0.5)
    axes[0].scatter(snapshot[Nh:N,0], snapshot[N+Nh:,0], s=0.3, color="red", alpha=0.5)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("v")
    axes[0].axis([xmin, xmax, vmin, vmax])
    axes[0].set_title(r"$t=0$")

    axes[1].cla()
    axes[1].scatter(snapshot[0:Nh,Nt//2], snapshot[N:N+Nh,Nt//2], s=0.3, color="blue", alpha=0.5)
    axes[1].scatter(snapshot[Nh:N,Nt//2], snapshot[N+Nh:,Nt//2], s=0.3, color="red", alpha=0.5)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].axis([xmin, xmax, vmin, vmax])
    axes[1].set_title(r"$t=t_{max}/2$")

    axes[2].cla()
    axes[2].scatter(snapshot[0:Nh,-1], snapshot[N:N+Nh,-1], s=0.3, color="blue", alpha=0.5)
    axes[2].scatter(snapshot[Nh:N,-1], snapshot[N+Nh:,-1], s=0.3, color="red", alpha=0.5)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("v")
    axes[2].axis([xmin, xmax, vmin, vmax])
    axes[2].set_title(r"$t=t_{max}$")

    fig.tight_layout()
    
    if filepath is not None:
        plt.savefig(filepath, dpi=120)
        
    return fig, axes

def plot_bump_on_tail_evolution(
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
    vmin:Optional[float] = -10.0,
    vmax:Optional[float] = 10.0,
    high_electron_indice:Optional[np.ndarray] = None,
):
    # check directory
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, filename)
    else:
        filepath = None
    
    # Snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]
    
    # Low electron indexing
    if high_electron_indice is not None:
        low_electron_indice = np.array([i for i in range(0,N) if i not in high_electron_indice])
    else:
        low_electron_indice = np.arange(0,N)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), facecolor="white", dpi=120)
    
    axes = axes.ravel()
    
    axes[0].cla()
    axes[0].scatter(snapshot[low_electron_indice,0], snapshot[low_electron_indice+N,0], s=0.3, color="blue", alpha=0.5)
    
    if high_electron_indice is not None:
        axes[0].scatter(snapshot[high_electron_indice,0], snapshot[high_electron_indice+N,0], s=0.3, color="red", alpha=0.5)
    
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("v")
    axes[0].axis([xmin, xmax, vmin, vmax])
    axes[0].set_title(r"$t=0$")
    
    axes[1].cla() 
    axes[1].scatter(snapshot[low_electron_indice,Nt//2], snapshot[low_electron_indice+N,Nt//2], s=0.3, color="blue", alpha=0.5)
    
    if high_electron_indice is not None:
        axes[1].scatter(snapshot[high_electron_indice,Nt//2], snapshot[high_electron_indice+N,Nt//2], s=0.3, color="red", alpha=0.5)
    
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].axis([xmin, xmax, vmin, vmax])
    axes[1].set_title(r"$t=t_{max}/2$")

    axes[2].cla()
    axes[2].scatter(snapshot[low_electron_indice,-1], snapshot[low_electron_indice+N,-1], s=0.3, color="blue", alpha=0.5)
    
    if high_electron_indice is not None:
        axes[2].scatter(snapshot[high_electron_indice,-1], snapshot[high_electron_indice+N,-1], s=0.3, color="red", alpha=0.5)
    
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("v")
    axes[2].axis([xmin, xmax, vmin, vmax])
    axes[2].set_title(r"$t=t_{max}$")

    fig.tight_layout()
    
    if filepath is not None:
        plt.savefig(filepath, dpi=120)
        
    return fig, axes

# Plot for analysis: E-field (log scale in time, profile), density (profile), and potential (profile)
def plot_log_E(
    tmax:float,
    L:float,
    dx:float,
    N_mesh:float,
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
):
    # check directory
    if save_dir is not None:
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        filepath = os.path.join(save_dir, filename)
        
    else:
        filepath = None

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    ts = np.linspace(0,tmax,Nt)

    # Gradient and Laplacian
    G = generate_grad(L,N_mesh)
    Lap = generate_laplacian(L,N_mesh)

    E_real_list = [compute_E(snapshot[:,i].reshape(-1,1), dx, N_mesh, 1.0, L, N, G, Lap)[1] for i in range(Nt)]
    E2_real = np.array([np.mean(E_real_list[i].ravel() ** 2) for i in range(Nt)])

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), facecolor="white", dpi=120)

    ax.plot(ts, E2_real)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"$\log <E^2>$")
    ax.set_yscale("log")
    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, ax

def plot_E_k_spectrum(
    tmax:float,
    L:float,
    dx:float,
    N_mesh:float,
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
):
    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)

    else:
        filepath = None

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    ts = np.linspace(0, tmax, Nt)

    ks_real, Ek_t_spectrum_real = compute_E_k_spectrum(1.0, L, dx, N_mesh, snapshot)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3), facecolor="white", dpi=120)
    
    ax.imshow(Ek_t_spectrum_real, extent=[0,tmax,ks_real[0], ks_real[-1]], aspect='auto', origin='lower')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$k$")
    ax.set_title(r"$E_k$")
    ax.set_ylim([0, 1.0])
    ax.grid(True)

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, ax

def plot_E_k_over_time(
    tmax:float,
    L:float,
    dx:float,
    N_mesh:float,
    max_mode:int,
    snapshot:np.ndarray,
    save_dir:Optional[str],
    filename:Optional[str],
):
    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)

    else:
        filepath = None

    # Snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    ts = np.linspace(0, tmax, Nt)

    ks_real, Ek_t_spectrum_real = compute_E_k_spectrum(1.0, L, dx, N_mesh, snapshot)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), facecolor="white", dpi=120)
    
    for i in range(1, max_mode + 1):
        ax.plot(ts, Ek_t_spectrum_real[i,:].ravel(), label = r"$n={}$".format(i))
    
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$E_k$")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    return fig, ax