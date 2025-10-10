import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional
from scipy.stats import gaussian_kde
from src.env.util import compute_E, generate_grad, generate_laplacian
from src.interpret.spectrum import compute_E_k_spectrum

def plot_x_dist_snapshot(
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
):

    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)

    else:
        filepath = None

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor="white", dpi=120)

    # snapshot info
    N = snapshot.shape[0] // 2

    # velocity array for plot
    x = np.linspace(xmin, xmax, 128)
    density_od = gaussian_kde(snapshot[:N])

    if prediction is not None:
        density_rd = gaussian_kde(prediction[:N])

    ax.cla()
    ax.plot(x, density_od(x), "b", label = "FOM")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$f(x,\cdot)$")
    ax.set_xlim([xmin, xmax])

    if prediction is not None:
        ax.plot(x, density_rd(x), "r", label = "ROM")

    ax.legend(loc="upper right")
    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, ax

def plot_v_dist_snapshot(
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
    save_dir:Optional[str],
    filename:Optional[str],
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

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor="white", dpi=120)

    # snapshot info
    N = snapshot.shape[0] // 2

    # velocity array for plot
    v = np.linspace(vmin, vmax, 128)
    density_od = gaussian_kde(snapshot[N:])

    if prediction is not None:
        density_rd = gaussian_kde(prediction[N:])

    ax.cla()
    ax.plot(v, density_od(v), "b", label = "FOM")
    ax.set_xlabel("v")
    ax.set_ylabel(r"$f(\cdot,v)$")
    ax.set_xlim([vmin, vmax])

    if prediction is not None:
        ax.plot(v, density_rd(v), "r", label = "ROM")

    ax.legend(loc="upper right")
    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, ax

def plot_dist_snapshot(
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
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

    if prediction is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white", dpi=120)
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5), facecolor="white", dpi=120)

    # snapshot info
    N = snapshot.shape[0] // 2

    # Scope
    extent = [xmin, xmax, vmin, vmax]

    if prediction is not None:

        dist, _, _ = np.histogram2d(snapshot[:N].ravel(), snapshot[N:].ravel(), bins=128, density=True)
        im_fom = axes[0].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("v")
        axes[0].set_title("FOM")

        fig.colorbar(im_fom, ax=axes[0])

        dist, _, _ = np.histogram2d(prediction[:N].ravel(), prediction[N:].ravel(), bins=128, density=True)
        im_rom = axes[1].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("v")
        axes[1].set_title("ROM")

        fig.colorbar(im_rom, ax=axes[1])

    else:
        dist, _, _ = np.histogram2d(snapshot[:N].ravel(), snapshot[N:].ravel(), bins=128, density=True)
        im = axes.imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
        axes.set_xlabel("x")
        axes.set_ylabel("v")
        axes.set_title("FOM")

        fig.colorbar(im, ax=axes)

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, axes

def plot_two_stream_snapshot(
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
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

    if prediction is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white", dpi=120)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4), facecolor="white", dpi=120)

    if prediction is not None:
        axes = axes.ravel()
        axes[0].cla()
        axes[0].scatter(snapshot[0:Nh], snapshot[N:N+Nh], s=0.4, color="blue", alpha=0.5)
        axes[0].scatter(snapshot[Nh:N], snapshot[N+Nh:], s=0.4, color="red", alpha=0.5)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("v")
        axes[0].axis([xmin, xmax, vmin, vmax])
        axes[0].set_title("FOM")
        
        axes[1].cla()
        axes[1].scatter(prediction[0:Nh], prediction[N:N+Nh], s=0.4, color="blue", alpha=0.5)
        axes[1].scatter(prediction[Nh:N], prediction[N+Nh:], s=0.4, color="red", alpha=0.5)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("v")
        axes[1].axis([xmin, xmax, vmin, vmax])
        axes[1].set_title("ROM")
        
    else:
        axes.cla()
        axes.scatter(snapshot[0:Nh], snapshot[N:N+Nh], s=0.4, color="blue", alpha=0.5)
        axes.scatter(snapshot[Nh:N], snapshot[N+Nh:], s=0.4, color="red", alpha=0.5)
        axes.set_xlabel("x")
        axes.set_ylabel("v")
        axes.axis([xmin, xmax, vmin, vmax])
        axes.set_title("FOM")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, axes

def plot_bump_on_tail_snapshot(
    snapshot: np.ndarray,
    prediction: Optional[np.ndarray],
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

    if prediction is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white", dpi=120)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4), facecolor="white", dpi=120)

    axes = axes.ravel()

    axes[0].cla()
    axes[0].scatter(snapshot[low_electron_indice], snapshot[low_electron_indice + N], s=0.4, color="blue", alpha=0.5)
    
    if high_electron_indice is not None:
        axes[0].scatter(snapshot[high_electron_indice], snapshot[high_electron_indice+N], s=0.4, color="red", alpha=0.5)
        
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("v")
    axes[0].axis([xmin, xmax, vmin, vmax])
    axes[0].set_title("FOM")

    if prediction is not None:
        
        axes[1].cla()
        axes[1].scatter(prediction[low_electron_indice], prediction[low_electron_indice+N], s=0.4, color="blue", alpha=0.5)
        
        if high_electron_indice is not None:
            axes[1].scatter(prediction[high_electron_indice], prediction[high_electron_indice+N], s=0.4, color="red", alpha=0.5)
        
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("v")
        axes[1].axis([xmin, xmax, vmin, vmax])
        axes[1].set_title("ROM")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, axes

def plot_x_dist_evolution(
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
    save_dir:Optional[str],
    filename:Optional[str],
    xmin:Optional[float] = 0.0,
    xmax:Optional[float] = 50.0,
):

    # check directory
    if save_dir is not None:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, filename)

    else:
        filepath = None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=120, sharey=True)
    axes = axes.ravel()

    # snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    # x array for plot
    x = np.linspace(xmin, xmax, 128)

    # t = 0
    density_od = gaussian_kde(snapshot[:N,0])

    if prediction is not None:
        density_rd = gaussian_kde(prediction[:N,0])

    axes[0].cla()
    axes[0].plot(x, density_od(x), "b", label = "FOM")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$f(x,\cdot)$")
    axes[0].set_xlim([xmin, xmax])
    axes[0].set_title(r"$t=0$")

    if prediction is not None:
        axes[0].plot(x, density_rd(x), "r", label = "ROM")

    axes[0].legend(loc="upper right")

    # t = tmax/2
    density_od = gaussian_kde(snapshot[:N, Nt//2])

    if prediction is not None:
        density_rd = gaussian_kde(prediction[:N, Nt//2])

    axes[1].cla()
    axes[1].plot(x, density_od(x), "b", label="FOM")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$f(x,\cdot)$")
    axes[1].set_xlim([xmin, xmax])
    axes[1].set_title(r"$t=t_{max}/2$")

    if prediction is not None:
        axes[1].plot(x, density_rd(x), "r", label="ROM")

    axes[1].legend(loc="upper right")

    # t = tmax
    density_od = gaussian_kde(snapshot[:N,-1])

    if prediction is not None:
        density_rd = gaussian_kde(prediction[:N,-1])

    axes[2].cla()
    axes[2].plot(x, density_od(x), "b", label="FOM")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel(r"$f(x,\cdot)$")
    axes[2].set_xlim([xmin, xmax])
    axes[2].set_title(r"$t=t_{max}$")

    if prediction is not None:
        axes[2].plot(x, density_rd(x), "r", label="ROM")

    axes[2].legend(loc="upper right")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, axes

def plot_v_dist_evolution(
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
    save_dir:Optional[str],
    filename:Optional[str],
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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=120, sharey = True)
    axes = axes.ravel()

    # snapshot info
    N = snapshot.shape[0] // 2
    Nt = snapshot.shape[1]

    # velocity array for plot
    v = np.linspace(vmin, vmax, 128)

    # t = 0
    density_od = gaussian_kde(snapshot[N:,0])

    if prediction is not None:
        density_rd = gaussian_kde(prediction[N:,0])

    axes[0].cla()
    axes[0].plot(v, density_od(v), "b", label = "FOM")
    axes[0].set_xlabel("v")
    axes[0].set_ylabel(r"$f(\cdot,v)$")
    axes[0].set_xlim([vmin, vmax])
    axes[0].set_title(r"$t=0$")

    if prediction is not None:
        axes[0].plot(v, density_rd(v), "r", label = "ROM")

    axes[0].legend(loc="upper right")

    # t = tmax/2
    density_od = gaussian_kde(snapshot[N:, Nt//2])

    if prediction is not None:
        density_rd = gaussian_kde(prediction[N:, Nt//2])

    axes[1].cla()
    axes[1].plot(v, density_od(v), "b", label="FOM")
    axes[1].set_xlabel("v")
    axes[1].set_ylabel(r"$f(\cdot,v)$")
    axes[1].set_xlim([vmin, vmax])
    axes[1].set_title(r"$t=t_{max}/2$")

    if prediction is not None:
        axes[1].plot(v, density_rd(v), "r", label="ROM")

    axes[1].legend(loc="upper right")

    # t = tmax
    density_od = gaussian_kde(snapshot[N:,-1])

    if prediction is not None:
        density_rd = gaussian_kde(prediction[N:,-1])

    axes[2].cla()
    axes[2].plot(v, density_od(v), "b", label="FOM")
    axes[2].set_xlabel("v")
    axes[2].set_ylabel(r"$f(\cdot,v)$")
    axes[2].set_xlim([vmin, vmax])
    axes[2].set_title(r"$t=t_{max}$")

    if prediction is not None:
        axes[2].plot(v, density_rd(v), "r", label="ROM")

    axes[2].legend(loc="upper right")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, axes

def plot_dist_evolution(
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
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
    Nt = snapshot.shape[1]

    # Scope
    extent = [xmin, xmax, vmin, vmax]

    if prediction is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="white", dpi=120, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="white", dpi=120, constrained_layout=True)

    if prediction is not None:

        axes[0,0].cla()

        dist, _, _ = np.histogram2d(snapshot[:N,0].ravel(), snapshot[N:,0].ravel(), bins=128, density=True)
        im_00 = axes[0,0].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
        axes[0,0].set_xlabel("x")
        axes[0,0].set_ylabel("v")
        axes[0,0].axis([xmin, xmax, vmin, vmax])
        axes[0,0].set_title("FOM at $t=0$")

        axes[0,1].cla()

        dist, _, _ = np.histogram2d(snapshot[:N,Nt//2].ravel(), snapshot[N:,Nt//2].ravel(), bins=128, density=True)
        im_01 = axes[0,1].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
        axes[0,1].set_xlabel("x")
        axes[0,1].set_ylabel("v")
        axes[0,1].axis([xmin, xmax, vmin, vmax])
        axes[0,1].set_title("FOM at $t=t_{max}/2$")

        axes[0,2].cla()

        dist, _, _ = np.histogram2d(snapshot[:N,-1].ravel(), snapshot[N:,-1].ravel(), bins=128, density=True)
        im_02 = axes[0,2].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
        axes[0,2].set_xlabel("x")
        axes[0,2].set_ylabel("v")
        axes[0,2].axis([xmin, xmax, vmin, vmax])
        axes[0,2].set_title("FOM at $t=t_{max}$")

        fig.colorbar(im_02, ax=axes[0,:].ravel().tolist())

        axes[1,0].cla()
        dist, _, _ = np.histogram2d(prediction[:N,0].ravel(), prediction[N:,0].ravel(), bins=128, density=True)
        im_10 = axes[1,0].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
        axes[1,0].set_xlabel("x")
        axes[1,0].set_ylabel("v")
        axes[1,0].axis([xmin, xmax, vmin, vmax])
        axes[1,0].set_title("ROM at $t=0$")

        axes[1,1].cla()
        dist, _, _ = np.histogram2d(prediction[:N,Nt//2].ravel(), prediction[N:,Nt//2].ravel(), bins=128, density=True)
        im_11 = axes[1,1].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")

        axes[1,1].set_xlabel("x")
        axes[1,1].set_ylabel("v")
        axes[1,1].axis([xmin, xmax, vmin, vmax])
        axes[1,1].set_title("ROM at $t=t_{max}/2$")

        axes[1,2].cla()
        dist, _, _ = np.histogram2d(prediction[:N,-1].ravel(), prediction[N:,-1].ravel(), bins=128, density=True)
        im_12 = axes[1,2].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")

        axes[1,2].set_xlabel("x")
        axes[1,2].set_ylabel("v")
        axes[1,2].axis([xmin, xmax, vmin, vmax])
        axes[1,2].set_title("ROM at $t=t_{max}$")

        fig.colorbar(im_12, ax=axes[1, :].ravel().tolist())

    else:
        axes = axes.ravel()

        axes[0].cla()
        dist, _, _ = np.histogram2d(snapshot[:N,0].ravel(), snapshot[N:,0].ravel(), bins=128, density=True)
        im_0 = axes[0].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")

        axes[0].set_xlabel("x")
        axes[0].set_ylabel("v")
        axes[0].axis([xmin, xmax, vmin, vmax])
        axes[0].set_title("FOM at $t=0$")

        axes[1].cla()
        dist, _, _ = np.histogram2d(snapshot[:N,Nt//2].ravel(), snapshot[N:,Nt//2].ravel(), bins=128, density=True)
        im_1 = axes[1].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")

        axes[1].set_xlabel("x")
        axes[1].set_ylabel("v")
        axes[1].axis([xmin, xmax, vmin, vmax])
        axes[1].set_title("FOM at $t=t_{max}/2$")

        axes[2].cla()
        dist, _, _ = np.histogram2d(snapshot[:N,-1].ravel(), snapshot[N:,-1].ravel(), bins=128, density=True)
        im_2 = axes[2].imshow(dist.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")

        axes[2].set_xlabel("x")
        axes[2].set_ylabel("v")
        axes[2].axis([xmin, xmax, vmin, vmax])
        axes[2].set_title("FOM at $t=t_{max}$")

        fig.colorbar(im_2, ax=axes.ravel().tolist())

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, axes

def plot_two_stream_evolution(
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
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

    if prediction is not None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white", dpi=120)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=120)

    if prediction is not None:
        
        axes[0,0].cla()
        axes[0,0].scatter(snapshot[0:Nh,0], snapshot[N:N+Nh,0], s=0.4, color="blue", alpha=0.5)
        axes[0,0].scatter(snapshot[Nh:N,0], snapshot[N+Nh:,0], s=0.4, color="red", alpha=0.5)
        axes[0,0].set_xlabel("x")
        axes[0,0].set_ylabel("v")
        axes[0,0].axis([xmin, xmax, vmin, vmax])
        axes[0,0].set_title("FOM at $t=0$")

        axes[0,1].cla()
        axes[0,1].scatter(snapshot[0:Nh,Nt//2], snapshot[N:N+Nh,Nt//2], s=0.4, color="blue", alpha=0.5)
        axes[0,1].scatter(snapshot[Nh:N,Nt//2], snapshot[N+Nh:,Nt//2], s=0.4, color="red", alpha=0.5)
        axes[0,1].set_xlabel("x")
        axes[0,1].set_ylabel("v")
        axes[0,1].axis([xmin, xmax, vmin, vmax])
        axes[0,1].set_title("FOM at $t=t_{max}/2$")

        axes[0,2].cla()
        axes[0,2].scatter(snapshot[0:Nh,-1], snapshot[N:N+Nh,-1], s=0.4, color="blue", alpha=0.5)
        axes[0,2].scatter(snapshot[Nh:N,-1], snapshot[N+Nh:,-1], s=0.4, color="red", alpha=0.5)
        axes[0,2].set_xlabel("x")
        axes[0,2].set_ylabel("v")
        axes[0,2].axis([xmin, xmax, vmin, vmax])
        axes[0,2].set_title("FOM at $t=t_{max}$")
    
        axes[1,0].cla()
        axes[1,0].scatter(prediction[0:Nh,0], prediction[N:N+Nh,0], s=0.4, color="blue", alpha=0.5)
        axes[1,0].scatter(prediction[Nh:N,0], prediction[N+Nh:,0], s=0.4, color="red", alpha=0.5)
        axes[1,0].set_xlabel("x")
        axes[1,0].set_ylabel("v")
        axes[1,0].axis([xmin, xmax, vmin, vmax])
        axes[1,0].set_title("ROM at $t=0$")

        axes[1,1].cla()
        axes[1,1].scatter(prediction[0:Nh,Nt//2], prediction[N:N+Nh,Nt//2], s=0.4, color="blue", alpha=0.5)
        axes[1,1].scatter(prediction[Nh:N,Nt//2], prediction[N+Nh:,Nt//2], s=0.4, color="red", alpha=0.5)
        axes[1,1].set_xlabel("x")
        axes[1,1].set_ylabel("v")
        axes[1,1].axis([xmin, xmax, vmin, vmax])
        axes[1,1].set_title("ROM at $t=t_{max}/2$")

        axes[1,2].cla()
        axes[1,2].scatter(prediction[0:Nh,-1], prediction[N:N+Nh,-1], s=0.4, color="blue", alpha=0.5)
        axes[1,2].scatter(prediction[Nh:N,-1], prediction[N+Nh:,-1], s=0.4, color="red", alpha=0.5)
        axes[1,2].set_xlabel("x")
        axes[1,2].set_ylabel("v")
        axes[1,2].axis([xmin, xmax, vmin, vmax])
        axes[1,2].set_title("ROM at $t=t_{max}$")
        
    else:
        axes = axes.ravel()
        
        axes[0].cla()
        axes[0].scatter(snapshot[0:Nh,0], snapshot[N:N+Nh,0], s=0.4, color="blue", alpha=0.5)
        axes[0].scatter(snapshot[Nh:N,0], snapshot[N+Nh:,0], s=0.4, color="red", alpha=0.5)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("v")
        axes[0].axis([xmin, xmax, vmin, vmax])
        axes[0].set_title("FOM at $t=0$")

        axes[1].cla()
        axes[1].scatter(snapshot[0:Nh,Nt//2], snapshot[N:N+Nh,Nt//2], s=0.4, color="blue", alpha=0.5)
        axes[1].scatter(snapshot[Nh:N,Nt//2], snapshot[N+Nh:,Nt//2], s=0.4, color="red", alpha=0.5)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("v")
        axes[1].axis([xmin, xmax, vmin, vmax])
        axes[1].set_title("FOM at $t=t_{max}/2$")

        axes[2].cla()
        axes[2].scatter(snapshot[0:Nh,-1], snapshot[N:N+Nh,-1], s=0.4, color="blue", alpha=0.5)
        axes[2].scatter(snapshot[Nh:N,-1], snapshot[N+Nh:,-1], s=0.4, color="red", alpha=0.5)
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("v")
        axes[2].axis([xmin, xmax, vmin, vmax])
        axes[2].set_title("FOM at $t=t_{max}$")

    fig.tight_layout()
    
    if filepath is not None:
        plt.savefig(filepath, dpi=120)
        
    plt.close()

    return fig, axes

def plot_bump_on_tail_evolution(
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
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

    if prediction is not None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white", dpi=120)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=120)

    if prediction is not None:
        
        axes[0,0].cla()
        axes[0,0].scatter(snapshot[low_electron_indice,0], snapshot[low_electron_indice+N,0], s=0.4, color="blue", alpha=0.5)
        
        if high_electron_indice is not None:
            axes[0,0].scatter(snapshot[high_electron_indice,0], snapshot[high_electron_indice+N,0], s=0.4, color="red", alpha=0.5)
        
        axes[0,0].set_xlabel("x")
        axes[0,0].set_ylabel("v")
        axes[0,0].axis([xmin, xmax, vmin, vmax])
        axes[0,0].set_title("FOM at $t=0$")

        axes[0,1].cla() 
        axes[0,1].scatter(snapshot[low_electron_indice,Nt//2], snapshot[low_electron_indice+N,Nt//2], s=0.4, color="blue", alpha=0.5)
        
        if high_electron_indice is not None:
            axes[0,1].scatter(snapshot[high_electron_indice,Nt//2], snapshot[high_electron_indice+N,Nt//2], s=0.4, color="red", alpha=0.5)
        
        axes[0,1].set_xlabel("x")
        axes[0,1].set_ylabel("v")
        axes[0,1].axis([xmin, xmax, vmin, vmax])
        axes[0,1].set_title("FOM at $t=t_{max}/2$")

        axes[0,2].cla()
        axes[0,2].scatter(snapshot[low_electron_indice,-1], snapshot[low_electron_indice+N,-1], s=0.4, color="blue", alpha=0.5)
        
        if high_electron_indice is not None:
            axes[0,2].scatter(snapshot[high_electron_indice,-1], snapshot[high_electron_indice+N,-1], s=0.4, color="red", alpha=0.5)
        
        axes[0,2].set_xlabel("x")
        axes[0,2].set_ylabel("v")
        axes[0,2].axis([xmin, xmax, vmin, vmax])
        axes[0,2].set_title("FOM at $t=t_{max}$")
        
        axes[1,0].cla()
        axes[1,0].scatter(prediction[low_electron_indice,0], prediction[low_electron_indice+N,0], s=0.4, color="blue", alpha=0.5)
    
        if high_electron_indice is not None:
            axes[1,0].scatter(prediction[high_electron_indice,0], prediction[high_electron_indice+N,0], s=0.4, color="red", alpha=0.5)
        
        axes[1,0].set_xlabel("x")
        axes[1,0].set_ylabel("v")
        axes[1,0].axis([xmin, xmax, vmin, vmax])
        axes[1,0].set_title("ROM at $t=0$")

        axes[1,1].cla()
        axes[1,1].scatter(prediction[low_electron_indice,Nt//2], prediction[low_electron_indice+N,Nt//2], s=0.4, color="blue", alpha=0.5)
    
        if high_electron_indice is not None:
            axes[1,1].scatter(prediction[high_electron_indice,Nt//2], prediction[high_electron_indice+N,Nt//2], s=0.4, color="red", alpha=0.5)
        
        axes[1,1].set_xlabel("x")
        axes[1,1].set_ylabel("v")
        axes[1,1].axis([xmin, xmax, vmin, vmax])
        axes[1,1].set_title("ROM at $t=t_{max}/2$")

        axes[1,2].cla()    
        axes[1,2].scatter(prediction[low_electron_indice,-1], prediction[low_electron_indice+N,-1], s=0.4, color="blue", alpha=0.5)
    
        if high_electron_indice is not None:
            axes[1,2].scatter(prediction[high_electron_indice,-1], prediction[high_electron_indice+N,-1], s=0.4, color="red", alpha=0.5)
        
        axes[1,2].set_xlabel("x")
        axes[1,2].set_ylabel("v")
        axes[1,2].axis([xmin, xmax, vmin, vmax])
        axes[1,2].set_title("ROM at $t=t_{max}$")
        
    else:
        axes = axes.ravel()
        
        axes[0].cla()
        axes[0].scatter(snapshot[low_electron_indice,0], snapshot[low_electron_indice+N,0], s=0.4, color="blue", alpha=0.5)
        
        if high_electron_indice is not None:
            axes[0].scatter(snapshot[high_electron_indice,0], snapshot[high_electron_indice+N,0], s=0.4, color="red", alpha=0.5)
        
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("v")
        axes[0].axis([xmin, xmax, vmin, vmax])
        axes[0].set_title("FOM at $t=0$")
        
        axes[1].cla() 
        axes[1].scatter(snapshot[low_electron_indice,Nt//2], snapshot[low_electron_indice+N,Nt//2], s=0.4, color="blue", alpha=0.5)
        
        if high_electron_indice is not None:
            axes[1].scatter(snapshot[high_electron_indice,Nt//2], snapshot[high_electron_indice+N,Nt//2], s=0.4, color="red", alpha=0.5)
        
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("v")
        axes[1].axis([xmin, xmax, vmin, vmax])
        axes[1].set_title("FOM at $t=t_{max}/2$")

        axes[2].cla()
        axes[2].scatter(snapshot[low_electron_indice,-1], snapshot[low_electron_indice+N,-1], s=0.4, color="blue", alpha=0.5)
        
        if high_electron_indice is not None:
            axes[2].scatter(snapshot[high_electron_indice,-1], snapshot[high_electron_indice+N,-1], s=0.4, color="red", alpha=0.5)
        
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("v")
        axes[2].axis([xmin, xmax, vmin, vmax])
        axes[2].set_title("FOM at $t=t_{max}$")

    fig.tight_layout()
    
    if filepath is not None:
        plt.savefig(filepath, dpi=120)
        
    plt.close()

    return fig, axes

# Plot for analysis: E-field (log scale in time, profile), density (profile), and potential (profile)
def plot_log_E(
    tmax:float,
    L:float,
    dx:float,
    N_mesh:float,
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
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

    if prediction is not None:
        E_pred_list = [compute_E(prediction[:,i].reshape(-1,1), dx, N_mesh, 1.0, L, N, G, Lap)[1] for i in range(Nt)]
        E2_pred = np.array([np.mean(E_pred_list[i].ravel() ** 2) for i in range(Nt)])
    else:
        E_pred_list = None
        E2_pred = None

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor="white", dpi=120)

    ax.plot(ts, E2_real, "b", label="FOM")

    if prediction is not None:
        ax.plot(ts, E2_pred, "r", label="ROM")

    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"$\log <E^2>$")
    ax.set_yscale("log")
    ax.legend(loc="upper right")

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()
    
    return fig, ax

def plot_E_k_spectrum(
    tmax:float,
    L:float,
    dx:float,
    N_mesh:float,
    snapshot:np.ndarray,
    prediction:Optional[np.ndarray],
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

    if prediction is not None:
        ks_pred, Ek_t_spectrum_pred = compute_E_k_spectrum(1.0, L, dx, N_mesh, prediction)
    else:
        ks_pred = None
        Ek_t_spectrum_pred = None

    if prediction is not None:
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), facecolor="white", dpi=120)
        axes = axes.ravel()
        
        axes[0].imshow(Ek_t_spectrum_real, extent=[0,tmax,ks_real[0], ks_real[-1]], aspect='auto', origin='lower')
        axes[0].set_ylabel(r"$k$")
        axes[0].set_title(r"$E_k$")
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(True)
        
        axes[1].imshow(Ek_t_spectrum_pred, extent=[0,tmax,ks_pred[0], ks_pred[-1]], aspect='auto', origin='lower')
        axes[1].set_xlabel(r"$t$")
        axes[1].set_ylabel(r"$k$")
        axes[1].set_title(r"$E_k$")
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), facecolor="white", dpi=120)
        
        axes.imshow(Ek_t_spectrum_real, extent=[0,tmax,ks_real[0], ks_real[-1]], aspect='auto', origin='lower')
        axes.set_xlabel(r"$t$")
        axes.set_ylabel(r"$k$")
        axes.set_title(r"$E_k$")
        axes.set_ylim([0, 1.0])
        axes.grid(True)

    fig.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, dpi=120)

    plt.close()

    return fig, axes