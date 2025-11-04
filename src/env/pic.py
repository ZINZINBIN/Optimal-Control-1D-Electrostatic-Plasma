import numpy as np
import scipy as sp
from tqdm.auto import tqdm
from typing import Literal, Optional, Union, List, Callable
from src.env.util import generate_grad, generate_laplacian
from src.env.integration import symplectic_4th_order
from src.env.solve import Gaussian_Elimination_Periodic
from src.env.util import compute_hamiltonian, compute_electric_energy, compute_E, compute_n
from src.env.dist import TwoStream, BumpOnTail

class PIC:
    np.random.seed(42)
    def __init__(
        self,
        N: int = 40000,
        N_mesh: int = 400,
        n0: float = 1.0,
        L: float = 50.0,
        dt: float = 1.0,
        tmin: float = 0.0,
        tmax: float = 50.0,
        gamma:float = 5.0,
        A: float = 0.1, # x perturbation
        n_mode:int = 4, # mode number of perturbation
        interpol: Literal["CIC", "TSC"] = "CIC",
        init_dist: Optional[Union[TwoStream, BumpOnTail]] = None,
    ):
        # setup
        self.N = N  # num of particles
        self.N_mesh = N_mesh  # num of mesh cell
        self.n0 = n0  # average electron density
        self.L = L  # box size
        self.dt = dt  # time difference
        self.tmin = tmin  # minimum time
        self.tmax = tmax  # maximum time
        self.dx = L / N_mesh

        self.gamma = gamma

        # Info of perturbation
        self.A = A  # Amplitude of the perturbation
        self.n_mode = n_mode  # mode number of perturbation

        # Initial distribution of particles
        self.init_dist = init_dist

        # Interpolation
        self.interpol = interpol

        # Gradient field and laplacian of potential
        # Mesh for 1st derivative and 2nd derivative
        self.grad = generate_grad(L, N_mesh)
        self.laplacian = generate_laplacian(L, N_mesh)
        
        # Field quantities
        self.phi_mesh = None
        self.E_mesh = None
        self.E = None

        # initialize x and v
        self.initialize()

    def initialize(self):
        self.init_dist.reinit()
        x, v = self.init_dist.get_sample()
        self.x = x.reshape(-1, 1)
        self.v = v.reshape(-1, 1)
        self.v *= (1 + self.A * np.sin(2 * np.pi * self.n_mode * self.x / self.L))  # add perturbation

        # check CFL condition for stability
        if self.dt > 2 / np.sqrt(self.N / self.L):
            self.dt = 2 / np.sqrt(self.N / self.L)
            print("CFL condtion invalid: change dt = {:.4f}".format(self.dt))

        # update density and corresponding electric field
        self.update_density()
        self.update_E_field()

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if hasattr(self, key) is True and kwargs[key] is not None:
                setattr(self, key, kwargs[key])

    def reinit(self):
        # initialize the simulation parameters
        self.initialize()

        # Field quantities
        self.E = None
        self.E_mesh = None
        self.phi_mesh = None

    def update_density(self):

        if self.interpol == "CIC":
            n, indx_l, indx_r, weight_l, weight_r = compute_n(self.x, self.dx, self.N_mesh, self.n0, self.L, self.N, True, self.interpol)

        elif self.interpol == "TSC":
            n, indx_l, indx_m, indx_r, weight_l, weight_m, weight_r = compute_n(self.x, self.dx, self.N_mesh, self.n0, self.L, self.N, True, self.interpol)

        self.n = n
        self.indx_l = indx_l
        self.indx_r = indx_r
        self.weight_l = weight_l
        self.weight_r = weight_r

        if self.interpol == "TSC":
            self.indx_m = indx_m
            self.weight_m = weight_m
        else:
            self.indx_m = None
            self.weight_m = None

    def update_E_field(self):

        self.phi_mesh = Gaussian_Elimination_Periodic(self.laplacian, self.n - self.n0, self.gamma).reshape(-1,1)
        self.E_mesh = (-1) * self.grad @ self.phi_mesh

        if self.interpol == "CIC":
            self.E = self.weight_l * self.E_mesh[self.indx_l[:, 0]] + self.weight_r * self.E_mesh[self.indx_r[:, 0]]

        elif self.interpol == "TSC":
            self.E = self.weight_l * self.E_mesh[self.indx_l[:, 0]] + self.weight_m * self.E_mesh[self.indx_m[:,0]] + self.weight_r * self.E_mesh[self.indx_r[:, 0]]

    def compute_state_gradient(self, eta:np.ndarray, E_external:Optional[np.ndarray]):
        xdot = eta[self.N:,:]
        vdot = (-1) * compute_E(eta, self.dx, self.N_mesh, self.n0, self.L, self.N, self.grad, self.laplacian, False, self.interpol, E_external)[0]
        grad_eta = np.concatenate([xdot, vdot], axis = 0)
        return grad_eta

    def update_state(self, E_external: Optional[np.ndarray]):
        eta = np.concatenate([self.x.reshape(-1, 1), self.v.reshape(-1, 1)], axis=0)
        eta_f = symplectic_4th_order(eta, lambda x : self.compute_state_gradient(x, E_external), self.dt)

        x = eta_f[:self.N]
        v = eta_f[self.N:]

        # Periodicity check
        x = np.mod(x, self.L)

        # Update information
        self.x = x
        self.v = v

        self.update_density()
        self.update_E_field()
        
    def update_state_w_input_func(self, input_func: Optional[Callable]):
        eta = np.concatenate([self.x.reshape(-1, 1), self.v.reshape(-1, 1)], axis=0)
        eta_f = symplectic_4th_order(eta, lambda x : self.compute_state_gradient(x, input_func(x)), self.dt)

        x = eta_f[:self.N]
        v = eta_f[self.N:]

        # Periodicity check
        x = np.mod(x, self.L)

        # Update information
        self.x = x
        self.v = v

        self.update_density()
        self.update_E_field()
        
    def get_state(self):
        state = np.concatenate([self.x.copy().reshape(-1,1), self.v.copy().reshape(-1,1)], axis = 0)
        return state
    
    def get_energy(self):
        return compute_hamiltonian(self.x, self.v, self.dx, self.N, self.N_mesh, self.n0, self.L, self.interpol)
    
    def get_electric_energy(self):
        return compute_electric_energy(self.x, self.dx, self.N, self.N_mesh, self.n0, self.L, self.interpol)

    def simulate(self, E_external_traj:Optional[List[np.ndarray]]):

        Nt = int(np.ceil((self.tmax - self.tmin) / self.dt))

        # Initialize
        self.update_density()

        # snapshot
        pos_list = []
        vel_list = []
        E_list = []
        PE_list = []

        E = compute_hamiltonian(self.x, self.v, self.dx, self.N, self.N_mesh, self.n0, self.L, self.interpol)
        PE = compute_electric_energy(self.x, self.dx, self.N, self.N_mesh, self.n0, self.L, self.interpol)

        pos_list.append(self.x.copy())
        vel_list.append(self.v.copy())
        E_list.append(E)
        PE_list.append(PE)

        for i in tqdm(range(Nt), "PIC simulation process..."):

            if E_external_traj is not None:
                E_external = E_external_traj[i]
            else:
                E_external = None

            self.update_state(E_external)

            pos_list.append(self.x.copy())
            vel_list.append(self.v.copy())

            E = compute_hamiltonian(self.x, self.v, self.dx, self.N, self.N_mesh, self.n0, self.L, self.interpol)
            PE = compute_electric_energy(self.x, self.dx, self.N, self.N_mesh, self.n0, self.L, self.interpol)

            E_list.append(E)
            PE_list.append(PE)

        print("# PIC simulation complete")

        qs = np.concatenate(pos_list, axis = 1)
        ps = np.concatenate(vel_list, axis = 1)
        snapshot = np.concatenate([qs, ps], axis=0)

        E = np.array(E_list)
        PE = np.array(PE_list)

        return snapshot, E, PE
