import numpy as np
from typing import Optional

class E_field:
    def __init__(self, L:float, N_mesh:int, max_mode:int):
        self.L = L
        self.N_mesh = N_mesh
        self.dx = L / N_mesh

        self.max_mode = max_mode

        # mesh coordinate
        self.xm = np.linspace(0, L, N_mesh)

        # Coefficient (all components in [-1,1] initially)
        self.coeff_cos = np.random.rand(max_mode) * 2 - 1.0 
        self.coeff_sin = np.random.rand(max_mode) * 2 - 1.0 

        # wave number
        self.k = np.array([2 * np.pi / L * n for n in range(1, max_mode + 1)])

        # basis
        self.basis_cos = np.concatenate([np.cos(k * self.xm).reshape(-1,1) for k in self.k], axis = 1)
        self.basis_sin = np.concatenate([np.sin(k * self.xm).reshape(-1,1) for k in self.k], axis = 1)

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if hasattr(self, key) is True and kwargs[key] is not None:
                setattr(self, key, kwargs[key])

    def reinit(self):
        # mesh coordinate
        self.xm = np.linspace(0, self.L, self.N_mesh)

        # Coefficient (all components in [-1,1] initially)
        self.coeff_cos = np.random.rand(self.max_mode) * 2 - 1.0 
        self.coeff_sin = np.random.rand(self.max_mode) * 2 - 1.0 

        # wave number
        self.k = np.array([2 * np.pi / self.L * n for n in range(1, self.max_mode + 1)])

        # basis
        self.basis_cos = np.concatenate([np.cos(k * self.xm).reshape(-1,1) for k in self.k], axis = 1)
        self.basis_sin = np.concatenate([np.sin(k * self.xm).reshape(-1,1) for k in self.k], axis = 1)
   
    def update_E(self, coeff_cos:Optional[np.ndarray] = None, coeff_sin:Optional[np.ndarray] = None):

        if coeff_cos is not None:
            self.coeff_cos = coeff_cos.copy()

        if coeff_sin is not None:
            self.coeff_sin = coeff_sin.copy()

    def compute_E(self, coeff_cos:Optional[np.ndarray] = None, coeff_sin:Optional[np.ndarray] = None):

        if coeff_cos is None:
            coeff_cos = self.coeff_cos.copy()

        if coeff_sin is None:
            coeff_sin = self.coeff_sin.copy()

        E = self.basis_cos @ coeff_cos.reshape(-1,1) + self.basis_sin @ coeff_sin.reshape(-1,1) 
        return E
