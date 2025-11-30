import numpy as np
from src.control.objective import estimate_KL_divergence, estimate_f, estimate_electric_energy
from typing import Optional

class Reward:
    def __init__(self, init_state:np.ndarray, N_mesh:int = 500, L:float = 50.0, vmin:float= -25.0, vmax:float = 25.0, n0:float = 1.0, alpha:float = 0.25):
        self.feq = estimate_f(init_state, N_mesh, L, vmin, vmax, n0)
        self.init_state = init_state
        self.N_mesh = N_mesh
        self.L = L
        self.vmin = vmin
        self.vmax = vmax
        self.n0 = n0
        
        # Multiplier
        self.alpha = alpha

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if hasattr(self, key) is True and kwargs[key] is not None:
                setattr(self, key, kwargs[key])

    def reinit(self):
        self.feq = estimate_f(self.init_state, self.N_mesh, self.L, self.vmin, self.vmax, self.n0)

    def compute_kl_divergence(self, state:np.ndarray):
        f = estimate_f(state, self.N_mesh, self.L, self.vmin, self.vmax, self.n0)
        kl = estimate_KL_divergence(f, self.feq, self.L / self.N_mesh, (self.vmax - self.vmin) / self.N_mesh)
        return kl

    def compute_electric_energy(self, state:np.ndarray, E_external:Optional[np.ndarray] = None):
        PE = estimate_electric_energy(state.reshape(-1,1), E_external, self.N_mesh, self.L, self.n0)
        return PE
    
    def compute_input_energy(self, actions:np.ndarray):
        PE = np.sum(actions ** 2) * self.L / 2
        return PE
    
    def compute_cost(self, state:np.ndarray, action:np.ndarray):
        r_kl = self.compute_kl_divergence(state)
        r_pe = self.compute_electric_energy(state)
        r_in = self.compute_input_energy(action)
        return r_kl + self.alpha * r_pe
    
    def compute_reward_kl_divergence(self, state:np.ndarray):
        return np.tanh(1 - np.sqrt(self.compute_kl_divergence(state) / 25))
    
    def compute_reward_electric_energy(self, state:np.ndarray, E_external:Optional[np.ndarray] = None):
        return np.tanh(1 - np.sqrt(self.compute_electric_energy(state, E_external) / 10.0))
    
    def compute_reward_input_energy(self, action:np.ndarray):
        return np.tanh(1 - np.sqrt(self.compute_input_energy(action) / 50.0))
    
    def compute_reward(self, state:np.ndarray, E_external:Optional[np.ndarray] = None):
        # r_kl = self.compute_reward_kl_divergence(state)
        r_pe = self.compute_reward_electric_energy(state, E_external)
        reward = r_pe * self.alpha
        return reward