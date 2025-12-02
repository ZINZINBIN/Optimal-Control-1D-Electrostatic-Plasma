import numpy as np
from abc import ABC

# Adaptive sampling method w.r.t customized distribution function
class BasicDistribution(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def get_target_prob(self, *args, **kwargs):
        pass

    def get_proposal_prob(self, *args, **kwargs):
        pass

    def get_sample(self, *args, **kwargs):
        pass

    def rejection_sampling(self, n_samples:int):
        pass
    
    def reinit(self):
        pass
    
    def update_params(self, **kwargs):
        pass

class TwoStream(BasicDistribution):
    def __init__(
        self,
        v0: float = 4.0,
        sigma: float = 0.5,
        n_samples: int = 40000,
        L:float = 50,
    ):
        # parameters
        self.v0 = v0
        self.sigma = sigma
        self.L = L
        self.n_samples = n_samples

        self.initialize(n_samples)

    def initialize(self, n_samples:int):
        state = self.rejection_sampling(n_samples)
        self.x_init = state[:, 0]
        self.v_init = state[:, 1]

    def reinit(self):
        self.initialize(self.n_samples)

    def get_sample(self):
        return self.x_init.copy(), self.v_init.copy()

    def get_init_state(self):
        return np.concatenate([self.x_init.copy().reshape(-1,1), self.v_init.copy().reshape(-1,1)], axis = 0)

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if hasattr(self, key) is True and kwargs[key] is not None:
                setattr(self, key, kwargs[key])

    def get_proposal_prob(self, v:float):
        prob = np.exp(-abs(v))
        return prob

    def get_target_prob(self, v:float, vb:float):
        prob = 1 / np.sqrt(2 * np.pi) / self.sigma * np.exp(-0.5 * (v-vb)**2 / self.sigma ** 2)
        return prob

    def rejection_sampling(self, n_samples: int, batch:int = 1000):
        pos = []
        vel = []

        # For electrom beam injected via +x direction
        while len(pos) <= n_samples // 2:
            x = np.random.uniform(0, self.L, size = batch)
            v = np.random.uniform(-10, 10, size = batch)
            u = np.random.uniform(0, 1.0, size = batch)

            pos += x[u < self.get_target_prob(v, self.v0)].tolist()
            vel += v[u < self.get_target_prob(v, self.v0)].tolist()

        pos = pos[:n_samples // 2]
        vel = vel[:n_samples // 2]

        # For electrom beam injected via -x direction
        while len(pos) < n_samples:

            x = np.random.uniform(0, self.L, size=batch)
            v = np.random.uniform(-10, 10, size=batch)
            u = np.random.uniform(0, 1.0, size=batch)

            pos += x[u < self.get_target_prob(v, (-1) * self.v0)].tolist()
            vel += v[u < self.get_target_prob(v, (-1) * self.v0)].tolist()

        pos = np.array(pos[:n_samples])
        vel = np.array(vel[:n_samples])

        samples = np.zeros((n_samples,2))
        samples[:,0] = pos
        samples[:,1] = vel
        return samples

class BumpOnTail(BasicDistribution):
    def __init__(
        self,
        a: float = 0.3,
        v0: float = 4.0,
        sigma: float = 0.5,
        n_samples: int = 40000,
        L:float = 10,
    ):
        # parameters
        self.a = a
        self.v0 = v0
        self.sigma = sigma
        self.L = L
        self.n_samples = n_samples

        self.initialize(n_samples)

    def initialize(self, n_samples:int):
        state = self.rejection_sampling(n_samples)
        self.x_init = state[:, 0]
        self.v_init = state[:, 1]
        self.high_indx = self.inject_high_electron_indice()

    def reinit(self):
        self.initialize(self.n_samples)

    def get_sample(self):
        return self.x_init.copy(), self.v_init.copy()

    def get_init_state(self):
        return np.concatenate([self.x_init.copy().reshape(-1,1), self.v_init.copy().reshape(-1,1)], axis = 0)

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if hasattr(self, key) is True and kwargs[key] is not None:
                setattr(self, key, kwargs[key])

    def get_proposal_prob(self, x:float, v:float):
        # x-dependecy
        prob = np.exp(-abs(v))
        return prob

    def get_target_prob(self, v: float, vb: float, sigma: float):
        prob = 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (v - vb) ** 2 / sigma**2)
        return prob

    def rejection_sampling(self, n_samples: int, batch:int = 1000):
        pos = []
        vel = []

        r1 = 1 / (1 + self.a)
        r2 = self.a / (1 + self.a)
        N1 = int(n_samples * r1)
        N2 = n_samples - N1

        # Background thermalized electron distribution
        while len(pos) < N1:

            x = np.random.uniform(0, self.L, size=batch)
            v = np.random.uniform(-10, 10, size=batch)
            u = np.random.uniform(0, 1.0, size=batch)

            pos += x[u < self.get_target_prob(v, 0.0, 1.0)].tolist()
            vel += v[u < self.get_target_prob(v, 0.0, 1.0)].tolist()

        pos = pos[:N1]
        vel = vel[:N1]

        while len(pos) < n_samples:

            x = np.random.uniform(0, self.L, size=batch)
            v = np.random.uniform(-10, 10, size=batch)
            u = np.random.uniform(0, 1.0, size=batch)

            pos += x[u < self.get_target_prob(v, self.v0, self.sigma)].tolist()
            vel += v[u < self.get_target_prob(v, self.v0, self.sigma)].tolist()

        pos = np.array(pos[:n_samples])
        vel = np.array(vel[:n_samples])

        samples = np.zeros((n_samples,2))
        samples[:,0] = pos
        samples[:,1] = vel

        return samples

    def inject_high_electron_indice(self):
        r1 = 1 / (1 + self.a)
        N1 = int(self.n_samples * r1)
        indice = np.array([i for i in range(N1, self.n_samples)])
        return indice