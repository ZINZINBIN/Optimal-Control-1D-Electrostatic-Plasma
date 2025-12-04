import torch
import torch.nn as nn

class ParticleEncoder(nn.Module):
    """
    Permutation-invariant encoder for particle phase-space data
    using DeepSets architecture.
    """

    def __init__(self, hidden_dim:int, output_dim:int, L=50.0):
        super().__init__()
        self.L = L
        self.output_dim = output_dim

        self.phi = nn.Sequential(
            nn.Linear(3, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x:torch.Tensor):
        N = x.size()[1] // 2
        q = x[:,:N]
        p = x[:,N:]
        
        sinq = torch.sin(q * 2 * torch.pi / self.L)
        cosq = torch.cos(q * 2 * torch.pi / self.L)

        z = torch.stack([cosq, sinq, p], dim=-1)
        phi = self.phi(z).mean(dim=1)
        rho = self.rho(phi)
        return rho
