import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, num_nodes:int, input_dim:int, hidden_dim:int, output_dim:int):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Create all possible edges (including self-loops)
        row = torch.arange(num_nodes).repeat(num_nodes)
        col = torch.arange(num_nodes).repeat_interleave(num_nodes)

        # Remove self-loops
        mask = row != col
        self.edge_index = torch.stack([row[mask], col[mask]], dim=0)
        
    def forward(self, x:torch.Tensor):
        # Graph Convolution Layers
        x = F.relu(self.conv1(x, self.edge_index.to(x.device)))
        x = F.relu(self.conv2(x, self.edge_index.to(x.device)))
        return x.mean(dim = 1)
    
class Encoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, kernel_size : int = 5, stride:int = 1, padding : int = 2, reduction : bool = False):
        super().__init__()
        
        dk = 1 if reduction else 0
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size= kernel_size + dk, stride = stride, padding = padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim,out_channels=output_dim, kernel_size= kernel_size, stride = stride, padding = padding),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x : torch.Tensor):
        return self.encoder(x)
    
    def compute_conv1d_output_dim(self, input_dim : int, kernel_size : int = 3, stride : int = 1, padding : int = 1, dilation : int = 1):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    
if __name__ == "__main__":
    
    batch_size = 10
    num_node = 100
    input_dim = 2
    hidden_dim = 10
    output_dim = 10
    
    # encoder = Encoder(num_node, input_dim, hidden_dim, output_dim)
    encoder = Encoder(input_dim, hidden_dim, output_dim, kernel_size = 5, stride = 3, padding = 1, reduction = False)
    
    x = torch.zeros((batch_size, input_dim, num_node))
    x_encode = encoder(x)
    
    print(x_encode.size())
    print(encoder.compute_conv1d_output_dim(
        encoder.compute_conv1d_output_dim(num_node, 5, 3, 1),
        5,3,1
        )
    )