import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    def __init__(self, num_nodes:int, input_dim:int, hidden_dim:int, output_dim:int):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Fully Connected Layer for Encoding
        self.fc = nn.Linear(hidden_dim, output_dim)
        
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

        # Fully Connected Layer
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    
    batch_size = 10
    num_node = 100
    input_dim = 2
    hidden_dim = 10
    output_dim = 10
    
    encoder = Encoder(num_node, input_dim, hidden_dim, output_dim)
    
    x = torch.zeros((batch_size, num_node, input_dim))
    x_encode = encoder(x)
    
    print(x_encode.size())