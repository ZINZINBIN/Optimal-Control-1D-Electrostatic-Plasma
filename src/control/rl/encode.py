import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim : int, conv_dim : int = 32, conv_kernel : int = 3, conv_stride : int = 2, conv_padding : int = 1):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        
        # temporal convolution
        self.layer_1 = nn.Sequential(
            nn.Conv1d(in_channels = input_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels = conv_dim, out_channels = conv_dim, kernel_size = conv_kernel, stride = conv_stride, padding = conv_padding),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
        )    
        
        self.feature_dim = conv_dim
        
    def forward(self, x : torch.Tensor):
        
        # normalization
        x = F.normalize(x, dim = 0)
        
        # x : (B, T, D)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        # x : (B, D, T)
        if x.size()[2] != self.seq_len:
            x = x.permute(0,2,1)
        
        # x : (B, conv_dim, T')
        x = self.layer_1(x)
        
        # x : (B, conv_dim, pred_len)
        x = self.layer_2(x)
        
        # x : (B, pred_len, conv_dim)
        x = x.permute(0,2,1)
        
        return x
    
    def compute_conv1d_output_dim(self, input_dim : int, kernel_size : int = 3, stride : int = 1, padding : int = 1, dilation : int = 1):
        return int((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)