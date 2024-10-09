import torch
import torch.nn as nn
import torch.nn.functional as F

class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FANLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Set d_p to 1/4 of the output dimension
        self.d_p = output_dim // 4
        self.d_p_bar = output_dim - 2 * self.d_p
        
        # Initialize learnable parameters
        self.W_p = nn.Parameter(torch.Tensor(input_dim, self.d_p))
        self.W_p_bar = nn.Parameter(torch.Tensor(input_dim, self.d_p_bar))
        self.B_p_bar = nn.Parameter(torch.Tensor(self.d_p_bar))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W_p)
        nn.init.xavier_uniform_(self.W_p_bar)
        nn.init.zeros_(self.B_p_bar)
        
    def forward(self, x):
        # Compute cos(W_p x)
        cos_term = torch.cos(F.linear(x, self.W_p))
        
        # Compute sin(W_p x)
        sin_term = torch.sin(F.linear(x, self.W_p))
        
        # Compute σ(B_p̄ + W_p̄ x)
        linear_term = F.linear(x, self.W_p_bar, self.B_p_bar)
        activation_term = F.gelu(linear_term)
        
        # Concatenate the results
        output = torch.cat([cos_term, sin_term, activation_term], dim=-1)
        
        return output
