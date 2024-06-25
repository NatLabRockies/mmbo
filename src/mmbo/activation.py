import torch
from torch import nn

class MeanSD(nn.Module):
    def __init__(self, split_dim=-1, stack_dim=-1):
        super(MeanSD, self).__init__()
        self._split_dim = split_dim
        self._stack_dim = stack_dim
        self._softplus = nn.Softplus()
    
    def forward(self, input):
        split_input = torch.chunk(input, 2, self._split_dim)
        return torch.stack(
            (split_input[0], self._softplus(split_input[1])),
            dim = self._stack_dim)
