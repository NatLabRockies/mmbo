import torch
from torch import nn

class MeanSD(nn.Module):
    """Activation function for the final layer of a Gaussian loss NN. Splits the final
    layer along the specified dimension into two groups. The second group is passed through
    a Softplus function and the two groups are returned in a tuple."""
    
    def __init__(self, split_dim=-1):
        super(MeanSD, self).__init__()
        self._split_dim = split_dim
        self._softplus = nn.Softplus()
    
    def forward(self, input):
        split_input = torch.chunk(input, 2, self._split_dim)
        return (split_input[0], self._softplus(split_input[1]))
