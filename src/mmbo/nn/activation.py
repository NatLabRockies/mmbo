import torch
from torch import nn
import pyro.distributions.transforms as tform
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist

__all__ = ["MeanSD", "LearnTanh", "bayestanh"]

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

class LearnTanh(nn.Module):
    """A Tanh activation function module with a learnable scale parameter."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor(1))
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        return self.tanh(self.scale * input)
    
def bayestanh(scale_prior):
    """Factory producing Bayesian version of the LearnTanh module with a
    user-specified prior."""
    tanh = PyroModule[LearnTanh]()
    tanh.scale = PyroSample(scale_prior)
    return tanh