import torch
from torch import nn
from torch.nn import init
import torch.functional as F
import itertools
from typing import Tuple

class Parallel(nn.Module):
    """Container module that splits the input along the specified dimension
    and applies its component modules to each piece of the input. Useful for
    applying different activation functions to different sections of the input,
    or having subnetworks of different depth."""

    def __init__(self, *args: Tuple[int, nn.Module], split_dim=1):
        super(Parallel, self).__init__()
        
        self._sizes = []
        self._modules = nn.ModuleList()
        self._dim = split_dim
        for n, mod in args:
            if not isinstance(n, int) or not isinstance(mod, nn.Module):
                raise Exception("Must pass tuples of int and Module")
            self._sizes.append(n)
            self._modules.append(mod)
        
    def forward(self, input):
        split_output = []
        split_input = torch.split(input, self._sizes, dim=self._dim)
        for i in range(len(self._sizes)):
            mod = self._modules[i]
            split_output.append(mod.forward(split_input[i]))
        return torch.cat(split_output, dim=self._dim)
