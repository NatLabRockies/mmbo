import torch
from torch import nn

class Parallel(nn.Module):
    """Container module that applies its consituent modules in parallel to
    the input, then combines them according to the specified function.
    The combine option can be any function which takes a list of tensors
    (in general, list of ouputs from the constituent modules). Its return
    value is returned by this module. For convenience, several options are
    provided as static method generators. 
    E.g. Parallel(Parallel.cat(), mod1, mod2, ...)
    """

    def __init__(self, combine, *modules: nn.Module):
        super(Parallel, self).__init__()
        self._combine = combine
        self._modules = modules

    def forward(self, input):
        outputs = []
        for mod in self._modules:
            outputs.append(mod.forward(input))
        return self._combine(outputs)

    @staticmethod
    def cat(dim=-1):
        def cat_(tensors):
            return torch.cat(tensors, dim=dim)
        return cat_
    
    @staticmethod
    def stack(dim=-1):
        def stack_(tensors):
            return torch.stack(tensors, dim=dim)
        return stack_
    
    @staticmethod
    def add():
        def add_(tensors):
            return torch.stack(tensors, dim=0).sum(dim=0)
        return add_
    
    @staticmethod
    def tuple():
        return tuple
    

