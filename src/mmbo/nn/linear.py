from pyro.nn import PyroModule, PyroSample
from torch import nn

__all__ = ["bayeslinear"]

def bayeslinear(*args, weight_prior, bias_prior = None, **kwargs):
    bl = PyroModule[nn.Linear](*args, **kwargs)
    bl.weight = PyroSample(weight_prior.expand(bl.weight.shape).to_event(len(bl.weight.shape)))
    if bl.bias is not None:
        if bias_prior is None:
            raise ValueError("Must supply a bias prior if Linear layer has bias.")
        bl.bias = PyroSample(bias_prior.expand(bl.bias.shape).to_event(len(bl.bias.shape)))
    return bl