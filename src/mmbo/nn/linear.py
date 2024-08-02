from pyro.nn import PyroModule, PyroSample, PyroParam
from torch import nn
import torch

__all__ = ["bayeslinear", "PyroMatrixLinear"]

def _expand_dist_to_shape(dist, goal_shape):
    if len(goal_shape) < dist.event_dim:
        raise ValueError("Distribution shape has too many dimensions")
    # Break up the goal shape into components
    if dist.event_dim == 0:
        goal_event_shape = torch.Size()
        goal_batch_shape = goal_shape
    else:
        goal_event_shape = goal_shape[-dist.event_dim:]
        goal_batch_shape = goal_shape[:-dist.event_dim]
    # The event shape must match the rightmost dimensions of the desired shape exactly
    if dist.event_shape != goal_event_shape:
        raise ValueError("Event shape does not match right-most dims")
    # The batch shape must be broadcastable to the remaining dimensions
    return dist.expand(goal_batch_shape).to_event(len(goal_batch_shape))

def bayeslinear(*args, weight_prior, bias_prior = None, **kwargs):
    """Factory producing Bayesian fully-connected linear NN layers, with
    the user-specified prior on weights/biases."""
    bl = PyroModule[nn.Linear](*args, **kwargs)
    bl.weight = PyroSample(_expand_dist_to_shape(weight_prior, bl.weight.shape))
    if bl.bias is not None:
        if bias_prior is None:
            raise ValueError("Must supply a bias prior if Linear layer has bias.")
        bl.bias = PyroSample(_expand_dist_to_shape(bias_prior, bl.bias.shape))
    return bl

class PyroMatrixLinear(PyroModule[nn.Linear]):
    """A dense linear layer intended to have its full bias/weight matrix parameterized
    as a MatrixNormal distribution."""
    full_matrix: torch.Tensor

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__(in_features, out_features, bias = True, device = device, dtype = dtype)
        self.full_matrix = PyroParam(
            torch.cat([self.bias.reshape((-1,1)), self.weight], dim = -1)
        )

    @property
    def weight(self):
        return self.full_matrix[:,1:]
    
    @property
    def bias(self):
        return self.full_matrix[:,0]