from pyro.nn import PyroModule, PyroSample
from torch import nn
import torch

__all__ = ["bayeslinear"]

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