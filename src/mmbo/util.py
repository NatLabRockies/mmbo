import torch
from pyro.distributions import constraints
import logging

def make_symmetric(A: torch.Tensor) -> torch.Tensor:
    """Make a matrix symmetric by averaging it with its transpose."""
    # Add a batch dimension if necessary
    batched = (A.ndim > 2)
    if not batched:
        A = A.unsqueeze(0)
    is_symmetric = constraints.symmetric.check(A)
    if not is_symmetric.all():
        # Only correct matrices where is_symmetric is False
        result = A.clone()
        result[is_symmetric.logical_not()] = (A[is_symmetric.logical_not()] + A[is_symmetric.logical_not()].mT) / 2
    else:
        result = A
    # Remove batch dim if necessary
    if not batched:
        result = result.squeeze(0)
    return result

def make_positive_definite(A: torch.Tensor, correction_factor = 1e-5) -> torch.Tensor:
    """Make a matrix positive definite, if necessary, by making it symmetric 
    and adding a ridge to its diagonal."""
    assert (correction_factor > 0.0) and (correction_factor <= 1.0)
    # Add a batch dimension if necessary
    batched = (A.ndim > 2)
    if not batched:
        A = A.unsqueeze(0)
    is_posdef = constraints.positive_definite.check(A)
    if not is_posdef.all():
        d = A.shape[-1]
        result = A.clone()
        # Only correct matrices where is_posdef is False
        A_sym = make_symmetric(A[is_posdef.logical_not()])
        #eps = (1e-5 * A[is_posdef.logical_not()].diagonal(dim1=-2, dim2=-1).max(dim=-1).values + 1e-8) * correction_factor
        #ridge = eps.unsqueeze(-1).expand((*eps.shape, d)).diag_embed()
        # If there's a negative or zero diagonal, raise an exception
        if (A_sym.diagonal(dim1=-2, dim2=-1) <= 0).any():
            raise Exception("Cannot correct matrix with non-positive elements on diagonal.")
        ridge = A_sym.diagonal(dim1=-2, dim2=-1).diag_embed()
        result[is_posdef.logical_not()] = A_sym - correction_factor * (A_sym - ridge)
    else:
        result = A
    # Remove batch dim if necessary
    if not batched:
        result = result.squeeze(0)
    return result