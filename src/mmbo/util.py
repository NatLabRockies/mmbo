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

def make_positive_definite(A: torch.Tensor) -> torch.Tensor:
    """Make a matrix positive definite, if necessary, by making it symmetric 
    and adding a ridge to its diagonal."""
    # Add a batch dimension if necessary
    batched = (A.ndim > 2)
    if not batched:
        A = A.unsqueeze(0)
    is_posdef = constraints.positive_definite.check(A)
    if not is_posdef.all():
        eps = 1e-5 * A.diagonal(dim1=-2, dim2=-1).mean(dim=-1) + 1e-8
        d = A.shape[-1]
        result = A.clone()
        # Only correct matrices where is_posdef is False
        A_sym = make_symmetric(A[is_posdef.logical_not()])
        # Remove gradients from the eigenvalues
        min_eigvals = torch.linalg.eigvals(A_sym.detach()).real.min(dim=-1).values
        to_add = (-min_eigvals).clamp(min=0) + eps
        ridge = to_add.unsqueeze(-1).expand((*to_add.shape, d)).diag_embed()
        result[is_posdef.logical_not()] = A_sym + ridge
    else:
        result = A
    # Remove batch dim if necessary
    if not batched:
        result = result.squeeze(0)
    return result