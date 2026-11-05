import torch
from pyro.distributions import constraints
from torch.distributions.multivariate_normal import _precision_to_scale_tril
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

def make_positive_definite_flipped(A: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return make_positive_definite(A.flip((-2, -1)), *args, **kwargs).flip((-2, -1))

def cholesky_with_retry(A: torch.Tensor, tries: int = 100) -> torch.Tensor:
    result = None
    cnt = 0
    while (result is None) and (cnt < tries):
        cnt += 1
        try:
            result = torch.linalg.cholesky(A)
        except torch.linalg.LinAlgError as e:
            if cnt == tries:
                raise Exception("Positive definite correction tries exceeded.")
            result = None
            A_new = make_positive_definite(A)
            if (A == A_new).all():
                raise Exception("Positive definite correction failed to change matrix.")
            A = A_new
    return result

def p2st_with_retry(A: torch.Tensor, tries: int = 100) -> torch.Tensor:
    result = None
    cnt = 0
    while (result is None) and (cnt < tries):
        cnt += 1
        try:
            result = _precision_to_scale_tril(A)
        except torch.linalg.LinAlgError as e:
            if cnt == tries:
                raise Exception("Positive definite correction tries exceeded.")
            result = None
            A_new = make_positive_definite_flipped(A)
            if (A == A_new).all():
                raise Exception("Positive definite correction failed to change matrix.")
            A = A_new
    return result

def pd_inv_with_retry(A: torch.Tensor, tries: int = 100) -> torch.Tensor:
    result = None
    cnt = 0
    while (result is None) and (cnt < tries):
        cnt += 1
        try:
            result = torch.linalg.inv(A)
        except torch.linalg.LinAlgError as e:
            if cnt == tries:
                raise Exception("Positive definite correction tries exceeded.")
            result = None
            A_new = make_positive_definite(A)
            if (A == A_new).all():
                raise Exception("Positive definite correction failed to change matrix.")
            A = A_new
    return result

def woodbury_inv(A_inv_diag: torch.Tensor, X: torch.Tensor, C_inv: torch.Tensor = None):
    """Inverts a matrix of the form M = (A + X^T * C * X), where A is diagonal,
    using a simplified Woodbury identity,
    M^(-1) = A^(-1) - A^(-1) * X^T * (C^(-1) + X * A^(-1) * X^T)^(-1) * X * A^(-1)
    Omitting C_inv (C_inv = None) implies C = I.
    A is p-by-p, X is n-by-p, and C is n-by-n. If n < p (X is wide), this is 
    faster than directly inverting M because only a n-by-n matrix needs to be 
    inverted rather than a p-by-p matrix.
    Notice that only the inverses of A and C are required in the formula so this
    function accepts a precomputed A_inv_diag and C_inv.
    """
    if C_inv is None:
        # C is the identity, auto-generate C_inv = I
        n = X.shape[-2]
        device = A_inv_diag.device
        dtype = A_inv_diag.dtype
        C_inv = torch.eye(n, device = device, dtype = dtype)
    X_scaled = X * A_inv_diag.unsqueze(-2) # Column-scaled X, as if post-multiplied by A^(-1) 
    B = C_inv + X_scaled @ X.mT
    return A_inv_diag.diag_embed() - X_scaled.mT @ torch.linalg.solve(B, X_scaled)