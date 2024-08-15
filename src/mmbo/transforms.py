from torch.distributions.constraint_registry import biject_to
import torch.distributions.constraints as constraints
import torch.distributions.transforms as transforms
import torch
import math
import logging

__all__ = []

# Check if there is already a bijection for lower cholesky factors,
# if not, define a bijective transform and register it.
try:
    _ = biject_to(constraints.lower_cholesky)
except NotImplementedError:
    class LowerCholeskyTransform(transforms.LowerCholeskyTransform):
        bijective = True
        sign = +1

        def log_abs_det_jacobian(self, x, y):
            return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        
    @biject_to.register(constraints.lower_cholesky)
    def _biject_to_lower_cholesky(constraint):
        return LowerCholeskyTransform()
else:
    msg = f"A bijection for LowerCholesky constraint alreay exists in your version of pytorch ({torch.__version_})"
    logging.info(msg)

# Check if there is already a bijection for positive definite matrices,
# if not, define a bijective transform and register it

try:
    _ = biject_to(constraints.positive_definite)
except NotImplementedError:
    class PositiveDefiniteTransform(transforms.Transform):
        """
        Transform from unconstrained matrices to positive-definite matrices.
        """

        domain = constraints.independent(constraints.real, 2)
        codomain = constraints.positive_definite  # type: ignore[assignment]
        bijective = True
        sign = +1

        def __eq__(self, other):
            return isinstance(other, PositiveDefiniteTransform)

        def _call(self, x):
            x = LowerCholeskyTransform()(x)
            return x @ x.mT

        def _inverse(self, y):
            y = torch.linalg.cholesky(y)
            return LowerCholeskyTransform().inv(y)

        def log_abs_det_jacobian(self, x, y):
            K = x.shape[-1]
            return K * math.log(2.0) + (torch.arange(K + 1, 1, -1) * x.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
        
    @biject_to.register(constraints.positive_definite)
    @biject_to.register(constraints.positive_semidefinite)
    def _biject_to_positive_definite(constraint):
        return PositiveDefiniteTransform()
else:
    msg = f"A bijection for PositiveDefinite constraint alreay exists in your version of pytorch ({torch.__version_})"
    logging.info(msg)