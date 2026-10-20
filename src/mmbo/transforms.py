import torch.distributions.constraints as constraints
import torch.distributions.transforms as transforms
import torch.distributions.constraint_registry as constraint_registry
import torch.distributions.utils as utils
import torch
import math
import logging

__all__ = []

#### Convenient Aliases

real_matrix = constraints.independent(constraints.real, 2)

#### Positive Ordered Vector, but with softplus transform to avoid overflow

class _SoftPlusPositiveOrderedVector(constraints.Constraint):
    """
    Constrains to a positive real-valued tensor where the elements are monotonically
    increasing along the `event_shape` dimension.
    """
    event_dim = 1

    def check(self, value):
        return constraints.ordered_vector.check(value) & constraints.independent(constraints.softplus_positive, 1).check(value)
softplus_positive_ordered_vector = _SoftPlusPositiveOrderedVector()

@constraint_registry.biject_to.register(softplus_positive_ordered_vector)
@constraint_registry.transform_to.register(softplus_positive_ordered_vector)
def _transform_to_softplus_positive_ordered_vector(constraint):
    return transforms.ComposeTransform([transforms.OrderedTransform(), transforms.SoftplusTransform()])

#### Real numbers but allow NA's

class _NanMaskedReal(constraints.Constraint):
    """Constrain to real numbers or nan"""

    def check(self, value):
        return torch.logical_or(value.isnan(), value == value)
    
nanmaskedreal = _NanMaskedReal()
nanmaskedreal_vector = constraints.independent(nanmaskedreal, 1)

@constraint_registry.biject_to.register(nanmaskedreal)
@constraint_registry.transform_to.register(nanmaskedreal)
def _transform_to_nanmaskedreal(constraint):
    return transforms.identity_transform

#### Cholesky Factors and Positive Definite Matrices

# Check if there is already a bijection for lower cholesky factors,
# if not, define a bijective transform and register it.
try:
    _ = constraint_registry.biject_to(constraints.lower_cholesky)
except NotImplementedError:
    class LowerCholeskyTransform(transforms.LowerCholeskyTransform):
        """
        Transform from unconstrained vectors to lower-triangular matrices with
        nonnegative diagonal entries.

        This is useful for parameterizing positive definite matrices in terms of
        their Cholesky factorization.
        """
        domain = constraints.real_vector
        codomain = constraints.lower_cholesky
        bijective = True
        sign = +1

        def __eq__(self, other):
            return isinstance(other, LowerCholeskyTransform)

        def _call(self, x):
            r = utils.vec_to_tril_matrix(x)
            return r.tril(-1) + r.diagonal(dim1=-2, dim2=-1).exp().diag_embed()

        def _inverse(self, y):
            s = y.tril(-1) + y.diagonal(dim1=-2, dim2=-1).log().diag_embed()
            return utils.tril_matrix_to_vec(s)

        def log_abs_det_jacobian(self, x, y):
            r = utils.vec_to_tril_matrix(x)
            return r.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        
    @constraint_registry.biject_to.register(constraints.lower_cholesky)
    def _biject_to_lower_cholesky(constraint):
        return LowerCholeskyTransform()
else:
    msg = f"A bijection for LowerCholesky constraint alreay exists in your version of pytorch ({torch.__version__})"
    logging.info(msg)

# Check if there is already a bijection for positive definite matrices,
# if not, define a bijective transform and register it

try:
    _ = constraint_registry.biject_to(constraints.positive_definite)
except NotImplementedError:
    class PositiveDefiniteTransform(transforms.Transform):
        """
        Transform from unconstrained vectors to positive-definite matrices.
        """
        domain = constraints.real_vector
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
            r = utils.vec_to_tril_matrix(x)
            K = r.shape[-1]
            return K * math.log(2.0) + (torch.arange(K + 1, 1, -1) * r.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
        
    @constraint_registry.biject_to.register(constraints.positive_definite)
    @constraint_registry.biject_to.register(constraints.positive_semidefinite)
    def _biject_to_positive_definite(constraint):
        return PositiveDefiniteTransform()
else:
    msg = f"A bijection for PositiveDefinite constraint alreay exists in your version of pytorch ({torch.__version_})"
    logging.info(msg)