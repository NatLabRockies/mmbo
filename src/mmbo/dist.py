from pyro.distributions import TorchDistribution
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.distributions import transforms
import functools

__all__ = ["OrderedNormal", "OrderedHalfNormal", "MatrixNormal"]

# Constraints and transforms for the distributions contained in this file

class _SoftPlusPositiveOrderedVector(constraints.Constraint):
    """
    Constrains to a positive real-valued tensor where the elements are monotonically
    increasing along the `event_shape` dimension.
    """
    event_dim = 1

    def check(self, value):
        return constraints.ordered_vector.check(value) & constraints.independent(constraints.softplus_positive, 1).check(value)
softplus_positive_ordered_vector = _SoftPlusPositiveOrderedVector()

@transforms.biject_to.register(softplus_positive_ordered_vector)
@transforms.transform_to.register(softplus_positive_ordered_vector)
def _transform_to_softplus_positive_ordered_vector(constraint):
    return transforms.ComposeTransform([transforms.OrderedTransform(), transforms.SoftplusTransform()])

real_matrix = constraints.independent(constraints.real, 2)

# Distributions

class OrderedNormal(TorchDistribution):
    """Distrubtion representing the joint distribution of order statistics
    of a normal distribution. Useful for placing priors on layer biases to 
    impose ordering constraints."""
    
    has_rsample = True
    support = constraints.ordered_vector
    arg_constraints = {
        "loc": constraints.real, 
        "scale": constraints.positive, 
        "n": constraints.positive_integer
        }

    def __init__(self, n, loc, scale, validate_args=None):
        self.loc, self.scale = dist.util.broadcast_all(loc, scale)
        self.loc = self.loc.unsqueeze(-1)
        self.scale = self.scale.unsqueeze(-1)
        self.n = torch.as_tensor(n)

        self.lognfact = torch.lgamma(self.n + 1)
        
        full_shape = torch.broadcast_shapes(self.loc.shape, (n,))
        self.basenorm = dist.Normal(
            self.loc.expand(full_shape),
            self.scale.expand(full_shape)
        ).to_event(1)
        batch_shape = self.basenorm.batch_shape
        event_shape = self.basenorm.event_shape
        self.stdnorm = dist.Normal(0, 1).expand(full_shape).to_event(1)

        super().__init__(batch_shape, event_shape, validate_args = validate_args)

    def rsample(self, sample_shape = torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.stdnorm.rsample(sample_shape).sort().values
        return self.loc + self.scale * eps
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.basenorm.log_prob(value) + self.lognfact


class OrderedHalfNormal(TorchDistribution):
    """Distrubtion representing the joint distribution of order statistics
    of a half ormal distribution. Useful for placing priors on layer biases to 
    impose ordering and positivity constraints."""
    
    has_rsample = True
    support = softplus_positive_ordered_vector
    arg_constraints = {
        "scale": constraints.positive, 
        "n": constraints.positive_integer
        }

    def __init__(self, n, scale, validate_args=None):
        self.scale = torch.as_tensor(scale).unsqueeze(-1)
        self.n = torch.as_tensor(n)

        self.lognfact = torch.lgamma(self.n + 1)
        
        full_shape = torch.broadcast_shapes(self.scale.shape, (n,))
        self.basenorm = dist.HalfNormal(self.scale.expand(full_shape)).to_event(1)
        batch_shape = self.basenorm.batch_shape
        event_shape = self.basenorm.event_shape
        self.stdnorm = dist.HalfNormal(1).expand(full_shape).to_event(1)

        super().__init__(batch_shape, event_shape, validate_args = validate_args)

    def rsample(self, sample_shape = torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.stdnorm.rsample(sample_shape).sort().values
        return self.scale * eps
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.basenorm.log_prob(value) + self.lognfact


def _matrixkron(A: torch.Tensor, B: torch.Tensor):
    """Performs the Kronecker product between two tensors only in their last
    two dimensions, rather than doing the product in all dimensions like
    torch.kron(). I.e., this treats the tensors like batches of matrices."""
    batch_shape = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2])
    result_shape = (A.shape[-2] * B.shape[-2], A.shape[-1] * B.shape[-1])
    return (A.unsqueeze(-1).unsqueeze(-3) * B.unsqueeze(-2).unsqueeze(-4)).reshape(*batch_shape, *result_shape)

class MatrixNormal(TorchDistribution):

    has_rsample = True
    support = real_matrix
    arg_constraints = {
        "loc": real_matrix,
        "col_scale_tril": constraints.lower_cholesky,
        "row_scale_tril": constraints.lower_cholesky
    }

    def __init__(self, loc, row_scale_tril, col_scale_tril, validate_args = None):
        # Validate shapes
        event_shape = loc.shape[-2:]
        batch_shape = torch.broadcast_shapes(
            loc.shape[:-2], row_scale_tril.shape[:-2], col_scale_tril.shape[:-2]
        )
        if (
            (row_scale_tril.shape[-2:] != (event_shape[0],) * 2)
            or (col_scale_tril.shape[-2:] != (event_shape[1],) * 2)
        ):
            raise ValueError(f"Incompatible parameter shapes: loc={loc.shape}, row_scale_tril={row_scale_tril.shape}, col_scale_tril={col_scale_tril.shape}")
        
        # Expand and store parameters
        loc_full_shape = batch_shape + event_shape
        row_full_shape = batch_shape + (event_shape[0],) * 2
        col_full_shape = batch_shape + (event_shape[1],) * 2

        self.loc = loc.expand(loc_full_shape)
        self.row_scale_tril = row_scale_tril.expand(row_full_shape)
        self.col_scale_tril = col_scale_tril.expand(col_full_shape)

        self.stdnorm = dist.Normal(0,1).expand(batch_shape + event_shape).to_event(2)

        super().__init__(batch_shape, event_shape, validate_args = validate_args)
    
    @property
    def n(self):
        return self.event_shape[0]
    
    @property
    def p(self):
        return self.event_shape[1]

    @functools.cached_property
    def V(self):
        return self.col_scale_tril @ self.col_scale_tril.mT

    @functools.cached_property
    def U(self):
        return self.row_scale_tril @ self.row_scale_tril.mT

    @functools.cached_property
    def V_inv(self):
        return torch.cholesky_inverse(self.col_scale_tril)

    @functools.cached_property
    def U_inv(self):
        return torch.cholesky_inverse(self.row_scale_tril)

    @functools.cached_property
    def V_logdet(self):
        return 2 * self.col_scale_tril.diagonal(offset=0, dim1=-1, dim2=-2).log().sum(-1)
    
    @functools.cached_property
    def U_logdet(self):
        return 2 * self.row_scale_tril.diagonal(offset=0, dim1=-1, dim2=-2).log().sum(-1)

    def rsample(self, sample_shape = torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.stdnorm.sample(sample_shape)
        return self.loc + self.row_scale_tril @ eps @ self.col_scale_tril.mT
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        normconst = torch.log(torch.tensor(2 * torch.pi)) * self.n * self.p / 2
        trace = (self.V_inv @ (value - self.loc).mT @ self.U_inv @ (value - self.loc)).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        return - trace / 2 - normconst - self.U_logdet * self.p / 2 - self.V_logdet * self.n / 2

        