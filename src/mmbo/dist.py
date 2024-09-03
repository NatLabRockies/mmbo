from pyro.distributions import TorchDistribution
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.distributions import transforms
import functools

from .transforms import softplus_positive_ordered_vector, real_matrix, nanmaskedreal_vector

__all__ = ["OrderedNormal", "OrderedHalfNormal", "MatrixNormal", "ConditionalMultivariateNormal"]


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


class ConditionalMultivariateNormal(TorchDistribution):
    """
    A class representing the conditional multivariate normal distribution
    of unobserved values conditioned on observed values. It's meant to serve
    as a kind of counterpart to the NanMaskedMultivariateNormal class.

    Because the observed values in this distribution function kind of like
    Delta distributions, this distribution should also not be used in MCMC.
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "condition": nanmaskedreal_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = nanmaskedreal_vector
    has_rsample = True

    def __init__(self, loc, condition, 
                 covariance_matrix=None, precision_matrix=None, scale_tril=None, 
                 validate_args=None):
        
        # Make sure any batch dimensions of condition get incorporated into the underlying distribution
        if loc.shape[-1] != condition.shape[-1]:
            raise ValueError(f"Last dimension of condition ({condition.shape[-1:]}) must match last dimension of loc ({event_shape})")

        self.unconditioned = dist.NanMaskedMultivariateNormal(
            loc.expand(torch.broadcast_shapes(loc.shape, condition.shape)), 
            covariance_matrix = covariance_matrix,
            precision_matrix = precision_matrix,
            scale_tril = scale_tril,
            validate_args = validate_args
        )

        batch_shape = self.unconditioned.batch_shape
        event_shape = self.unconditioned.event_shape

        # Save the condition values broadcasted to the full distribution shape
        self.condition = condition.expand(batch_shape + event_shape)

        super().__init__(batch_shape, event_shape, validate_args = validate_args)
        
    @property
    def covariance_matrix(self):
        return self.unconditioned.covariance_matrix
    
    @property
    def precision_matrix(self):
        return self.unconditioned.precision_matrix
    
    @property
    def scale_tril(self):
        return self.unconditioned.scale_tril
    
    @property
    def loc(self):
        return self.unconditioned.loc
    
    def rsample(self, sample_shape = torch.Size()):

        shape = self._extended_shape(sample_shape)

        result = self.condition.expand(shape).clone() # sample_shape + batch_shape + event_shape
        is_observed =  self.condition.isfinite() # batch_shape + event_shape
        patterns = sorted(set(map(tuple, is_observed.reshape(-1, *self.event_shape).tolist()))) 

        for pattern in patterns:
            col_obs = torch.tensor(pattern) # event_shape
            col_miss = col_obs.logical_not() # event_shape
            row_mask = (is_observed == col_obs).all(-1) # batch_shape
            if all(pattern):
                continue
            elif not any(pattern):
                # No conditioning, just use full distribution
                conditional = dist.MultivariateNormal(self.loc[row_mask], scale_tril = self.scale_tril[row_mask])
            else:
                # Compute conditional distribution
                mu1 = self.loc[row_mask][:,col_miss]
                mu2 = self.loc[row_mask][:,col_obs]
                Sigma11 = self.covariance_matrix[row_mask][:,col_miss][:,:,col_miss]
                Sigma22 = self.covariance_matrix[row_mask][:,col_obs][:,:,col_obs]
                Sigma12 = self.covariance_matrix[row_mask][:,col_miss][:,:,col_obs]
                x = self.condition[row_mask][:,col_obs]
                mu = mu1 + (Sigma12 @ torch.linalg.solve(Sigma22, (x - mu2).unsqueeze(-1))).squeeze(-1)
                Sigma = Sigma11 - Sigma12 @ torch.linalg.solve(Sigma22, Sigma12.mT)
                conditional = dist.MultivariateNormal(mu, Sigma)
            # Insert random samples into the appropriate places
            write_to = torch.logical_and(row_mask.unsqueeze(-1), col_miss)
            result[...,write_to]= conditional.rsample(sample_shape).reshape(*sample_shape, -1)
        
        return result
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # Instances where value does not match conditioned value
        # Automatically broadcast value and distribution shape.
        # nonmatch will have result shape
        is_observed = self.condition.isfinite()
        nonmatch = torch.logical_and(self.condition != value, is_observed).any(-1)
        # Calculate conditional density as p(y|x) = p(y,x)/p(x)
        # Automatically broadcasts in subtraction, result will have
        # result shape.
        # Fill in places where conditional values don't match with -inf
        # Performs some unnecessary computation where nonmatch == True, but
        # simplifies the code significantly.
        result = self.unconditioned.log_prob(value) - self.unconditioned.log_prob(self.condition)
        result[nonmatch] = -torch.inf
        return result