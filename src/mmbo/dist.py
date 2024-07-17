from pyro.distributions import TorchDistribution
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.distributions import transforms

__all__ = ["OrderedNormal", "OrderedHalfNormal"]

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

