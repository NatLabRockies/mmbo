from pyro.distributions import TorchDistribution, Uniform
import pyro.distributions as dist
import torch
from pyro.distributions import constraints, transforms
from pyro.distributions.constraints import Constraint
from torch.distributions.constraint_registry import transform_to, biject_to

__all__ = ["OrderedUniform", "OrderedNormal"]

class _OrderedUnitInterval(Constraint):
    event_dim = 1

    def check(self, value):
        return torch.logical_and(
            constraints.unit_interval.check(value).all(dim=-1), 
            constraints.ordered_vector.check(value)
        )

ordered_unit_interval = _OrderedUnitInterval()

@biject_to.register(ordered_unit_interval)
@transform_to.register(ordered_unit_interval)
def _transform_to_oui(constraint):
    return transforms.ComposeTransform(
        [
            transforms.OrderedTransform(),
            transforms.SigmoidTransform()
        ]
    )


class OrderedUniform(TorchDistribution):
    """Distribution which represents an ordered sample from a multivariate Uniform(0,1)^n
    distribution. I.e. the components are jointly distributed as the order statistics of
    a Uniform distribution. Each component is thus marginally Beta-distributed."""
    
    has_rsample = True
    support = ordered_unit_interval
    arg_constraints = {}

    def __init__(self, n: int, validate_args=None):
        self.n = n
        self.lognfact = torch.lgamma(torch.tensor(n + 1))
        self.unif = Uniform(0, 1).expand((n,)).to_event(1)

        batch_shape = torch.Size(())
        event_shape = self.unif.event_shape

        super().__init__(batch_shape, event_shape, validate_args = validate_args)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.unif.sample(sample_shape).sort().values
        return eps
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.ones(value.shape[:-1]) * self.lognfact

def OrderedNormal(n, *args, **kwargs):
    """Function returning a transformed OrderedUniform distribution with the 
    inverse normal CDF - resulting in a joint distribution of the order 
    statistics of the normal distribution."""
    return dist.TransformedDistribution(
        OrderedUniform(n),
        transforms.CumulativeDistributionTransform(dist.Normal(*args, **kwargs)).inv
    )
