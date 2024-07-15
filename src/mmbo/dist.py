from pyro.distributions import TorchDistribution
import pyro.distributions as dist
import torch
from pyro.distributions import constraints

__all__ = ["OrderedNormal"]

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

