import torch
from torch import nn
import math
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import pyro

from .linear import bayeslinear
from .activation import bayestanh
from .normalizer import PyroNormalizer, PyroUnNormalizer

__all__ = ["TIBNN"]

class TIBNN(PyroModule):

    def __init__(self, n_feat, n_out, width, depth = 1,
                 feature_means = None, feature_stds = None,
                 out_means = None, out_stds = None):
        
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be at least one")
        if width < 1:
            raise ValueError("width must be at least one")

        # Save the dimensions for easy reference
        self._n_feat = n_feat
        self._n_out = n_out

        if feature_means is None:
            feature_means = torch.zeros(n_feat)
        if feature_stds is None:
            feature_stds = torch.ones(n_feat)
        if out_means is None:
            out_means = torch.zeros(n_out)
        if out_stds is None:
            out_stds = torch.ones(n_out)

        self._norm = PyroNormalizer(feature_means, feature_stds)
        self._unnorm = PyroUnNormalizer(out_means, out_stds)
        
        # First layer: features -> hidden
        self._chain = PyroModule[nn.Sequential](
            bayeslinear(n_feat, width, 
                        weight_prior = dist.Normal(0, 1 / math.sqrt(n_feat)), 
                        bias_prior = dist.Normal(0, 1)),
            bayestanh(dist.HalfCauchy(1))
        )
        # Inner layers: hidden -> hidden
        for _ in range(depth-1):
            self._chain.append(
                bayeslinear(width, width, 
                            weight_prior = dist.Normal(0, 1 / math.sqrt(width)), 
                            bias_prior = dist.Normal(0, 1))
            )
            self._chain.append(bayestanh(dist.HalfCauchy(1)))
        # Last layer: hidden -> output
        self._chain.append(
            bayeslinear(width, n_out, 
                        weight_prior = dist.Normal(0, 1 / math.sqrt(width)), 
                        bias = False)
        ) 

        #self.sigma = PyroSample(dist.HalfCauchy(1).expand((1,)).to_event(1))
        self.sigma = PyroSample(dist.HalfCauchy(1))

    @classmethod
    def from_data(cls, X_norm, Y_norm, *args, **kwargs):
        n_feat = X_norm.shape[1]
        n_out = Y_norm.shape[1]

        net = cls(n_feat, n_out, *args, **kwargs)
        net._norm = PyroNormalizer.from_data(X_norm)
        net._unnorm = PyroUnNormalizer.from_data(Y_norm)

        return net

    def forward(self, X, y=None):
        mu = pyro.deterministic("mu", self._unnorm(self._chain(self._norm(X))), event_dim = 1)

        sigma = self.sigma

        with pyro.plate_stack("data", mu.shape[:-1], rightmost_dim=-1):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)

        return mu