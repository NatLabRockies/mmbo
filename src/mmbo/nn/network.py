import torch
from torch import nn
import math
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import pyro
from functools import partial

from .linear import bayeslinear
from .activation import bayestanh
from .normalizer import PyroNormalizer, PyroUnNormalizer
from ..dist import OrderedNormal

__all__ = ["TIBNN"]

class TIBNN(PyroModule):
    """Truncated Infinite Bayesian Neural Netowork (TIBNN)
    
    A class approximating an infinite-width Bayesian neural network by using the priors
    described in Neal (1996) and Lee et al (2018). The prior variance of the weights 
    exiting a layer of width N have variance 1/N. In the limit as N->infinity, this produces
    a Gaussian Process.

    Parameters:
    n_feat = feature dimension of input. The right-most tensor dimension *must* have this length, i.e.,
        a 1d tensor is assumed to be a single observation rather than a vector of 1d observations. A 
        vector of 1d observations should be passed in as a k-by-1 tensor.
    n_out = dimension of output. Same caveat about dimension as n_feat.
    width = width of each hidden layer
    depth = number of hidden layers
    output_bias = whether the output layer should have a bias.
    bias_ordering = whether to impose a bias ordering constraint on the hidden
        layer nodes. This removes a kind of non-identifiability from the network.
    feature_means, feature_stds = used to normalize input before passing through network
    out_means, out_stds = used to un-normalize output after coming out of the network. Having these
        norm/unnorm layers in the network lets the Bayesian layers work with observations that are
        all mean zero variance one, which is much nicer statistically.
    """

    def __init__(self, n_feat, n_out, width, depth = 1, output_bias = False,
                 bias_ordering = False,
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
        
        if bias_ordering:
            hidden_bias_prior = partial(OrderedNormal, width)
        else:
            hidden_bias_prior = dist.Normal

        # First layer: features -> hidden
        self._chain = PyroModule[nn.Sequential](
            bayeslinear(n_feat, width, 
                        weight_prior = dist.Normal(0, 1 / math.sqrt(n_feat)), 
                        bias_prior = hidden_bias_prior(0, 1)),
            bayestanh(dist.HalfCauchy(1))
        )
        # Inner layers: hidden -> hidden
        for _ in range(depth-1):
            self._chain.append(
                bayeslinear(width, width, 
                            weight_prior = dist.Normal(0, 1 / math.sqrt(width)), 
                            bias_prior = hidden_bias_prior(0, 1))
            )
            self._chain.append(bayestanh(dist.HalfCauchy(1)))
        # Last layer: hidden -> output
        self._chain.append(
            bayeslinear(width, n_out, 
                        weight_prior = dist.Normal(0, 1 / math.sqrt(width)),
                        bias = output_bias,
                        bias_prior = dist.Normal(0, 1))
        ) 

        #self.sigma = PyroSample(dist.HalfCauchy(1).expand((1,)).to_event(1))
        self.sigma = PyroSample(dist.HalfCauchy(1))

    @classmethod
    def from_data(cls, X_norm, Y_norm, *args, **kwargs):
        """Construct a TIBNN network, automatically creating the pre- and post-
        normalization layers from the empirical mean and std dev of the provided data."""
        n_feat = X_norm.shape[1]
        n_out = Y_norm.shape[1]

        net = cls(n_feat, n_out, *args, **kwargs)
        net._norm = PyroNormalizer.from_data(X_norm)
        net._unnorm = PyroUnNormalizer.from_data(Y_norm)

        return net

    def forward(self, X, y=None):

        sigma = self.sigma
        _mu = self._unnorm(self._chain(self._norm(X)))

        with pyro.plate_stack("data", X.shape[:-1], rightmost_dim=-1):
            mu = pyro.deterministic("mu", _mu, event_dim = 1)
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)

        return mu