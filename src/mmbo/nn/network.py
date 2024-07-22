import torch
from torch import nn
import math
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import pyro
from functools import partial
import pyro.infer.autoguide as autoguide
import pyro.poutine as poutine
from pyro.contrib.autoname import autoname, scope

from .linear import bayeslinear
from .activation import bayestanh
from .normalizer import PyroNormalizer, PyroUnNormalizer
from ..dist import OrderedNormal, OrderedHalfNormal

__all__ = ["TIBNN", "LayeredTIBNN"]


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
    residual_prior = prior for the residual standard deviation, sigma.
    bias_ordering = whether to impose a bias ordering constraint on the hidden
        layer nodes. This removes a kind of non-identifiability from the network's
        parameters.
    bias_positive = whether to restrict biases to a Tanh activated node
        to be positive. This removes another kind of non-identifiability from the
        network's parameters.
    feature_means, feature_stds = used to normalize input before passing through network
    out_means, out_stds = used to un-normalize output after coming out of the network. Having these
        norm/unnorm layers in the network lets the Bayesian layers work with observations that are
        all mean zero variance one, which is much nicer statistically.
    """

    def __init__(self, n_feat, n_out, width, depth = 1, output_bias = True,
                 residual_prior = dist.HalfCauchy(1),
                 bias_ordering = False, bias_positive = False,
                 feature_means = None, feature_stds = None,
                 out_means = None, out_stds = None):
        
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be at least one")
        if width < 1:
            raise ValueError("width must be at least one")

        # Save the input parameters for easy reference
        self._n_feat = n_feat
        self._n_out = n_out
        self._width = width
        self._depth = depth
        self._output_bias = output_bias
        self._bias_ordering = bias_ordering
        self._bias_positive = bias_positive

        if feature_means is None:
            feature_means = torch.zeros(n_feat)
        if feature_stds is None:
            feature_stds = torch.ones(n_feat)
        if out_means is None:
            out_means = torch.zeros(n_out)
        if out_stds is None:
            out_stds = torch.ones(n_out)

        self.norm = PyroNormalizer(feature_means, feature_stds)
        self.unnorm = PyroUnNormalizer(out_means, out_stds)
        
        # Determine the prior of the biases based on supplied constraints
        if bias_ordering and bias_positive:
            hidden_bias_prior = partial(OrderedHalfNormal, width)
        elif bias_ordering:
            hidden_bias_prior = partial(OrderedNormal, width, 0)
        elif bias_positive:
            hidden_bias_prior = dist.HalfNormal
        else:
            hidden_bias_prior = partial(dist.Normal, 0)

        # First layer: features -> hidden
        self.chain = PyroModule[nn.Sequential](
            bayeslinear(n_feat, width, 
                        weight_prior = dist.Normal(0, 1), 
                        bias_prior = hidden_bias_prior(1)),
            bayestanh(dist.HalfCauchy(1))
        )
        # Inner layers: hidden -> hidden
        for _ in range(depth-1):
            self.chain.append(
                bayeslinear(width, width, 
                            weight_prior = dist.Normal(0, 1 / math.sqrt(width)), 
                            bias_prior = hidden_bias_prior(1))
            )
            self.chain.append(bayestanh(dist.HalfCauchy(1)))
        # Last layer: hidden -> output
        self.chain.append(
            bayeslinear(width, n_out, 
                        weight_prior = dist.Normal(0, 1 / math.sqrt(width)),
                        bias = output_bias,
                        bias_prior = dist.Normal(0, 1))
        ) 

        self.sigma = PyroSample(residual_prior.expand((n_out,)).to_event(1))

    @classmethod
    def from_data(cls, X_norm, Y_norm, *args, **kwargs):
        """Construct a TIBNN network, automatically creating the pre- and post-
        normalization layers from the empirical mean and std dev of the provided data."""
        n_feat = X_norm.shape[1]
        n_out = Y_norm.shape[1]

        net = cls(n_feat, n_out, *args, **kwargs)
        net.norm = PyroNormalizer.from_data(X_norm)
        net.unnorm = PyroUnNormalizer.from_data(Y_norm)

        return net

    def forward(self, X, y=None):

        sigma = self.sigma
        _mu = self.unnorm(self.chain(self.norm(X)))

        with pyro.plate_stack("data", X.shape[:-1], rightmost_dim=-1):
            mu = pyro.deterministic("mu", _mu, event_dim = 1)
            obs = pyro.sample("obs", dist.NanMaskedNormal(mu, sigma).to_event(1), obs=y)

        return mu


class LayeredTIBNN(PyroModule):
    """An architecture of two TIBNN networks, layered such that the output of one
    is concatenated to the features and passed as input to the second."""

    def __init__(self, n_feat, n_out1, n_out2, *args, 
                 feature_means = None, feature_stds = None,
                 out1_means = None, out1_stds = None, 
                 out2_means = None, out2_stds = None, **kwargs):
        
        super().__init__()

        self.first = TIBNN(n_feat, n_out1, *args, 
                           feature_means = feature_means, feature_stds = feature_stds,
                           out_means = out1_means, out_stds = out1_stds, **kwargs)
        if feature_means is None or out1_means is None:
            feature2_means = None
        else:
            feature2_means = torch.cat((feature_means, out1_means), dim = -1)
        if feature_stds is None or out1_stds is None:
            feature2_stds = None
        else:
            feature2_stds = torch.cat((feature_stds, out1_stds), dim = -1)
        self.second = TIBNN(n_feat + n_out1, n_out2, *args,
                            feature_means = feature2_means, feature_stds = feature2_stds,
                            out_means = out2_means, out_stds = out2_stds, **kwargs)

    @classmethod
    def from_data(cls, X_norm, Y1_norm, Y2_norm, *args, **kwargs):
        """Construct a LayeredTIBNN network, automatically creating the pre- and post-
        normalization layers from the empirical mean and std dev of the provided data."""
        n_feat = X_norm.shape[1]
        n_out1 = Y1_norm.shape[1]
        n_out2 = Y2_norm.shape[1]

        net = cls(n_feat, n_out1, n_out2, *args, **kwargs)

        net.first.norm = PyroNormalizer.from_data(X_norm)
        net.first.unnorm = PyroUnNormalizer.from_data(Y1_norm)

        net.second.norm = PyroNormalizer.from_data(
            torch.cat((X_norm, Y1_norm), dim=-1)
        )
        net.second.unnorm = PyroUnNormalizer.from_data(Y2_norm)

        return net
    
    def forward(self, X, y1=None, y2=None):
        with scope(prefix = "first"):
            first_mu = self.first(X, y1)
        last_input = torch.cat((X, first_mu), dim=-1)
        with scope(prefix = "second"):
            last_mu = self.second(last_input, y2)
        return last_mu