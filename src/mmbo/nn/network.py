import math
from functools import partial, cached_property
import collections

import torch
from torch import nn
from torch.nn.functional import linear, tanh

from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import pyro
import pyro.infer.autoguide as autoguide
import pyro.poutine as poutine
from pyro.contrib.autoname import autoname, scope

from .linear import bayeslinear, PyroMatrixLinear
from .activation import bayestanh
from .normalizer import PyroNormalizer, PyroUnNormalizer
from ..dist import OrderedNormal, OrderedHalfNormal, MatrixNormal

__all__ = ["TIBNN2", "LayeredTIBNN2"]


class TIBNN2(object):
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
    prior_variance = expected residual variance, on the normalized output scale.
    feature_means, feature_stds = used to normalize input before passing through network
    out_means, out_stds = used to un-normalize output after coming out of the network. Having these
        norm/unnorm layers in the network lets the Bayesian layers work with observations that are
        all mean zero variance one, which is much nicer statistically.
    """

    def __init__(self, n_feat, n_out, width, depth = 1, prior_variance = 0.01,
                 feature_means = None, feature_stds = None,
                 out_means = None, out_stds = None):
        
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be at least one")
        if width < 1:
            raise ValueError("width must be at least one")

        # Save the input parameters for easy reference
        self.n_feat = n_feat
        self.n_out = n_out
        self.width = width
        self.depth = depth
        self.prior_variance = prior_variance
        if feature_means is None:
            feature_means = torch.zeros(n_feat)
        if feature_stds is None:
            feature_stds = torch.ones(n_feat)
        if out_means is None:
            out_means = torch.zeros(n_out)
        if out_stds is None:
            out_stds = torch.ones(n_out)
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.out_means = out_means
        self.out_stds = out_stds

        # # First layer: features -> hidden
        # self.chain = PyroModule[nn.Sequential](
        #     bayeslinear(n_feat, width, 
        #                 weight_prior = dist.Normal(0, 1), 
        #                 bias_prior = dist.Normal(0, 1)),
        #     bayestanh(dist.LogNormal(1, 1))
        # )
        # # Inner layers: hidden -> hidden
        # for _ in range(depth-1):
        #     self.chain.append(
        #         bayeslinear(width, width, 
        #                     weight_prior = dist.Normal(0, 1 / math.sqrt(width)), 
        #                     bias_prior = dist.Normal(0, 1))
        #     )
        #     self.chain.append(bayestanh(dist.LogNormal(1, 1)))

        # self.precision = PyroSample(dist.Wishart(df = n_out + 5, scale_tril = torch.eye(n_out) / math.sqrt(4 * prior_variance)))

        # # Last layer: hidden -> output, conditional on precision matrix
        # self.lastlayer = PyroMatrixLinear(width, n_out)
        # colscale = torch.eye(width+1) / math.sqrt(width)
        # colscale[1,1] = 1
        # colscale /= math.sqrt(prior_variance)
        # self.lastlayer.full_matrix = PyroSample(
        #     lambda self: MatrixNormal(torch.zeros((n_out, width+1)), 
        #                               row_scale_tril = torch.linalg.cholesky(torch.linalg.inv(self.precision)), 
        #                               col_scale_tril = colscale)
        # )

    @classmethod
    def from_data(cls, X_norm, Y_norm, *args, **kwargs):
        """Construct a TIBNN network, automatically creating the pre- and post-
        normalization layers from the empirical mean and std dev of the provided data."""
        n_feat = X_norm.shape[1]
        n_out = Y_norm.shape[1]

        net = cls(n_feat, n_out, *args, **kwargs)
        norm = PyroNormalizer.from_data(X_norm)
        unnorm = PyroUnNormalizer.from_data(Y_norm)
        net.feature_means = norm.means
        net.feature_stds = norm.stds
        net.out_means = unnorm.means
        net.out_stds = unnorm.stds

        return net
    
    @cached_property
    def norm(self):
        return PyroNormalizer(self.feature_means, self.feature_stds)
    
    @cached_property
    def unnorm(self):
        return PyroUnNormalizer(self.out_means, self.out_stds)
    
    @cached_property
    def nu0(self):
        """Prior parameter for the precision matrix.
        precision ~ Wishart(nu0, V0_tril @ V0_tril.mT)"""
        return self.n_out + 5
    
    @cached_property
    def V0_tril(self):
        """Prior parameter for the precision matrix.
        precision ~ Wishart(nu0, V0_tril @ V0_tril.mT)"""
        return torch.eye(self.n_out) / math.sqrt(4 * self.prior_variance)
    
    @cached_property
    def V0(self):
        return torch.eye(self.n_out) / (4 * self.prior_variance)
    
    @cached_property 
    def V0_inv(self):
        return torch.eye(self.n_out) * (4 * self.prior_variance)
    
    @cached_property
    def Lambda0_tril(self):
        """Prior parameter for last layer matrix.
        full_matrix|precision ~ MatrixNormal(0, precision^(-1), Lambda0_tril @ Lambda0_tril.mT)
        In full_matrix, the first column is the biases, and the remaining matrix is
        the weights."""
        colscale_tril = torch.eye(self.width+1) / math.sqrt(self.width)
        colscale_tril[1,1] = 1
        colscale_tril /= math.sqrt(self.prior_variance)
        return colscale_tril
    
    @cached_property
    def Lambda0(self):
        mat = torch.eye(self.width + 1) / self.width
        mat[1,1] = 1
        mat /= self.prior_variance
        return mat
    
    @cached_property
    def Lambda0_inv(self):
        mat = torch.eye(self.width + 1) * self.width
        mat[1,1] = 1
        mat *= self.prior_variance
        return mat

    def __call__(self, X, y=None):

        # Normalize data to have mean zero and variance one
        X_normed = self.norm.norm(X)
        if y is None:
            y_normed = None
        else:
            y_normed = self.unnorm.norm(y)

        # Plate to reuse throughout
        def data_plate():
            return pyro.plate_stack("data", X.shape[:-1], rightmost_dim=-1)

        # Construct network layers and pass input through them
        activation = X_normed
        for layer in range(1, self.depth+1):
            insize = activation.shape[-1]
            outsize = self.width
            with scope(prefix = f"layer{layer}"):
                weight = pyro.sample(
                    "weight", 
                    dist.Normal(0, 1 / math.sqrt(insize)).expand((outsize, insize)).to_event(2)
                )
                bias = pyro.sample(
                    "bias", 
                    dist.Normal(0, 1).expand((outsize,)).to_event(1)
                )
                scale = pyro.sample(
                    "scale", 
                    dist.LogNormal(1, 1)
                )
                with data_plate():
                    activation = pyro.deterministic(
                        "activation", 
                        tanh(scale * linear(activation, weight, bias)),
                        event_dim = 1
                    )

        # Last layer
        # Output multivariate normal precision matrix
        # Diagonal elements of the covariance matrix (inverse of the precision 
        # matrix) have expectation self.prior_variance
        precision = pyro.sample(
            "precision", 
            dist.Wishart(self.nu0, scale_tril = self.V0_tril)
        )

        # Weights and biases with conjugate Matrix Normal prior distribution
        with scope(prefix = "last"):
            full_matrix = pyro.sample(
                "full_matrix", 
                MatrixNormal(torch.zeros((self.n_out, self.width+1)), 
                             row_scale_tril = torch.linalg.cholesky(torch.linalg.inv(precision)), 
                             col_scale_tril = self.Lambda0_tril)
            )
            weight = pyro.deterministic("weight", full_matrix[:,1:])
            bias = pyro.deterministic("bias", full_matrix[:,0])
            with data_plate():
                activation = pyro.deterministic("activation", linear(activation, weight, bias), event_dim = 1)
        
        # Output mean and observation
        mu_normed = activation
        with data_plate():
             mu = pyro.deterministic("mu", self.unnorm(mu_normed), event_dim = 1)
             obs = pyro.sample("obs", dist.MultivariateNormal(mu_normed, precision_matrix = precision), obs=y_normed)

        return mu


class LayeredTIBNN2(PyroModule):
    """An architecture of two TIBNN networks, layered such that the output of one
    is concatenated to the features and passed as input to the second."""

    def __init__(self, n_feat, n_out1, n_out2, *args, 
                 feature_means = None, feature_stds = None,
                 out1_means = None, out1_stds = None, 
                 out2_means = None, out2_stds = None, **kwargs):
        
        super().__init__()

        self.first = TIBNN2(n_feat, n_out1, *args, 
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
        self.second = TIBNN2(n_feat + n_out1, n_out2, *args,
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