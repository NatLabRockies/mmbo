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
from ..dist import OrderedNormal, OrderedHalfNormal, MatrixNormal, ConditionalMultivariateNormal

__all__ = ["TIBNN2", "LayeredTIBNN2"]

def _gen_Lambda0_tril(width, prior_variance):
        mat = torch.eye(width+1) / math.sqrt(width)
        mat[1,1] = 1
        mat /= math.sqrt(prior_variance)
        return mat
    
def _gen_Lambda0(width, prior_variance):
    mat = torch.eye(width + 1) / width
    mat[1,1] = 1
    mat /= prior_variance
    return mat

def _gen_Lambda0_inv(width, prior_variance):
    mat = torch.eye(width + 1) * width
    mat[1,1] = 1
    mat *= prior_variance
    return mat

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

    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor,
                 width, depth = 1, prior_variance = 0.01):
        
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be at least one")
        if width < 1:
            raise ValueError("width must be at least one")

        # Save the input parameters for easy reference
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_feat = X_train.shape[-1]
        self.n_out = Y_train.shape[-1]
        self.n_train = X_train.shape[:-1].numel()
        self.width = width
        self.depth = depth
        self.prior_variance = prior_variance
        
        # Create normalizers
        self.norm = PyroNormalizer.from_data(self.X_train)
        self.unnorm = PyroUnNormalizer.from_data(self.Y_train)

        # Create last layer prior hyperparameters
        # precision ~ Wishart(V0, nu0)
        # V0_inv = V0^(-1), V0_tril = chol(V0)
        # full_matrix|precision ~ MatrixNormal(0, precision^(-1), Lambda0)
        # Lambda0_inv = Lambda0^(-1), Lambda0_tril = chol(Lambda0)
        self.nu0 = self.n_out + 5
        self.V0_tril = torch.eye(self.n_out) / math.sqrt(4 * self.prior_variance)
        self.V0 = torch.eye(self.n_out) / (4 * self.prior_variance)
        self.V0_inv = torch.eye(self.n_out) * (4 * self.prior_variance)
        self.Lambda0_tril = _gen_Lambda0_tril(self.width, self.prior_variance)
        self.Lambda0 = _gen_Lambda0(self.width, self.prior_variance)
        self.Lambda0_inv = _gen_Lambda0_inv(self.width, self.prior_variance)

    def no_missing(self):
        """Return a view of the model without the sample site for missing data,
        for use with HMC/NUTS/MCMC or pyro Autoguides."""
        return pyro.poutine.block(self, hide=["Y_normed_full"])

    def __call__(self, X_pred: torch.Tensor = None):

        do_pred = (X_pred is not None)

        # Normalize data to have mean zero and variance one
        X_normed = self.norm.norm(self.X_train)
        Y_normed = self.unnorm.norm(self.Y_train)
        if do_pred:
            X_normed_pred = self.norm.norm(X_pred)

        # Plates to reuse throughout
        def data_plate():
            return pyro.plate_stack("data", X_normed.shape[:-1], rightmost_dim=-1)
        def pred_plate():
            return pyro.plate_stack("preds", X_normed_pred.shape[:-1], rightmost_dim=-1)

        # Construct network layers and pass input through them
        activation = X_normed
        if do_pred:
            activation_pred = X_normed_pred
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
                if do_pred:
                    with pred_plate():
                        activation_pred = pyro.deterministic(
                            "activation_pred", 
                            tanh(scale * linear(activation_pred, weight, bias)),
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
            if do_pred:
                with pred_plate():
                    activation_pred = pyro.deterministic("activation_pred", linear(activation_pred, weight, bias), event_dim = 1)
        
        # Output mean and observation
        mu_normed = activation
        if do_pred:
            mu_normed_pred = activation_pred
        with data_plate():
            mu = pyro.deterministic("mu", self.unnorm(mu_normed), event_dim = 1)
            obs_normed = pyro.sample("obs_normed", dist.NanMaskedMultivariateNormal(mu_normed, precision_matrix = precision), obs=Y_normed)
        if do_pred:
            with pred_plate():
                mu_pred = pyro.deterministic("mu_pred", self.unnorm(mu_normed_pred), event_dim = 1)
        # If Y_normed is None: obs_normed is sampled, Y_normed_full is fully conditioned (Delta distribution)
        # If Y_normed is not None: obs_normed is a matrix with some nans, as expected for ConditionalMultivariateNormal
        Y_normed_full = pyro.sample("Y_normed_full", ConditionalMultivariateNormal(mu_normed, obs_normed, precision_matrix = precision).to_event(mu_normed.ndim - 1))
        
        return mu_pred if do_pred else mu


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