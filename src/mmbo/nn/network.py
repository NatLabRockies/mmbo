import math
from functools import partial, cached_property
import collections

import torch
from torch import nn
from torch.nn.functional import linear, tanh

from pyro.nn import PyroModule
import pyro.distributions as dist
import pyro
import pyro.infer.autoguide as autoguide
import pyro.poutine as poutine
from pyro.contrib.autoname import autoname, scope
from torch.distributions.multivariate_normal import _precision_to_scale_tril

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


class TIBNN2(PyroModule):
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
    prior_variance = expected residual variance, on the normalized output scale.
    fixed_scale = If provided, a constant scale value to use inside tanh function. If None, 
        the scale is considered a parameter and given a reasonable prior distribution.
    """

    def __init__(self, n_feat, n_out, width, depth = 1, 
                 feature_means = None, feature_stds = None,
                 out_means = None, out_stds = None,
                 prior_variance = 0.01, fixed_scale = None):
        
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
        self.fixed_scale = fixed_scale
        
        # Training data, if supplied
        self.register_buffer("X_train", None)
        self.register_buffer("Y_train", None)

        # Create normalizers
        self.norm = PyroNormalizer(feature_means, feature_stds)
        self.unnorm = PyroUnNormalizer(out_means, out_stds)

        # Create last layer prior hyperparameters, specifying the priors as below
        # precision ~ Wishart(V0, nu0)
        # V0_inv = V0^(-1), V0_tril = chol(V0)
        # full_matrix|precision ~ MatrixNormal(0, precision^(-1), Lambda0)
        # Lambda0_inv = Lambda0^(-1), Lambda0_tril = chol(Lambda0)
        self.register_buffer("nu0", torch.as_tensor(self.n_out + 5))
        self.register_buffer("V0_tril", torch.eye(self.n_out) / math.sqrt(4 * self.prior_variance))
        self.register_buffer("V0", torch.eye(self.n_out) / (4 * self.prior_variance))
        self.register_buffer("V0_inv", torch.eye(self.n_out) * (4 * self.prior_variance))
        self.register_buffer("Lambda0_tril", _gen_Lambda0_tril(self.width, self.prior_variance))
        self.register_buffer("Lambda0", _gen_Lambda0(self.width, self.prior_variance))
        self.register_buffer("Lambda0_inv", _gen_Lambda0_inv(self.width, self.prior_variance))

    @staticmethod
    def from_data(X_train: torch.Tensor, Y_train: torch.Tensor, *args, **kwargs):
        """Wrapper around the initializer for this class that determines n_feat, n_out,
        feature_means, feature_stds, out_means, and out_stds automatically from supplied
        tensors X_train and Y_train"""
        n_feat = X_train.shape[-1]
        n_out = Y_train.shape[-1]
        model = TIBNN2(n_feat, n_out, *args, **kwargs)
        model.norm = PyroNormalizer.from_data(X_train)
        model.unnorm = PyroUnNormalizer.from_data(Y_train)
        model.X_train = X_train
        model.Y_train = Y_train
        return model

    def no_missing(self):
        """Return a view of the model without the sample site for missing data,
        for use with HMC/NUTS/MCMC or pyro Autoguides."""
        return poutine.block(self, hide=["Y_normed_full"])

    def forward(self, *, 
                X_train: torch.Tensor = None, Y_train: torch.Tensor = None, 
                X_pred: torch.Tensor = None):

        # Fill in training data
        X_train = X_train if X_train is not None else self.X_train
        if X_train is None:
            raise ValueError("Must supply X_train")
        Y_train = Y_train if Y_train is not None else self.Y_train
        if Y_train is None:
            raise ValueError("Must supply Y_train")

        do_pred = (X_pred is not None)

        # Normalize data to have mean zero and variance one
        X_normed = self.norm.norm(X_train)
        Y_normed = self.unnorm.norm(Y_train)
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
        for layer in range(self.depth):
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
                if self.fixed_scale is None:
                    scale = pyro.sample(
                        "scale", 
                        dist.LogNormal(1, 1)
                    )
                else:
                    scale = self.fixed_scale
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
                             row_scale_tril = _precision_to_scale_tril(precision), 
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
        pyro.sample("Y_normed_full", ConditionalMultivariateNormal(mu_normed, obs_normed, precision_matrix = precision).to_event(mu_normed.ndim - 1))
        
        return mu_pred if do_pred else mu


def _vector_pack(tensors):
    """Packs an iterable of tensors, each representing a batch of vectors of
    equal dimension, into one 2d tensor with the same length last dimension."""
    return torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=0)

def _vector_unpack(tensor, batch_shapes):
    """Undoes the operation of _vector_pack, by unpacking the first dimension
    of the 2d input tensor into one tensor per element in the iterable
    batch_shapes. Each resulting tensor will have leading dimensions according
    to the values in batch_shapes, and final dimension equal to the last dimension
    of the input tensor."""
    assert tensor.ndim == 2
    vecdim = tensor.shape[-1]
    splits = torch.split(tensor, [s.numel() for s in batch_shapes], dim=0)
    return tuple(t.reshape(*s, vecdim) for t,s in zip(splits, batch_shapes))


class LayeredTIBNN2(PyroModule):
    """An architecture of two TIBNN networks, layered such that the output of one
    is concatenated to the features and passed as input to the second."""

    def __init__(self, n_feat, n_out1, n_out2, *args, 
                 feature_means = None, feature_stds = None,
                 out1_means = None, out1_stds = None, 
                 out2_means = None, out2_stds = None, **kwargs):
        
        super().__init__()

        self.aux = TIBNN2(n_feat, n_out1, *args, 
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
        self.main = TIBNN2(n_feat + n_out1, n_out2, *args,
                            feature_means = feature2_means, feature_stds = feature2_stds,
                            out_means = out2_means, out_stds = out2_stds, **kwargs)
        
        # Training data, if supplied
        self.register_buffer("X1_train", None)
        self.register_buffer("X2_train", None)
        self.register_buffer("Y1_train", None)
        self.register_buffer("Y2_train", None)

    @staticmethod
    def from_data(X1_train: torch.Tensor, X2_train: torch.Tensor, Y1_train: torch.Tensor, Y2_train: torch.Tensor, *args, **kwargs):
        """Construct a LayeredTIBNN network, automatically creating the pre- and post-
        normalization layers from the empirical mean and std dev of the provided data."""
        X_norm = _vector_pack([X1_train, X2_train])
        n_feat = X_norm.shape[1]
        n_out1 = Y1_train.shape[1]
        n_out2 = Y2_train.shape[1]

        net = LayeredTIBNN2(n_feat, n_out1, n_out2, *args, **kwargs)

        net.aux.norm = PyroNormalizer.from_data(X_norm)
        net.aux.unnorm = PyroUnNormalizer.from_data(Y1_train)

        net.main.norm = PyroNormalizer(
            torch.cat([net.aux.norm.means, net.aux.unnorm.means]),
            torch.cat([net.aux.norm.stds, net.aux.unnorm.stds])
        )
        net.main.unnorm = PyroUnNormalizer.from_data(Y2_train)

        net.X1_train = X1_train
        net.X2_train = X2_train
        net.Y1_train = Y1_train
        net.Y2_train = Y2_train

        return net
    
    def no_missing(self):
        return poutine.block(self, hide=["aux/Y_normed_full", "main/Y_normed_full"])

    def forward(self, *, 
                X1_train:torch.Tensor = None, X2_train:torch.Tensor = None, 
                Y1_train:torch.Tensor = None, Y2_train:torch.Tensor = None, 
                X_pred:torch.Tensor = None):
        # Fill in training data
        X1_train = X1_train if X1_train is not None else self.X1_train
        if X1_train is None:
            raise ValueError("Must supply X1_train")
        X2_train = X2_train if X2_train is not None else self.X2_train
        if X2_train is None:
            raise ValueError("Must supply X2_train")
        Y1_train = Y1_train if Y1_train is not None else self.Y1_train
        if Y1_train is None:
            raise ValueError("Must supply Y1_train")
        Y2_train = Y2_train if Y2_train is not None else self.Y2_train
        if Y2_train is None:
            raise ValueError("Must supply Y2_train")

        do_pred = (X_pred is not None)

        if do_pred:
            aux_X_pred = _vector_pack([X2_train, X_pred])
        else:
            aux_X_pred = X2_train
        with scope(prefix = "aux"):
            aux_mu = self.aux(X_train = X1_train, Y_train = Y1_train, X_pred = aux_X_pred)
        if do_pred:
            main_feed_train, main_feed_pred = _vector_unpack(aux_mu, [X2_train.shape[:-1], X_pred.shape[:-1]])
        else:
            main_feed_train = aux_mu
        
        main_X_train = torch.cat((X2_train, main_feed_train), dim=-1)
        if do_pred:
            main_X_pred = torch.cat((X_pred, main_feed_pred), dim=-1)
        else:
            main_X_pred = None
        with scope(prefix = "main"):
            main_mu = self.main(X_train = main_X_train, Y_train = Y2_train, X_pred = main_X_pred)
        return main_mu