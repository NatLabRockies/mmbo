import math
import functools
from typing import Sequence, Optional

import torch
from torch.nn.functional import linear, tanh, relu, elu, softplus, gelu, leaky_relu
from torch.distributions.multivariate_normal import _precision_to_scale_tril

from pyro.nn import PyroModule, PyroModuleList, PyroSample
import pyro.distributions as dist
import pyro
import pyro.poutine as poutine
from pyro.contrib.autoname import scope

from .normalizer import PyroNormalizer, PyroUnNormalizer
from ..dist import MatrixNormal, ConditionalMultivariateNormal

__all__ = ["TIBNN2", "LayeredTIBNN2", "RegBNN"]

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

class BNNLinearLayer(PyroModule):

    def __init__(self, insize: int, outsize: int, *, weight_scale = 1.0, bias_scale = 1.0):
        super().__init__()
        if (insize < 1) or (outsize < 1):
            raise ValueError(f"insize and outsize must be at least 1. Got {insize} and {outsize}.")
        self.register_buffer("loc", torch.zeros((outsize, insize + 1)))
        self.register_buffer(
            "scale",
            torch.cat(
                 [torch.ones(outsize, 1) * bias_scale, torch.ones(outsize, insize) * weight_scale],
                 dim = -1
            )
        )
        # Scale the weights by the inverse square root of the input size,
        # known as the Neural Tangent Kernel parameterization?
        self.register_buffer("k", torch.as_tensor(1.0 / math.sqrt(insize)))
        # Sample site for full matrix
        self.bW = PyroSample(lambda self: dist.Normal(self.loc, self.scale).to_event(2))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        bias = self.bW[:,0]
        weight = self.bW[:,1:]
        return linear(X, self.k * weight, bias)

class BNNActivationLayer(PyroModule):
    
    def __init__(self, func: str = "relu", *, scale_concentration = 2.0, scale_rate = 1.0):
        super().__init__()
        self.register_buffer("scale_concentration", torch.as_tensor(scale_concentration))
        self.register_buffer("scale_rate", torch.as_tensor(scale_rate))
        self._funcs = {
            "elu": elu, 
            "gelu": gelu,
            "leaky_relu": leaky_relu,
            "relu": relu, 
            "softplus": softplus,
            "tanh": tanh
        }
        self.func = self._funcs[func]
        # Sample site for scaling factor
        self.scale = PyroSample(lambda self: dist.Gamma(self.scale_concentration, self.scale_rate))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.func(self.scale * X)

class RegBNN(PyroModule):
    """Bayesian Neural Network for Regression class.
    
    Parameters:
    n_feat = feature dimension of input. The right-most tensor dimension *must* have this length, i.e.,
        a 1d tensor is assumed to be a single observation rather than a vector of 1d observations. A 
        vector of 1d observations should be passed in as a k-by-1 tensor.
    n_out = dimension of output. Same caveat about dimension as n_feat.
    hidden_widths = an iterable of the number of neurons in each hidden layer. Must be at least length 1.
    activation = Which activation function to use after hidden layers
    feature_means, feature_stds = used to normalize input before passing through network
    out_means, out_stds = used to un-normalize output after coming out of the network. Having these
        norm/unnorm layers in the network lets the Bayesian layers work with observations that are
        all mean zero variance one, which is much nicer statistically.
    """

    def __init__(self, n_feat, n_out, hidden_widths=None, activation="relu",
                 feature_means = None, feature_stds = None,
                 out_means = None, out_stds = None,
                 prior_variance = 0.01):
        
        super().__init__()
        
        if hidden_widths is None:
            hidden_widths = [2 * max(n_feat, n_out)]
        
        self.n_feat = n_feat
        self.n_out = n_out
        self.hidden_widths = hidden_widths
        self.prior_variance = prior_variance
        self.activation = activation

        # Training data, if supplied
        self.register_buffer("X_train", None)
        self.register_buffer("Y_train", None)

        # Create normalizers
        self.norm = PyroNormalizer(feature_means, feature_stds)
        self.unnorm = PyroUnNormalizer(out_means, out_stds)

        # Construct hidden layers
        self.layers = PyroModuleList([])
        insize = n_feat
        for outsize in hidden_widths:
            self.layers.append(
                PyroModule[torch.nn.Sequential](
                    BNNLinearLayer(insize, outsize),
                    BNNActivationLayer(activation)
                )    
            )
            insize = outsize

        # Last Layer Priors
        # precision ~ Wishart(V0, nu0)
        # V0_inv = V0^(-1), V0_tril = chol(V0)
        # bW|precision ~ MatrixNormal(0, precision^(-1), Lambda0)
        # Lambda0_inv = Lambda0^(-1), Lambda0_tril = chol(Lambda0)
        self.register_buffer("nu0", torch.as_tensor(self.n_out + 5))
        self.register_buffer("V0_tril", torch.eye(self.n_out) / math.sqrt(4 * self.prior_variance))
        self.register_buffer("V0", torch.eye(self.n_out) / (4 * self.prior_variance))
        self.register_buffer("V0_inv", torch.eye(self.n_out) * (4 * self.prior_variance))
        h_last = self.hidden_widths[-1]
        self.register_buffer("Lambda0_tril", torch.eye(h_last + 1) / math.sqrt(self.prior_variance))
        self.register_buffer("Lambda0", torch.eye(h_last + 1) / self.prior_variance)
        self.register_buffer("Lambda0_inv", torch.eye(h_last + 1) * self.prior_variance)

    @staticmethod
    def from_data(X_train: torch.Tensor, Y_train: torch.Tensor, *args, **kwargs):
        """Wrapper around the initializer for this class that determines n_feat, n_out,
        feature_means, feature_stds, out_means, and out_stds automatically from supplied
        tensors X_train and Y_train"""
        n_feat = X_train.shape[-1]
        n_out = Y_train.shape[-1]
        model = RegBNN(n_feat, n_out, *args, **kwargs)
        model.norm = PyroNormalizer.from_data(X_train)
        model.unnorm = PyroUnNormalizer.from_data(Y_train)
        model.X_train = X_train
        model.Y_train = Y_train
        return model

    def no_missing(self):
        return functools.partial(self, sample_missing = False)

    def forward(self, *, 
                X_train: torch.Tensor = None, Y_train: torch.Tensor = None, 
                X_pred: torch.Tensor = None, sample_missing: bool = True):
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

        # Pass through layers
        activation = X_normed
        if do_pred:
            activation_pred = X_normed_pred
        for idx,layer in enumerate(self.layers):
            activation = layer(activation)
            if do_pred:
                activation_pred = layer(activation_pred)
            with scope(prefix = f"layer{idx+1}"):
                with data_plate():
                    pyro.deterministic("activation", activation, event_dim = 1)
                if do_pred:
                    with pred_plate():
                        pyro.deterministic("activation_pred", activation_pred, event_dim = 1)
        
        # Last layer

        # Output multivariate normal precision matrix
        # Diagonal elements of the covariance matrix (inverse of the precision 
        # matrix) have both prior mean and prior sd of self.prior_variance
        precision = pyro.sample(
            "precision", 
            dist.Wishart(self.nu0, scale_tril = self.V0_tril, validate_args = False)
        )

        # Weights and biases with conjugate Matrix Normal prior distribution
        h_last = self.hidden_widths[-1]
        with scope(prefix = "last"):
            bW = pyro.sample(
                "bW", 
                MatrixNormal(torch.zeros((self.n_out, h_last + 1)), 
                             row_scale_tril = _precision_to_scale_tril(precision), 
                             col_scale_tril = self.Lambda0_tril)
            )
            weight = pyro.deterministic("weight", bW[:,1:])
            bias = pyro.deterministic("bias", bW[:,0])
            with data_plate():
                activation = pyro.deterministic("activation", linear(activation, weight / math.sqrt(h_last), bias), event_dim = 1)
            if do_pred:
                with pred_plate():
                    activation_pred = pyro.deterministic("activation_pred", linear(activation_pred, weight / math.sqrt(h_last), bias), event_dim = 1)
        
        # Output mean and observation
        mu_normed = activation
        if do_pred:
            mu_normed_pred = activation_pred
        with data_plate():
            mu = pyro.deterministic("mu", self.unnorm(mu_normed), event_dim = 1)
            obs_normed = pyro.sample(
                "obs_normed", 
                dist.NanMaskedMultivariateNormal(mu_normed, precision_matrix = precision, validate_args = False), 
                obs = Y_normed
            )
        if do_pred:
            with pred_plate():
                mu_pred = pyro.deterministic("mu_pred", self.unnorm(mu_normed_pred), event_dim = 1)
        # If Y_normed is None: obs_normed is sampled, Y_normed_full is fully conditioned (Delta distribution)
        # If Y_normed is not None: obs_normed is a matrix with some nans, as expected for ConditionalMultivariateNormal
        if sample_missing:
            pyro.sample(
                "Y_normed_full", 
                ConditionalMultivariateNormal(mu_normed, obs_normed, precision_matrix = precision, validate_args = False).to_event(mu_normed.ndim - 1)
            )
        
        return mu_pred if do_pred else mu


def _vector_pack(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Packs an iterable of tensors, each representing a batch of vectors of
    equal dimension, into one 2d tensor with the same length last dimension."""
    return torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=0)

def _vector_unpack(tensor: torch.Tensor, batch_shapes: Sequence[torch.Size]) -> Sequence[torch.Tensor]:
    """Undoes the operation of _vector_pack, by unpacking the first dimension
    of the 2d input tensor into one tensor per element in the iterable
    batch_shapes. Each resulting tensor will have leading dimensions according
    to the values in batch_shapes, and final dimension equal to the last dimension
    of the input tensor."""
    assert tensor.ndim == 2
    vecdim = tensor.shape[-1]
    splits = torch.split(tensor, [s.numel() for s in batch_shapes], dim=0)
    return tuple(t.reshape(*s, vecdim) for t,s in zip(splits, batch_shapes))


class LayeredRegBNN(PyroModule):
    """An architecture of two layers of RegBNN networks, with one main and all
    the others feeding their output into the main as additional features. All networks
    have the same input dimension (ignoring the additional inputs) and the inputs are assumed
    to be from the same distribution."""

    def __init__(self, n_feat: int, n_out_main: int, n_out_aux: Sequence[int] = [], *args,
                 feature_means: Optional[torch.Tensor] = None, feature_stds: Optional[torch.Tensor] = None, 
                 out_main_means: Optional[torch.Tensor] = None, out_main_stds: Optional[torch.Tensor] = None,
                 out_aux_means: Optional[Sequence[torch.Tensor]] = None, out_aux_stds: Optional[Sequence[torch.Tensor]] = None,
                 **kwargs):
        super().__init__()
        
        # Create list of auxiliary networks
        self.aux = PyroModuleList()
        for out, means, stds in zip(n_out_aux, out_aux_means, out_aux_stds):
            self.aux.append(
                RegBNN(n_feat, out, *args,
                       feature_means = feature_means, feature_stds = feature_stds,
                       out_means = means, out_stds = stds, **kwargs)
            )
        
        # Assemble the full feature means and stds if provided for the main network
        if out_aux_means is None or feature_means is None:
            feature_main_means = None
        else:
            feature_main_means = torch.cat((feature_means, *out_aux_means), dim = -1)
        if out_aux_stds is None or feature_stds is None:
            feature_main_stds = None
        else:
            feature_main_stds = torch.cat((feature_stds, *out_aux_stds), dim = -1)
        
        # Main network
        self.main = RegBNN(n_feat + sum(n_out_aux), n_out_main, *args,
                           feature_means = feature_main_means, feature_stds = feature_main_stds,
                           out_means = out_main_means, out_stds = out_main_stds)

        self.register_buffer("X_train_main")
        self.register_buffer("Y_train_main")
        self.X_train_aux = [None] * len(n_out_aux)
        self.Y_train_aux = [None] * len(n_out_aux)
        
    @staticmethod
    def from_data(X_train_main: torch.Tensor, Y_train_main: torch.Tensor,
                  X_train_aux: Sequence[torch.Tensor], Y_train_aux: Sequence[torch.Tensor],
                  *args, **kwargs) -> "LayeredRegBNN":
        X_norm = _vector_pack([X_train_main, *X_train_aux])
        n_feat = X_norm.shape[-1]
        n_out_main = Y_train_main.shape[-1]
        n_out_aux = [y.shape[-1] for y in Y_train_aux]

        net = LayeredRegBNN(n_feat, n_out_main, n_out_aux, *args, **kwargs)

        # Set normalizers in nets
        net.main.norm = PyroNormalizer.from_data(X_norm)
        net.main.unnorm = PyroUnNormalizer.from_data(Y_train_main)
        for i in range(len(n_out_aux)):
            net.aux[i].norm = PyroNormalizer.from_data(X_norm)
            net.aux[i].unnorm = PyroUnNormalizer.from_data(Y_train_aux[i])
        
        # Set internal training data buffers
        net.X_train_main = X_train_main
        net.Y_train_main = Y_train_main
        for i in range(len(n_out_aux)):
            # Hack to register a bunch of buffers because there's no such thing as a buffer list (yet)
            net.register_buffer(f"X_train_aux_{i}", X_train_aux[i])
            net.register_buffer(f"Y_train_aux_{i}", Y_train_aux[i])
            net.X_train_aux[i] = net.__getattr__(f"X_train_aux_{i}")
            net.Y_train_aux[i] = net.__getattr__(f"Y_train_aux_{i}")
        
        return net
    
    def no_missing(self):
        return functools.partial(self, sample_missing = False)

    def forward(self, *, 
                X_train_main: Optional[torch.Tensor] = None, Y_train_main: Optional[torch.Tensor] = None,
                X_train_aux: Optional[Sequence[torch.Tensor]] = None, Y_train_aux: Optional[Sequence[torch.Tensor]] = None, 
                X_pred: Optional[torch.Tensor] = None, sample_missing: bool = True):
        # Fill in training data from internal buffers if not provided
        if X_train_main is None:
            X_train_main = self.X_train_main
        if Y_train_main is None:
            Y_train_main = self.Y_train_main
        if X_train_aux is None:
            X_train_aux = self.X_train_aux
        if Y_train_aux is None:
            Y_train_aux = self.Y_train_aux

        do_pred = (X_pred is not None)

        # Auxiliary networks
        aux_out_train = [None] * len(self.aux)
        if do_pred:
            aux_out_pred = [None] * len(self.aux)
        for i in range(len(self.aux)):
            if do_pred:
                X_aux_pred = _vector_pack([X_train_main, X_pred])
            else:
                X_aux_pred = X_train_main
            with scope(prefix = f"aux_{i}"):
                aux_out = self.aux[i](X_train = X_train_aux[i], Y_train = Y_train_aux[i], X_pred = X_aux_pred, sample_missing = sample_missing)
            if do_pred:
                aux_out_train[i], aux_out_pred[i] = _vector_unpack(aux_out, (X_train_main.shape[:-1], X_pred.shape[:-1]))
            else:
                aux_out_train[i] = aux_out
        
        # Main network
        X_train_main_full = torch.cat([X_train_main, *aux_out_train], dim = -1)
        if do_pred:
            X_pred_full = torch.cat([X_pred, *aux_out_pred], dim = -1)
        else:
            X_pred_full = None
        with scope(prefix = "main"):
            out_main = self.main(X_train = X_train_main_full, Y_train = Y_train_main, X_pred = X_pred_full, sample_missing = sample_missing)
        # Rename overall predicted mean to conform to other naming
        if do_pred:
            pyro.deterministic("mu_pred", out_main, event_dim = 1)
        return out_main
