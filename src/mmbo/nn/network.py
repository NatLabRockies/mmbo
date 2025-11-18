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
from pyro.infer import Predictive

from .normalizer import PyroNormalizer, PyroUnNormalizer, mean_std
from ..dist import MatrixNormal, ConditionalMultivariateNormal

__all__ = ["RegBNN", "LayeredRegBNN", "predictor", "predictor_batch"]

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

"""A batched version of torch.nn.functional.linear using unsqueeze,
matrix multiply, and squeeze. Allows for batched weights/biases, e.g.,
for posterior samples."""
def _batch_linear(X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    return (X.unsqueeze(-2) @ weight.mT).squeeze(-2) + bias

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
        if self.training:
            bias = self.bW[:,0]
            weight = self.bW[:,1:]
            return linear(X, self.k * weight, bias)
        else:
            # Can this be done all the time?
            bias = self.bW[...,0]
            weight = self.bW[...,1:]
            return _batch_linear(X, self.k * weight, bias)

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
        if self.training:
            return self.func(self.scale * X)
        else:
            # Can this be done all the time?
            return self.func(self.scale.unsqueeze(-1) * X)

class _BaseBNN(PyroModule):
    def __init__(self, n_feat, n_out, hidden_widths, activation="relu",
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

        self.norm = PyroNormalizer(feature_means, feature_stds)
        self.unnorm = PyroUnNormalizer(out_means, out_stds)

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
        self.register_buffer("last_bW_mean", torch.zeros((self.n_out, h_last + 1)))

        self.precision = PyroSample(dist.Wishart(self.nu0, scale_tril = self.V0_tril, validate_args = False))
        self.last_bW = PyroSample(lambda self:
                MatrixNormal(self.last_bW_mean, 
                             row_scale_tril = _precision_to_scale_tril(self.precision), 
                             col_scale_tril = self.Lambda0_tril)
            )
        
    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None, label: str = "train", sample_missing: bool = True):
        
        X_normed = self.norm.norm(X)
        Y_normed = self.norm.norm(Y) if Y is not None else None

        def data_plate():
            return pyro.plate_stack("data", X_normed.shape[:-1], rightmost_dim=-1)

        # Hidden layers
        activation = X_normed
        for idx, layer in enumerate(self.layers):
            activation = layer(activation)
            with scope(prefix=label):
                with data_plate():
                    pyro.deterministic(f"activation_{idx+1}", activation, event_dim=1)
        
        # Last layer
        h_last = self.hidden_widths[-1]
        if self.training:
            ll_linear = linear
        else:
            ll_linear = _batch_linear
        weight = self.last_bW[:,1:]
        bias = self.last_bW[:,1:]
        precision = self.precision
        with scope(prefix = label):
            with data_plate():
                activation = pyro.deterministic("activation_last", ll_linear(activation, weight / math.sqrt(h_last), bias), event_dim = 1)
                mu_normed = activation
                mu = pyro.deterministic("mu", self.unnorm(mu_normed), event_dim = 1)
                obs_normed = pyro.sample(
                    "obs_normed", 
                    dist.NanMaskedMultivariateNormal(mu_normed, precision_matrix = precision, validate_args = False), 
                    obs = Y_normed
                )
            if sample_missing:
                pyro.sample(
                    "Y_normed_full", 
                    ConditionalMultivariateNormal(mu_normed, obs_normed, precision_matrix = precision, validate_args = False).to_event(mu_normed.ndim - 1)
                )
        
        return mu

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
        # The constituient base BNN. Contains all the single forward pass logic.
        self.net = _BaseBNN(n_feat, n_out, hidden_widths, activation = activation,
                            feature_means = feature_means, feature_stds = feature_stds,
                            out_means = out_means, out_stds = out_stds,
                            prior_variance = prior_variance)


    @staticmethod
    def from_data(X_train: torch.Tensor, Y_train: torch.Tensor, *args, **kwargs):
        """Wrapper around the initializer for this class that determines n_feat, n_out,
        feature_means, feature_stds, out_means, and out_stds automatically from supplied
        tensors X_train and Y_train"""
        n_feat = X_train.shape[-1]
        n_out = Y_train.shape[-1]
        X_mean,X_std = mean_std(X_train)
        Y_mean,Y_std = mean_std(Y_train)
        model = RegBNN(n_feat, n_out, *args,
                       feature_means = X_mean, feature_stds = X_std,
                       out_means = Y_mean, out_stds = Y_std,
                       **kwargs)
        return model

    def forward(self, *, 
                X_train: torch.Tensor = None, Y_train: torch.Tensor = None, 
                X_pred: torch.Tensor = None):
        
        # Feed the training data through first (if present)
        if X_train is not None:
            mu = self.net(X_train, Y_train, label="train")
        else:
            mu = None
        # Feed the prediction data through second (if present)
        if X_pred is not None:
            mu_pred = self.net(X_pred, label="pred")
        else:
            mu_pred = None

        return mu_pred if mu_pred is not None else mu


def _vector_pack(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Packs an iterable of tensors, each representing a batch of vectors of
    equal dimension, into one 2d tensor with the same length last dimension."""
    assert all(t.ndim >= 2 for t in tensors)
    return torch.cat(tensors, dim=-2)

def _vector_unpack(tensor: torch.Tensor, batch_shapes: Sequence[int]) -> Sequence[torch.Tensor]:
    """Undoes the operation of _vector_pack, by unpacking the first dimension
    of the 2d input tensor into one tensor per element in the iterable
    batch_shapes. Each resulting tensor will have leading dimensions according
    to the values in batch_shapes, and final dimension equal to the last dimension
    of the input tensor."""
    assert tensor.ndim >= 2, f"Shape is {tensor.shape}"
    return torch.split(tensor, batch_shapes, dim=-2)


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
        if out_aux_means is None:
            out_aux_means = [None] * len(n_out_aux)
        if out_aux_stds is None:
            out_aux_stds = [None] * len(n_out_aux)
        self.aux = PyroModuleList([])
        for out, means, stds in zip(n_out_aux, out_aux_means, out_aux_stds):
            self.aux.append(
                _BaseBNN(n_feat, out, *args,
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
        self.main = _BaseBNN(n_feat + sum(n_out_aux), n_out_main, *args,
                           feature_means = feature_main_means, feature_stds = feature_main_stds,
                           out_means = out_main_means, out_stds = out_main_stds)
        
    @staticmethod
    def from_data(X_train_main: torch.Tensor, Y_train_main: torch.Tensor,
                  X_train_aux: Sequence[torch.Tensor], Y_train_aux: Sequence[torch.Tensor],
                  *args, **kwargs) -> "LayeredRegBNN":
        X_norm = _vector_pack([X_train_main, *X_train_aux])
        n_feat = X_norm.shape[-1]
        n_out_main = Y_train_main.shape[-1]
        n_out_aux = [y.shape[-1] for y in Y_train_aux]
        
        X_mean,X_std = mean_std(X_norm)
        Y_main_mean,Y_main_std = mean_std(Y_train_main)
        Y_aux_mean,Y_aux_std = list(zip(*(mean_std(y) for y in Y_train_aux)))

        net = LayeredRegBNN(n_feat, n_out_main, n_out_aux, *args,   
                            feature_means = X_mean, feature_stds = X_std,
                            out_main_means = Y_main_mean, out_main_stds = Y_main_std,
                            out_aux_means = Y_aux_mean, out_aux_stds = Y_aux_std,
                            **kwargs)
        
        return net

    def forward(self, *, 
                X_train_main: Optional[torch.Tensor] = None, Y_train_main: Optional[torch.Tensor] = None,
                X_train_aux: Optional[Sequence[torch.Tensor]] = None, Y_train_aux: Optional[Sequence[torch.Tensor]] = None, 
                X_pred: Optional[torch.Tensor] = None, sample_missing: bool = True):
        
        # Training
        if X_train_main is not None:
            # Training data through aux networks
            for i in range(len(self.aux)):
                self.aux[i](X_train_aux[i], Y_train_aux[i], label=f"train/aux{i}")
            
            # Pass main training X through aux networks
            aux_out_train = [net(X_train_main, label=f"maintrain/aux{i}") for i,net in enumerate(self.aux)]
            
            # Training data through main network
            # Concatenate training data
            aux_out_train_cat = torch.cat(aux_out_train, dim=-1) # Need same shape as X_train_main except last dimension
            if X_train_main.shape[:-1] != aux_out_train_cat.shape[:-1]:
                X_train_main = X_train_main.unsqueeze(0).expand(*aux_out_train_cat.shape[:-1], -1)
            X_train_main_full = torch.cat([X_train_main, aux_out_train_cat], dim = -1)
            mu = self.main(X_train_main_full, Y_train_main, label="train/main")
        else:
            mu = None

        # Prediction
        if X_pred is not None:
            # Pass prediction X through aux networks
            aux_out_pred = [net(X_train_main, label=f"pred/aux{i}") for i,net in enumerate(self.aux)]
            aux_out_pred_cat = torch.cat(aux_out_pred, dim=-1) # Needs same shape as X_pred except last dimension
            if X_pred.shape[:-1] != aux_out_pred_cat.shape[:-1]:
                X_pred = X_pred.unsqueeze(0).expand(*aux_out_pred_cat.shape[:-1], -1)
            X_pred_full = torch.cat([X_pred, aux_out_pred_cat], dim = -1)

            # Combined prediction X through main network
            mu_pred = self.main(X_pred_full, label="pred/main")
        else:
            mu_pred = None
        
        return mu_pred if mu_pred is not None else mu


def predictor(model, posterior_samples):
    numsamples = next(iter(posterior_samples.values())).shape[0]
    def conditioned_model(idx):
        return pyro.poutine.condition(model, data = {n:posterior_samples[n][idx] for n in posterior_samples})
    def f(X_test):
        return torch.cat(
            [conditioned_model(i)(X_pred = X_test).unsqueeze(0) for i in range(numsamples)],
            dim = 0
        )
    return f

def predictor_batch(model, posterior_samples):
    """A predictor wrapper that functions by batched matrix operations, not looping over posterior samples.
    Requires the model to be in eval mode, using the batched linear layers, etc."""
    expanded_samples = {n:posterior_samples[n].unsqueeze(1) for n in posterior_samples}
    def f(X_test: torch.Tensor):
        assert X_test.ndim == 2
        with pyro.poutine.condition(data = expanded_samples):
            return model(X_pred = X_test)
    return f