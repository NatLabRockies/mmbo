import math
from typing import Optional, Sequence, Union

from pyro.nn import PyroModule, PyroSample, PyroParam, PyroModuleList
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.distributions.transforms as transforms
import pyro
from pyro.contrib.autoname import scope
import torch
from torch import nn
from torch.nn.functional import linear, tanh, relu, elu, softplus, gelu, leaky_relu
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from pyro.infer import Predictive

from .network import RegBNN, LayeredRegBNN, _BaseBNN, _vector_pack, _vector_unpack, _batch_linear
from ..dist import MatrixNormal, ConditionalMultivariateNormal
from ..util import make_positive_definite

__all__ = ["RegBNNGuide", "LayeredRegBNNGuide", "posterior_samples"]

def make_positive_definite_flipped(A: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return make_positive_definite(A.flip((-2, -1)), *args, **kwargs).flip((-2, -1))

def cholesky_with_retry(A: torch.Tensor, tries: int = 100) -> torch.Tensor:
    result = None
    cnt = 0
    while (result is None) and (cnt < tries):
        cnt += 1
        try:
            result = torch.linalg.cholesky(A)
        except torch.linalg.LinAlgError as e:
            if cnt == tries:
                raise Exception("Positive definite correction tries exceeded.")
            result = None
            A_new = make_positive_definite(A)
            if (A == A_new).all():
                raise Exception("Positive definite correction failed to change matrix.")
            A = A_new
    return result

def p2st_with_retry(A: torch.Tensor, tries: int = 100) -> torch.Tensor:
    result = None
    cnt = 0
    while (result is None) and (cnt < tries):
        cnt += 1
        try:
            result = _precision_to_scale_tril(A)
        except torch.linalg.LinAlgError as e:
            if cnt == tries:
                raise Exception("Positive definite correction tries exceeded.")
            result = None
            A_new = make_positive_definite_flipped(A)
            if (A == A_new).all():
                raise Exception("Positive definite correction failed to change matrix.")
            A = A_new
    return result

def pd_inv_with_retry(A: torch.Tensor, tries: int = 100) -> torch.Tensor:
    result = None
    cnt = 0
    while (result is None) and (cnt < tries):
        cnt += 1
        try:
            result = torch.linalg.inv(A)
        except torch.linalg.LinAlgError as e:
            if cnt == tries:
                raise Exception("Positive definite correction tries exceeded.")
            result = None
            A_new = make_positive_definite(A)
            if (A == A_new).all():
                raise Exception("Positive definite correction failed to change matrix.")
            A = A_new
    return result

def _woodbury_inv(A_inv_diag: torch.Tensor, X: torch.Tensor, C_inv: torch.Tensor = None):
    """Inverts a matrix of the form M = (A + X^T * C * X), where A is diagonal,
    using a simplified Woodbury identity,
    M^(-1) = A^(-1) - A^(-1) * X^T * (C^(-1) + X * A^(-1) * X^T)^(-1) * X * A^(-1)
    Omitting C_inv (C_inv = None) implies C = I.
    A is p-by-p, X is n-by-p, and C is n-by-n. If n < p (X is wide), this is 
    faster than directly inverting M because only a n-by-n matrix needs to be 
    inverted rather than a p-by-p matrix.
    Notice that only the inverses of A and C are required in the formula so this
    function accepts a precomputed A_inv_diag and C_inv.
    """
    if C_inv is None:
        # C is the identity, auto-generate C_inv = I
        n = X.shape[-2]
        device = A_inv_diag.device
        dtype = A_inv_diag.dtype
        C_inv = torch.eye(n, device = device, dtype = dtype)
    X_scaled = X * A_inv_diag.unsqueze(-2) # Column-scaled X, as if post-multiplied by A^(-1) 
    B = C_inv + X_scaled @ X.mT
    return A_inv_diag.diag_embed() - X_scaled.mT @ torch.linalg.solve(B, X_scaled)

def _init_scale(n: int) -> torch.Tensor:
    """Generate a random cholesky factor of a positive definite matrix
    suitable for initializing MatrixNormal scale parameters."""
    k = 2 * n
    sigma = torch.rand(n)
    x = sigma * torch.randn((k,n))
    return torch.linalg.cholesky((torch.eye(n) + x.mT @ x / k) / 2)

def _add_intercept(X: torch.Tensor) -> torch.Tensor:
    """Add a 'column' of ones to the left size of the last dimension.
    If X.shape == (..., k) then the return shape will be (..., k+1)."""
    intercept = torch.ones((*X.shape[:-1], 1), dtype = X.dtype, device = X.device)
    return torch.cat([intercept, X], dim=-1)


class BNNLinearLayerGuide(PyroModule):

    def __init__(self, insize: int, outsize: int):
        super().__init__()
        self.loc = PyroParam(torch.randn((outsize, insize + 1)))
        #self.row_scale = PyroParam(_init_scale(outsize), constraint = constraints.lower_cholesky)
        #self.col_scale = PyroParam(_init_scale(insize + 1), constraint = constraints.lower_cholesky)
        self.scale = PyroParam(torch.rand((outsize, insize + 1)), constraint = constraints.positive)
        # Scale the weights by the inverse square root of the input size,
        # known as the Neural Tangent Kernel parameterization?
        self.register_buffer("k", torch.as_tensor(1.0 / math.sqrt(insize)))
        # Sample site for weight/bias matrix - Matrixnormal approximate distribution
        #self.bW = PyroSample(lambda self: MatrixNormal(self.loc, self.row_scale, self.col_scale))
        self.bW = PyroSample(lambda self: dist.Normal(self.loc, self.scale).to_event(2))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        bias = self.bW[:,0]
        weight = self.bW[:,1:]
        return linear(X, self.k * weight, bias)

class BNNActivationLayerGuide(PyroModule):

    def __init__(self, func: str = "relu"):
        super().__init__()
        self.scale_val = PyroParam(torch.as_tensor(1.0), constraint = constraints.positive)
        # Sample site for scaling factor - MAP estimate
        self.scale = PyroSample(lambda self: dist.Delta(self.scale_val))
        self._funcs = {
            "elu": elu, 
            "gelu": gelu,
            "leaky_relu": leaky_relu,
            "relu": relu, 
            "softplus": softplus,
            "tanh": tanh 
        }
        self.func = self._funcs[func]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.func(self.scale * X)

class _BaseBNNGuide(PyroModule):

    def __init__(self, model: _BaseBNN):
        super().__init__()

        self._model = (model,) # Do not register model as submodule

        self.layers = PyroModuleList([])
        insize = model.n_feat
        for outsize in model.hidden_widths:
            self.layers.append(
                PyroModule[torch.nn.Sequential](
                    BNNLinearLayerGuide(insize, outsize),
                    BNNActivationLayerGuide(model.activation)
                )    
            )
            insize = outsize
        
        # Values used to calculate the full conditional distribution of the last layer parameters
        self._training_activations = None
        self._Y_train = None
        self._posterior_params = {}

        # Important note: Because of the caching in PyroModules with PyroSample members,
        # these sampling statements will only be executed once, on the first forward() pass
        # through the module. In the intended use of this class (as a submodule of an outer 
        # module), therefore the first call of the model MUST have the training data on which the
        # posterior is calculated.
        self.precision = PyroSample(lambda self: 
                dist.Wishart(self._training_nu, scale_tril = self._training_V_tril, validate_args = False)
            )
        self.last_bW = PyroSample(lambda self:
                MatrixNormal(self._training_last_bW_mean.mT, 
                             row_scale_tril = _precision_to_scale_tril(self.precision), 
                             col_scale_tril = self._training_Lambda_tril)
            )

    @property
    def model(self):
        return self._model[0]
    
    def _set_last_layer_training(self, activations: torch.Tensor, Y:torch.Tensor):
        assert activations.shape[:-1] == Y.shape[:-1]
        self._training_activations = activations
        self._Y_train = Y
        self._posterior_params = {}

    def _clear_last_layer_training(self):
        self._training_activations = None
        self._Y_train = None
        self._posterior_params = None

    @property
    def _last_layer_training_is_set(self):
        return (self._training_activations is not None) and (self._Y_train is not None)

    def _missing_posterior_predictive(self, X, Y, j):
        """Posterior predictive parameters for missing observations in a single
        output. The joint posterior predictive distribution of new observations in
        Bayesian linear regression is multivariate T with 'df' degrees of freedom,
        location 'mu', and lower triangular scale 'Sigma_tril'.
        """
        # Extract/process data
        y = Y[...,j]
        k = self.model.n_out
        y_present = y.isfinite()
        y_obs = y[y_present].unsqueeze(-1) # Make n-by-1
        X_obs = X[y_present,:]
        X_miss = X[y_present.logical_not(),:]
        n_obs = y_present.sum().item()
        n_miss = y_present.logical_not().sum().item()
        # Prior parameters
        a0 = (self.model.nu0 - k + 1) / 2
        b0 = self.model.V0_inv[j,j] / 2
        Lambda0_inv = self.model.Lambda0_inv
        # Posterior parameters
        Lambda_inv = Lambda0_inv + X_obs.mT @ X_obs
        Lambda = pd_inv_with_retry(Lambda_inv)
        #mu = torch.linalg.solve(Lambda_inv, X_obs.mT @ y_obs)
        mu = Lambda @ X_obs.mT @ y_obs
        a = a0 + n_obs / 2
        #b = b0 + (y_obs.mT @ y_obs - mu.mT @ Lambda_inv @ mu).squeeze() / 2
        b = b0 + (y_obs.mT @ (torch.eye(n_obs, dtype=X_miss.dtype, device=X_miss.device) - X_obs @ Lambda @ X_obs.mT) @ y_obs).squeeze().clamp(min = -1.99 * b0) / 2
        # Compile into posterior predictive and return
        Sigma = (b / a) * (torch.eye(n_miss, dtype=X_miss.dtype, device=X_miss.device) + 
                           X_miss @ Lambda @ X_miss.mT)
        Sigma_tril = cholesky_with_retry(Sigma)
        return {
            "df": 2 * a,
            "mu": (X_miss @ mu).squeeze(-1),
            "Sigma_tril": Sigma_tril
        }
    
    def _get_full_Y(self, X, Y_normed):
        Y_normed_full = Y_normed.clone()
        Y_ismissing = Y_normed_full.isfinite().logical_not()
        col_idx = torch.arange(0, Y_normed_full.shape[-1])
        with scope(prefix = "missing_data"):
            for j in range(Y_normed_full.shape[-1]):
                if Y_ismissing[...,j].any():
                    post_pred = self._missing_posterior_predictive(X, Y_normed_full, j)
                    with scope(prefix = f"{j}"):
                        y_new = pyro.sample(
                            "y_new",
                            dist.MultivariateStudentT(post_pred["df"], post_pred["mu"], post_pred["Sigma_tril"]),
                            infer = {'is_auxiliary': True}
                        )
                    write_to = torch.logical_and(Y_ismissing, col_idx == j)
                    Y_normed_full[write_to] = y_new
        return pyro.sample("Y_normed_full", dist.Delta(Y_normed_full, event_dim = Y_normed_full.ndim))

    @property
    def _training_nu(self):
        if "training_nu" not in self._posterior_params:
            assert self._last_layer_training_is_set
            training_nu = self.model.nu0 + self._Y_train.shape[:-1].numel()
            self._posterior_params["training_nu"] = training_nu
        return self._posterior_params["training_nu"]

    @property
    def _training_V_tril(self):
        if "training_V_tril" not in self._posterior_params:
            assert self._last_layer_training_is_set
            B = self._training_last_bW_mean
            X = self._training_activations
            Y = self._Y_train
            SSE_dist = (Y - X @ B).mT @ (Y - X @ B)
            Prior_dist = B.mT @ self.model.Lambda0_inv @ B
            training_V_tril = p2st_with_retry(self.model.V0_inv + SSE_dist + Prior_dist)
            self._posterior_params["training_V_tril"] = training_V_tril
        return self._posterior_params["training_V_tril"]
    
    @property
    def _training_last_bW_mean(self):
        if "training_last_bW_mean" not in self._posterior_params:
            assert self._last_layer_training_is_set
            Lambda_tril = self._training_Lambda_tril
            X = self._training_activations
            Y = self._Y_train
            training_last_bW_mean = torch.linalg.multi_dot([Lambda_tril, Lambda_tril.mT, X.mT, Y])
            self._posterior_params["training_last_bW_mean"] = training_last_bW_mean
        return self._posterior_params["training_last_bW_mean"]

    @property
    def _training_Lambda_tril(self):
        if "training_Lambda_tril" not in self._posterior_params:
            assert self._last_layer_training_is_set
            X = self._training_activations
            Lambda_inv = self.model.Lambda0_inv + X.mT @ X
            self._posterior_params["training_Lambda_tril"] = p2st_with_retry(Lambda_inv)
        return self._posterior_params["training_Lambda_tril"]

    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None, label: str = "train"):
        
        X_normed = self.model.norm.norm(X)
        Y_normed = self.model.unnorm.norm(Y) if Y is not None else None

        # Hidden layers
        activation = X_normed
        for layer in self.layers:
            activation = layer(activation)

        # Last layer

        # Append an intercept to the activations, at the training sites,
        # and scale down by the sqrt of the last width
        h_last = self.model.hidden_widths[-1]
        ll_X = _add_intercept(activation / math.sqrt(h_last))
        
        # Calculate missing Y values (if any) (if needed)
        # If a Y is passed in, calculate missing values. Otherwise, don't
        if Y is not None:
            with scope(prefix=label):
                # Calculate missing Y, store full Y, issue sample statement.
                Y_full = self._get_full_Y(ll_X, Y_normed)
            # Set training data
            self._set_last_layer_training(ll_X, Y_full)

        # Sample the last layer parameters
        # (just need to reference them to issue pyro.sample satements)
        weight = self.last_bW[:,1:]
        bias = self.last_bW[:,0]
        precision = self.precision

        # Clear training data
        self._clear_last_layer_training()

        return self.model.unnorm(linear(activation, weight, bias))

        

class RegBNNGuide(PyroModule):
    """A custom guide for RegBNN models that captures the kind of posterior dependence
    that matters for the model's predictive abilities."""

    def __init__(self, model: RegBNN, sub_ll_train: bool = False):
        super().__init__()

        self.net = _BaseBNNGuide(model.net)
        self._model = (model,)

    @property
    def model(self):
        return self._model[0]
    
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


class LayeredRegBNNGuide(PyroModule):

    def __init__(self, model: LayeredRegBNN):
        super().__init__()
        self.aux = PyroModuleList([])
        for net in model.aux:
            self.aux.append(_BaseBNNGuide(net))
        self.main = _BaseBNNGuide(model.main)
        self._model = (model,)

    @property
    def model(self):
        return self._model[0]

    def forward(self, *, 
                X_train_main: Optional[torch.Tensor] = None, Y_train_main: Optional[torch.Tensor] = None,
                X_train_aux: Optional[Sequence[torch.Tensor]] = None, Y_train_aux: Optional[Sequence[torch.Tensor]] = None, 
                X_pred: Optional[torch.Tensor] = None):
        # Training
        if X_train_main is not None:
            # Training data through aux networks
            for i in range(len(self.aux)):
                self.aux[i](X_train_aux[i], Y_train_aux[i], label=f"train_aux{i}")
            
            # Pass main training X through aux networks
            aux_out_train = [self.aux[i](X_train_main, label=f"maintrain_aux{i}") for i in range(len(self.aux))]
            
            # Training data through main network
            # Concatenate training data
            aux_out_train_cat = torch.cat(aux_out_train, dim=-1) # Need same shape as X_train_main except last dimension
            if X_train_main.shape[:-1] != aux_out_train_cat.shape[:-1]:
                X_train_main = X_train_main.unsqueeze(0).expand(*aux_out_train_cat.shape[:-1], -1)
            X_train_main_full = torch.cat([X_train_main, aux_out_train_cat], dim = -1)
            mu = self.main(X_train_main_full, Y_train_main, label="train_main")
        else:
            mu = None

        # Prediction
        if X_pred is not None:
            # Pass prediction X through aux networks
            aux_out_pred = [self.aux[i](X_pred, label=f"pred_aux{i}") for i in range(len(self.aux))]
            aux_out_pred_cat = torch.cat(aux_out_pred, dim=-1) # Needs same shape as X_pred except last dimension
            if X_pred.shape[:-1] != aux_out_pred_cat.shape[:-1]:
                X_pred = X_pred.unsqueeze(0).expand(*aux_out_pred_cat.shape[:-1], -1)
            X_pred_full = torch.cat([X_pred, aux_out_pred_cat], dim = -1)

            # Combined prediction X through main network
            mu_pred = self.main(X_pred_full, label="pred_main")
        else:
            mu_pred = None
        
        return mu_pred if mu_pred is not None else mu


def posterior_samples(guide: Union[RegBNNGuide, LayeredRegBNNGuide], params, num_samples, *args, **kwargs):
    """Take a fitted guide and produce samples from it."""
    with pyro.get_param_store().scope(params):
        guide.eval()
        pred = Predictive(guide.model, guide = guide, num_samples = num_samples)
        samples = pred.get_samples(*args, **kwargs)
        return {n:samples[n].squeeze(1) for n in samples.keys() if not n.startswith("pred")}