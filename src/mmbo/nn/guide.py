import math
from typing import Optional, Sequence

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

from .network import RegBNN, LayeredRegBNN, _vector_pack, _vector_unpack
from ..dist import MatrixNormal, ConditionalMultivariateNormal
from ..util import make_positive_definite

__all__ = ["RegBNNGuide", "LayeredRegBNNGuide"]

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


class RegBNNGuide(PyroModule):
    """A custom guide for RegBNN models that captures the kind of posterior dependence
    that matters for the model's predictive abilities."""

    def __init__(self, model: RegBNN, sub_ll_train: bool = False):
        super().__init__()

        self._model = (model,) # Do not register model as submodule
        self.sub_ll_train = sub_ll_train

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

        self.training_correlation = PyroParam(torch.eye(model.n_out), constraint = constraints.corr_cholesky)
        self.training_bW_mean = PyroParam(torch.randn((model.n_out, insize + 1)))
        self.training_bW_lambda_scale = PyroParam(model.Lambda0_tril @ _init_scale(insize + 1), constraint = constraints.lower_cholesky)


    @property
    def model(self):
        return self._model[0]

    def lastlayer_posterior(self, X, Y):
        """Posterior parameter of precision matrix and weight/bias matrix.
        precision|X,Y ~ Wishart(nu, V_tril @ V_tril.mT)
        full_matrix|X,Y,precision ~ MatrixNormal(B.mT, precision^(-1), Lambda_tril @ Lambda_tril.mT)

        This function does all of the expensive matrix calculations as few times
        as possible.
        """
        # Reshape the inputs to be 2-d tensors, making matrix multipliations
        # later work appropriately
        X = X.reshape((-1, X.shape[-1]))
        Y = Y.reshape((-1, Y.shape[-1]))

        nu = self.model.nu0 + X.shape[:-1].numel()
        Lambda_inv = self.model.Lambda0_inv + X.mT @ X
        #Lambda = torch.linalg.inv(Lambda_inv)
        #Lambda_tril = torch.linalg.cholesky(Lambda)
        Lambda_tril = p2st_with_retry(Lambda_inv)
        #B = Lambda_tril @ Lambda_tril.mT @ X.mT @ Y # Is this better?
        #B = torch.linalg.multi_dot([Lambda_tril, Lambda_tril.mT, X.mT, Y]) # Is this better?
        #B = torch.linalg.solve(Lambda_inv, X.mT @ Y)
        B = torch.linalg.multi_dot([Lambda_tril, Lambda_tril.mT, X.mT, Y])
        SSE_dist = (Y - X @ B).mT @ (Y - X @ B)
        Prior_dist = B.mT @ self.model.Lambda0_inv @ B
        #V_tril = torch.linalg.cholesky(torch.linalg.inv(self.model.V0_inv + SSE_dist + Prior_dist))
        V_tril = p2st_with_retry(self.model.V0_inv + SSE_dist + Prior_dist)
        return {
            "nu": nu,
            "Lambda_tril": Lambda_tril,
            "B": B,
            "V_tril": V_tril
        }
    
    def missing_posterior_predictive(self, X, Y, j):
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
    
    def forward(self, *, 
                X_train: torch.Tensor = None, Y_train: torch.Tensor = None, 
                X_pred: torch.Tensor = None):
        # Fill in training data
        X_train = X_train if X_train is not None else self.model.X_train
        if X_train is None:
            raise ValueError("Must supply X_train")
        Y_train = Y_train if Y_train is not None else self.model.Y_train
        if Y_train is None:
            raise ValueError("Must supply Y_train")

        do_pred = (X_pred is not None)

        # Normalize data with model's normalizers
        X_normed = self.model.norm.norm(X_train)
        Y_normed = self.model.unnorm.norm(Y_train)
        if do_pred:
            X_normed_pred = self.model.norm.norm(X_pred)
        
        # Pass through layers
        activation = X_normed
        if do_pred:
            activation_pred = X_normed_pred
        for layer in self.layers:
            activation = layer(activation)
            if do_pred:
                activation_pred = layer(activation_pred)
        
        # Last layer
        # Output multivariate normal precision matrix
        # Conjugate prior with the previous layer activations serving as X

        # Append an intercept to the activations, at the training and prediciton sites,
        # and scale down by the sqrt of the last width
        h_last = self.model.hidden_widths[-1]
        ll_X = _add_intercept(activation / math.sqrt(h_last))
        if do_pred:
            ll_X_pred = _add_intercept(activation_pred / math.sqrt(h_last))
        else:
            ll_X_pred = ll_X

        if self.training and self.sub_ll_train:
            # While training, use a standin distribution for the last layer with fixed
            # variance equal to the prior, effectively weighting the observations vs the prior
            precision = pyro.sample(
                "precision",
                dist.Delta(torch.linalg.inv(self.model.prior_variance * self.training_correlation), event_dim = 2)
            )
            with scope(prefix = "last"):
                bW = pyro.sample(
                    "bW",
                    MatrixNormal(self.training_bW_mean, 
                                 row_scale_tril = torch.linalg.cholesky(self.training_correlation) * math.sqrt(self.model.prior_variance),
                                 col_scale_tril = self.training_bW_lambda_scale)
                )
        else:
            # When not training, calculate the conditional last layer using all
            # training data (and imputed missing data)
            # Fill in any missing data
            Y_normed_full = Y_normed.clone()
            Y_ismissing = Y_normed_full.isfinite().logical_not()
            col_idx = torch.arange(0, Y_normed_full.shape[-1])
            with scope(prefix = "missing_data"):
                for j in range(Y_normed_full.shape[-1]):
                    if Y_ismissing[...,j].any():
                        post_pred = self.missing_posterior_predictive(ll_X, Y_normed_full, j)
                        with scope(prefix = f"{j}"):
                            y_new = pyro.sample(
                                "y_new",
                                dist.MultivariateStudentT(post_pred["df"], post_pred["mu"], post_pred["Sigma_tril"]),
                                infer = {'is_auxiliary': True}
                            )
                        write_to = torch.logical_and(Y_ismissing, col_idx == j)
                        Y_normed_full[write_to] = y_new
            pyro.sample("Y_normed_full", dist.Delta(Y_normed_full, event_dim = Y_normed_full.ndim))
            
            # Calculate last layer full conditional parameters and sample last layer sites
            posterior = self.lastlayer_posterior(ll_X, Y_normed_full)
            # Precision matrix from full conditional
            precision = pyro.sample(
                "precision", 
                dist.Wishart(posterior["nu"], scale_tril = posterior["V_tril"], validate_args = False)
            )
            # Weights and bias full conditional with conjugate Matrix Normal prior distribution
            with scope(prefix = "last"):
                bW = pyro.sample(
                    "bW", 
                    MatrixNormal(posterior["B"].mT, 
                                #row_scale_tril = torch.linalg.cholesky(torch.linalg.inv(precision)), 
                                row_scale_tril = _precision_to_scale_tril(precision),
                                col_scale_tril = posterior["Lambda_tril"])
                )

        # Return the estimated means (at prediction sites if supplied)
        return self.model.unnorm(ll_X_pred @ bW.mT)


class LayeredRegBNNGuide(PyroModule):

    def __init__(self, model: LayeredRegBNN):
        super().__init__()
        self.aux = PyroModuleList([])
        for net in model.aux:
            self.aux.append(RegBNNGuide(net))
        self.main = RegBNNGuide(model.main)
        self._model = (model,)

    @property
    def model(self):
        return self._model[0]

    def expose_main(self):
        prefix = 'main'
        def expose(msg):
            if (msg['type'] == 'sample'):
                return True
            elif (msg['name'] is not None) and msg['name'].startswith(prefix):
                return True
            else:
                return False
        return pyro.poutine.block(self, expose_fn = expose)
    
    def expose_aux(self, num=None):
        sample_prefix = 'aux' if num is None else f'aux_{num}'
        param_prefix = 'aux' if num is None else f'aux.{num}'
        def expose(msg):
            if (msg['type'] == 'param') and msg['name'].startswith(param_prefix):
                return True
            elif (msg['type'] == 'sample') and msg['name'].startswith(sample_prefix):
                return True
            else:
                return False
        return pyro.poutine.block(self, expose_fn = expose)

    def forward(self, *, 
                X_train_main: Optional[torch.Tensor] = None, Y_train_main: Optional[torch.Tensor] = None,
                X_train_aux: Optional[Sequence[torch.Tensor]] = None, Y_train_aux: Optional[Sequence[torch.Tensor]] = None, 
                X_pred: Optional[torch.Tensor] = None):
        # Fill in training data from internal buffers if not provided
        X_train_main = X_train_main if X_train_main is not None else self.model.X_train_main
        if X_train_main is None:
            raise ValueError("Must supply X_train_main")
        Y_train_main = Y_train_main if Y_train_main is not None else self.model.Y_train_main
        if Y_train_main is None:
            raise ValueError("Must supply Y_train_main")
        X_train_aux = X_train_aux if X_train_aux is not None else self.model.X_train_aux
        if X_train_aux is None:
            raise ValueError("Must supply X_train_aux")
        Y_train_aux = Y_train_aux if Y_train_aux is not None else self.model.Y_train_aux
        if Y_train_aux is None:
            raise ValueError("Must supply Y_train_aux")

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
                aux_out = self.aux[i](X_train = X_train_aux[i], Y_train = Y_train_aux[i], X_pred = X_aux_pred)
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
            out_main = self.main(X_train = X_train_main_full, Y_train = Y_train_main, X_pred = X_pred_full)
        
        return out_main
