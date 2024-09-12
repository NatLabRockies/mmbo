import math
from pyro.nn import PyroModule, PyroSample, PyroParam, PyroModuleList
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.distributions.transforms as transforms
import pyro
from pyro.contrib.autoname import scope
import torch
from torch import nn
from torch.nn.functional import linear, tanh
from torch.distributions.multivariate_normal import _precision_to_scale_tril

from .network import TIBNN2
from ..dist import MatrixNormal
from .activation import LearnTanh

__all__ = ["TIBNN2Guide"]


def _woodbury_inv(A_inv, X, C_inv = None):
    """Inverts a matrix of the form M = (A + X^T * C * X) using a simplified
    Woodbury identity,
    M^(-1) = A^(-1) - A^(-1) * X^T * (C^(-1) + X * A^(-1) * X^T)^(-1) * X * A^(-1)
    or
    M^(-1) = A^(-1) - A^(-1) * X^T * (I + X * A^(-1) * X^T)^(-1) * X * A^(-1)
    if C is the identity matrix. Omitting C_inv (C_inv = None) implies C = I.
    A is p-by-p, X is n-by-p, and C is n-by-n. If n < p (X is wide), this is 
    faster than directly inverting M because only a n-by-n matrix needs to be 
    inverted rather than a p-by-p matrix.
    Notice that only the inverses of A and C are required in the formula so this
    function accepts a precomputed A_inv and C_inv.
    """
    if C_inv is None:
        # C is the identity, auto-generate C_inv = I
        n = X.shape[-2]
        smaller_inv = torch.linalg.inv(torch.eye(n) + X @ A_inv @ X.mT)
    else:
        smaller_inv = torch.linalg.inv(C_inv + X @ A_inv @ X.mT)
    return A_inv - A_inv @ X.mT @ smaller_inv @ X @ A_inv

def _layer_params(insize, outsize, init_style, include_scale):
    """Create a PyroModule containing PyroParams for a dense layer."""
    if init_style == "prior":
        weight_loc_init = lambda: torch.zeros((outsize, insize))
        weight_scale_init = lambda: torch.ones((outsize, insize)) / math.sqrt(insize)
        bias_loc_init = lambda: torch.zeros((outsize,))
        bias_scale_init = lambda: torch.ones((outsize,))
        if include_scale:
            scale_loc_init = lambda: torch.tensor(1.0)
            scale_scale_init = lambda: torch.tensor(1.0)
    else:
        weight_loc_init = lambda: torch.randn((outsize, insize)) / math.sqrt(insize)
        weight_scale_init = lambda: torch.ones((outsize, insize)) / (math.sqrt(insize))
        bias_loc_init = lambda: torch.randn((outsize,))
        bias_scale_init = lambda: torch.ones((outsize,))
        if include_scale:
            scale_loc_init = lambda: dist.Normal(1, 0.1).sample()
            scale_scale_init = lambda: torch.tensor(1.0)
    # Set parameter values
    pars = PyroModule()
    pars.weight_loc = PyroParam(weight_loc_init)
    pars.weight_scale = PyroParam(weight_scale_init, constraint = constraints.positive)
    pars.bias_loc = PyroParam(bias_loc_init)
    pars.bias_scale = PyroParam(bias_scale_init, constraint = constraints.positive)
    if include_scale:
        pars.scale_loc = PyroParam(scale_loc_init)
        pars.scale_scale = PyroParam(scale_scale_init, constraint = constraints.positive)
    return pars


class TIBNN2Guide(PyroModule):
    """A custom guide for TIBNN models that captures the kinds of posterior dependence
    that matters for the model's predictive abilities."""

    def __init__(self, model: TIBNN2, init_style = "prior"):
        super().__init__()

        self._model = (model,) # Do not register model as submodule
        if init_style in ["prior","random"]:
            self.init_style = init_style
        else:
            raise ValueError(f"Invalid value '{init_style}' for init_style")
        
        # Parameters
        self.layer_params = PyroModuleList([])
        insize = model.n_feat
        outsize = model.width
        for _ in range(model.depth):
            self.layer_params.append(_layer_params(insize, outsize, init_style, model.fixed_scale is None))
            insize = model.width

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
        Lambda_tril = _precision_to_scale_tril(Lambda_inv)
        #B = Lambda @ (X.mT @ Y)
        B = torch.linalg.solve(Lambda_inv, X.mT @ Y)
        SSE_dist = (Y - X @ B).mT @ (Y - X @ B)
        Prior_dist = B.mT @ self.model.Lambda0_inv @ B
        #V_tril = torch.linalg.cholesky(torch.linalg.inv(self.model.V0_inv + SSE_dist + Prior_dist))
        V_tril = _precision_to_scale_tril(self.model.V0_inv + SSE_dist + Prior_dist)
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
        n_obs = y_present.sum()
        n_miss = y_present.logical_not().sum()
        # Prior parameters
        a0 = (self.model.nu0 - k + 1) / 2
        b0 = self.model.V0_inv[j,j] / 2
        Lambda0_inv = self.model.Lambda0_inv
        # Posterior parameters
        Lambda_inv = Lambda0_inv + X_obs.mT @ X_obs
        mu = torch.linalg.solve(Lambda_inv, X_obs.mT @ y_obs)
        a = a0 + n_obs / 2
        b = b0 + (y_obs.mT @ y_obs - mu.mT @ Lambda_inv @ mu).squeeze() / 2
        # Compile into posterior predictive and return
        Sigma = (b / a) * (torch.eye(n_miss) + X_miss @ torch.linalg.solve(Lambda_inv, X_miss.mT))
        return {
            "df": 2 * a,
            "mu": (X_miss @ mu).squeeze(-1),
            "Sigma_tril": torch.linalg.cholesky(Sigma)
        }

    def forward(self, *, X_train: torch.Tensor, Y_train: torch.Tensor, X_pred: torch.Tensor = None):

        do_pred = (X_pred is not None)

        # Normalize data with model's normalizers
        X_normed = self.model.norm.norm(X_train)
        Y_normed = self.model.unnorm.norm(Y_train)
        if do_pred:
            X_normed_pred = self.model.norm.norm(X_pred)

        # Plates to use throughout
        def data_plate():
            return pyro.plate_stack("data", X_normed.shape[:-1], rightmost_dim=-1)
        def pred_plate():
            return pyro.plate_stack("preds", X_normed_pred.shape[:-1], rightmost_dim=-1)

        # Construct all but last network layers and pass input through them
        activation = X_normed
        if do_pred:
            activation_pred = X_normed_pred
        for layer in range(self.model.depth):
            with scope(prefix = f"layer{layer}"):
                #weight = pyro.sample("weight", dist.Delta(weight_loc, event_dim=2))
                weight = pyro.sample("weight", dist.Normal(self.layer_params[layer].weight_loc, self.layer_params[layer].weight_scale).to_event(2))
                #bias = pyro.sample("bias", dist.Delta(bias_loc, event_dim=1))
                bias = pyro.sample("bias", dist.Normal(self.layer_params[layer].bias_loc, self.layer_params[layer].bias_scale).to_event(1))
                #scale = pyro.sample("scale", dist.Delta(scale_loc))
                if self.model.fixed_scale is None:
                    scale = pyro.sample("scale", dist.LogNormal(self.layer_params[layer].scale_loc, self.layer_params[layer].scale_scale))
                else:
                    scale = self.model.fixed_scale
            activation = tanh(scale * linear(activation, weight, bias))
            if do_pred:
                activation_pred = tanh(scale * linear(activation_pred, weight, bias))

        # Last layer
        # Output multivariate normal precision matrix
        # Conjugate prior with the previous layer activations serving as X

        # Append an intercept to the activations, at the training and prediciton sites
        ll_X = torch.cat([torch.ones((*activation.shape[:-1], 1)), activation], dim=-1)
        if do_pred:
            ll_X_pred = torch.cat([torch.ones((*activation_pred.shape[:-1], 1)), activation_pred], dim=-1)
        else:
            ll_X_pred = ll_X

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
            dist.Wishart(posterior["nu"], scale_tril = posterior["V_tril"])
        )
        # Weights and bias full conditional with conjugate Matrix Normal prior distribution
        with scope(prefix = "last"):
            full_matrix = pyro.sample(
                "full_matrix", 
                MatrixNormal(posterior["B"].mT, 
                             row_scale_tril = torch.linalg.cholesky(torch.linalg.inv(precision)), 
                             col_scale_tril = posterior["Lambda_tril"])
            )

        # Return the estimated means (at prediction sites if supplied)
        return self.model.unnorm(ll_X_pred @ full_matrix.mT)
