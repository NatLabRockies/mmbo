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

from .network import TIBNN2, LayeredTIBNN2, _vector_pack, _vector_unpack
from ..dist import MatrixNormal
from .activation import LearnTanh

__all__ = ["TIBNN2Guide", "LayeredTIBNN2Guide"]


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

class _IndependentLinearGuide(PyroModule):
    """A PyroModule that contains parameters for a dense linear layer, and samples
    weights and biases from independent Gaussian distributions. For use in a guide."""

    def __init__(self, insize, outsize, init_style, fixed_scale = None):
        super().__init__()
        # Initial parameter values based on init_style
        weight_scale_init = torch.ones((outsize, insize)) / math.sqrt(insize)
        bias_scale_init = torch.ones((outsize,))
        if fixed_scale is None:
                scale_scale_init = torch.tensor(1.0)
        if init_style == "prior":
            weight_loc_init = torch.zeros((outsize, insize))
            bias_loc_init = torch.zeros((outsize,))
            if fixed_scale is None:
                scale_loc_init = torch.tensor(1.0)
        else:
            weight_loc_init = torch.randn((outsize, insize)) / math.sqrt(insize)
            bias_loc_init = torch.randn((outsize,))
            if fixed_scale is None:
                scale_loc_init = dist.Normal(1, 0.1).sample()
        # Create parameters
        self.weight_loc = PyroParam(weight_loc_init)
        self.weight_scale = PyroParam(weight_scale_init, constraint = constraints.positive)
        self.bias_loc = PyroParam(bias_loc_init)
        self.bias_scale = PyroParam(bias_scale_init, constraint = constraints.positive)
        self.fixed_scale = fixed_scale
        if fixed_scale is None:
            self.scale_loc = PyroParam(scale_loc_init)
            self.scale_scale = PyroParam(scale_scale_init, constraint = constraints.positive)
    
    def forward(self):
        weight = pyro.sample("weight", dist.Normal(self.weight_loc, self.weight_scale).to_event(2))
        bias = pyro.sample("bias", dist.Normal(self.bias_loc, self.bias_scale).to_event(1))
        if self.fixed_scale is None:
            scale = pyro.sample("scale", dist.LogNormal(self.scale_loc, self.scale_scale))
        else:
            scale = self.fixed_scale
        return weight,bias,scale


class _MatrixNormalLinearGuide(PyroModule):
    """A PyroModule that contains parameters for a dense linear layer, and samples
    weights and biases together from a MatrixNormal distribution. For use in a guide."""

    def __init__(self, insize, outsize, init_style, fixed_scale = None):
        super().__init__()
        # Initial parameter values based on init_style
        full_rowscale_init = torch.eye(outsize)
        full_colscale_init = torch.eye(insize + 1) / math.sqrt(insize)
        full_colscale_init[0,0] = 1.0
        if fixed_scale is None:
            scale_scale_init = torch.tensor(1.0)
        if init_style == "prior":
            full_loc_init = torch.zeros((outsize, insize + 1))
            if fixed_scale is None:
                scale_loc_init = torch.tensor(1.0)
        else:
            full_loc_init = torch.randn((outsize, insize + 1)) / math.sqrt(insize)
            if fixed_scale is None:
                scale_loc_init = dist.Normal(1, 0.1).sample()
        # Create parameters
        self.full_loc = PyroParam(full_loc_init)
        self.full_rowscale = PyroParam(full_rowscale_init, constraint = constraints.lower_cholesky)
        self.full_colscale = PyroParam(full_colscale_init, constraint = constraints.lower_cholesky)
        self.fixed_scale = fixed_scale
        if fixed_scale is None:
            self.scale_loc = PyroParam(scale_loc_init)
            self.scale_scale = PyroParam(scale_scale_init, constraint = constraints.positive)
    
    def forward(self):
        full_matrix = pyro.sample(
            "full_matrix", 
            MatrixNormal(self.full_loc, self.full_rowscale, self.full_colscale),
            infer = {"is_auxiliary": True}
        )
        weight = pyro.sample("weight", dist.Delta(full_matrix[:,1:], event_dim=2))
        bias = pyro.sample("bias", dist.Delta(full_matrix[:,0], event_dim=1))
        if self.fixed_scale is None:
            scale = pyro.sample("scale", dist.LogNormal(self.scale_loc, self.scale_scale))
        else:
            scale = self.fixed_scale
        return weight,bias,scale


class TIBNN2Guide(PyroModule):
    """A custom guide for TIBNN models that captures the kinds of posterior dependence
    that matters for the model's predictive abilities."""

    def __init__(self, model: TIBNN2, init_style = "prior", family = "independent"):
        super().__init__()

        self._model = (model,) # Do not register model as submodule
        if init_style in ["prior","random"]:
            self.init_style = init_style
        else:
            raise ValueError(f"Invalid value '{init_style}' for init_style")
        if family in ["independent", "matrixnormal"]:
            self.family = family
        else:
            raise ValueError(f"Invalid value '{family}' for family")
        
        # Parameters
        self.layers = PyroModuleList([])
        insize = model.n_feat
        outsize = model.width
        for _ in range(model.depth):
            if self.family == "independent":
                self.layers.append(_IndependentLinearGuide(insize, outsize, init_style, model.fixed_scale))
            elif self.family == "matrixnormal":
                self.layers.append(_MatrixNormalLinearGuide(insize, outsize, init_style, model.fixed_scale))
            else:
                raise Exception("How did I get here?")
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
                weight, bias, scale = self.layers[layer]()
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


class LayeredTIBNN2Guide(PyroModule):

    def __init__(self, model: LayeredTIBNN2, init_style = "prior", family = "independent"):
        super().__init__()
        self.aux = TIBNN2Guide(model.aux, init_style, family)
        self.main = TIBNN2Guide(model.main, init_style, family)
        self._model = (model,) # Do not register model as submodule

    @property
    def model(self):
        return self._model[0]

    def forward(self, 
                X1_train:torch.Tensor = None, X2_train:torch.Tensor = None, 
                Y1_train:torch.Tensor = None, Y2_train:torch.Tensor = None, 
                X_pred:torch.Tensor = None):
        # Fill in training data
        X1_train = X1_train if X1_train is not None else self.model.X1_train
        if X1_train is None:
            raise ValueError("Must supply X1_train")
        X2_train = X2_train if X2_train is not None else self.model.X2_train
        if X2_train is None:
            raise ValueError("Must supply X2_train")
        Y1_train = Y1_train if Y1_train is not None else self.model.Y1_train
        if Y1_train is None:
            raise ValueError("Must supply Y1_train")
        Y2_train = Y2_train if Y2_train is not None else self.model.Y2_train
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