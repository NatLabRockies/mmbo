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

from .network import TIBNN2
from ..dist import MatrixNormal
from .activation import LearnTanh

__all__ = ["TIBNN2Guide"]

class MNBayesLinearGuide(PyroModule):
    # Serves as a guide for Bayesian linear layers

    def __init__(self, layer: PyroModule[nn.Linear], bias_positive = False, bias_ordering = False):
        super().__init__()

        # Weight guide
        nrow = layer.weight.shape[0]
        ncol = layer.weight.shape[1]
        self.weight_loc = PyroParam(torch.randn((nrow, ncol)) / ncol ** 0.5)
        self.weight_row_ltri = PyroParam(torch.eye(nrow) / ncol ** 0.25, constraints.lower_cholesky)
        self.weight_col_ltri = PyroParam(torch.eye(ncol) / ncol ** 0.25, constraints.lower_cholesky)
        self.weight = PyroSample(lambda self: MatrixNormal(self.weight_loc, self.weight_row_ltri, self.weight_col_ltri))
        # Bias Guide
        if layer.bias is not None:
            nbias = layer.bias.shape[0]
            if nbias == 1:
                self.bias_loc = PyroParam(torch.tensor([0.0]))
                self.bias_scale = PyroParam(torch.tensor([1.0]), constraints.positive)
                self.bias = PyroSample(lambda self: dist.Normal(self.bias_loc, self.bias_scale).to_event(1))
            elif bias_positive and bias_ordering:
                self.bias_loc = PyroParam(torch.zeros(nbias) - math.log(math.log(nbias)))
                self.bias_ltri = PyroParam(torch.eye(nbias) / 2, constraints.lower_cholesky)
                self.bias = PyroSample(
                    lambda self: dist.TransformedDistribution(
                        dist.MultivariateNormal(self.bias_loc, scale_tril = self.bias_ltri),
                        transforms.ComposeTransform([transforms.OrderedTransform(), transforms.SoftplusTransform()])
                    )
                )
            elif bias_positive:
                self.bias_loc = PyroParam(torch.zeros(nbias))
                self.bias_ltri = PyroParam(torch.eye(nbias), constraints.lower_cholesky)
                self.bias = PyroSample(
                    lambda self: dist.TransformedDistribution(
                        dist.MultivariateNormal(self.bias_loc, scale_tril = self.bias_ltri),
                        transforms.SoftplusTransform()
                    )
                )
            elif bias_ordering:
                initloc = torch.zeros(nbias) - math.log(nbias)
                initloc[0] = -1
                self.bias_loc = PyroParam(initloc)
                self.bias_ltri = PyroParam(torch.eye(nbias), constraints.lower_cholesky)
                self.bias = PyroSample(
                    lambda self: dist.TransformedDistribution(
                        dist.MultivariateNormal(self.bias_loc, scale_tril = self.bias_ltri),
                        transforms.OrderedTransform()
                    )
                )
            else:
                self.bias_loc = PyroParam(torch.randn(nbias))
                self.bias_scale = PyroParam(torch.ones(nbias) / (10 * nbias), constraints.positive)
                self.bias = PyroSample(lambda self: dist.Normal(self.bias_loc, self.bias_scale).to_event(1))
            
                

    def forward(self):
        # Access all of the parameters to issue sample statements
        # TODO: this is not ideal or clean looking. Future - change to
        #    pyro.sample() everywhere instead of PyroSample?
        _ = self.weight
        _ = self.bias

class BayesTanhGuide(PyroModule):
    # Serves as a guide for Tanh activation with a random scale parameter

    def __init__(self, layer: PyroModule[nn.Tanh]):
        super().__init__()

        # Scale guide
        self.scale_loc = PyroParam(torch.tensor(0.0))
        self.scale_scale = PyroParam(torch.tensor(1.0), constraints.positive)
        self.scale = PyroSample(lambda self: dist.LogNormal(self.scale_loc, self.scale_scale))
    
    def forward(self):
        # Access all of the parameters to issue sample statements
        # TODO: this is not ideal or clean looking. Future - change to
        #    pyro.sample() everywhere instead of PyroSample?
        _ = self.scale


class TIBNN2Guide(object):
    """A custom guide for TIBNN models that captures the kinds of posterior dependence
    that matters for the model's predictive abilities."""

    def __init__(self, model: TIBNN2, X: torch.Tensor, Y: torch.Tensor):
        self.model = model
        self.X = X
        self.Y = Y
        self.X_normed = self.model.norm.norm(X)
        self.Y_normed = self.model.unnorm.norm(Y)
        self.n = X.shape[:-1].numel()

    def lastlayer_posterior(self, X, Y):
        """Posterior parameter of precision matrix.
        precision|X,Y ~ Wishart(nu, V_tril @ V_tril.mT)
        full_matrix|X,Y,precision ~ MatrixNormal(B.mT, precision^(-1), Lambda_tril @ Lambda_tril.mT)

        This function does all of the expensive matrix calculations as few times
        as possible.
        """
        nu = self.model.nu0 + X.shape[:-1].numel()
        Lambda = torch.linalg.inv(self.model.Lambda0_inv + X.mT @ X)
        Lambda_tril = torch.linalg.cholesky(Lambda)
        B = Lambda @ (X.mT @ Y)
        SSE_dist = (Y - X @ B).mT @ (Y - X @ B)
        Prior_dist = B.mT @ self.model.Lambda0_inv @ B
        V_tril = torch.linalg.cholesky(torch.linalg.inv(self.model.V0_inv + SSE_dist + Prior_dist))
        return {
            "nu": nu,
            "Lambda_tril": Lambda_tril,
            "B": B,
            "V_tril": V_tril
        }

    # def nun(self, X, Y):
    #     """Posterior parameter of precision matrix.
    #     precision|X,Y ~ Wishart(nun, Vn_tril @ Vn_tril.mT)"""
    #     return self.model.nu0 + X.shape[:-1].numel()
    
    # def Vn_tril(self, X, Y):
    #     """Posterior parameter of precision matrix.
    #     precision|X,Y ~ Wishart(nun, Vn_tril @ Vn_tril.mT)"""
    #     V0 = self.model.V0
    #     Lambda0 = self.model.Lambda0
    #     Bn = self.Bn(X, Y)
    #     SSE_dist = (Y - X @ Bn).mT @ (Y - X @ Bn)
    #     Prior_dist = Bn.mT @ torch.linalg.inv(Lambda0) @ Bn
    #     Vn = torch.linalg.inv(torch.linalg.inv(V0) + SSE_dist + Prior_dist)
    #     return torch.linalg.cholesky(Vn)

    # def _Lambdan(self, X, Y):
    #     Lambda0 = self.model.Lambda0
    #     return torch.linalg.inv(torch.linalg.inv(Lambda0) + X.mT @ X)

    # def Lambdan_tril(self, X, Y):
    #     """Posterior parameter of last layer weights and biases.
    #     full_matrix|X,Y,precision ~ MatrixNormal(Bn, precision^(-1), Lambdan_tril @ Lambdan_tril.mT)"""
    #     return torch.linalg.cholesky(self._Lambdan(X, Y))

    # def Bn(self, X, Y):
    #     """Posterior parameter of last layer weights and biases.
    #     full_matrix|X,Y,precision ~ MatrixNormal(Bn, precision^(-1), Lambdan_tril @ Lambdan_tril.mT)"""
    #     return self._Lambdan(X, Y) @ (X.mT @ Y)

    def __call__(self, *args, **kwargs):

        # Construct network layers and pass input through them
        activation = self.X_normed
        for layer in range(1, self.model.depth+1):
            insize = activation.shape[-1]
            outsize = self.model.width
            weight_loc = pyro.param(f"layer{layer}/weight_loc", torch.randn((outsize, insize)) / math.sqrt(insize))
            weight_scale = pyro.param(f"layer{layer}/weight_scale", torch.ones((outsize, insize)) / (math.sqrt(self.n * insize)))
            bias_loc = pyro.param(f"layer{layer}/bias_loc", torch.randn((outsize,)))
            bias_scale = pyro.param(f"layer{layer}/bias_scale", torch.ones((outsize,)) / math.sqrt(self.n))
            scale_loc = pyro.param(f"layer{layer}/scale_loc", dist.Normal(1, 1).sample())
            scale_scale = pyro.param(f"layer{layer}/scale_scale", torch.tensor(1 / math.sqrt(self.n)))
            with scope(prefix = f"layer{layer}"):
                #weight = pyro.sample("weight", dist.Delta(weight_loc, event_dim=2))
                weight = pyro.sample("weight", dist.Normal(weight_loc, weight_scale).to_event(2))
                #bias = pyro.sample("bias", dist.Delta(bias_loc, event_dim=1))
                bias = pyro.sample("bias", dist.Normal(bias_loc, bias_scale).to_event(1))
                #scale = pyro.sample("scale", dist.Delta(scale_loc))
                scale = pyro.sample("scale", dist.LogNormal(scale_loc, scale_scale))
            activation = tanh(scale * linear(activation, weight, bias))

        # Last layer
        # Output multivariate normal precision matrix
        # Conjugate prior with the previous layer activations serving as X
        # Append an intercept to the activations
        ll_X = torch.cat([torch.ones((*activation.shape[:-1], 1)), activation], dim=-1)

        posterior = self.lastlayer_posterior(ll_X, self.Y_normed)

        precision = pyro.sample(
            "precision", 
            dist.Wishart(posterior["nu"], scale_tril = posterior["V_tril"])
        )

        # Weights and biases with conjugate Matrix Normal prior distribution
        with scope(prefix = "last"):
            # Previous layer activations serve as the X.
            full_matrix = pyro.sample(
                "full_matrix", 
                MatrixNormal(posterior["B"].mT, 
                             row_scale_tril = torch.linalg.cholesky(torch.linalg.inv(precision)), 
                             col_scale_tril = posterior["Lambda_tril"])
            )
