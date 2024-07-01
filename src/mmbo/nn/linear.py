import torch
import torchbnn as bnn
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
import abc

def _kl_gaussian(mu_0: torch.Tensor, log_sigma_0: torch.Tensor, mu_1: float, log_sigma_1: float) :
    """
    An method for calculating KL divergence between two Normal distribtuion,
    DKL(P0||P1), where P0 = N(mu0, sigma0), and P1 = N(mu1, sigma1).

    Arguments:
        mu_0 (Tensor) : mean of normal distribution.
        log_sigma_0 (Tensor): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).
   
    """
    kl = log_sigma_1 - log_sigma_0 + \
    (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5
    return kl.sum()

def _sum_to_zero(tens: torch.Tensor):
    """Take a 1d tensor and append a value causing it to sum to zero"""
    torch.cat(tens, -tens.sum())

class _VarBayesLayer(nn.Module, metaclass=abc.ABCMeta):
    """A base bayesian NN layer with key functions that should be overridden by
    subclasses"""

    def __init__(self, *args, **kwargs):
        super(_VarBayesLayer, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def kl(self):
        """Subclasses should return the KL divergence from their prior to their
        current variational posterior."""
        raise NotImplementedError()

class IndBayesLinear(_VarBayesLayer):
    """A Bayes dense linear NN layer with allowance for separate priors for
    weights and biases. Heavily based on torchbnn.BayesLinear, which is MIT
    licensed: https://github.com/Harry24k/bayesian-neural-network-pytorch"""

    __constants__ = ['weight_prior_mu', 'weight_prior_sigma', 
                     'bias', 'bias_prior_mu', 'bias_prior_sigma', 
                     'in_features', 'out_features']

    def __init__(self, in_features, out_features,
                 weight_prior_mu = 0, weight_prior_sigma = 1,
                 bias = True, bias_prior_mu = 0, bias_prior_sigma = 1,
                 initialize = True):
        super(IndBayesLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight parameters
        self.weight_prior_mu = weight_prior_mu
        self.weight_prior_sigma = weight_prior_sigma
        self.weight_prior_log_sigma = math.log(weight_prior_sigma)
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)

        # Create bias parameters    
        if bias is None or bias is False :
            self.bias = False
            self.bias_prior_mu = None
            self.bias_prior_sigma = None
            self.bias_prior_log_sigma = None
        else :
            self.bias = True
            self.bias_prior_mu = bias_prior_mu
            self.bias_prior_sigma = bias_prior_sigma
            self.bias_prior_log_sigma = math.log(bias_prior_sigma)
            
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
        self.register_buffer('bias_eps', None)

        if initialize:
            self.initialize_()

    def freeze(self):
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias :
            self.bias_eps = torch.randn_like(self.bias_log_sigma)
        
    def unfreeze(self):
        self.weight_eps = None
        if self.bias :
            self.bias_eps = None
    
    def forward(self, input):
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
        
        if self.bias:
            if self.bias_eps is None :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps                
        else :
            bias = None
            
        return F.linear(input, weight, bias)
    
    def kl(self):
        kl = _kl_gaussian(self.weight_mu, self.weight_log_sigma, 
                          self.weight_prior_mu, self.weight_prior_log_sigma)
        if self.bias:
            kl += _kl_gaussian(self.bias_mu, self.bias_log_sigma, 
                               self.bias_prior_mu, self.bias_prior_log_sigma)
        return kl
    
    @torch.no_grad
    def initialize_(self):
        self.weight_log_sigma.uniform_(0, self.weight_prior_sigma).log_()
        init.normal_(self.weight_mu, mean = self.weight_prior_mu, std = self.weight_prior_sigma)
        if self.bias:
            self.bias_log_sigma.uniform_(0, self.bias_prior_sigma).log_()
            init.normal_(self.bias_mu, mean = self.bias_prior_mu, std = self.bias_prior_sigma)

# class MNBayesLinear(nn.Module):
#     """A Bayes dense linear NN layer that operates in a very similar way to
#     torchbnn.BayesLinear, except, the variational distribution is MatrixNormal
#     instead of independent gaussian distributions. It also allows for separate
#     prior mean and standard deviation for weights and biases."""

#     def __init__(self, in_features, out_features, weight_prior_mu=0, weight_prior_sigma=1, bias=True, bias_prior_mu=0, bias_prior_sigma=1):
#         super(MNBayesLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight_prior_mu = weight_prior_mu
#         self.weight_prior_sigma = weight_prior_sigma
#         self.bias = bias
#         self.bias_prior_mu = bias_prior_mu
#         self.bias_prior_sigma = bias_prior_sigma

#         # Set up Prior M, U, V - MatrixNormal parameters
#         if bias:
#             prior_M = torch.ones(self.out_features, self.in_features + 1) * weight_prior_mu
#             prior_M[:,0] = bias_prior_mu
#             prior_V_chol = torch.ones(self.in_features + 1, self.in_features + 1) * weight_prior_sigma
#             prior_V_chol[0,0] = bias_prior_sigma
#         else:
#             prior_M = torch.ones(self.out_features, self.in_features) * weight_prior_mu
#             prior_V_chol = torch.ones(self.in_features, self.in_features) * weight_prior_sigma
#         prior_U_chol = torch.eye(self.out_features)
#         self.register_buffer("prior_M", prior_M)
#         self.register_buffer("prior_U_chol", prior_U_chol)
#         self.register_buffer("prior_V_chol", prior_V_chol)

#         # Posterior M, U, V - MatrixNormal parameters
#         if self.bias:
#             self.M = nn.Parameter(torch.Tensor(self.out_features, self.in_features + 1))
#             self.log_V_chol_diag = nn.Parameter(torch.Tensor(self.in_features + 1))
#             self.V_chol_ltri = nn.Parameter(torch.Tensor(self.in_features * (self.in_features + 1) // 2))
#         else:
#             self.M = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
#             self.log_V_chol_diag = nn.Parameter(torch.Tensor(self.in_features))
#             self.V_chol_ltri = nn.Parameter(torch.Tensor((self.in_features - 1) * self.in_features // 2))
#         # We restrict U to have determinant 1 by restricting the log diagonal entries of
#         # its Cholesky decomposition to sum to zero
#         self.log_U_chol_diag = nn.Parameter(torch.Tensor(self.out_features- 1))
#         self.U_chol_ltri = nn.Parameter(torch.Tensor((self.out_features - 1) * self.out_features // 2))