import torch
import torchbnn as bnn
from torch import nn


#TODO: out of date, redo when we decide what kind of error structure we want to use.
class GaussianNetwork(nn.Module):
    """
    Neural Network model with a mix of frequentist and Bayesian layers.
    Hidden layers are ordered frequentist first, then bayesian
    """
    def __init__(self, num_feat, width, freq_layers, bayes_layers, num_out, act=nn.LeakyReLU, prior_mu=0, prior_sigma=1, **kwargs):
        super(GaussianNetwork, self).__init__(**kwargs)

        # Copy params into self
        self._num_feat = num_feat
        self._width = width
        if freq_layers < 0 or bayes_layers < 0 or (freq_layers + bayes_layers) <= 0:
            raise Exception("Not enough layers.")
        self._freq_layers = freq_layers
        self._bayes_layers = bayes_layers
        self._num_out = num_out
        self._act = act
        self._prior_mu = prior_mu
        self._prior_sigma = prior_sigma

        # Construct the chain of layers
        # First layer
        self._chain = nn.Sequential()
        if self._freq_layers > 0:
            self._chain.append(nn.Linear(self._num_feat, self._width))
        else:
            self._chain.append(
                bnn.BayesLinear(in_features = self._num_feat, out_features = self._width,
                                prior_mu = self._prior_mu, prior_sigma = self._prior_sigma)
            )
        self._chain.append(self._act())
        # Intermediate layers
        for _ in range(self._freq_layers - 1):
            self._chain.append(nn.Linear(self._width, self._width))
            self._chain.append(self._act())
        for _ in range(self._bayes_layers - 1):
            self._chain.append(
                bnn.BayesLinear(in_features = self._width, out_features = self._width,
                                prior_mu = self._prior_mu, prior_sigma = self._prior_sigma)
            )
            self._chain.append(self._act())
        # Last layer
        if self._bayes_layers > 0:
            self._chain.append(
                bnn.BayesLinear(in_features = self._width, out_features = self._num_out,
                                prior_mu = self._prior_mu, prior_sigma = self._prior_sigma)
            )
        else:
            self._chain.append(nn.Linear(self._width, self._num_out))

        # Log error standard deviation
        self._log_sigma = nn.Parameter(torch.Tensor(1))
    
    def forward(self, inputs):
        return self._chain.forward(inputs)
    
    @property
    def log_error_sd(self):
        return self._log_sigma

