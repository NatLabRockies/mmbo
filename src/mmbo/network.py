import torch
import torchbnn as bnn
from torch import nn


class GaussianNetwork(nn.Module):
    """
    Neural Network model with a mix of frequentist and Bayesian layers.
    First layer is always frequentist, last layer is always Bayesian. Hidden layers
    are ordered frequentist first, then bayesian
    """
    def __init__(self, num_feat, width, freq_hidden_layers, bayes_hidden_layers, num_out, act=nn.LeakyReLU, prior_mu=0, prior_sigma=1, **kwargs):
        super(GaussianNetwork, self).__init__(**kwargs)

        # Copy params into self
        self._num_feat = num_feat
        self._width = width
        self._freq_layers = freq_hidden_layers
        self._bayes_layers = bayes_hidden_layers
        self._num_out = num_out
        self._act = act
        self._prior_mu = prior_mu
        self._prior_sigma = prior_sigma

        # Construct the chain of layers
        self._chain = nn.Sequential(
            nn.Linear(self._num_feat, self._width),
            self._act()
        )
        for _ in range(self._freq_layers):
            self._chain.append(nn.Linear(self._width, self._width))
            self._chain.append(self._act())
        for _ in range(self._bayes_layers):
            self._chain.append(
                bnn.BayesLinear(in_features = self._width, out_features = self._width,
                                prior_mu = self._prior_mu, prior_sigma = self._prior_sigma)
            )
            self._chain.append(self._act())
        self._chain.append(
            bnn.BayesLinear(in_features = self._width, out_features = self._num_out,
                            prior_mu = self._prior_mu, prior_sigma = self._prior_sigma)
        )

        # Log error standard deviation
        self._log_sigma = nn.Parameter(torch.Tensor(1))
    
    def forward(self, inputs):
        return self._chain.forward(inputs)
    
    @property
    def log_error_sd(self):
        return self._log_sigma

