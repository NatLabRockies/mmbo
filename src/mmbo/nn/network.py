import torch
from torch import nn
import math
from .normalizer import Normalizer, UnNormalizer
from .activation import MeanSD
from .parallel import Parallel
from .linear import IndBayesLinear

class TIBNN(nn.Module):

    def __init__(self, n_feat, n_out, width, depth = 1,
                 feature_means = None, feature_stds = None,
                 out_means = None, out_stds = None,
                 sigma_prior_median = 0.1, sigma_prior_scale = 10):
        
        super().__init__()

        if feature_means is None:
            feature_means = torch.zeros(n_feat)
        if feature_stds is None:
            feature_stds = torch.ones(n_feat)
        if out_means is None:
            out_means = torch.zeros(n_out)
        if out_stds is None:
            out_stds = torch.ones(n_out)

        if depth < 1:
            raise ValueError("depth must be at least one")
        if width < 1:
            raise ValueError("width must be at least one")

        sigma_raw_median = math.log(math.expm1(sigma_prior_median))
        sigma_raw_scale = math.log(math.sqrt(sigma_prior_scale))

        self._input = nn.Sequential(
            Normalizer(feature_means, feature_stds),
            IndBayesLinear(n_feat, width),
            nn.Tanh()
        )

        self._output = Parallel(
            Parallel.tuple(),
            # Mean component
            nn.Sequential(
                IndBayesLinear(width, n_out, weight_prior_sigma = 1 / math.sqrt(width)),
                UnNormalizer(out_means, out_stds)
            ),
            # SD Component
            nn.Sequential(
                IndBayesLinear(width, n_out, weight_prior_sigma = sigma_raw_scale / math.sqrt(width),
                               bias_prior_mu = sigma_raw_median, bias_prior_sigma = sigma_raw_scale),
                nn.Softplus()
            )
        )

        self._chain = nn.Sequential()
        for i in range(depth-1):
            self._chain.append(IndBayesLinear(width, width, weight_prior_sigma = 1 / math.sqrt(width)))
            self._chain.append(nn.Tanh)

    @classmethod
    def from_data(cls, X_norm, Y_norm, width, depth = 1,
                  sigma_prior_median = 0.1, sigma_prior_scale = 10):
        n_feat = X_norm.shape[1]
        n_out = Y_norm.shape[1]
        feature_means = X_norm.nanmean(dim=0)
        feature_stds = (X_norm - feature_means).pow(2).nanmean(dim=0).sqrt()
        out_means = Y_norm.nanmean(dim=0)
        out_stds = (Y_norm - out_means).pow(2).nanmean(dim=0).sqrt()

        return cls(n_feat, n_out, width, depth, 
                   feature_means, feature_stds, out_means, out_stds,
                   sigma_prior_median, sigma_prior_scale)

    def forward(self, X):
        return self._output(self._chain(self._input(X)))

