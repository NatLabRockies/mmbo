import torch
from torch import nn
import numpy as np

class Normalizer(nn.Module):
    feature_means: torch.Tensor
    feature_stds: torch.Tensor

    @classmethod
    def from_data(cls, X: torch.Tensor, ndim=1):
        dims = tuple(range(ndim))
        feature_means = X.nanmean(dim=dims)
        feature_stds = (X - feature_means).pow(2).nanmean(dim=dims).sqrt()

        return cls(feature_means, feature_stds)

    def __init__(self, feature_means: torch.Tensor, feature_stds: torch.Tensor):
        super().__init__()
        self.register_buffer("feature_means", feature_means)
        self.register_buffer("feature_stds", feature_stds)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.feature_means) / self.feature_stds
    
class UnNormalizer(Normalizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.feature_stds + self.feature_means