import torch
from torch import nn
import pyro.nn as pnn

__all__ = ["Normalizer", "PyroNormalizer", "UnNormalizer", "PyroUnNormalizer"]

class Normalizer(nn.Module):
    """Normalizes data by first subtracting the means and then dividing by
    the standard deviations. Normalizes along the first k dimensions of
    the data, by using the standard tensor broadcasting when performing
    the subtraction and division."""
    feature_means: torch.Tensor
    feature_stds: torch.Tensor

    @classmethod
    def from_data(cls, X: torch.Tensor, ndim=1):
        """Creates a Normalizer such that the given data would be normalized
        to have mean zero and standard deviation one."""
        dims = tuple(range(ndim))
        feature_means = X.nanmean(dim=dims)
        feature_stds = (X - feature_means).pow(2).nanmean(dim=dims).sqrt()

        return cls(feature_means, feature_stds)
    
    @classmethod
    def from_data_range(cls, X: torch.Tensor, ndim=1):
        """Creates a Normalizer such that the given data would be normalized
        to lie between -1 and 1."""
        X_min = X.view(-1, *X.shape[ndim:]).min(dim=0).values
        X_max = X.view(-1, *X.shape[ndim:]).max(dim=0).values
        
        return cls((X_max + X_min) / 2, (X_max - X_min) / 2)

    def __init__(self, feature_means: torch.Tensor, feature_stds: torch.Tensor):
        super().__init__()
        self.register_buffer("feature_means", feature_means)
        self.register_buffer("feature_stds", feature_stds)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.feature_means) / self.feature_stds

PyroNormalizer = pnn.PyroModule[Normalizer]

class UnNormalizer(Normalizer):
    """Performs the inverse operation of a normalizer, first multiplying by
    the standard deviations then adding the means."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.feature_stds + self.feature_means
    
PyroUnNormalizer = pnn.PyroModule[UnNormalizer]