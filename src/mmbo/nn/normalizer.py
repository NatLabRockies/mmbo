import torch
from torch import nn
import pyro.nn as pnn

__all__ = ["Normalizer", "PyroNormalizer", "UnNormalizer", "PyroUnNormalizer"]


class _MeanStdTransform(nn.Module):
    """Base class for normalization modules based on feature-wise means
    and standard deviations. Allows for both normalization and
    un-normalization based on mean and standard deviation. Is inherited
    by two more specific classes: Normalizer and UnNormalizer, whose
    forward() methods wrap norm() and unnorm(), respectively."""
    means: torch.Tensor
    stds: torch.Tensor

    def __init__(self, means: torch.Tensor, stds: torch.Tensor):
        super().__init__()
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.eps = 1e-5
    
    @classmethod
    def from_data(cls, X: torch.Tensor, ndim=1):
        """Creates a Normalizer such that the given data would be normalized
        to have mean zero and standard deviation one."""
        dims = tuple(range(ndim))
        means = X.nanmean(dim=dims)
        stds = (X - means).pow(2).nanmean(dim=dims).sqrt()

        return cls(means, stds)
    
    def norm(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.means) / (self.stds + self.eps)
    
    def unnorm(self, Z: torch.Tensor) -> torch.Tensor:
        return Z * (self.stds + self.eps) + self.means

class Normalizer(_MeanStdTransform):
    """Normalizes data by first subtracting the means and then dividing by
    the standard deviations. Normalizes along the first k dimensions of
    the data, by using the standard tensor broadcasting when performing
    the subtraction and division."""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.norm(X)

PyroNormalizer = pnn.PyroModule[Normalizer]

class UnNormalizer(_MeanStdTransform):
    """Performs the inverse operation of a normalizer, first multiplying by
    the standard deviations then adding the means."""

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.unnorm(Z)
    
PyroUnNormalizer = pnn.PyroModule[UnNormalizer]