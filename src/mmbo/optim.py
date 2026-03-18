import pyro
from pyro.infer import Predictive
import torch
from torch.quasirandom import SobolEngine
from .nn import predictor


def expected_improvement(model, posterior_samples, X_test, current_min):
    pred = predictor(model, posterior_samples)
    Y_samples = pred(X_test)
    return (current_min - Y_samples).clamp(min=0).mean(dim=0)

def expected_improvement_softplus(model, posterior_samples, X_test, current_min):
    pred = predictor(model, posterior_samples)
    Y_samples = pred(X_test)
    return torch.nn.functional.softplus(current_min - Y_samples).mean(dim=0)

def lcb(model, posterior_samples, X_test, quantile=0.05):
    pred = predictor(model, posterior_samples)
    Y_samples = pred(X_test)
    return Y_samples.quantile(quantile, dim=0)

def probability_improvement(model, posterior_samples, X_test, current_min):
    pred = predictor(model, posterior_samples)
    Y_samples = pred(X_test)
    return ((current_min - Y_samples > 0) * 1.0).mean(dim=0)

def probability_improvement_sigmoid(model, posterior_samples, X_test, current_min):
    pred = predictor(model, posterior_samples)
    Y_samples = pred(X_test)
    return torch.nn.functional.sigmoid(current_min - Y_samples).mean(dim=0)

def rect_sample(n: int, low_lim: torch.Tensor, high_lim: torch.Tensor) -> torch.Tensor:
    assert low_lim.shape == high_lim.shape
    return torch.rand((n, *low_lim.shape)) * (high_lim - low_lim) + low_lim

def rect_sobol(n: int, low_lim: torch.Tensor, high_lim: torch.Tensor) -> torch.Tensor:
    assert low_lim.shape == high_lim.shape
    shape = low_lim.shape
    d = low_lim.numel()
    sobol = SobolEngine(d)
    return sobol.draw(n).reshape((n, *shape)) * (high_lim - low_lim) + low_lim


def select_next_point(objective, num_candidates, low_lim, high_lim):
    initial_samples = rect_sobol(num_candidates, low_lim, high_lim)
    # Select the top 'num_candidates' of the initial samples to use as starting points
    sample_objective = objective(initial_samples)
    _,bestidx = torch.topk(sample_objective.squeeze(), 1)
    return initial_samples[bestidx]