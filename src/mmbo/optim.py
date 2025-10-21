import pyro
from pyro.infer import Predictive
import torch
import botorch
from .nn import predictor, predictor_batch


def expected_improvement(model, posterior_samples, X_test, current_min):
    pred = predictor(model, posterior_samples)
    Y_samples = pred(X_test)
    return (current_min - Y_samples).clamp(min=0).mean(dim=0)

def expected_improvement_softplus(model, posterior_samples, X_test, current_min):
    pred = predictor(model, posterior_samples)
    Y_samples = pred(X_test)
    return torch.nn.functional.softplus(current_min - Y_samples).mean(dim=0)

def expected_improvement_softplus_batch(model, posterior_samples, X_test, current_min):
    pred = predictor_batch(model, posterior_samples)
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

def probability_improvement_sigmoid_batch(model, posterior_samples, X_test, current_min):
    pred = predictor_batch(model, posterior_samples)
    Y_samples = pred(X_test)
    return torch.nn.functional.sigmoid(current_min - Y_samples).mean(dim=0)

def rect_sample(n: int, low_lim: torch.Tensor, high_lim: torch.Tensor) -> torch.Tensor:
    assert low_lim.shape == high_lim.shape
    return torch.rand((n, *low_lim.shape)) * (high_lim - low_lim) + low_lim

def select_next_point(objective, num_candidates, low_lim, high_lim, num_raw_samples=None, initial_samples=None):
    if num_raw_samples is None and initial_samples is None:
        raise ValueError("Must supply either num_raw_samples or initial_samples")
    if num_raw_samples is not None and initial_samples is not None:
        raise Warning("num_raw_samples is ignored if initial_samples is given")
    if initial_samples is None:
        initial_samples = rect_sample(num_raw_samples, low_lim, high_lim)
    # Select the top 'num_candidates' of the initial samples to use as starting points
    raw_objective = objective(initial_samples)
    _,bestidx = torch.topk(raw_objective.squeeze(), num_candidates)
    candidates = initial_samples[bestidx]
    # Use the botorch SLSQP optimizer and select the best final position
    final_samples,final_objective = botorch.gen_candidates_scipy(candidates, objective, low_lim, high_lim, options={"method":"SLSQP"})
    _,finalidx = torch.topk(final_objective.squeeze(), 1)
    return final_samples[finalidx]