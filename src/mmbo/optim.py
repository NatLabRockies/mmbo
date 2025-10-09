import pyro
from pyro.infer import Predictive
import torch
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