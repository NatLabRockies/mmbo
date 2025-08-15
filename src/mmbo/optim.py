import pyro
from pyro.infer import Predictive


def expected_improvement(model, posterior_samples, X_test, current_min):
    pred = Predictive(model, posterior_samples = posterior_samples, return_sites=['mu_pred'])
    Y_samples = pred.get_samples(X_pred = X_test)['mu_pred']
    return (current_min - Y_samples).clamp(min=0).mean(dim=0)

def lcb(model, posterior_samples, X_test, quantile=0.05):
    pred = Predictive(model, posterior_samples = posterior_samples, return_sites=['mu_pred'])
    Y_samples = pred.get_samples(X_pred = X_test)['mu_pred']
    return Y_samples.quantile(quantile, dim=0)

def probability_improvement(model, posterior_samples, X_test, current_min):
    pred = Predictive(model, posterior_samples = posterior_samples, return_sites=['mu_pred'])
    Y_samples = pred.get_samples(X_pred = X_test)['mu_pred']
    return ((current_min - Y_samples > 0) * 1.0).mean(dim=0)

