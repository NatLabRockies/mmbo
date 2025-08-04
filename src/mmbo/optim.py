import pyro
from pyro.infer import Predictive


def expected_improvement(model, guide, params, X_test, num_samples, current_min):
    with pyro.get_param_store().scope(params):
        guide.eval()
        pred = Predictive(model, guide = guide, num_samples = num_samples, return_sites=['mu_pred'])
        Y_samples = pred.get_samples(X_pred = X_test)['mu_pred']
    return (current_min - Y_samples).clamp(min=0).mean(dim=0)

def lcb(model, guide, params, X_test, num_samples, quantile=0.05):
    with pyro.get_param_store().scope(params):
        guide.eval()
        pred = Predictive(model, guide = guide, num_samples = num_samples, return_sites=['mu_pred'])
        Y_samples = pred.get_samples(X_pred = X_test)['mu_pred']
    return Y_samples.quantile(quantile, dim=0)

def probability_improvement(model, guide, params, X_test, num_samples, current_min):
    with pyro.get_param_store().scope(params):
        guide.eval()
        pred = Predictive(model, guide = guide, num_samples = num_samples, return_sites=['mu_pred'])
        Y_samples = pred.get_samples(X_pred = X_test)['mu_pred']
    return (current_min - Y_samples > 0).mean(dim=0)

