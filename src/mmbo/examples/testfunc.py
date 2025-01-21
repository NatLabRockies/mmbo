from typing import Tuple

import scipy.interpolate as interpolate
import importlib.resources as resources
import pandas as pd
import numpy as np
import torch

__all__ = ["li2_energy", "be2_energy", "era_wind", "nice_periodic", "noisy_uvadlc",
           "branin_hf", "branin_lf", "paciorek_hf", "paciorek_lf", "trid_hf", "trid_lf",
           "timeseries_full", "timeseries_stat"]

# Generate test functions from pandas dataframes by cubic interpolation
def _make_tablefunc(data, xcol, default=None): 
    functions = {
        col:interpolate.CubicSpline(
            data.loc[~data.loc[:,col].isna(),xcol], 
            data.loc[~data.loc[:,col].isna(),col], 
            extrapolate=False
        ) for col in data.columns if col != xcol
    }
    def func(x, m=default):
        return torch.tensor(functions[m](x), dtype=x.dtype)
    func.modes = [col for col in data.columns if col != xcol]
    func.default = default
    return func

# Load Li2 and Be2 energy data and create test functions
_li2_data = pd.read_csv(resources.files("mmbo.examples").joinpath("Table_Li2.txt"), sep=r"\s+")
_be2_data = pd.read_csv(resources.files("mmbo.examples").joinpath("Table_Be2.txt"), sep=r"\s+")

li2_energy = _make_tablefunc(_li2_data, "Dist", "HF")
be2_energy = _make_tablefunc(_be2_data, "Dist", "HF")

# Load ERA wind speed data and create test functions
_wind_data = pd.read_csv(resources.files("mmbo.examples").joinpath("il_argonne_obs_era.csv"))
# X variable as scaled time
_wind_data['Time'] = pd.to_datetime(_wind_data['Time'])
_wind_data['X'] = (_wind_data['Time'] - _wind_data['Time'].min()).apply(lambda x: x.days + x.seconds / (24*60*60))
del _wind_data['Time']
# Decompose wind direction
for dircol in ['obs_Dir10m', 'obs_Dir60m', 'era_Dir10m', 'era_Dir100m']:
    _wind_data[dircol+'_sin'] = np.sin(_wind_data[dircol] * np.pi / 180)
    _wind_data[dircol+'_cos'] = np.cos(_wind_data[dircol] * np.pi / 180)
    del _wind_data[dircol]
# Hourly change in wind direction
# TODO

era_wind = _make_tablefunc(_wind_data, "X")

# "Friendlier" test functions that are designed to have discrete additive components
def nice_periodic(X: torch.Tensor, m=0):
    if m < 0:
        raise Exception("m must be nonnegative")
    y = X + 1
    for i in range(m):
        f = i + 1
        y += torch.sin(f * X + (torch.pi * f / 2)) * torch.pi * 2 / (f ** 1.01)
    return y

def noisy_uvadlc(X: torch.Tensor):
    # Based on the test function here: 
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.html
    noise = 0.02 * torch.randn_like(X)
    wave1 = 0.3 * torch.sin(2 * torch.pi * (X + noise))
    wave2 = 0.3 * torch.sin(4 * torch.pi * (X + noise))
    return X + wave1 + wave2 + noise

#################################################################
## Somewhat standard optimization / multifidelity test functions
#################################################################

def branin_hf(X: torch.Tensor):
    assert X.shape[-1] == 2
    x1 = X[...,0].unsqueeze(-1)
    x2 = X[...,1].unsqueeze(-1)
    if not torch.all(
        torch.logical_and(
            torch.logical_and(x1 <= 10, x1 >= -5),
            torch.logical_and(x2 <= 15, x2 >= 0))):
        raise ValueError("X values out of bounds: x1 in [-5, 10], x2 in [0, 15]")

    parabola = (x2 - (5.1 / (4 * torch.pi ** 2)) * (x1 ** 2) + (5 / torch.pi) * x1 - 6) ** 2
    periodic = 10 * (1 - 1 / (8 * torch.pi)) * torch.cos(x1)
    return parabola + periodic + 10

def branin_lf(X: torch.Tensor, A1):
    assert X.shape[-1] == 2
    assert (A1 >= 0) and (A1 <= 1)
    x1 = X[...,0].unsqueeze(-1)
    x2 = X[...,1].unsqueeze(-1)
    parabola = (x2 - (5.1 / (4 * torch.pi ** 2)) * (x1 ** 2) + (5 / torch.pi) * x1 - 6) ** 2
    return branin_hf(X) - (A1 + 0.5) * parabola

def paciorek_hf(X: torch.Tensor):
    d = X.shape[-1]
    if not torch.all(torch.logical_and(X >= 0.3, X <= 1.0)):
        raise ValueError("X values out of bounds: [0.3, 1]")
    return torch.sin(1 / X.prod(dim=-1, keepdim = True))

def paciorek_lf(X: torch.Tensor, A2):
    d = X.shape[-1]
    assert (A2 >= 0) and (A2 <= 1)
    return paciorek_hf(X) - 9 * A2**2 * torch.cos(1 / X.prod(dim=-1, keepdim = True))

def trid_hf(X: torch.Tensor):
    d = X.shape[-1]
    parabola = ((X - 1.0) ** 2).sum(dim=-1, keepdim = True)
    interact = (X[...,1:] * X[...,:-1]).sum(dim=-1, keepdim = True)
    return parabola - interact

def trid_lf(X: torch.Tensor, A4):
    d = X.shape[-1]
    assert (A4 >= 0) and (A4 <= 1)
    parabola = ((X - A4) ** 2).sum(dim=-1, keepdim = True)
    interact = (A4 - 0.65) * (torch.arange(2, d+1) * X[...,1:] * X[...,:-1]).sum(dim=-1, keepdim = True)
    return parabola - interact

########################################################
## Multimodal data test with time series
########################################################

def _ts_x_unpack(X: torch.Tensor) -> Tuple[torch.Tensor]:
    assert X.shape[-1] == 4
    rate = X[...,0].exp()
    period = X[...,1].exp()
    amp = X[...,2].exp()
    phase = torch.tanh(X[...,3]) * torch.pi
    return rate, period, amp, phase

def timeseries_full(X: torch.Tensor, noise: float = 0.0, res: int = 100):
    rate, period, amp, phase = _ts_x_unpack(X)
    time = torch.linspace(0.0, 1.0, steps = res)
    y = time.pow(rate.unsqueeze(-1)) + amp.unsqueeze(-1) * torch.cos(2.0 * torch.pi * time / period.unsqueeze(-1) + phase.unsqueeze(-1))
    return y + torch.randn_like(y) * noise

def timeseries_stat(X: torch.Tensor, m: str, res: int = 1000):
    assert m in ["max", "min", "max_loc", "min_loc", "path_length", "avg_slope", "avg_value", "total_dist"]
    ts = timeseries_full(X, noise = 0.0, res = res)
    if m == "max":
        return ts.max(dim = -1).values
    elif m == "min":
        return ts.min(dim = -1).values
    elif m == "max_loc":
        return ts.max(dim = -1).indices / (res - 1.0)
    elif m == "min_loc":
        return ts.min(dim = -1).indices / (res - 1.0)
    elif m == "path_length":
        delta = 1.0 / (res - 1.0)
        changes = ts[...,1:] - ts[...,:(res-1)]
        return (changes.pow(2) + delta ** 2).sqrt().sum(dim = -1)
    elif m == "avg_slope":
        return ts[...,res-1] - ts[...,0]
    elif m == "avg_value":
        return ts.mean(dim = -1)
    elif m == "total_dist":
        changes = ts[...,1:] - ts[...,:(res-1)]
        return changes.abs().sum(dim = -1)
timeseries_stat.modes = ["max", "min", "max_loc", "min_loc", "path_length", "avg_slope", "avg_value", "total_dist"]
