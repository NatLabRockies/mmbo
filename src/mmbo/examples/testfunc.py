import scipy.interpolate as interpolate
import importlib.resources as resources
import pandas as pd
import numpy as np

# Generate test functions from pandas dataframes by cubic interpolation
def _make_tablefunc(data, xcol, default=None): 
    functions = {
        col:interpolate.CubicSpline(data.loc[:,xcol], data.loc[:,col], extrapolate=False) 
        for col in data.columns if col != xcol
    }
    def func(x, m=default):
        return functions[m](x)
    func.modes = [col for col in data.columns if col != xcol]
    func.default = default
    return func

# Load Li2 and Be2 energy data and create test functions
_li2_data = pd.read_csv(resources.files().joinpath("Table_Li2.txt"), sep=r"\s+")
_be2_data = pd.read_csv(resources.files().joinpath("Table_Be2.txt"), sep=r"\s+")

li2_energy = _make_tablefunc(_li2_data, "Dist", "HF")
be2_energy = _make_tablefunc(_be2_data, "Dist", "HF")

# "Friendlier" test functions that are designed to have discrete additive components
def nice_periodic(x, m=0):
    if m < 0:
        raise Exception("m must be nonnegative")
    x = np.array(x)
    y = x + 1
    for i in range(m):
        f = i + 1
        y += np.sin(f * x + (np.pi * f / 2)) / (f ** 2)
    return y
