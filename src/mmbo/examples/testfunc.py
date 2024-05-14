import scipy.interpolate as interpolate
import importlib.resources as resources
import pandas as pd
import numpy as np

_li2_data = pd.read_csv(resources.files().joinpath("Table_Li2.txt"), sep=r"\s+")
_be2_data = pd.read_csv(resources.files().joinpath("Table_Be2.txt"), sep=r"\s+")

def _make_tablefunc(data, xcol, default=None):
    functions={col:interpolate.CubicSpline(data.loc[:,xcol], data.loc[:,col]) for col in data.columns if col != xcol}
    def func(x, m=default):
        return functions[m](x)
    func.modes = [col for col in data.columns if col != xcol]
    func.default = default
    return func

li2_energy = _make_tablefunc(_li2_data, "Dist", "HF")
be2_energy = _make_tablefunc(_be2_data, "Dist", "HF")