from typing import Tuple
import itertools

import scipy.interpolate as interpolate
import importlib.resources as resources
import pandas as pd
import numpy as np
import torch

from .testfunc import paciorek_hf, paciorek_lf, branin_hf, branin_lf

__all__ = ["paciorek_demo", "branin_demo"]

def _product_tensor(*args, **kwargs) -> torch.Tensor:
    prod = itertools.product(*args, **kwargs)
    return torch.cat([torch.tensor(x).unsqueeze(0) for x in prod], dim = 0)

## Paciorek Demo

def _make_paciorek_demo():
    # X values
    _xvals = [0.475, 0.5625, 0.7375, 0.825]
    _xvals_mm = [0.3875, 0.51875, 0.65, 0.71825, 0.9125]
    _X_train_main = _product_tensor(_xvals, repeat = 4)
    _X_train_mm = [torch.cat([_X_train_main, _product_tensor(_xvals_mm, repeat = 4)], dim = 0)]
    _X_test = torch.cat([_X_train_main, _product_tensor(_xvals_mm, repeat = 4)], dim = 0)

    # Y values
    _modes_aux = [0.25, 0.5, 0.75, 1.0]
    _Y_train_main = paciorek_hf(_X_train_main)
    _Y_train_mm = [torch.cat([paciorek_lf(_X_train_mm[0], A) for A in _modes_aux], dim = -1)]
    _Y_test_main = paciorek_hf(_X_test)
    _Y_test_mm = [torch.cat([paciorek_lf(_X_test, A) for A in _modes_aux], dim = -1)]

    return {
        "X_train_main": _X_train_main,
        "X_train_mm": _X_train_mm,
        "X_test": _X_test,
        "Y_train_main": _Y_train_main,
        "Y_train_mm": _Y_train_mm,
        "Y_test_main": _Y_test_main,
        "Y_test_mm": _Y_test_mm
    }

paciorek_demo = _make_paciorek_demo()

## Branin Demo

def _make_branin_demo():
    # X values
    dim1_vals = [-5 + 1.5*x for x in [2,3,4,6,7,8]]
    dim2_vals = [0 + 1.5*x for x in [2,3,4,6,7,8]]
    dim1_vals_full = [-5 + 1.5*x for x in [1,2,3,4,5,6,7,8,9]]
    dim2_vals_full = [0 + 1.5*x for x in [1,2,3,4,5,6,7,8,9]]
    X_train_main = _product_tensor(dim1_vals, dim2_vals)
    X_train_mm = [_product_tensor(dim1_vals_full, dim2_vals_full)]
    X_test = _product_tensor(dim1_vals_full, dim2_vals_full)

    # Y values
    modes_aux = [0.0, 0.514, 1.0]
    Y_train_main = branin_hf(X_train_main)
    Y_train_mm = [torch.cat([branin_lf(X_train_mm[0], A) for A in modes_aux], dim = -1)]
    Y_test_main = branin_hf(X_test)
    Y_test_mm = [torch.cat([branin_lf(X_test, A) for A in modes_aux], dim = -1)]

    return {
        "X_train_main": X_train_main,
        "X_train_mm": X_train_mm,
        "X_test": X_test,
        "Y_train_main": Y_train_main,
        "Y_train_mm": Y_train_mm,
        "Y_test_main": Y_test_main,
        "Y_test_mm": Y_test_mm
    }

branin_demo = _make_branin_demo()
