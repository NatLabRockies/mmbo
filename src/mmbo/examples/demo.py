from typing import Tuple
import itertools

import scipy.interpolate as interpolate
import importlib.resources as resources
import pandas as pd
import numpy as np
import torch

from .testfunc import paciorek_hf, paciorek_lf

__all__ = ["paciorek_demo"]

def _product_tensor(*args, **kwargs) -> torch.Tensor:
    prod = itertools.product(*args, **kwargs)
    return torch.cat([torch.tensor(x).unsqueeze(0) for x in prod], dim = 0)

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

paciorek_demo = {
    "X_train_main": _X_train_main,
    "X_train_mm": _X_train_mm,
    "X_test": _X_test,
    "Y_train_main": _Y_train_main,
    "Y_train_mm": _Y_train_mm,
    "Y_test_main": _Y_test_main,
    "Y_test_mm": _Y_test_mm
}