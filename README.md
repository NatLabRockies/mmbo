# mmbo
Multi-modal Bayesian Neural Network Surrogate Models

## Installation

```
pip install git+https://github.com/NatLabRockies/mmbo
```

## Overview

This package implements Bayesian Neural Network (BNN) based surrogate models for multi-modal data. These models are fit using a custom Variational Bayes procedure leveraging conjugate posteriors in the last layer of the network, offering more accurate predictions with better-calibrated uncertainty quantification than mean-field Variational Bayes.

## Basic Usage

This package offers two main surrogate models based on Bayesian neural networks (BNNs) implemented in Pyro. First, models are created using one of the built-in model classes. Then, using a built-in guide class and Pyro's SVI interface, the model's parameters can be estimated.

### Base Class

We can test each model using a demo dataset based on the Paciorek function:

```
from mmbo.nn import RegBNN, LayeredRegBNN, RegBNNGuide, LayeredRegBNNGuide
from mmbo.examples.demo import paciorek_demo

# Unpack the included data from dictionary
X_train_main = paciorek_demo["X_train_main"]
X_train_mm =   paciorek_demo["X_train_mm"]
X_test =       paciorek_demo["X_test"]
Y_train_main = paciorek_demo["Y_train_main"]
Y_train_mm =   paciorek_demo["Y_train_mm"]
Y_test_main =  paciorek_demo["Y_test_main"]
Y_test_mm =    paciorek_demo["Y_test_mm"]

# Initialize the model: from_data method uses training data to set input/output
# dimensions and normalizer layersl It also stores the training data within the model,
# so that no data needs to be passed during training.
model_single = RegBNN.from_data(X_train_main, Y_train_main, hidden_widths = [256, 256])

# Initialize the guide: uses model to reference layers and their widths, and
# priors if custom hyperparemters are supplied.
guide_single = RegBNNGuide(model_single)
```

### Layered BNN

The layered model can take both main modality training data and a list of auxiliary modality training data without much data manipulation:

```
model_layered = LayeredRegBNN.from_data(X_train_main, Y_train_main, X_train_mm, Y_train_mm, hidden_widths = [256, 256])
guide_layered = LayeredRegBNNGuide(model_layered)
```

### Joint multimodal model

The joint model is just a regular `RegBNN` under the hood. For this, we need to combine all modalities of data into two tensors: one for the union of all X values, and one for the Y values of each modality:

```
def _combine_X(X1: torch.Tensor, X2: torch.Tensor):
    """Combine two compatible X tensors into on deduplicated tensor,
    e.g., for merging two sets of observations."""
    assert X1.ndim == 2 and X2.ndim == 2
    assert X1.shape[-1] == X2.shape[-1]
    X2_missing_sel = []
    for row in range(X2.shape[0]):
        if not X1.isclose(X2[row]).all(dim = -1).any():
            X2_missing_sel.append(row)
    return torch.cat([X1, X2[X2_missing_sel]], dim = 0)

def _expand_Y(X_orig: torch.Tensor, Y_orig: torch.Tensor, X_new: torch.Tensor):
    """For matching X_orig and Y_orig of independent and dependent variables,
    respectively, and an X_new which is a superset of X_orig, expand Y to match
    X_new with nan's everywhere Y_orig has no observation."""
    assert X_orig.ndim == 2 and Y_orig.ndim == 2 and X_new.ndim == 2
    assert X_orig.shape[0] == Y_orig.shape[0]
    Y_new = torch.full((X_new.shape[0], Y_orig.shape[1]), torch.nan, dtype = Y_orig.dtype, device = Y_orig.device)
    for i in range(X_orig.shape[0]):
        Y_new[X_new.isclose(X_orig[i]).all(dim = -1)] = Y_orig[i]
    return Y_new

X_train_joint = X_train_main
for X2 in X_train_mm:
    X_train_joint = _combine_X(X_train_joint, X2)
Y_train_joint = torch.cat(
    [_expand_Y(X_train_main, Y_train_main, X_train_joint)] +
    [_expand_Y(X_train_mm[i], Y_train_mm[i], X_train_joint) for i in range(len(X_train_mm))],
    dim = -1
)

model_joint = RegBNN.from_data(X_train_joint, Y_train_joint, hidden_widths = hidden_widths)
guide_joint = RegBNNGuide(model_joint)
```

### Training

This package includes a convenience function for training these models using Stochastic Variational Inference (SVI). It performs SVI using pyro's built-in model/guide architecture, and stops
when there is no longer a statistically significant downward slope in the loss function (equivalently, upward slope in the ELBO). It returns the best parameters as determined by
the rolling average of the loss function over training.

```
import pyro
from mmbo.training import do_svi

adam = pyro.optim.Adam({"lr": 0.001})
elbo = pyro.infer.Trace_ELBO(num_particles = 1)
# model,guide are created by code like above
svi = pyro.infer.SVI(model, guide, adam, elbo)

# The optional params argument can be used to restart training from a
# previously returned set of parameters (e.g. with a smaller learning
# rate).
params, losses = do_svi(svi, params = None)
```

## References

Released under NLR software record SWR-25-119.
