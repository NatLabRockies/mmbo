# mmbo
Multi-modal Bayesian Neural Network Surrogate Models

## Installation

```
pip install git+https://github.com/NatLabRockies/mmbo
```

## Overview

This package implements Bayesian Neural Network (BNN) based surrogate models for multi-modal data. These models are fit using a custom Variational Bayes procedure leveraging conjugate posteriors in the last layer of the network, offering more accurate predictions with better-calibrated uncertainty quantification than mean-field Variational Bayes.

## Changes in version 2.0

Version 2.0 refactors the model and guide classes. Code written for version 1.x needs these changes:

* **Training data is no longer stored inside the model.** `from_data` still uses the training data to set the input/output dimensions and the normalization layers, but the data must now be passed as keyword arguments whenever the model is run: to `svi.step` (or `do_svi`), and to `posterior_samples`. See [Training](#training) and [Posterior samples and prediction](#posterior-samples-and-prediction) below.
* **The `no_missing()` method has been removed.** Pass `sample_missing = False` as a keyword argument instead (see [Missing data](#missing-data)).

Version 2.0 also fixes several bugs in the layered model, adds a `min_iter` option to `do_svi`, adds a batched predictor (`predictor_batch`), and adds a Branin demo dataset.

## Basic Usage

This package offers two main surrogate models based on Bayesian neural networks (BNNs) implemented in Pyro. First, a model is created using one of the built-in model classes. Then, using the matching built-in guide class and Pyro's SVI interface, the model's parameters are estimated. Finally, samples from the fitted posterior are used to make predictions.

All data are `torch.Tensor`s with two dimensions: `X` has shape `(n, n_features)` and `Y` has shape `(n, n_outputs)`. A single input or output dimension still needs its own axis, e.g. shape `(n, 1)`.

### Demo data

The examples below use a demo dataset based on the Paciorek function, with one main (high fidelity) modality and one auxiliary modality containing four low fidelity outputs:

```python
import torch
import pyro
from mmbo.nn import RegBNN, LayeredRegBNN, RegBNNGuide, LayeredRegBNNGuide
from mmbo.examples.demo import paciorek_demo

# Unpack the included data from dictionary
X_train_main = paciorek_demo["X_train_main"]  # (256, 4)
Y_train_main = paciorek_demo["Y_train_main"]  # (256, 1)
X_train_mm =   paciorek_demo["X_train_mm"]    # list of auxiliary inputs:  [(881, 4)]
Y_train_mm =   paciorek_demo["Y_train_mm"]    # list of auxiliary outputs: [(881, 4)]
X_test =       paciorek_demo["X_test"]
Y_test_main =  paciorek_demo["Y_test_main"]
Y_test_mm =    paciorek_demo["Y_test_mm"]
```

A second demo dataset based on the Branin function, `branin_demo`, has the same structure and can be imported from the same module.

### Base Class

```python
# Initialize the model: from_data uses the training data to set the input/output
# dimensions and the normalizer layers. The data itself is not stored in the model.
model_single = RegBNN.from_data(X_train_main, Y_train_main, hidden_widths = [256, 256])

# Initialize the guide: uses the model to reference its layers, their widths,
# and its prior hyperparameters.
guide_single = RegBNNGuide(model_single)
```

Model options (accepted by `from_data` and by both model classes):

* `hidden_widths`: the number of neurons in each hidden layer (at least one layer). Defaults to a single layer of width `2 * max(n_features, n_outputs)`.
* `activation`: the activation function after each hidden layer: one of `"relu"` (default), `"elu"`, `"gelu"`, `"leaky_relu"`, `"softplus"` or `"tanh"`.
* `prior_variance`: scale of the prior on the last layer (default `0.01`). Inputs and outputs are normalized to mean zero and standard deviation one inside the network, so this is relative to the normalized outputs.

### Layered BNN

The layered model trains one auxiliary network per auxiliary modality and feeds their outputs into a main network as additional features. It takes the main modality training data and a list of auxiliary modality training data without much data manipulation. All modalities must have the same input dimension, but they can be observed at different input locations:

```python
model_layered = LayeredRegBNN.from_data(X_train_main, Y_train_main, X_train_mm, Y_train_mm, hidden_widths = [256, 256])
guide_layered = LayeredRegBNNGuide(model_layered)
```

### Joint multimodal model

The joint model is just a regular `RegBNN` under the hood, whose outputs are all the modalities side by side. For this, we need to combine all modalities of data into two tensors: one for the union of all X values, and one for the Y values of each modality, with `nan` wherever a modality was not observed (see [Missing data](#missing-data)):

```python
def _combine_X(X1: torch.Tensor, X2: torch.Tensor):
    """Combine two compatible X tensors into one deduplicated tensor,
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

model_joint = RegBNN.from_data(X_train_joint, Y_train_joint, hidden_widths = [256, 256])
guide_joint = RegBNNGuide(model_joint)
```

The first `Y_train_main.shape[-1]` outputs of the joint model are the main modality.

### Training

This package includes a convenience function, `do_svi`, for training these models using Stochastic Variational Inference (SVI). It performs SVI using Pyro's built-in model/guide architecture, and stops when there is no longer a statistically significant downward slope in the loss function (equivalently, upward slope in the ELBO). It returns the best parameters as determined by the rolling average of the loss function over training, along with the loss at every step.

Any keyword arguments not used by `do_svi` itself are passed on to `svi.step`, and from there to the model and guide. This is how the training data is supplied. Each model class takes its own keyword arguments:

```python
from mmbo.training import do_svi

adam = pyro.optim.Adam({"lr": 0.001})
elbo = pyro.infer.Trace_ELBO(num_particles = 1)

# RegBNN (including the joint model) takes X_train and Y_train
svi_single = pyro.infer.SVI(model_single, guide_single, adam, elbo)
params_single, losses_single = do_svi(svi_single, X_train = X_train_main, Y_train = Y_train_main)

# LayeredRegBNN takes the main modality data and lists of auxiliary modality data
svi_layered = pyro.infer.SVI(model_layered, guide_layered, adam, elbo)
params_layered, losses_layered = do_svi(
    svi_layered,
    X_train_main = X_train_main, Y_train_main = Y_train_main,
    X_train_aux = X_train_mm, Y_train_aux = Y_train_mm
)
```

Options for `do_svi`:

* `check_interval` (default `1000`): number of steps between convergence checks. A progress line is printed at each check.
* `max_nonimproved` (default `3`): training stops after this many consecutive checks without a significant decrease in the loss.
* `min_iter` (default `0`): minimum number of steps to run before stopping is allowed.
* `params` (default `None`): each call to `do_svi` trains in its own Pyro parameter store scope, so several models can be trained in the same session without their parameters colliding. Pass a set of parameters returned by an earlier call to continue training from it, e.g. with a smaller learning rate.

### Posterior samples and prediction

`posterior_samples` draws samples of the model parameters from the fitted guide. It must be given **the same training data keyword arguments used in training**, because the guide computes the last-layer posterior from the training data. (Without them, it returns an empty dictionary.)

`predictor` turns a model and a set of posterior samples into a function that predicts at new inputs. Predictions are in the original (unnormalized) units of `Y`, and have shape `(num_samples, n_test, n_outputs)`: one prediction of the mean function per posterior sample. For the layered model, the outputs are those of the main modality.

```python
from mmbo.nn import posterior_samples, predictor

n_samples = 1000

samples_single = posterior_samples(guide_single, params_single, n_samples,
                                   X_train = X_train_main, Y_train = Y_train_main)
pred_single = predictor(model_single, samples_single)
Y_test_pred_single = pred_single(X_test)  # (n_samples, n_test, 1)

samples_layered = posterior_samples(guide_layered, params_layered, n_samples,
                                    X_train_main = X_train_main, Y_train_main = Y_train_main,
                                    X_train_aux = X_train_mm, Y_train_aux = Y_train_mm)
pred_layered = predictor(model_layered, samples_layered)
Y_test_pred_layered = pred_layered(X_test)

# Posterior mean and a 90% credible interval for the mean function
Y_mean = Y_test_pred_layered.mean(dim = 0)
Y_lower, Y_upper = Y_test_pred_layered.quantile(torch.tensor([0.05, 0.95]), dim = 0)
```

`predictor` loops over the posterior samples. `predictor_batch` gives the same predictions using batched matrix operations, which is much faster for many samples. It requires the model to be in evaluation mode, and 2-dimensional test inputs:

```python
from mmbo.nn import predictor_batch

model_layered.eval()
Y_test_pred_layered = predictor_batch(model_layered, samples_layered)(X_test)
model_layered.train()  # switch back before training the model further
```

### Missing data

Outputs may contain `nan` values, which are treated as missing observations. This is what makes the joint model possible, but missing values are allowed in the outputs of any model and any modality. Inputs may not contain missing values.

During training, the guide imputes the missing outputs from the last-layer posterior predictive distribution. The models' `sample_missing` keyword argument (default `True`) controls whether the model also includes a sample site for the imputed outputs. Imputing missing data incurs some computational cost, so avoiding this step can be beneficial if it's not needed. It is passed like the training data, e.g. `do_svi(svi, X_train = X, Y_train = Y, sample_missing = False)`. With `sample_missing = False`, Pyro warns that the guide has a sample site that the model does not; this is expected.

## References

Released under NLR software record SWR-25-119.

To cite, please use

```
@misc{ doecode_181165,
title = {mmbo (Multi-modal Bayesian Optimization) [SWR-25-119]},
author = {Taylor, Ian and Mueller, Juliane and Bessac, Julie},
abstractNote = {This package implements Bayesian Neural Network (BNN) based surrogate models for multi-modal data. These models are fit using a custom Variational Bayes procedure leveraging conjugate posteriors in the last layer of the network, offering more accurate predictions with better-calibrated uncertainty quantification than mean-field Variational Bayes.},
doi = {10.11578/dc.20260519.1},
url = {https://doi.org/10.11578/dc.20260519.1},
howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20260519.1}},
year = {2026},
month = {apr}
}
```
