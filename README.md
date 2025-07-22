# mmbo
Surrogate Models for Multi-modal Bayesian Optimization

## Installation

```
pip install git+https://github.nrel.gov/itaylor/mmbo
```

## Overview

This package implements Bayesian Neural Network (BNN) based surrogate models for multi-modal data. These models are fit using a custom Variational Bayes procedure leveraging conjugate posteriors in the last layer of the network, offering more accurate predictions with better-calibrated uncertainty quantification than mean-field Variational Bayes.

## Basic Usage

This package offers two main surrogate models based on Bayesian neural networks (BNNs) implemented in Pyro. First, models are created using one of the built-in model classes. Then, using a built-in guide class and Pyro's SVI interface, the model's parameters can be estimated.

### Base Class

RegBNN usage

### Layered BNN

LayeredRegBNN usage

## Example datasets

Examples usage
