### This file contains code for processing and extracting features from time series data

import math

import torch_pca

import torch

from .vae import VAE

__all__ = ["autocovar", "TSPCA", "VarTSPCA"]

###############################################################################
## Auto-cross correlation
###############################################################################

def _cross_covar(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Calculate the cross-covariance matrix for tensors X and Y, with shapes:
    (*X_batch, p, n) and (*Y_batch, q, n) respectively, where X_batch and Y_batch are
    broadcastable. The last second-to-last dimension is the variables and the last
    dimension is the observations. Returns a tensor of shape (*broadcasted_batch, p, q)."""
    X = X.unsqueeze(-2)
    Y = Y.unsqueeze(-3)
    X_mean = X.mean(dim=-1)
    Y_mean = Y.mean(dim=-1)
    XY_mean = (X * Y).mean(dim=-1)
    return XY_mean - X_mean * Y_mean


def autocovar(T: torch.Tensor, k: int = 1, reshape: bool = True) -> torch.Tensor:
    """T is a tensor representing (batches of) time series with shape:
    (batch_shape, obs, time), with obs being the dimension of each observation
    and time being the dimension in the time axis.
    If reshape is false: It returns a tensor with shape (batch_shape, obs, obs, k+1),
    where the last dimension is the lag (first index 0 is no lag, i.e. just the 
    covariance of the series).
    If reshape is true (default): the last three dimensions are stacked into a vector
    to return a tensor with shape (batch_shape, n) where n = (k+1)*obs^2."""
    p = T.shape[-2]
    batch_shape = T.shape[:-2]
    result = T.new_empty((*T.shape[:-2], p, p, k+1))
    result[...,0] = _cross_covar(T,T)
    for i in range(1,k+1):
        result[...,i] = _cross_covar(T[...,i:], T[...,:-i])
    if reshape:
        return result.reshape((*batch_shape, -1))
    else:
        return result

###############################################################################
## PCA 
###############################################################################

class TSPCA:
    """Class for performing PCA on (multivariate) time series data. Time series data
    is a Tensor with shape (*batch_shape, var, time). i.e. it is always at least
    2-dimensional, i.e. even univariate time series have the var dimension of length 1."""
    def __init__(self, X: torch.Tensor = None, n_components = 0.95):
        self._pca = torch_pca.PCA(n_components = n_components)
        self._nvar = self._ntime = None
        if X is not None:
            self.fit(X)

    @property
    def _n_components(self):
        return self._pca.n_components_

    def fit(self, X:torch.Tensor) -> "TSPCA":
        batch_shape = X.shape[:-2]
        self._nvar = X.shape[-2]
        self._ntime = X.shape[-1]
        self._pca.fit(X.reshape((-1, self._nvar * self._ntime)))
        return self

    def encode(self, X: torch.Tensor):
        batch_shape = X.shape[:-2]
        assert X.shape[-2] == self._nvar
        assert X.shape[-1] == self._ntime
        return self._pca.transform(X.reshape((-1, self._nvar * self._ntime))).reshape((*batch_shape, -1))

    def decode(self, X: torch.Tensor):
        batch_shape = X.shape[:-1]
        assert X.shape[-1] == self._n_components
        return self._pca.inverse_transform(X.reshape(-1, self._n_components)).reshape((*batch_shape, self._nvar, self._ntime))

class VarTSPCA:
    """Class for performing PCA on (multivariate) time series data. Time series data
    is a Tensor with shape (*batch_shape, var, time). i.e. it is always at least
    2-dimensional, i.e. even univariate time series have the var dimension of length 1.
    
    This class performs PCA separately on each variable in the time series before concatenating
    their PC representations"""
    def __init__(self, X: torch.Tensor = None, pct_var: float = 0.95):
        self._nvar = self._ntime = None
        self._timepca = []
        assert pct_var > 0.0
        assert pct_var <= 1.0
        self._pct_var = pct_var
        if X is not None:
            self.fit(X)
    
    @property
    def _ntimecomp(self):
        return [x.n_components_ for x in self._timepca]

    @property
    def n_components(self):
        return sum(self._ntimecomp)

    def fit(self, X: torch.Tensor) -> "VarTSPCA":
        self._nvar = X.shape[-2]
        self._ntime = X.shape[-1]
        self._timepca = [None] * self._nvar
        for i in range(self._nvar):
            self._timepca[i] = torch_pca.PCA(n_components = self._pct_var).fit(X[...,i,:].reshape((-1, self._ntime)))
        return self

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        batch_shape = X.shape[:-2]
        pieces = [None] * self._nvar
        for i in range(self._nvar):
            pieces[i] = self._timepca[i].transform(X[...,i,:].reshape((-1, self._ntime)))
        return torch.cat(pieces, dim=-1).reshape((*batch_shape, -1))

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        batch_shape = X.shape[:-1]
        decoded_pieces = [None] * self._nvar
        pieces = X.reshape((-1, self.n_components)).split(self._ntimecomp, dim=-1)
        for i,piece in enumerate(pieces):
            decoded_pieces[i] = self._timepca[i].inverse_transform(piece).unsqueeze(-2)
        return torch.cat(decoded_pieces, dim=-2).reshape((*batch_shape, self._nvar, self._ntime))