import sklearn.neighbors as neighbors
import numpy as np
import scipy.stats as stats
import functools

class CopulaKNNDensity():
    def __init__(self, data, k=None, k_adjust=1.0, **kwargs):
        if data.ndim != 2:
            raise TypeError("must pass 2-dimensional array")
        if (data < 0).any() or (data > 1).any():
            raise ValueError("data must only contain values between zero and one")
        nsamples, nfeatures = data.shape
        if k is None:
            # (Approximately) optimal k, see Blumentritt and Schmid 2012, Hall 1983
            # Bound below by 2, making density finite at observed locations
            k = max(2, round(k_adjust * nsamples ** (4 / (nfeatures + 4))))
        self._neighbors = neighbors.NearestNeighbors(n_neighbors = k, metric="chebyshev", **kwargs)
        self._neighbors.fit(data)
    
    @property
    def _k(self):
        return self._neighbors.get_params()['n_neighbors']

    @property
    def _fit_data(self):
        return self._neighbors._fit_X
    
    @property
    def _nsamples(self):
        return self._neighbors.n_samples_fit_
    
    @property
    def _nfeatures(self):
        return self._neighbors.n_features_in_

    def __call__(self, U=None, k=None, log=False):
        logpdf = self.logpdf(U, k)
        if log:
            return logpdf
        else:
            return np.exp(logpdf)
    
    def logpdf(self, U=None, k=None):
        if U is None:
            U = self._fit_data
        if (U < 0).any() or (U > 1).any():
            raise ValueError("array must only contain values between zero and one")
        if k is None:
            k = self._k
        distances = np.array(self._neighbors.kneighbors(U, n_neighbors=k, return_distance=True)[0][:,-1], ndmin=2).T
        boxdims = np.minimum(U, distances) + np.minimum(1 - U, distances)
        return np.log(k) - np.log(self._nsamples) - np.sum(np.log(boxdims), axis=1)
        # Non-log version:
        #return (k / self._nsamples) / np.prod(boxdims, axis=1)
    
    @functools.cached_property
    def _sample_logpdf(self):
        # Log PDF at sample points with default k
        return self.logpdf()

    @property
    def entropy(self):
        return -np.mean(self._sample_logpdf)
    
    @property
    def entropy_stderr(self):
        return np.sqrt(np.var(self._sample_logpdf) / self._nsamples)

    @property
    def mutualinfo(self):
        return -self.entropy
    
    @property
    def mutualinfo_stderr(self):
        return self.entropy_stderr
        
def pseudoobs(X):
    if X.ndim != 2:
        raise TypeError("must pass 2-dimensional array")
    n = X.shape[0]
    return stats.rankdata(X, method="average", axis=0) / (n + 1)

