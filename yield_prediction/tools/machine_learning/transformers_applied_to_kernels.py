# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 13:46:25 2021.

@author: alexe
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class rbf_of_kernel(BaseEstimator, TransformerMixin):
    """
    Calculate the RBF kernel of a kernel (X).

        K(x) = exp(-gamma 2*(1-x))

    for each entry (x) in X.

    Parameters
    ----------
    gamma : float, default=None
         If None, defaults to 1.0 / n_features.

    """

    def __init__(self, gamma=None):
        self.gamma = gamma

    def kmat_to_dist(self, kmat):
        """
        Convert to distances.

        Parameters
        ----------
        kmat : numpy.ndarray
            Kernel matrix.

        Returns
        -------
        dist : numpy.ndarray
            Distance matrix.

        """
        dist = 2 * (1 - kmat)
        return dist

    def nonlinear_rbf(self, kmat):
        """
        Calculate rbf kernel.

        Parameters
        ----------
        kmat : numpy.ndarray
            Kernel matrix.

        Returns
        -------
        numpy.ndarray
            RBF kernel matrix.

        """
        if self.gamma is None:
            self.gamma = 1.0 / kmat.shape[1]

        dist_mat = self.kmat_to_dist(kmat)

        return np.exp(-self.gamma*dist_mat)

    def fit(self, X=None, y=None):
        """No fit method required. Returns self."""
        return self

    def transform(self, X, y=None):
        """
        Transform data.

        Parameters
        ----------
        X : array-like
            Data to transform.

        Returns
        -------
        X_ : array-like
            Transformed array.

        """
        X_ = X.copy()
        X_ = self.nonlinear_rbf(X_)
        return X_

    @property
    def _pairwise(self):
        return True


class poly_of_kernel(BaseEstimator, TransformerMixin):
    """
    Calculate the polynomial kernel of a kernel (X).

        K(x) = (gamma * x + c)^d

    for each entry (x) in X.

    Parameters
    ----------
    gamma : float, default=None
         If None, defaults to 1.0 / n_features.
    degree : int, default=3
    coef0 : float, default=1

    """

    def __init__(self, gamma=None, c=1, d=3):
        self.gamma = gamma
        self.c = c
        self.d = d

    def nonlinear_polynomial(self, kmat):
        """
        Calculate rbf kernel.

        Parameters
        ----------
        kmat : numpy.ndarray
            Kernel matrix.

        Returns
        -------
        numpy.ndarray
            Polynomial kernel matrix.

        """
        if self.gamma is None:
            self.gamma = 1.0 / kmat.shape[1]

        return np.power(kmat*self.gamma + self.c, self.d)

    def fit(self, X=None, y=None):
        """No fit method required. Returns self."""
        return self

    def transform(self, X, y=None):
        """
        Transform data.

        Parameters
        ----------
        X : array-like
            Data to transform.

        Returns
        -------
        X_ : array-like
            Transformed array.

        """
        X_ = X.copy()
        X_ = self.nonlinear_polynomial(X_)
        return X_

    @property
    def _pairwise(self):
        return True


class sigmoid_of_kernel(BaseEstimator, TransformerMixin):
    """
    Calculate the sigmoid kernel of a kernel (X).

        K(x) = tanh(gamma * x + c)

    for each entry (x) in X.

    Parameters
    ----------
    gamma : float, default=None
         If None, defaults to 1.0 / n_features.
    coef0 : float, default=1

    """

    def __init__(self, gamma=None, c=1):
        self.gamma = gamma
        self.c = c

    def nonlinear_sigmoid(self, kmat):
        """
        Calculate rbf kernel.

        Parameters
        ----------
        kmat : numpy.ndarray
            Kernel matrix.

        Returns
        -------
        numpy.ndarray
            Sigmoid kernel matrix.

        """
        if self.gamma is None:
            self.gamma = 1.0 / kmat.shape[1]

        return np.tanh(kmat*self.gamma + self.c)

    def fit(self, X=None, y=None):
        """No fit method required. Returns self."""
        return self

    def transform(self, X, y=None):
        """
        Transform data.

        Parameters
        ----------
        X : array-like
            Data to transform.

        Returns
        -------
        X_ : array-like
            Transformed array.

        """
        X_ = X.copy()
        X_ = self.nonlinear_sigmoid(X_)
        return X_

    @property
    def _pairwise(self):
        return True


class kernel_features(BaseEstimator, ClassifierMixin):
    """Convert kernel into kernel features."""

    def __init__(self):
        self

    def fit(self, X, y=None):
        eigvals, eigvecs = np.linalg.eigh(X)

        # Make zero eigenvalues a small positive number
        eigvals += 1e-10  # equivalent to K = K + 1e-10*I

        self.linear_map = (eigvecs*np.sqrt(np.reciprocal(eigvals)))

        return self

    def transform(self, X, y=None):
        return X@self.linear_map

    @property
    def _pairwise(self):
        return True
