# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 13:46:25 2021.

@author: alexe
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin

from rdkit.DataStructs import BulkTanimotoSimilarity


class tanimoto_kernel(BaseEstimator, ClassifierMixin):
    """A class that calculates tanimoto kernels."""

    def __init__(self):
        self

    def calculate_kernel(self, X_test, X_train):
        """
        Calculate the training kernel.

        Parameters
        ----------
        X_test : list of fingerprints
            Test set of fingerprints.
        X_train : list of fingerprints
            Train set of fingerprints.

        Returns
        -------
        k : numpy.ndarray
            Tanimoto kernel matrix.

        """
        # len_X_train = len(X_train)
        # len_X_test = len(X_test)
        # k = np.ones((len_X_test, len_X_train))

        # for n, fp in enumerate(X_test):
        #     bulk_tan = BulkTanimotoSimilarity(fp, X_train)
        #     k[n, :] = bulk_tan

        bulk_tan_list = [
            BulkTanimotoSimilarity(fp, X_train)
            for n, fp in enumerate(X_test)
            ]

        k = np.row_stack(bulk_tan_list)

        return k

    def fit(self, X, y=None):
        """
        Generate training set Tanimoto kernel matrix.

        If each row in X is a list of fingerprints (representing each reaction
        component), calculate the Hadamard product of the kernel matrices.

        Parameters
        ----------
        X : array-like object
            Contains fingerprints of the training set. Can be 1D or 2D.
            If 2D, each column should represent a different reaction component.
        y : None, optional
            The default is None. Added to allow use as part of a pipeline.

        Returns
        -------
        k_train : numpy.ndarray
            Training set Tanimoto kernel matrix.

        """
        if X.ndim == 2:
            self.X_train = defaultdict()
            for n, col_name in enumerate(X):
                self.X_train[n] = X[col_name]

        elif X.ndim == 1:
            self.X_train = X

        else:
            raise ValueError(
                'Invalid dimension of X ({}). \
                    Valid options are 1D or 2D'.format(X.ndim)
                    )

        return self

    def transform(self, X, y=None):
        """
        Generate test set Tanimoto kernel matrix.

        If each row in X is a list of fingerprints (representing each reaction
        component), calculate the Hadamard product of the kernel matrices.

        Parameters
        ----------
        X : array-like object
            Contains fingerprints of the training set. Can be 1D or 2D.
            If 2D, each column should represent a different reaction component.
        y : None, optional
            The default is None. Added to allow use as part of a pipeline.

        Returns
        -------
        k : numpy.ndarray
            Test set Tanimoto kernel matrix.

        """
        if X.ndim == 2:
            X_test_list = X.copy()

            k = 1

            for n, col_name in enumerate(X_test_list):
                test = self.calculate_kernel(
                    X_test_list[col_name],
                    self.X_train[n]
                    )
                k = k * test

        elif X.ndim == 1:
            X_test = X.copy()
            k = self.calculate_kernel(X_test, self.X_train)

        else:
            raise ValueError(
                'Invalid dimension of X ({}). \
                    Valid options are 1D or 2D'.format(X.ndim)
                    )

        return k


class tanimoto_kernel_with_missing_mols(BaseEstimator, ClassifierMixin):
    """
    A class that calculates tanimoto kernels.

    Takes into account missing molecules.
    """

    def __init__(self):
        self

    def calculate_tanimoto_scores(self, X_test, X_train):
        """
        Calculate the training kernel.

        Parameters
        ----------
        X_test : list of fingerprints
            Test set of fingerprints.
        X_train : list of fingerprints
            Train set of fingerprints.

        Returns
        -------
        k : numpy.ndarray
            Tanimoto kernel matrix.

        """
        bulk_tan_list = [
            BulkTanimotoSimilarity(fp, X_train)
            for n, fp in enumerate(X_test)
            ]

        k = np.row_stack(bulk_tan_list)

        return k

    def calcualte_reduced_X(self, X):
        """
        Remove missing molecules from X.

        Parameters
        ----------
        X : list of fingerprints
            Training set of fingerprints. Contains nan values.

        Returns
        -------
        reduced_X : list of fingerprints
            Training set of fingerprints. Does not contain nan values.
        missing_mol_indices : list
            Indicies of the missing molecules (nan values).
        present_mol_indices : list
            Indicies of the present molecules (real values).

        """
        missing_mol_indices = []
        present_mol_indices = []
        reduced_X = []

        # Get indices where mols are missing and create new list with no
        # missing mols.
        if X.isnull().values.any():
            for i, x in enumerate(X):
                if pd.isnull(x):
                    missing_mol_indices.append(i)
                else:
                    present_mol_indices.append(i)
                    reduced_X.append(x)

            present_mol_indices = np.array(present_mol_indices)
            reduced_X = pd.Series(reduced_X, index=present_mol_indices)

        else:
            present_mol_indices = list(range(len(X)))
            present_mol_indices = np.array(present_mol_indices)
            reduced_X = X

        return reduced_X, missing_mol_indices, present_mol_indices

    def get_info(self, X):
        reduced_X, missing_mol_indices, present_mol_indices \
            = self.calcualte_reduced_X(X)

        len_X = len(X)
        len_reduced_X = len(reduced_X)

        return reduced_X, missing_mol_indices, present_mol_indices, \
            len_X, len_reduced_X

    def calculate_kernel(self, X_train, X_test):
        reduced_X_train, missing_mol_indices_X_train, \
            present_mol_indices_X_train, len_X_train, len_reduced_X_train \
                = self.get_info(X_train)
        reduced_X_test, missing_mol_indices_X_test, \
            present_mol_indices_X_test, len_X_test, len_reduced_X_test \
                = self.get_info(X_test)

        # Calculate kernel matrix on non-missing molecules
        reduced_k_test = self.calculate_tanimoto_scores(
            X_test=reduced_X_test,
            X_train=reduced_X_train
            )

        # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
        np.add(
            reduced_k_test,
            np.ones((len_reduced_X_test, len_reduced_X_train)),
            reduced_k_test
            )

        # missing molecules have value 1 or 2, initialise with ones
        k_test = np.ones((len_X_test, len_X_train))

        reduced_index_X_train = present_mol_indices_X_train[
            np.arange(reduced_X_train.shape[0])
            ]
        reduced_index_X_test = present_mol_indices_X_test[
            np.arange(reduced_X_test.shape[0])
            ]

        for i in range(len_reduced_X_test):
            k_test[reduced_index_X_test[i], reduced_index_X_train] \
                = reduced_k_test[i, :]

        for i in missing_mol_indices_X_test:
            for j in missing_mol_indices_X_train:
                k_test[i, j] = 2

        return k_test

    def fit(self, X, y=None):
        """
        Generate training set Tanimoto kernel matrix.

        If each row in X is a list of fingerprints (representing each reaction
        component), calculate the Hadamard product of the kernel matrices.

        Parameters
        ----------
        X : array-like object
            Contains fingerprints of the training set. Can be 1D or 2D.
            If 2D, each column should represent a different reaction component.
        y : None, optional
            The default is None. Added to allow use as part of a pipeline.

        Returns
        -------
        k_train : numpy.ndarray
            Training set Tanimoto kernel matrix.

        """
        if X.ndim == 2:
            self.X_train = defaultdict()

            for n, col_name in enumerate(X):
                self.X_train[n] = X[col_name]

        elif X.ndim == 1:
            self.X_train = X

        else:
            raise ValueError(
                'Invalid dimension of X ({}). \
                    Valid options are 1D or 2D'.format(X.ndim)
                    )

        return self

    def transform(self, X, y=None):
        """
        Generate test set Tanimoto kernel matrix.

        If each row in X is a list of fingerprints (representing each reaction
        component), calculate the Hadamard product of the kernel matrices.

        Parameters
        ----------
        X : array-like object
            Contains fingerprints of the training set. Can be 1D or 2D.
            If 2D, each column should represent a different reaction component.
        y : None, optional
            The default is None. Added to allow use as part of a pipeline.

        Returns
        -------
        k : numpy.ndarray
            Test set Tanimoto kernel matrix.

        """
        if X.ndim == 2:
            X_test_list = X.copy()

            k = 1

            for n, col_name in enumerate(X_test_list):
                test = self.calculate_kernel(
                    X_train=self.X_train[n],
                    X_test=X_test_list[col_name]
                    )
                k = k * test

        elif X.ndim == 1:
            X_test = X.copy()
            k = self.calculate_kernel(X_test, self.X_train)

        else:
            raise ValueError(
                'Invalid dimension of X ({}). \
                    Valid options are 1D or 2D'.format(X.ndim)
                    )

        return k

    # def calculate_k_train_with_missing_mols(self, X_train):
    #     """
    #     Calculate the training set kernel matrix.

    #     Contains missing molecules.

    #     Parameters
    #     ----------
    #     X_train : list of fingerprints
    #         Training set of fingerprints. Contains nan values.

    #     Returns
    #     -------
    #     fitted_kernel_params : dict
    #         Parameters required for the calculation of the test set kernel
    #         matrix.
    #     k_train : numpy.ndarray
    #         Training set Tanimoto kernel matrix.

    #     """
    #     reduced_X_train, missing_mol_indices_X_train, \
    #         present_mol_indices_X_train = self.calcualte_reduced_X(X_train)

    #     len_X_train = len(X_train)
    #     len_reduced_X_train = len(reduced_X_train)

    #     # Calculate kernel matrix on non-missing molecules
    #     reduced_k_train = self.calculate_train_kernel(reduced_X_train)

    #     # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
    #     np.add(
    #         reduced_k_train,
    #         np.ones((len_reduced_X_train, len_reduced_X_train)),
    #         reduced_k_train
    #         )

    #     # missing molecules have value 1 or 2, initialise with ones
    #     k_train = np.ones((len_X_train, len_X_train))

    #     reduced_index_X_train = present_mol_indices_X_train[
    #         np.arange(reduced_k_train.shape[0])
    #         ]

    #     for i in range(len_reduced_X_train):
    #         k_train[reduced_index_X_train[i], reduced_index_X_train] \
    #             = reduced_k_train[i, :]

    #     for i in missing_mol_indices_X_train:
    #         for j in missing_mol_indices_X_train:
    #             k_train[i, j] = 2

    #     fitted_kernel_params = {
    #         'missing_mol_indices_X_train': missing_mol_indices_X_train,
    #         'len_X_train': len_X_train,
    #         'len_reduced_X_train': len_reduced_X_train,
    #         'reduced_index_X_train': reduced_index_X_train,
    #         'reduced_X_train': reduced_X_train
    #         }

    #     return fitted_kernel_params, k_train

    # def calculate_k_train_without_missing_mols(self, X_train):
    #     """
    #     Calculate the training set kernel matrix.

    #     Does not contain missing molecules.

    #     Parameters
    #     ----------
    #     X_train : list of fingerprints
    #         Training set of fingerprints. Contains nan values.

    #     Returns
    #     -------
    #     fitted_kernel_params : dict
    #         Parameters required for the calculation of the test set kernel
    #         matrix.
    #     k_train : numpy.ndarray
    #         Training set Tanimoto kernel matrix.

    #     """
    #     k_train = self.calculate_train_kernel(X_train)

    #     len_X_train = len(X_train)

    #     # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
    #     np.add(
    #         k_train,
    #         np.ones((len_X_train, len_X_train)),
    #         k_train
    #         )

    #     len_reduced_X_train = len(X_train)
    #     reduced_index_X_train = np.arange(k_train.shape[0])
    #     missing_mol_indices_X_train = []

    #     fitted_kernel_params = {
    #         'missing_mol_indices_X_train': missing_mol_indices_X_train,
    #         'len_X_train': len_X_train,
    #         'len_reduced_X_train': len_reduced_X_train,
    #         'reduced_index_X_train': reduced_index_X_train,
    #         'reduced_X_train': X_train
    #         }

    #     return fitted_kernel_params, k_train

    # def calculate_k_test_with_missing_mols(
    #         self, fitted_kernel_params, X_test
    #         ):
    #     """
    #     Calculate the test set kernel matrix.

    #     Contains missing molecules.

    #     Parameters
    #     ----------
    #     fitted_kernel_params : dict
    #         Parameters required for the calculation of the test set kernel
    #         matrix.
    #     X_test : list of fingerprints
    #         Test set of fingerprints.

    #     Returns
    #     -------
    #     k_test : numpy.ndarray
    #         Test set Tanimoto kernel matrix.

    #     """
    #     missing_mol_indices_X_train \
    #         = fitted_kernel_params['missing_mol_indices_X_train']
    #     len_X_train = fitted_kernel_params['len_X_train']
    #     len_reduced_X_train = fitted_kernel_params['len_reduced_X_train']
    #     reduced_index_X_train = fitted_kernel_params['reduced_index_X_train']
    #     reduced_X_train = fitted_kernel_params['reduced_X_train']

    #     reduced_X_test, missing_mol_indices_X_test, \
    #         present_mol_indices_X_test = self.calcualte_reduced_X(X_test)

    #     len_X_test = len(X_test)
    #     len_reduced_X_test = len(reduced_X_test)

    #     # Calculate kernel matrix on non-missing molecules
    #     reduced_k_test = self.calculate_test_kernel(
    #         reduced_X_test,
    #         reduced_X_train
    #         )

    #     # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
    #     np.add(
    #         reduced_k_test,
    #         np.ones((len_reduced_X_test, len_reduced_X_train)),
    #         reduced_k_test
    #         )

    #     # missing molecules have value 1 or 2, initialise with ones
    #     k_test = np.ones((len_X_test, len_X_train))

    #     reduced_index_X_test = present_mol_indices_X_test[
    #         np.arange(reduced_k_test.shape[0])
    #         ]

    #     for i in range(len_reduced_X_test):
    #         k_test[reduced_index_X_test[i], reduced_index_X_train] \
    #             = reduced_k_test[i, :]

    #     for i in missing_mol_indices_X_test:
    #         for j in missing_mol_indices_X_train:
    #             k_test[i, j] = 2

    #     return k_test

    # def calculate_k_test_without_missing_mols(
    #         self, fitted_kernel_params, X_test
    #         ):
    #     """
    #     Calculate the test set kernel matrix.

    #     Does not contain missing molecules.

    #     Parameters
    #     ----------
    #     kernel : object of grakel.kernels
    #         Object generated by grakel to compute test set kernel.
    #     fitted_kernel_params : dict
    #         Parameters required for the calculation of the test set kernel
    #         matrix.
    #     X_test : list of fingerprints
    #         Test set of fingerprints.

    #     Returns
    #     -------
    #     k_test : numpy.ndarray
    #         Test set graph kernel matrix.

    #     """
    #     len_X_train = fitted_kernel_params['len_X_train']
    #     len_reduced_X_train = fitted_kernel_params['len_reduced_X_train']
    #     reduced_index_X_train = fitted_kernel_params['reduced_index_X_train']
    #     reduced_X_train = fitted_kernel_params['reduced_X_train']

    #     reduced_k_test = self.calculate_test_kernel(X_test, reduced_X_train)

    #     len_X_test = len(X_test)

    #     np.add(
    #         reduced_k_test,
    #         np.ones((len_X_test, len_reduced_X_train)),
    #         reduced_k_test
    #         )

    #     k_test = np.ones((len_X_test, len_X_train))

    #     reduced_index_X_test = np.arange(k_test.shape[0])

    #     for i in range(len_X_test):
    #         k_test[reduced_index_X_test[i], reduced_index_X_train] \
    #             = reduced_k_test[i, :]

    #     return k_test

    

    # def fit(self, X, y=None):
    #     """
    #     Generate training set Tanimoto kernel matrix.

    #     If each row in X is a list of fingerprints (representing each reaction
    #     component), calculate the Hadamard product of the kernel matrices.

    #     Parameters
    #     ----------
    #     X : array-like object
    #         Contains fingerprints of the training set. Can be 1D or 2D.
    #         If 2D, each column should represent a different reaction component.
    #     y : None, optional
    #         The default is None. Added to allow use as part of a pipeline.

    #     Returns
    #     -------
    #     k_train : numpy.ndarray
    #         Training set Tanimoto kernel matrix.

    #     """
    #     if X.ndim == 2:
    #         X_train_list = X.copy()

    #         self.fitted_kernel_params_list = defaultdict()
    #         k_train = 1

    #         for n, col_name in enumerate(X_train_list):
    #             X_train = X_train_list[col_name]

    #             if X_train.isnull().values.any():
    #                 self.fitted_kernel_params_list[n], train \
    #                     = self.calculate_k_train_with_missing_mols(X_train)
    #             else:
    #                 self.fitted_kernel_params_list[n], train \
    #                     = self.calculate_k_train_without_missing_mols(X_train)

    #             k_train = k_train * train

    #     elif X.ndim == 1:
    #         X_train = X.copy()

    #         if X_train.isnull().values.any():
    #             self.fitted_kernel_params, k_train \
    #                 = self.calculate_k_train_with_missing_mols(X_train)
    #         else:
    #             self.fitted_kernel_params, k_train \
    #                 = self.calculate_k_train_without_missing_mols(X_train)

    #     else:
    #         raise ValueError(
    #             'Invalid dimension of X ({}). \
    #                 Valid options are 1D or 2D'.format(X.ndim)
    #                 )

    #     return k_train

    # def predict(self, X, y=None):
    #     """
    #     Generate test set Tanimoto kernel matrix.

    #     If each row in X is a list of fingerprints (representing each reaction
    #     component), calculate the Hadamard product of the kernel matrices.

    #     Parameters
    #     ----------
    #     X : array-like object
    #         Contains fingerprints of the training set. Can be 1D or 2D.
    #         If 2D, each column should represent a different reaction component.
    #     y : None, optional
    #         The default is None. Added to allow use as part of a pipeline.

    #     Returns
    #     -------
    #     k_test : numpy.ndarray
    #         Test set Tanimoto kernel matrix.

    #     """
    #     if X.ndim == 2:
    #         X_test_list = X.copy()

    #         k_test = 1

    #         for n, col_name in enumerate(X_test_list):
    #             X_test = X_test_list[col_name]

    #             if X_test.isnull().values.any():
    #                 test = self.calculate_k_test_with_missing_mols(
    #                     self.fitted_kernel_params_list[n],
    #                     X_test
    #                     )
    #             else:
    #                 test = self.calculate_k_test_without_missing_mols(
    #                     self.fitted_kernel_params_list[n],
    #                     X_test
    #                     )

    #             k_test = k_test * test

    #     elif X.ndim == 1:
    #         X_test = X.copy()

    #         if X_test.isnull().values.any():
    #             k_test = self.calculate_k_test_with_missing_mols(
    #                 self.fitted_kernel_params,
    #                 X_test
    #                 )
    #         else:
    #             k_test = self.calculate_k_test_without_missing_mols(
    #                 self.fitted_kernel_params,
    #                 X_test
    #                 )

    #     else:
    #         raise ValueError(
    #             'Invalid dimension of X ({}). \
    #                 Valid options are 1D or 2D'.format(X.ndim)
    #                 )

    #     return k_test
