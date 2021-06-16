#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel modules.
"""
import pandas as pd
import numpy as np

import grakel.kernels as kernels
import sklearn.metrics.pairwise as sklearn_kernels

class kernel():
    """A class that defines and calculates kernels using GraKel."""
    
    def __init__(self, kernel_name, base_kernel=None):
        self.kernel_name = kernel_name
        self.base_kernel = base_kernel
        
    def define_kernel(self, *args, **kwargs):
        """
        Defines the graph kernel.

        Parameters
        ----------
        *args :
            Graph kernel parameters.
        **kwargs :
            Graph kernel parameters.

        Returns
        -------
        None.

        """
        base_kernel = self.base_kernel
        if base_kernel is None:
            base_kernel = 'VertexHistogram'
        k = getattr(kernels, self.kernel_name)
        k_base = getattr(kernels, base_kernel)
        self.kernel = k(base_kernel=k_base, *args, **kwargs)
        # self.kernel = k(base_graph_kernel=k_base, *args, **kwargs)
        
    def fit_and_transform(self, X):
        """
        Fit and transform on the same dataset. Calculates X_fit by X_fit 
        kernel matrix.
        """
        self.fitted_kernel = self.kernel.fit_transform(X)
    
    def transform_data(self, X):
        """
        Calculates X_fit by X_transform kernel matrix.
        """
        self.transformed_kernel = self.kernel.transform(X)
    
    def calcualte_reduced_X(self, X):
        """
        
        """
        missing_mol_indices = []
        present_mol_indices = []
        reduced_X = []
         
        # Get indices where mols are missing and create new list with no 
        # missing mols.
        for i, x in enumerate(X):
            if pd.isnull(x):
                 missing_mol_indices.append(i)
            else:
                present_mol_indices.append(i)
                reduced_X.append(x)
                
        present_mol_indices = np.array(present_mol_indices)             
        
        return reduced_X, missing_mol_indices, present_mol_indices
        
    def calculate_kernel_matrices(self, X_train, X_test, **kernel_params):
        """
        Fit and transform the X_train data. Calculate the kernel matrix between
        the fitted data (X_train) and X_test.

        Parameters
        ----------
        X_train : Series, list, numpy array 
            Training set of molecular graphs. Input must be iterable.
        X_test : Series, list, numpy array 
            Test set of molecular graphs. Input must be iterable.

        Returns
        -------
        k_train : numpy array
            The kernel matrix between all pairs of graphs in X_train.
        k_test : TYPE
            The kernel matrix between all pairs of graphs in X_train and 
            X_test.

        """
        self.define_kernel(normalize=True, **kernel_params)
        
        self.fit_and_transform(X_train)
        if X_test is not None:
            self.transform_data(X_test)
    
        k_train = self.fitted_kernel
        if X_test is not None:
            k_test = self.transformed_kernel
            return k_train, k_test   
        else:
            return k_train
    
    def calculate_kernel_matrices_with_missing_mols(self, X_train, X_test, **kernel_params):
        """
        Fit and transform the X_train data. Calculate the kernel matrix between
        the fitted data (X_train) and X_test.

        Parameters
        ----------
        X_train : Series, list, numpy array 
            Training set of molecular graphs. Input must be iterable.
        X_test : Series, list, numpy array 
            Test set of molecular graphs. Input must be iterable.

        Returns
        -------
        k_train : numpy array
            The kernel matrix between all pairs of graphs in X_train.
        k_test : TYPE
            The kernel matrix between all pairs of graphs in X_train and 
            X_test.

        """
        self.define_kernel(normalize=True, **kernel_params)
        
        if X_train.isnull().values.any():
            print('nan in X_train')
            reduced_X_train, \
                missing_mol_indices_X_train, \
                    present_mol_indices_X_train \
                        = self.calcualte_reduced_X(X_train)
            
            len_X_train = len(X_train)
            len_reduced_X_train = len(reduced_X_train)
        
            # Calculate kernel matrix on non-missing molecules
            reduced_k_train = self.kernel.fit_transform(reduced_X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                reduced_k_train, 
                np.ones((len_reduced_X_train, len_reduced_X_train)), 
                reduced_k_train
                ) 
        
            # missing molecules have value 1 or 2, initialise with ones
            k_train = np.ones((len_X_train, len_X_train)) 
            
            reduced_index_X_train = present_mol_indices_X_train[
                np.arange(reduced_k_train.shape[0])
                ]
        
            for i in range(len_reduced_X_train):
                k_train[reduced_index_X_train[i], reduced_index_X_train] \
                    = reduced_k_train[i, :]
        
            for i in missing_mol_indices_X_train:
                for j in missing_mol_indices_X_train:
                    k_train[i, j] = 2
                
        else:
            print('no nan in X_train')            
            k_train = self.kernel.fit_transform(X_train)
            
            len_X_train = len(X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                k_train, 
                np.ones((len_X_train, len_X_train)), 
                k_train
                ) 
                        
            len_reduced_X_train = len(X_train)
            reduced_index_X_train = np.arange(k_train.shape[0])
            missing_mol_indices_X_train = []
        
        if X_test.isnull().values.any():
            print('nan in X_test')
            reduced_X_test, \
                missing_mol_indices_X_test, \
                    present_mol_indices_X_test \
                        = self.calcualte_reduced_X(X_test)
            
            len_X_test = len(X_test)
            len_reduced_X_test = len(reduced_X_test)
            
            # Calculate kernel matrix on non-missing molecules
            reduced_k_test = self.kernel.transform(reduced_X_test)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                reduced_k_test, 
                np.ones((len_reduced_X_test, len_reduced_X_train)), 
                reduced_k_test
                ) 
        
            # missing molecules have value 1 or 2, initialise with ones
            k_test = np.ones((len_X_test, len_X_train)) 
            
            reduced_index_X_test = present_mol_indices_X_test[
                np.arange(reduced_k_test.shape[0])
                ]
        
            for i in range(len_reduced_X_test):
                k_test[reduced_index_X_test[i], reduced_index_X_train] \
                    = reduced_k_test[i, :]
        
            for i in missing_mol_indices_X_test:
                for j in missing_mol_indices_X_train:
                    k_test[i, j] = 2
                        
        else:
            print('no nan in X_test')
            reduced_k_test = self.kernel.transform(X_test)
            
            len_X_test = len(X_test)
            
            np.add(
                reduced_k_test, 
                np.ones((len_X_test, len_reduced_X_train)), 
                reduced_k_test
                )
            
            k_test = np.ones((len_X_test, len_X_train)) 
            
            reduced_index_X_test = np.arange(k_test.shape[0])
            
            for i in range(len_X_test):
                k_test[reduced_index_X_test[i], reduced_index_X_train] \
                    = reduced_k_test[i, :]
    
        return k_train, k_test   
    
        
    def multiple_descriptor_types(self, X_train, X_test, **kernel_params):
        k_train = 1
        
        if X_test is not None:
            k_test = 1
            
            if X_train.isnull().values.any() or X_test.isnull().values.any(): 
                for i in X_train:
                    train, test = self.calculate_kernel_matrices_with_missing_mols(
                        X_train[i], X_test[i], **kernel_params
                        )
                    k_train = k_train * train
                    k_test = k_test * test
            else:
                for i in X_train:
                    train, test = self.calculate_kernel_matrices(
                        X_train[i], X_test[i], **kernel_params
                        )
                    k_train = k_train * train
                    k_test = k_test * test
                    
        else:
            k_test = None
            for i in X_train:
                train = self.calculate_kernel_matrices(
                    X_train[i], None, **kernel_params
                    )
                k_train = k_train * train          
            
        return k_train, k_test
    
class sklearn_kernel():
    """A class that defines and calculates kernels using Sklearn."""
    
    def __init__(self, kernel_name):
        self.kernel_name = kernel_name
        
    # def define_kernel(self):
    #     """
    #     Defines the kernel.

    #     Parameters
    #     ----------
    #     None.
        
    #     Returns
    #     -------
    #     None.

    #     """.
        k = getattr(sklearn_kernels, self.kernel_name)
        self.kernel = k
    
    def calcualte_reduced_X(self, X):
        """
        
        """
        missing_mol_indices = []
        present_mol_indices = []
        reduced_X = []
         
        # Get indices where mols are missing and create new list with no 
        # missing mols.
        for i, x in X.iterrows():
            if x.isnull().values.any():
                 missing_mol_indices.append(X.index.get_loc(i))
            else:
                present_mol_indices.append(X.index.get_loc(i))
                reduced_X.append(x.values)
                
        present_mol_indices = np.array(present_mol_indices)             
        
        return reduced_X, missing_mol_indices, present_mol_indices
        
    def calculate_kernel_matrices(self, X_train, X_test):
        """
        Fit and transform the X_train data. Calculate the kernel matrix between
        the fitted data (X_train) and X_test.

        Parameters
        ----------
        X_train : Series, list, numpy array 
            Training set. Input must be iterable.
        X_test : Series, list, numpy array 
            Test set. Input must be iterable.

        Returns
        -------
        k_train : numpy array
            The kernel matrix between all pairs in X_train.
        k_test : TYPE
            The kernel matrix between all pairs in X_train and 
            X_test.

        """
    
        k_train = self.kernel(X_train)
        k_test = self.kernel(X_train, X_test)
        return k_train, k_test   
    
    def calculate_kernel_matrices_with_missing_mols(self, X_train, X_test):
        """
        Fit and transform the X_train data. Calculate the kernel matrix between
        the fitted data (X_train) and X_test.

        Parameters
        ----------
        X_train : Series, list, numpy array 
            Training set. Input must be iterable.
        X_test : Series, list, numpy array 
            Test set of. Input must be iterable.

        Returns
        -------
        k_train : numpy array
            The kernel matrix between all pairs in X_train.
        k_test : TYPE
            The kernel matrix between all pairs in X_train and 
            X_test.

        """
        # self.define_kernel(normalize=True, **kernel_params)
        
        if X_train.isnull().values.any():
            print('nan in X_train')
            reduced_X_train, \
                missing_mol_indices_X_train, \
                    present_mol_indices_X_train \
                        = self.calcualte_reduced_X(X_train)
            
            len_X_train = len(X_train)
            len_reduced_X_train = len(reduced_X_train)
        
            # Calculate kernel matrix on non-missing molecules
            reduced_k_train = self.kernel(reduced_X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                reduced_k_train, 
                np.ones((len_reduced_X_train, len_reduced_X_train)), 
                reduced_k_train
                ) 
        
            # missing molecules have value 1 or 2, initialise with ones
            k_train = np.ones((len_X_train, len_X_train)) 
            
            reduced_index_X_train = present_mol_indices_X_train[
                np.arange(reduced_k_train.shape[0])
                ]
        
            for i in range(len_reduced_X_train):
                k_train[reduced_index_X_train[i], reduced_index_X_train] \
                    = reduced_k_train[i, :]
        
            for i in missing_mol_indices_X_train:
                for j in missing_mol_indices_X_train:
                    k_train[i, j] = 2
                
        else:
            print('no nan in X_train')            
            k_train = self.kernel(X_train)
            
            len_X_train = len(X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                k_train, 
                np.ones((len_X_train, len_X_train)), 
                k_train
                ) 
            
            reduced_X_train = X_train
            len_reduced_X_train = len(X_train)
            reduced_index_X_train = np.arange(k_train.shape[0])
            missing_mol_indices_X_train = []
        
        if X_test.isnull().values.any():
            print('nan in X_test')
            reduced_X_test, \
                missing_mol_indices_X_test, \
                    present_mol_indices_X_test \
                        = self.calcualte_reduced_X(X_test)
            
            len_X_test = len(X_test)
            len_reduced_X_test = len(reduced_X_test)
            
            # Calculate kernel matrix on non-missing molecules
            reduced_k_test = self.kernel(reduced_X_test, reduced_X_train)
            
            # new_kernel(mol_i, mol_j) = base_kernel(mol_i, mol_j) + 1
            np.add(
                reduced_k_test, 
                np.ones((len_reduced_X_test, len_reduced_X_train)), 
                reduced_k_test
                ) 
        
            # missing molecules have value 1 or 2, initialise with ones
            k_test = np.ones((len_X_test, len_X_train)) 
            
            reduced_index_X_test = present_mol_indices_X_test[
                np.arange(reduced_k_test.shape[0])
                ]
        
            for i in range(len_reduced_X_test):
                k_test[reduced_index_X_test[i], reduced_index_X_train] \
                    = reduced_k_test[i, :]
        
            for i in missing_mol_indices_X_test:
                for j in missing_mol_indices_X_train:
                    k_test[i, j] = 2
                        
        else:
            print('no nan in X_test')
            reduced_k_test = self.kernel(X_test, reduced_X_train)
            
            len_X_test = len(X_test)
            
            np.add(
                reduced_k_test, 
                np.ones((len_X_test, len_reduced_X_train)), 
                reduced_k_test
                )
            
            k_test = np.ones((len_X_test, len_X_train)) 
            
            reduced_index_X_test = np.arange(k_test.shape[0])
            
            for i in range(len_X_test):
                k_test[reduced_index_X_test[i], reduced_index_X_train] \
                    = reduced_k_test[i, :]
    
        return k_train, k_test   
    