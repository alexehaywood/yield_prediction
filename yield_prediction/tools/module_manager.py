# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 20:12:54 2021

@author: alexe
"""
from collections import defaultdict
from sklearn import svm, linear_model, neighbors, naive_bayes, tree, ensemble
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from tools.descriptors.generate_descriptors import descriptor_generator
from tools.descriptors.graph_kernels import graph_kernel, graph_kernel_with_missing_mols
from tools.descriptors.tanimoto_kernels import tanimoto_kernel, tanimoto_kernel_with_missing_mols
from tools.machine_learning.transformers_applied_to_kernels import kernel_features, poly_of_kernel, rbf_of_kernel, sigmoid_of_kernel
from tools.splitter import splitter


class module_loader():
    __DESCRIPTORS = defaultdict()
    __PREPROCESSING = defaultdict()
    __KERNEL = defaultdict()
    __KERNEL_PARAM_GRID = defaultdict()
    __MODEL_CLASSES = defaultdict()
    __MODEL = defaultdict()
    __PARAM_GRID = defaultdict()
    __SPLIITER = defaultdict()

    def __init__(self, n_jobs=1):
        super()
        self.__load_descriptors()
        self.__load_preprocessing()
        self.__load_kernel()
        self.__load_model_classes()
        self.__load_model()
        self.__load_param_grid()
        self.__load_splitter()

        self.n_jobs = n_jobs

    def __load_descriptors(self):
        self

    def __load_preprocessing(self):
        self.__PREPROCESSING['quantum'] = [
            ('scaler', StandardScaler())
            ]
        self.__PREPROCESSING['wl_kernel'] = [
            ('wl_kernel', graph_kernel('WeisfeilerLehman'))
            ]
        self.__PREPROCESSING['wl_kernel_missing_mols'] = [
            ('wl_kernel', graph_kernel_with_missing_mols('WeisfeilerLehman'))
            ]
        self.__PREPROCESSING['wl_features'] = [
            self.__PREPROCESSING['wl_kernel'][0],
            ('wl_features', kernel_features())
            ]
        self.__PREPROCESSING['wl_features_missing_mols'] = [
            self.__PREPROCESSING['wl_kernel_missing_mols'][0],
            # ('wl_kernel', graph_kernel_with_missing_mols('WeisfeilerLehman')),
            ('wl_features', kernel_features())
            ]
        self.__PREPROCESSING['tanimoto_kernel'] = [
            ('tanimoto_kernel', tanimoto_kernel())
            ]
        self.__PREPROCESSING['tanimoto_kernel_missing_mols'] = [
            ('tanimoto_kernel', tanimoto_kernel_with_missing_mols())
            ]
        self.__PREPROCESSING['tanimoto_features'] = [
            ('tanimoto_kernel', tanimoto_kernel()),
            ('tanimoto_features', kernel_features())
            ]
        self.__PREPROCESSING['tanimoto_features_missing_mols'] = [
            ('tanimoto_kernel', tanimoto_kernel_with_missing_mols()),
            ('tanimoto_features', kernel_features())
            ]
        self.__PREPROCESSING['fingerprints'] = []
        self.__PREPROCESSING['one-hot'] = [
            ('one_hot_encoder',
             OneHotEncoder(sparse=False, handle_unknown='ignore'))
            ]

    def __load_kernel(self):
        self.__KERNEL['poly_kernel'] = [('poly_kernel', poly_of_kernel())]
        self.__KERNEL['RBF_kernel'] = [('RBF_kernel', rbf_of_kernel())]
        self.__KERNEL['sigmoid_kernel'] = [('sigmoid_kernel', sigmoid_of_kernel())]

        self.__KERNEL_PARAM_GRID['poly_kernel'] = {
            'poly_kernel__gamma': [1, 10, 100, 1000],
            'poly_kernel__c': [1]
            }
        self.__KERNEL_PARAM_GRID['RBF_kernel'] = {
            'RBF_kernel__gamma': [1, 10, 100, 1000]
            }
        self.__KERNEL_PARAM_GRID['sigmoid_kernel'] = {
            'sigmoid_kernel__gamma': [1, 10, 100, 1000]
            }

    def __load_model_classes(self):
        self.__MODEL_CLASSES['svm'] = svm
        self.__MODEL_CLASSES['linear_model'] = linear_model
        self.__MODEL_CLASSES['neighbors'] = neighbors
        self.__MODEL_CLASSES['naive_bayes'] = naive_bayes
        self.__MODEL_CLASSES['tree'] = tree
        self.__MODEL_CLASSES['ensemble'] = ensemble

    def __load_model(self):
        self.__MODEL['svr-linear_kernel'] =      self._get_sklearn_model('svm', 'SVR', {'kernel': 'linear'})
        self.__MODEL['svr-poly_kernel'] =        self._get_sklearn_model('svm', 'SVR', {'kernel': 'poly'})
        self.__MODEL['svr-RBF_kernel'] =         self._get_sklearn_model('svm', 'SVR', {'kernel': 'rbf'})
        self.__MODEL['svr-sigmoid_kernel'] =     self._get_sklearn_model('svm', 'SVR', {'kernel': 'sigmoid'})
        self.__MODEL['svr-precomputed_kernel'] = self._get_sklearn_model('svm', 'SVR', {'kernel': 'precomputed'})
        self.__MODEL['linear_regression'] =      self._get_sklearn_model('linear_model', 'LinearRegression')
        self.__MODEL['lasso'] =                  self._get_sklearn_model('linear_model', 'Lasso')
        self.__MODEL['ridge'] =                  self._get_sklearn_model('linear_model', 'Ridge')
        self.__MODEL['elastic_net'] =            self._get_sklearn_model('linear_model', 'ElasticNet')
        self.__MODEL['bayesian_ridge'] =         self._get_sklearn_model('linear_model', 'BayesianRidge')
        self.__MODEL['k-nearest_neighbours'] =   self._get_sklearn_model('neighbors', 'KNeighborsRegressor')
        self.__MODEL['random_forest'] =          self._get_sklearn_model('ensemble', 'RandomForestRegressor')
        self.__MODEL['gradient_boosting'] =      self._get_sklearn_model('ensemble', 'GradientBoostingRegressor')
        self.__MODEL['decision_tree'] =          self._get_sklearn_model('tree', 'DecisionTreeRegressor')
        
    def __load_param_grid(self):
        self.__PARAM_GRID['svr-linear_kernel'] =      {
            'C': [1, 10, 100, 1000],
            'epsilon': [1, 5, 10]
            }
        self.__PARAM_GRID['svr-poly_kernel'] =        {
            'C': [1, 10, 100, 1000],
            'epsilon': [1, 5, 10]
            }
        self.__PARAM_GRID['svr-RBF_kernel'] =         {
            'C': [1, 10, 100, 1000],
            'epsilon': [1, 5, 10]
            }
        self.__PARAM_GRID['svr-sigmoid_kernel'] =     {
            'C': [1, 10, 100, 1000],
            'epsilon': [1, 5, 10]
            }
        self.__PARAM_GRID['svr-precomputed_kernel'] = {
            'C': [1, 10, 100, 1000],
            'epsilon': [1, 5, 10]
            }
        self.__PARAM_GRID['linear_regression'] =      {
            'fit_intercept': ['True', 'False']
            }
        self.__PARAM_GRID['lasso'] =                  {
            'alpha': [1, 1e-1, 1e-2, 1e-3, 1e-4]
            }
        self.__PARAM_GRID['ridge'] =                  {
            'alpha': [1, 1e-1, 1e-2, 1e-3, 1e-4]
            }
        self.__PARAM_GRID['elastic_net'] =            {
            'alpha': [0.01, 0.1, 0.2, 0.5]
            }
        self.__PARAM_GRID['bayesian_ridge'] =         {
            'alpha_1': [1e-4, 1e-6, 1e-8],
            'alpha_2': [1e-4, 1e-6, 1e-8],
            'lambda_1': [1e-4, 1e-6, 1e-8],
            'lambda_2': [1e-4, 1e-6, 1e-8]
            }
        self.__PARAM_GRID['k-nearest_neighbours'] =   {
            'n_neighbors': [5, 10, 15, 20],
            'weights': ['uniform', 'distance']
            }
        self.__PARAM_GRID['random_forest'] =          {
            'n_estimators': [250, 500, 750, 1000]
            }
        self.__PARAM_GRID['gradient_boosting'] =      {
            'n_estimators': [250, 500, 750, 1000],
            'learning_rate': [0.05, 0.1, 0.15, 0.2]
            }
        self.__PARAM_GRID['decision_tree'] =          {
            }

    def __load_splitter(self):
        self

    def _get_sklearn_model(self, model_class, model_name, kwargs={}):
        if model_class in self.__MODEL_CLASSES.keys():
            sklearn_model = getattr(
                self.__MODEL_CLASSES[model_class],
                model_name
                )

        else:
            raise KeyError(
                'model_class {}, not in model_classes.'.format(model_class)
                )

        model = sklearn_model(**kwargs)

        return model

    def get_descriptors(self, descriptor_type, data, **kwargs):
        rxn_components = list(data.index.names)
        reactions = data.index.to_frame()

        if descriptor_type == 'quantum':
            descriptors = descriptor_generator().assemble_quantum_descriptors(
                rxn_components, reactions, **kwargs
                )

        elif 'wl' in descriptor_type:
            # kwargs.update({'n_jobs': self.n_jobs})
            kwargs.update({'n_jobs': 1})
            descriptors = descriptor_generator().assemble_graph_descriptors(
                rxn_components, reactions, rxn_smiles=data
                )
            if not data.isnull().values.any():
                self.__PREPROCESSING['wl_kernel'] \
                    = [(
                        self.__PREPROCESSING['wl_kernel'][0][0],
                        self.__PREPROCESSING['wl_kernel'][0][1].set_params(
                            kernel_params=kwargs)
                        )]
            else:
                self.__PREPROCESSING['wl_kernel_missing_mols'] \
                    = [(
                        self.__PREPROCESSING['wl_kernel_missing_mols'][0][0],
                        # self.__PREPROCESSING[descriptor_type][0][1].set_params(
                        #     kernel_params=kwargs)
                        #     )]
                        self.__PREPROCESSING['wl_kernel_missing_mols'][0][1].set_params(
                            kernel_params=kwargs)
                            )]

        elif 'tanimoto' in descriptor_type:
            descriptors = descriptor_generator().assemble_fingerprint_descriptors(
                rxn_components, reactions, rxn_smiles=data, return_raw=True,
                **kwargs
                )

        elif descriptor_type == 'fingerprints':
            descriptors = descriptor_generator().assemble_fingerprint_descriptors(
                rxn_components, reactions, rxn_smiles=data, return_concat=True,
                **kwargs
                )

        elif descriptor_type == 'one-hot':
            descriptors = reactions.copy()

        return descriptors

    def get_split(self, X, y, splitter_name, settings):
        if splitter_name == 'activity_ranking':
            test_sets = splitter.activity_ranking(X, y, **settings)
        if splitter_name == 'leave-one-component-out':
            test_sets = splitter.leave_one_out(X, y, **settings)
        if splitter_name == 'cross-validation':
            test_sets = splitter.cross_validation(X, y)
        if splitter_name == 'user-defined_mols':
            test_sets = splitter.user_defined_mols(X, y, **settings)
        return test_sets

    def get_preprocessing(
            self, descriptor_type, data, model_name, kernel_kwargs={}
            ):
        if descriptor_type in ['quantum', 'fingerprints', 'one-hot']:
            return self.__PREPROCESSING[descriptor_type]

        elif 'wl' in descriptor_type:
            descriptor_type = 'wl'

        elif 'tanimoto' in descriptor_type:
            descriptor_type = 'tanimoto'

        else:
            raise ValueError('{} not a valid descriptor_type. Options: \
                             quantum, wl_kernel, fingerprints, tanimoto or \
                                 one-hot')

        if 'svr' not in model_name:
            if not data.isnull().values.any():
                return self.__PREPROCESSING[
                    '{}_features'.format(descriptor_type)
                    ]
            else:
                return self.__PREPROCESSING[
                    '{}_features_missing_mols'.format(descriptor_type)
                    ]

        else:
            if not data.isnull().values.any():
                preprocessing = self.__PREPROCESSING[
                    '{}_kernel'.format(descriptor_type)
                    ].copy()
            else:
                preprocessing = self.__PREPROCESSING[
                    '{}_kernel_missing_mols'.format(descriptor_type)
                    ].copy()

            kernel_name = model_name[4:]
            if kernel_name != 'precomputed_kernel':
                preprocessing.extend(
                    self.get_kernel(kernel_name, **kernel_kwargs)
                    )
                self.__MODEL[model_name] = self.__MODEL['svr-precomputed_kernel']
                self.__PARAM_GRID[model_name].update(
                    self.__KERNEL_PARAM_GRID[kernel_name]
                    )

            return preprocessing

    def get_kernel(self, kernel, **kwargs):
        if kwargs:
            self.__KERNEL[kernel] = self.__KERNEL[kernel].set_params(**kwargs)

        return self.__KERNEL[kernel]

    def get_model(self, model_name, model_kwargs={}):
        if model_kwargs:
            self.__MODEL[model_name].set_params(**model_kwargs)

        return self.__MODEL[model_name]

    def get_param_grid(self, model_name):
        return self.__PARAM_GRID[model_name]

    def get_pipeline(self, preprocess, model, model_name):
        pipeline = preprocess.copy()
        pipeline.extend([('model', model)])

        model_pipeline = Pipeline(pipeline)

        keys = [k for k in self.__PARAM_GRID[model_name].keys() 
                if '__' not in k]
        for k in keys:
            self.__PARAM_GRID[model_name]['model__{}'.format(k)] \
                = self.__PARAM_GRID[model_name].pop(k)
            # self.__PARAM_GRID[model_name] = {
            # 'model__{}'.format(k): v
            # for k, v in self.__PARAM_GRID[model_name].items()
            # }

        return model_pipeline
