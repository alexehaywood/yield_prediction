#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import logging
from collections import defaultdict
from collections.abc import Iterable
from time import time
from inspect import getmembers, isfunction
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
from sklearn import svm, linear_model, neighbors, naive_bayes, tree, ensemble
from sklearn import neural_network
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin


# from tools.machine_learning import kernels
# import kernels

model_classes = {
    'svm': svm,
    'linear_model': linear_model,
    'neighbors': neighbors,
    'naive_bayes': naive_bayes,
    'tree': tree,
    'ensemble': ensemble,
    'neural_network': neural_network
    }

sklearn_scorers = [name for name, _ in getmembers(metrics, isfunction)]


class scaler(BaseEstimator, TransformerMixin):
    """
    Wrapper for scalers of scikit-learn.

    Parameters
    ----------
    scaler_name : str, optional
        Possible values are:
            Binarizer, FunctionTransformer, KBinsDiscretizer,
            KernelCenterer, LabelBinarizer, LabelEncoder,
            MultilabelBinarizer, MaxAbsScaler, MinMaxScaler, Normalizer,
            OneHotEncoder, OrdinalEncoder, PolynomialFeatures,
            PowerTransformer, QuantileTransformer, RobustScaler,
            StandardScaler.
        The default is 'StandardScaler'.

    """

    def __init__(self, scaler_name='StandardScaler'):
        self.name = scaler_name
        sklearn_scaler = getattr(preprocessing, self.name)
        self.scaler = sklearn_scaler()

    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return X_scaled


# class split_data:

#     def random_split(*arrays, test_size=0.3):
#         return train_test_split(*arrays, test_size=test_size, random_state=0)


class scorer():
    """
    
    Parameters
        ----------
        scorer_name_from_metrics_module : TYPE, optional
            DESCRIPTION. The default is None.
        scorer_name_from_metrics_SCORES : TYPE, optional
            DESCRIPTION. The default is None.
        add_scorer : TYPE, optional
            DESCRIPTION. The default is None.
    
    """

    def __init__(
            self, scorer_name_from_metrics_module=None,
            scorer_name_from_metrics_SCORES=None, add_scorer=None
            ):
        if scorer_name_from_metrics_module is not None:
            self.scorer = getattr(metrics, scorer_name_from_metrics_module)

        if scorer_name_from_metrics_SCORES is not None:
            self.scorer = get_scorer(scorer_name_from_metrics_SCORES)

        if add_scorer is not None:
            self.scorer = add_scorer

    def score(self, y_true, y_pred, **kwargs):
        z = self.scorer(y_true, y_pred, **kwargs)
        return z

    def predict_and_score(self, model, X, y_true):
        z = self.scorer(model, X, y_true)
        return z

    def make(self, **kwargs):
        scorer = make_scorer(self.scorer, **kwargs)
        return scorer

    def run(scoring, model=None, X_test=None, y_test=None, y_pred=None):
        scores = defaultdict()
        for scorer_name, scorer_fn in scoring.items():
            if isinstance(scorer_fn, str):
                if scorer_fn in metrics.SCORERS.keys():
                    scores[scorer_name] = scorer(
                        scorer_name_from_metrics_SCORES=scorer_fn
                        ).predict_and_score(model, X_test, y_test)
                else:
                    try:
                        scores[scorer_name] = scorer(
                            scorer_name_from_metrics_module=scorer_fn
                            ).score(y_test, y_pred)
                    except AttributeError:
                        print('Module "{}" not found in sklearn.metrics,'
                              .format(scorer_fn))
            else:
                scores[scorer_name] = scorer(
                    add_scorer=scorer_fn
                    ).predict_and_score(model, X_test, y_test)

        return scores


class model_selector(BaseEstimator, RegressorMixin):
    """
    Wrapper for training and testing models.

    Parameters
    ----------
    model_class : str
        Name of model class. Possible values are defined in model_classes.
        model_classes = {
            'svm' : svm,
            'linear_model' : linear_model,
            'neighbours' : neighbors,
            'naive_bayes' : naive_bayes,
            'tree' : tree,
            'ensemble' : ensemble,
            'neural_network' : neural_network
            }
    model_name : str
        Name of model within class.
    model_kwargs : dict
        Sklearn model parameters.
    pipeline_pre_model: list
        Transformers to apply before model.
    pipeline_post_model: list
        Transformers to apply after model.
    hyperparameter_tuning_method: str, default=None
        Either the exhaustive 'GridSearchCV' or 'RandomizedSearchCV.
    hyperparameter_tuning_kwargs: dict
        Sklearn cross-validation search method parameters.

    Attributes
    ----------
    model: object
        Sklearn model or sklearn search method.

    """

    def __init__(
            # self, model_class, model_name, model_kwargs={},
            # pipeline_pre_model=[], pipeline_post_model=[]
            self, model
            ):
        # self.model_class = model_class
        # self.model_name = model_name
        # self.model_kwargs = model_kwargs
        # self.pipeline_pre_model = pipeline_pre_model
        # self.pipeline_post_model = pipeline_post_model
        self.model = model

    # def initialise_model(self):
    #     # Initialise model.
    #     print('Model: {}'.format(self.model_name))

    #     try:
    #         sklearn_model = getattr(
    #             model_classes[self.model_class],
    #             self.model_name
    #             )
    #     except KeyError:
    #         print('model_class: {}, not in model_classes.'
    #               .format(self.model_class))

    #     if not self.pipeline_pre_model or self.pipeline_post_model:
    #         model = sklearn_model(**self.model_kwargs)
    #     else:
    #         pipeline = self.pipeline_pre_model.copy()
    #         pipeline.extend([('model', sklearn_model(**self.model_kwargs))])
    #         pipeline.extend(self.pipeline_post_model)

    #         model = Pipeline(pipeline)

    #     print(pipeline)

    #     self.model = model

    def initialise_hyperparameter_tuning(
            self, hyperparameter_tuning_method,
            **hyperparameter_tuning_kwargs
            ):
        # Initialise hyperparameter tuning.
        hyperparameter_tuning = getattr(
            model_selection,
            hyperparameter_tuning_method
            )

        self.model = hyperparameter_tuning(
            self.model, **hyperparameter_tuning_kwargs
            )

    def fit(self, X, y):
        logging.info('\t\tModel fitting started...')
        t0 = time()
        self.model.fit(X, y)
        logging.info('\t\t...finished ({:.1f}s)'.format(time() - t0))

    def predict(self, X):
        logging.info('\t\tPrediction started...')
        t0 = time()
        y_pred = self.model.predict(X)
        logging.info('\t\t...finished ({:.1f}s)'.format(time() - t0))
        return y_pred

    def get_score(self, scorer_name, y_true, y_pred):
        if type(scorer_name) is str:

            if scorer_name in metrics.SCORERS.keys():
                scorer_function = get_scorer(scorer_name)._score_func
                kwargs = get_scorer(scorer_name)._kwargs
                score = scorer_function(y_true, y_pred, **kwargs)
            elif scorer_name in sklearn_scorers:
                scorer_function = getattr(metrics, scorer_name)
                score = scorer_function(y_true, y_pred)
            else:
                logging.error(
                    'ERROR: Invalid str - {} not found in \
                    sklearn.SCORERS.keys() nor sklearn.metrics'
                    .format(scorer_name)
                    )
                raise ValueError(
                    'Invalid str - {} not found in \
                    sklearn.SCORERS.keys() nor sklearn.metrics'
                    .format(scorer_name)
                    )

        else:
            try:
                scorer_function = scorer_name._score_func
                kwargs = scorer_name._kwargs
                score = scorer_function(y_true, y_pred, **kwargs)
                scorer_name = scorer_function._score_func.__name__
                print(scorer_name, scorer_function)

            except ValueError:
                logging.error('ERROR: Invalid scorer function')
                print('Invalid scorer function')

        return scorer_name, score

    def score(self, y_true, y_pred, scoring):
        logging.info('\t\tModel scoring started...')
        t0 = time()

        if (type(scoring) is list) or (type(scoring) is tuple):

            scores = defaultdict()
            for scorer_name in scoring:
                scorer_name, score \
                    = self.get_score(scorer_name, y_true, y_pred)
                scores[scorer_name] = score

        elif type(scoring) is dict:

            scores = defaultdict()
            for scorer_name, scorer_function in scoring.items():
                n, score \
                    = self.get_score(scorer_function, y_true, y_pred)
                scores[scorer_name] = score

        elif (type(scoring) is str) or (not isinstance(scoring, Iterable)):
            scorer_name, score = self.get_score(scoring, y_true, y_pred)
            scores = score

        logging.info('\t\t...finished ({:.1f}s)'.format(time() - t0))

        return scores
