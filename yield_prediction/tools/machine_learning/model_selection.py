#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from collections import defaultdict
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import sklearn.model_selection as model_selection
from sklearn import svm, linear_model, neighbors, naive_bayes, tree, ensemble
from sklearn import neural_network
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import get_scorer

model_classes = {'svm': svm,
              'linear_model': linear_model,
              'neighbors': neighbors,
              'naive_bayes': naive_bayes,
              'tree': tree,
              'ensemble': ensemble,
              'neural_network': neural_network
              }
              
class scaler():
    
    def __init__(self, scaler_name='StandardScaler'):
        """
        Initialise scaler class.

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

        Returns
        -------
        None.

        """
        self.name = scaler_name
        sklearn_scaler = getattr(preprocessing, self.name)
        self.scaler = sklearn_scaler()
        
    def fit(self, X):
        self.scaler.fit(X)
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def scale_descriptors(self, X_train, X_test):
        """
        Scale descriptors.

        Parameters
        ----------
        X_train : array-like
            Raw descriptors to fit to model.
        X_test : array_like
            Raw descriptors to test the model with.

        Returns
        -------
        X_train_scaled : array_like
            Scaled descriptors to fit to model.
        X_test_scaled : array_like
            Scaled descriptors to test the model with.

        """
        self.fit(X_train)
        X_train_scaled = self.transform(X_train)
        X_test_scaled = self.transform(X_test)
        return X_train_scaled, X_test_scaled

class split_data:
    
    def random_split(*arrays, test_size=0.3):
        return train_test_split(*arrays, test_size=test_size, random_state=0)
        
class scorer():
    def __init__(self, scorer_name_from_metrics_module=None, 
                 scorer_name_from_metrics_SCORES=None, add_scorer=None):
        
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
                        print('Module "{}" not found in sklearn.metrics,'\
                              .format(scorer_fn))
            else:
                scores[scorer_name] = scorer(
                    add_scorer=scorer_fn
                    ).predict_and_score(model, X_test, y_test)
        return scores

class model_selector():
    
    def __init__(self, model_class, model_name, **kwargs):
        """
        Initialise model_selector class.

        Parameters
        ----------
        model_class : str
            Name of model class. Possible values are defined in model_classes.
            model_classes = {'svm' : svm,
              'linear_model' : linear_model,
              'neighbours' : neighbors,
              'naive_bayes' : naive_bayes,
              'tree' : tree,
              'ensemble' : ensemble,
              'neural_network' : neural_network}        
        model_name : str
            Name of model within class. 
        **kwargs : 
            Sklearn model parameters.

        Returns
        -------
        None.

        """
        self.model_name = model_name
        try:
            sklearn_model = getattr(model_classes[model_class], model_name)
        except KeyError:
            print('model_class: {}, not in model_classes.'.format(model_class))
        self.model = sklearn_model(**kwargs)
    
    def cv(self, X, y, scoring=None, **kwargs):
        """
        Cross validation.

        Parameters
        ----------
        X : array-like
            The descriptors to fit to model.
        y : array-like
            The target.
        scoring : string, callable, list/tuple, dict or None, default: None.
            Used to evaluate the predictions on the test set.
        **kwargs : 
            Sklearn cross_validate parameters.

        Returns
        -------
        cv_results : dict of float arrays
            Scores and times of the model for the cross validation.

        """
        cv_results = cross_validate(
            self.model, X, y, scoring=scoring, **kwargs)
        return cv_results
    
    def tune_hyperparameters(self, search_cv_method, X_train, y_train,
                             param_grid, scoring=None, X_test=None, 
                             y_test=None,  **kwargs):
        """
        

        Parameters
        ----------
        search_cv_method : str
            Either the exhaustive 'GridSearchCV' or 'RandomizedSearchCV.
        X_train : array-like
            Descriptors to fit to model.
        y_train : array_like
            The target.
        param_grid : dict or list of dicts
            The parameter names as keys and the list of parameters to search 
            as values.
        X_test : array-like, optional, default: None
            Descriptors to test the model with.
        **kwargs : 
            Sklearn cross-validation search method parameters.
        
        Returns
        -------
        dict
            Dictionary containg `best parameters`, `best_scores` and 
            `best_estimator. If X_test is set it also returns `y_pred`.

        """
        sklearn_search_cv_method = getattr(model_selection, search_cv_method)
        
        model = sklearn_search_cv_method(
            self.model, param_grid, scoring, **kwargs)
        
        self.tuned_model = model.fit(X_train, y_train)
        
        best_params = model.best_params_
        best_estimator = model.best_estimator_
        best_score = {k:v[model.best_index_] for (k,v) in 
                      model.cv_results_.items() if 'mean_test' in k}
        
        
        if X_test is None:
            return self.tuned_model, best_params, best_score, best_estimator
        
        elif y_test is None:
            y_pred = self.tuned_model.predict(X_test)
            return self.tuned_model, best_params, best_score, best_estimator, y_pred
        
        elif scoring is None:
            y_pred = self.tuned_model.predict(X_test)
            score =  self.tuned_model.score(X_test, y_test)
            return self.tuned_model, best_params, best_score, best_estimator, y_pred, score
        
        else:
            y_pred = self.tuned_model.predict(X_test)            
            score = scorer.run(scoring, model=self.tuned_model, X_test=X_test, y_test=y_test, y_pred=y_pred)            
            return self.tuned_model, best_params, best_score, best_estimator, y_pred, score
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def get_params(self):
        params = self.model.get_params()
        return params