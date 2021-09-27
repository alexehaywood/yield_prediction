#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:18:00 2020

@author: pcxah5
"""

import logging
import os
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from joblib import dump, load

from tools.machine_learning.model_selection import scorer, model_selector

class machine_learning_grid_search():

    def __init__(
            self, model_names, pipelines, param_grids, X_train, y_train,
            X_test=None, y_test=None, n_jobs=1
            ):

        self.model_names = model_names
        self.pipelines = pipelines
        self.param_grids = param_grids

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.n_jobs = n_jobs

    def r2_and_rmse_scorer(self):
        rmse_scorer = scorer(
            scorer_name_from_metrics_module='mean_squared_error'
            ).make(greater_is_better=False, squared=False)

        scoring = {
            'R-squared': 'r2',
            'RMSE': rmse_scorer
            }

        return scoring

    def setup_models_wrappers(self):
        self.models_wrappers = []

        for pipeline in self.pipelines:
            model_wrapper = model_selector(pipeline)
            self.models_wrappers.append(model_wrapper)

    def run(self):
        self.setup_models_wrappers()

        models = defaultdict()
        results = defaultdict(dict)

        if self.X_test is not None:
            results['y_pred'] = pd.DataFrame()

        # scoring = self.r2_and_rmse_scorer()
        scoring = {'R-squared': 'r2', 'RMSE': 'neg_root_mean_squared_error'}

        for model_name, model_wrapper, param_grid in zip(
                self.model_names, self.models_wrappers, self.param_grids
                ):
            logging.info('\n\tModel Name:\t{}'.format(model_name))
            logging.info('\tPipeline:\t{}'.format(model_wrapper.model))
            logging.info('\tParameter Grid:\t{}\n'.format(param_grid))

            logging.info('\tTuning Hyperparameters')
            model_wrapper.initialise_hyperparameter_tuning(
                'GridSearchCV',
                param_grid=param_grid,
                scoring=scoring,
                refit='R-squared',
                n_jobs=self.n_jobs,
                cv=KFold(n_splits=5, shuffle=True, random_state=0),
                verbose=2
                )

            model_wrapper.fit(self.X_train, self.y_train)

            results['best_params'][model_name] \
                = model_wrapper.model.best_params_

            best_mean_cv_score = {
                k: v[model_wrapper.model.best_index_]
                for (k, v)
                in model_wrapper.model.cv_results_.items()
                if 'mean_test' in k
                }
            best_mean_cv_score['mean_test_RMSE'] \
                = best_mean_cv_score['mean_test_RMSE']*-1
            results['best_mean_cv_scores'][model_name] = best_mean_cv_score

            logging.info('\tCalculating Training Errors')
            training_y_pred = model_wrapper.predict(self.X_train)
            training_score = model_wrapper.score(
                y_true=self.y_train,
                y_pred=training_y_pred,
                scoring=scoring
                )
            results['training_scores'][model_name] = training_score

            if self.X_test is not None:
                logging.info('\tPredicting Test Set')
                y_pred = model_wrapper.predict(self.X_test)
                y_pred = pd.DataFrame(
                    y_pred,
                    index=self.X_test.index,
                    columns=[model_name]
                    )
                results['y_pred'] = pd.concat([results['y_pred'], y_pred])

                if self.y_test is not None:
                    logging.info('\tCalculating Test Errors')
                    score = model_wrapper.score(
                        self.y_test,
                        y_pred,
                        scoring=scoring
                        )
                    results['scores'][model_name] = score

            models[model_name] = model_wrapper.model

        results['best_params'] = pd.DataFrame(results['best_params'])
        results['best_mean_cv_scores'] = pd.DataFrame.from_dict(
            results['best_mean_cv_scores']
            )
        results['training_scores'] = pd.DataFrame.from_dict(
            results['training_scores']
            )

        if 'scores' in results.keys():
            results['scores'] = pd.DataFrame(results['scores'])

        self.models = models
        self.results = results

    def save_models(self, saveas_dir='.', saveas_fnames=None):
        logging.info('\n\tSaving Models')

        if not os.path.exists(saveas_dir):
            os.makedirs(saveas_dir)

        if not saveas_fnames:
            saveas_fnames = [model_name for model_name in self.models.keys()]

        for saveas_fname, model in zip(saveas_fnames, self.models.values()):
            fpath = '{}/{}.pk'.format(
                saveas_dir, saveas_fname.replace(' ', '_')
                )

            if type(model.best_estimator_) is Pipeline:
                pipeline_data = {
                    'pipeline_steps': list(
                        model.best_estimator_.named_steps.keys()
                        ),
                    'model': model.best_estimator_.named_steps['model'],
                    'X_train': self.X_train,
                    'y_train': self.y_train
                    }
                for k, v in model.best_estimator_.named_steps.items():
                    if k != 'model':
                        pipeline_data.update({k: v.get_params()})
                        params = [
                            i for i in model.best_params_.keys()
                            if i.startswith(k)
                            ]
                        for param in params:
                            pipeline_data.update({
                                k: {param.split('__')[-1]:
                                    model.best_params_[param]}
                                })
                dump(
                    pipeline_data,
                    '{}'.format(fpath)
                    )
            else:
                dump(
                    model.best_estimator_,
                    '{}'.format(fpath),
                    )

    def save_results(self, saveas_dir='.', saveas_fname='results'):
        logging.info('\n\tSaving Results Table')

        if not os.path.exists(saveas_dir):
            os.makedirs(saveas_dir)

        fpath = '{}/{}.xlsx'.format(saveas_dir, saveas_fname)

        with pd.ExcelWriter(fpath) as writer:
            for name, data in self.results.items():
                data.to_excel(
                    writer, sheet_name=name, merge_cells=False
                    )
