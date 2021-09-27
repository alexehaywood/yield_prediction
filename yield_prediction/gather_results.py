#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:17:41 2021

@author: alexehaywood
"""

import os
from collections import defaultdict
import pandas as pd
import numpy as np


class gather_results():
    def __init__(
            self, path_head, descriptor_names, test_name,
            sheet_names, index_cols, paths=None
            ):
        self.path_head = path_head
        self.descriptor_names = descriptor_names
        self.test_name = test_name
        self.sheet_names = sheet_names
        self.index_cols = index_cols

        self.results = defaultdict()

        if paths:
            self.paths = paths
            self.test_set_names = None

        else:
            self.paths = []
            self.test_set_names = defaultdict()

            for descriptor_name in descriptor_names:
                temp_path_head = os.path.join(
                    path_head, descriptor_name, test_name
                    )
                test_set_names = [
                    path
                    for path in os.listdir(temp_path_head)
                    if os.path.isdir(os.path.join(temp_path_head, path))
                    ]
                self.test_set_names[descriptor_name] = test_set_names

                if test_set_names:
                    self.paths.append([
                        os.path.join(
                            self.path_head, descriptor_name, test_name,
                            test_set_name, 'results.xlsx'
                            )
                        for test_set_name in test_set_names
                        ])

                else:
                    self.paths.append([
                        os.path.join(
                            self.path_head, descriptor_name, test_name,
                            'results.xlsx'
                            )
                        ])

        self.get_results()

    def clean_df(self, df, keys, names, sheet_name):
        # Name df index and columns
        df.index.name = sheet_name
        df.columns = df.columns.str.replace('_', ' ')
        df.columns = df.columns.str.replace('-', ' - ')
        df.columns = df.columns.str.title()
        df.columns = df.columns.str.replace('Svr', 'SVR')
        df.columns = df.columns.str.replace('Wl', 'WL')
        df.columns = df.columns.str.replace('Rbf', 'RBF')

        # Add test, descriptor, test set and test name to index
        for key, name in zip(keys, names):
            df = pd.concat(
                [df],
                keys=[key],
                names=[name]
                )

        return df

    def get_results(self):
        for sheet_name, index_col in zip(self.sheet_names, self.index_cols):
            results = pd.DataFrame()

            for descriptor_name, path_list in zip(
                    self.descriptor_names, self.paths
                    ):
                test_set_name_list = self.test_set_names[descriptor_name]

                # Read data
                for n, path in enumerate(path_list):
                    if isinstance(index_col, int) or any(isinstance(x, int) for x in index_col):
                        df = pd.read_excel(
                            path,
                            sheet_name=sheet_name,
                            index_col=index_col
                            )
                    else:
                        df = pd.read_excel(
                            path,
                            sheet_name=sheet_name,
                            ).set_index(index_col)

                    if df.index.to_frame().isnull().values.any():
                        if len(df.index.names) > 1:
                            df.index = pd.MultiIndex.from_frame(
                                df.index.to_frame().fillna('none'),
                                names=df.index.names
                                )
                        else:
                            df.index = df.index.fillna('none')

                    if sheet_name == 'y_pred':
                        df = df.reset_index().pivot_table(
                            index=['additive', 'aryl_halide', 'base', 'ligand']
                            )

                    # # Remame SVR - Precomputed Kernel based on descriptor_name
                    # if 'wl_kernel' in descriptor_name:
                    #     df.rename(
                    #         columns={'svr-precomputed_kernel':
                    #                  'svr-WL_kernel'},
                    #         inplace=True
                    #         )
                    # elif 'tanimoto_kernel' in descriptor_name:
                    #     df.rename(
                    #         columns={'svr-precomputed_kernel':
                    #                  'svr-tanimoto_kernel'},
                    #         inplace=True
                    #         )

                    keys = ['_'.join(os.path.split(descriptor_name))]
                    names = ['Descriptor']
                    if test_set_name_list:
                        keys.append(test_set_name_list[n])
                        names.append('Test Set')
                    df = self.clean_df(df, keys, names, sheet_name)

                    # Add df to results dataframe
                    results = pd.concat([results, df])
                    if results.columns.name != 'Model':
                        results.columns.name = 'Model'

                    self.results[sheet_name] = results

    def save(
            self, saveas_path, saveas_index_order=None,
            saveas_column_order=None, saveas_mean=False, saveas_cols=None,
            saveas_index=None
            ):
        results_to_save = defaultdict()
        for sheet_name, df in self.results.items():

            if sheet_name == 'y_pred':
                for descriptor_name in self.descriptor_names:
                    descriptor_name = '_'.join(os.path.split(descriptor_name))
                    results_to_save[descriptor_name] = df[
                        df.index.get_level_values('Descriptor')
                        == descriptor_name
                        ]

            else:
                df = df.stack().reset_index()
                for k in df[sheet_name].drop_duplicates().values:
                    df_k = df[df[sheet_name] == k]
                    df_k = df_k.drop(columns=[sheet_name])
                    df_k = df_k.pivot_table(
                        index=saveas_index_order.copy(),
                        columns=saveas_column_order.copy()
                        )
                    df_k.columns = df_k.columns.droplevel(0)
                    df_k.sort_index(
                        axis='index',
                        level=saveas_index_order,
                        inplace=True
                        )
                    df_k.sort_index(
                        axis='columns',
                        level=saveas_column_order,
                        inplace=True
                        )
                    if saveas_cols is not None:
                        df_k = df_k[saveas_cols]
                    if (saveas_index is not None) & (type(saveas_index) is dict):
                        self.df_k = df_k
                        for ind_name, ind_val in saveas_index.items():
                            df_k = df_k[df_k.index.get_level_values(
                                ind_name
                                ).isin(ind_val)]
                            if df_k.index.name:
                                df_k = df_k.reindex(ind_val)
                            else:
                                df_k = df_k.reindex(ind_val, level=ind_name)
                    if saveas_mean is True:
                        mean = df_k.mean(axis='columns')
                        std = df_k.std(axis='columns')
                        df_k['Mean'], df_k['Std'] = mean, std
                    results_to_save[
                        '{}_{}'.format(sheet_name, k.split('_')[-1])
                        ] = df_k.copy()

            writer = pd.ExcelWriter('{}.xlsx'.format(saveas_path))
            for k, df in results_to_save.items():
                results_to_save[k].to_excel(
                    writer, sheet_name=k, merge_cells=False
                    )
            writer.save()


# =============================================================================
# Cross-Validation
# =============================================================================
path_head = 'C:/Users/alexe/Documents/PhD/yield_prediction/output/results'
test_name = 'cv'
sheet_names = ['best_mean_cv_scores', 'training_scores']
index_cols = [0, 0]
saveas_index_order = ['Descriptor']
saveas_column_order = ['Model']
saveas_mean = True

# Fingerprints and Tanimoto Kernel
descriptor_names = [
    '{}/morgan{}_{}'.format(folder, radius, bit_size)
    for folder in ['fingerprints', 'tanimoto_kernel']
    for radius in [1, 2, 3]
    for bit_size in [32, 64, 128, 256, 512, 1024, 2048]
    ]
descriptor_names.extend([
    '{}/fmorgan{}_{}'.format(folder, radius, bit_size)
    for folder in ['fingerprints', 'tanimoto_kernel']
    for radius in [1, 2, 3]
    for bit_size in [32, 64, 128, 256, 512, 1024, 2048]
    ])
descriptor_names.extend([
    '{}/rdk_{}'.format(folder, bit_size)
    for folder in ['fingerprints', 'tanimoto_kernel']
    for bit_size in [32, 64, 128, 256, 512, 1024, 2048]
    ])

fn = gather_results(
    path_head, descriptor_names, test_name, sheet_names, index_cols
    )

results_to_save = fn.save(
    'output/cv_fps_svr', saveas_index_order, saveas_column_order, saveas_mean,
    saveas_cols=[
        'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'SVR - Precomputed Kernel'
        ]
    )
results_to_save = fn.save(
    'output/cv_fps_linear', saveas_index_order, saveas_column_order,
    saveas_mean, saveas_cols=[
        'Linear Regression', 'Lasso', 'Ridge', 'Elastic Net', 'Bayesian Ridge'
        ]
    )
results_to_save = fn.save(
    'output/cv_fps_tree', saveas_index_order, saveas_column_order, saveas_mean,
    saveas_cols=[
        'Decision Tree', 'Gradient Boosting', 'Random Forest'
        ]
    )

# WL Kernel
descriptor_names = [
    'wl_kernel/wl{}'.format(n_iter)
    for n_iter in np.arange(2, 11)
    ]

fn = gather_results(
    path_head, descriptor_names, test_name, sheet_names, index_cols
    )

results_to_save = fn.save(
    'output/cv_wl_svr', saveas_index_order, saveas_column_order, saveas_mean,
    saveas_cols=[
        'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'SVR - Precomputed Kernel'
        ]
    )
results_to_save = fn.save(
    'output/cv_wl_linear', saveas_index_order, saveas_column_order,
    saveas_mean, saveas_cols=[
        'Linear Regression', 'Lasso', 'Ridge', 'Elastic Net', 'Bayesian Ridge'
        ]
    )
results_to_save = fn.save(
    'output/cv_wl_tree', saveas_index_order, saveas_column_order, saveas_mean,
    saveas_cols=[
        'Decision Tree', 'Gradient Boosting', 'Random Forest'
        ]
    )

# All Models
saveas_index_order = ['Model']
saveas_column_order = ['Descriptor']
saveas_mean = True

descriptor_names = [
    'one-hot_encodings', 'quantum', 'fingerprints/morgan1_512',
    'fingerprints/rdk_512', 'fingerprints/maccs',
    'tanimoto_kernel/morgan1_512', 'tanimoto_kernel/rdk_512',
    'tanimoto_kernel/maccs', 'wl_kernel/wl5'
    ]

fn = gather_results(
    path_head, descriptor_names, test_name, sheet_names, index_cols
    )

results_to_save = fn.save(
    'output/cv_all_linear', saveas_index_order, saveas_column_order,
    saveas_mean,
    saveas_cols=[
        'Linear Regression', 'Lasso', 'Ridge', 'Elastic Net', 'Bayesian Ridge',
        ]
    )
results_to_save = fn.save(
    'output/cv_all_svr', saveas_index_order, saveas_column_order, saveas_mean,
    saveas_cols=[
        'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'SVR - Precomputed Kernel',
        ]
    )
results_to_save = fn.save(
    'output/cv_all_tree', saveas_index_order, saveas_column_order, saveas_mean,
    saveas_cols=[
        'Decision Tree', 'Gradient Boosting', 'Random Forest'
        ]
    )

# =============================================================================
# Out-of-sample
# =============================================================================
path_head = 'C:/Users/alexe/Documents/PhD/yield_prediction/output/results'
descriptor_names = [
    'one-hot_encodings', 'quantum', 'fingerprints/morgan1_512',
    'fingerprints/rdk_512', 'fingerprints/maccs',
    'tanimoto_kernel/morgan1_512', 'tanimoto_kernel/rdk_512',
    'tanimoto_kernel/maccs', 'wl_kernel/wl5'
    ]

# Scores
sheet_names = ['best_mean_cv_scores', 'training_scores', 'scores']
index_cols = [0, 0, 0]
saveas_index_order = ['Descriptor', 'Model']
saveas_column_order = ['Test Set']
saveas_mean = True

for test_name in [
        'additive_ranking', 'aryl_halide_ranking',
        # 'additive_plates', 'aryl_halide_aryls', 
        # 'aryl_halide_halides',
        # 'base_loo', 
        'ligand_loo'
        ]:
    fn = gather_results(
        path_head, descriptor_names, test_name, sheet_names, index_cols
        )

    results_to_save = fn.save(
        'output/{}'.format(test_name), saveas_index_order, saveas_column_order,
        saveas_mean, saveas_index={'Model': [
            'Linear Regression', 'Lasso', 'Ridge', 'Bayesian Ridge',
            'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
            'SVR - Sigmoid Kernel',  'SVR - Precomputed Kernel',
            'Gradient Boosting', 'Random Forest'
            ]}
        )

# Predicted Yields
sheet_names = ['y_pred']
index_cols = [[0, 1, 2, 3]]

for test_name in [
        'additive_ranking', 'aryl_halide_ranking',
        # 'additive_plates', 'aryl_halide_aryls', 
        # 'aryl_halide_halides',
        # 'base_loo', 'ligand_loo'
        ]:
    fn = gather_results(
        path_head, descriptor_names, test_name, sheet_names, index_cols
        )

    results_to_save = fn.save(
        'output/{}_ypred'.format(test_name)
        )


# =============================================================================
# Validation
# =============================================================================
path_head = 'C:/Users/alexe/Documents/PhD/yield_prediction/output/results'
descriptor_names = [
    'one-hot_encodings', 'quantum', 'fingerprints/morgan1_512',
    'tanimoto_kernel/morgan1_512', 'wl_kernel/wl5'
    ]

sheet_names = ['best_params', 'best_mean_cv_scores', 'training_scores']
index_cols = [0, 0, 0]
saveas_index_order = ['Descriptor', 'Model']
saveas_column_order = ['Test Set']
saveas_mean = False

for test_name in [
        'subset'
        ]:
    fn = gather_results(
        path_head, descriptor_names, test_name, sheet_names, index_cols
        )

    results_to_save = fn.save(
        'output/validation_{}'.format(test_name.replace('/', '_')), saveas_index_order, saveas_column_order,
        saveas_mean, saveas_index={'Model': [
            'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
            'SVR - Sigmoid Kernel',  'SVR - Precomputed Kernel',
            ]}
        )

sheet_names = ['y_pred']
index_cols = [['additive', 'aryl_halide', 'base', 'ligand']]
for test_name in [
        'subset'
        ]:
    fn = gather_results(
        path_head, descriptor_names, test_name, sheet_names, index_cols
        )

    results_to_save = fn.save(
        'output/validation_{}_ypred'.format(test_name)
        )

descriptor_names = [
    'one-hot_encodings', 'fingerprints/morgan1_512',
    'tanimoto_kernel/morgan1_512', 'wl_kernel/wl5'
    ]

sheet_names = ['best_params', 'best_mean_cv_scores', 'training_scores']
index_cols = [0, 0, 0]
saveas_index_order = ['Descriptor', 'Model']
saveas_column_order = ['Test Set']
saveas_mean = False

for test_name in [
        'all'
        ]:
    fn = gather_results(
        path_head, descriptor_names, test_name, sheet_names, index_cols
        )

    results_to_save = fn.save(
        'output/validation_{}'.format(test_name.replace('/', '_')), saveas_index_order, saveas_column_order,
        saveas_mean, saveas_index={'Model': [
            'SVR - Poly Kernel',
            'SVR - Precomputed Kernel',
            ]}
        )

sheet_names = ['y_pred']
index_cols = [['additive', 'aryl_halide', 'base', 'ligand']]
for test_name in [
        'all'
        ]:
    fn = gather_results(
        path_head, descriptor_names, test_name, sheet_names, index_cols
        )

    results_to_save = fn.save(
        'output/validation_{}_ypred'.format(test_name)
        )