# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:44:23 2021

@author: alexe
"""
import numpy as np
import pandas as pd
from collections import defaultdict


def activity_ranking(X, y, rxn_component, n_splits=4):
    mols = X.index.get_level_values(rxn_component).drop_duplicates()

    ranked_mols = []
    for mol in mols:
        ranked_mols.append({
            'mol': mol,
            'mean_y': y[
                y.index.get_level_values(rxn_component) == mol
                ].mean()
            })

    ranked_mols = pd.DataFrame(ranked_mols)
    ranked_mols.sort_values(by=['mean_y'], inplace=True)
    ranked_mols.reset_index(inplace=True, drop=True)
    ranked_mols.index = ranked_mols.index.set_names(['order'])

    test_sets = defaultdict()
    for n in range(n_splits):
        n = n + 1
        test_sets['ranking_set{}'.format(n)] \
            = split_descriptors_out_of_sample(
                X, y, rxn_component, ranked_mols.mol[n:-1:n_splits].tolist()
                )

    return test_sets


def leave_one_out(X, y, rxn_component):
    mols = X.index.get_level_values(rxn_component).drop_duplicates()

    test_sets = defaultdict()
    for mol in mols:
        test_sets['loo_{}'.format(mol)] \
            = split_descriptors_out_of_sample(
                X, y, rxn_component, [mol]
                )

    return test_sets


def user_defined_mols(X, y, rxn_component, test_sets_mols, test_sets_names=None):
    if not test_sets_names:
        test_sets_names = [
            'test_set{}'.format(n+1)
            for n in range(len(test_sets_mols))
            ]

    test_sets = defaultdict()
    for mols, name in zip(test_sets_mols, test_sets_names):
        test_sets[name] \
            = split_descriptors_out_of_sample(
                X, y, rxn_component, mols
                )

    return test_sets


def split_descriptors_out_of_sample(X, y, rxn_component, molecule_test_list):
    X_train = X[~X.index.get_level_values(
        rxn_component).isin(molecule_test_list)]

    y_train = y[~y.index.get_level_values(
        rxn_component).isin(molecule_test_list)]

    X_test = X[X.index.get_level_values(
        rxn_component).isin(molecule_test_list)]

    y_test = y[y.index.get_level_values(
        rxn_component).isin(molecule_test_list)]

    return {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test}


def cross_validation(X, y):
    X_train = X
    y_train = y
    test_sets = defaultdict()
    test_sets['cv'] = {'X_train': X_train, 'y_train': y_train}
    return test_sets
