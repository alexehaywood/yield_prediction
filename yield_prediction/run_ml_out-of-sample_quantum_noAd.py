#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import os
from collections import defaultdict
import numpy as np

from generate_descriptors import assemble_graph_descriptors, assemble_fingerprint_descriptors, assemble_one_hot_encondings
import tools.data.rdkit_tools as rd
import tools.machine_learning.machine_learning_tests as ml_tests

class ranking():
        
    def rank_mols(mols, molecule_keys, rxn_component, yields):
        ranked_mols = []
        for mol in mols:
            ranked_mols.append({
                rxn_component: mol,
                'key': molecule_keys[rxn_component][
                    molecule_keys[rxn_component] == mol].index[0],
                'mean_yield': np.mean(
                    yields.loc[yields[rxn_component] == mol].yield_exp
                    )
                })
        
        ranked_mols = pd.DataFrame(ranked_mols)
        ranked_mols.sort_values(by=['mean_yield'], inplace=True)
        ranked_mols.reset_index(inplace=True, drop=True)
        ranked_mols.index = ranked_mols.index.set_names(['order'])
        
        return ranked_mols
        
    def make_test_sets(ranked_mols, n_sets=4):
        test_sets = defaultdict()
        n=1
        for test_n in np.arange(1,n_sets+1):
            test_sets[test_n] = ranked_mols.key[n:-1:n_sets].tolist()
            n = n+1
            
        return test_sets


def dir_setup(descriptor_names, test_types, test_names, dir_results='.'):
    if 'results' not in os.listdir(dir_results):
        os.mkdir('{}/results'.format(dir_results))
    cwd = '{}/results'.format(dir_results)
        
    for descriptor_name in descriptor_names:
        cwd1 = '{}/{}'.format(cwd, descriptor_name)
        if not os.path.exists(cwd1):
            os.makedirs('{}'.format(cwd1))
        
        for test_type in test_types:
            cwd2 = '{}/{}'.format(cwd1, test_type)
            if not os.path.exists(cwd2):
                os.makedirs('{}'.format(cwd2))
            
            if test_names[test_type] is not None:
                for test_name in test_names[test_type]:
                    cwd3 = '{}/{}'.format(cwd2, test_name)
                    if not os.path.exists(cwd3):
                        os.makedirs('{}'.format(cwd3))                        
    

def main():
    rxn_components = ['additive', 'aryl_halide', 'base', 'ligand']
    
    reactions = pd.read_excel(
        './data/original/reactions/rxns_subset_missing_additive.xlsx',
        )
    
    info = defaultdict()
    
    info['quantum_missing_add'] = defaultdict()
    info['quantum_missing_add']['dir'] = 'quantum_descriptors_missing_additive'
    info['quantum_missing_add']['X_type'] = 'quantum'
    info['quantum_missing_add']['model_names'] = [
        'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'Random Forest'
        ]
    info['quantum_missing_add']['X'] = pd.read_excel(
        'data/original/quantum_descriptors_missing_additive/quantum_descriptors.xlsx',
        index_col=[1,2,3,4]
        )
    info['quantum_missing_add']['X'] = info['quantum_missing_add'][
        'X'].set_index('additive', append=True)
    
    dir_setup(
        descriptor_names=[info[k]['dir'] for k in info.keys()],
        test_types=['out_of_sample'],
        test_names={
            'out_of_sample': ['aryl_halide'],
            }
        )
    
    # Load in molecule keys.
    molecule_keys = pd.read_excel(
        './data/original/molecule_keys.xlsx', 
        index_col=0,
        sheet_name=None,
        squeeze=True
        )

    ranked_mols = defaultdict()
    for rxn_component in rxn_components:
        if rxn_component == 'aryl_halide':
            ranked_mols[rxn_component] = ranking.rank_mols(
                mols=reactions[rxn_component].drop_duplicates(), 
                molecule_keys=molecule_keys, 
                rxn_component=rxn_component, 
                yields=reactions
                )
    
    for d, info_d in info.items():
        
        # Split into descriptors (X) and targets (y).
        X = info_d['X'].reset_index('yield_exp', drop=True)
        y = info_d['X'].reset_index('yield_exp').yield_exp
        
        # Missing additive test
            
        # Aryl Halide Tests.
        test_name = 'out_of_sample'
        rxn_component = 'aryl_halide'
        
        aryl_halide_ranking = ranking.make_test_sets(
            ranked_mols=ranked_mols[rxn_component],
            n_sets=3
            )
        
        mol_sets = []
        mol_sets.extend(aryl_halide_ranking.values())
        
        names = []
        names.extend(
            ['ranking_test{}'.format(i) for i in aryl_halide_ranking.keys()]
            )
    
        for mol_set, name in zip(mol_sets, names):
            # Molecules held out of training.
            molecule_test_list = [
                molecule_keys[rxn_component][k] for k in mol_set
                ]
        
            ml_tests.out_of_sample(
                X, y, 
                models={k: ml_tests.models[k] for k in info_d['model_names']},
                param_grid={k: ml_tests.param_grid[k] 
                            for k in info_d['model_names']},
                X_type=info_d['X_type'],
                molecule_test_list=molecule_test_list, 
                molecule_keys=molecule_keys,
                rxn_component=rxn_component, 
                saveas='./results/{}/{}/{}/{}'.format(
                    info_d['dir'], test_name, rxn_component, name)
                )
        
     
if __name__ == '__main__':
    main()