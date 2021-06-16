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
        './data/original/reactions/rxns_subset_no_nan_yields.xlsx',
        index_col=0
        )
    
    rxn_smiles = pd.read_excel(
        './data/original/reactions/rxns_smi.xlsx',
        index_col=[0, 1, 2, 3, 4]
        )
    
    info = defaultdict()
    
    info['quantum'] = defaultdict()
    info['quantum']['dir'] = 'quantum_descriptors'
    info['quantum']['X_type'] = 'quantum'
    info['quantum']['model_names'] = [
        'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'Random Forest',
        'Linear Regression', 'k-Nearest Neighbours', 
        'Bayes Generalised Linear Model',
        'Gradient Boosting', 'Decision Tree'
        ]
    info['quantum']['X'] = pd.read_excel(
        'data/original/quantum_descriptors/quantum_descriptors.xlsx',
        index_col=[0,1,2,3,4]
        )
    info['quantum']['kwargs'] = None
    
    info['one-hot'] = defaultdict()
    info['one-hot']['dir'] = 'one_hot_encodings'
    info['one-hot']['X_type'] = 'one-hot'
    info['one-hot']['model_names'] = [
        'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'Random Forest',
        'Linear Regression', 'k-Nearest Neighbours', 
        'Bayes Generalised Linear Model',
        'Gradient Boosting', 'Decision Tree'
        ]
    info['one-hot']['X'] = assemble_one_hot_encondings(
        rxn_components, reactions
        )
    info['one-hot']['kwargs'] = None

    info['graphs'] = defaultdict()
    info['graphs']['dir'] = 'graph_descriptors'
    info['graphs']['X_type'] = 'graphs'
    info['graphs']['model_names'] = [
        'SVR - Precomputed Kernel', 
        'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'Random Forest',
        'Linear Regression', 'k-Nearest Neighbours', 
        'Bayes Generalised Linear Model',
        'Gradient Boosting', 'Decision Tree']
    info['graphs']['X'] = assemble_graph_descriptors(
        rxn_components, reactions, rxn_smiles
        )
    graphs = assemble_graph_descriptors(
            rxn_components, reactions, rxn_smiles
            )
    for n in np.arange(2, 11):
        info['graphs_WL{}'.format(n)] = defaultdict()
        info['graphs_WL{}'.format(n)]['dir'] = 'graph_descriptors/WL{}'.format(n)
        info['graphs_WL{}'.format(n)]['X_type'] = 'graphs'
        info['graphs_WL{}'.format(n)]['model_names'] = [
            'SVR - Precomputed Kernel', 
            'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
            'SVR - Sigmoid Kernel', 'Random Forest',
            'Linear Regression', 'k-Nearest Neighbours', 
            'Bayes Generalised Linear Model',
            'Gradient Boosting', 'Decision Tree'
            ]
        info['graphs_WL{}'.format(n)]['X'] = graphs
        info['graphs_WL{}'.format(n)]['kwargs'] = {'niter': int(n)}
        
    for fp, fp_type, fps_kw in zip(
            [
                'FMorgan1_32', 'FMorgan1_64', 
                    'FMorgan1_128',  'FMorgan1_256', 
                    'FMorgan1_1024', 'FMorgan1_2048',
                'FMorgan2_32', 'FMorgan2_64', 
                    'FMorgan2_128',  'FMorgan2_256', 
                    'FMorgan2_1024', 'FMorgan2_2048',
                'FMorgan3_32', 'FMorgan3_64', 
                    'FMorgan3_128',  'FMorgan3_256', 
                    'FMorgan3_1024', 'FMorgan3_2048',
                'Morgan1_32', 'Morgan1_64', 
                    'Morgan1_128',  'Morgan1_256', 
                    'Morgan1_1024', 'Morgan1_2048',
                'Morgan2_32', 'Morgan2_64', 
                    'Morgan2_128',  'Morgan2_256', 
                    'Morgan2_1024', 'Morgan2_2048',
                'Morgan3_32', 'Morgan3_64', 
                    'Morgan3_128',  'Morgan3_256', 
                    'Morgan3_1024', 'Morgan3_2048',
                'MACCS', 
                'RDK_32', 'RDK_64',
                    'RDK_128',  'RDK_256',
                    'RDK_1024', 'RDK_2048'
                ],
            [
                rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, 
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, 
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, 
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, 
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, 
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, 
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                    rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
                rd.GenMACCSKeys, 
                rd.RDKFingerprint, rd.RDKFingerprint,
                    rd.RDKFingerprint, rd.RDKFingerprint,
                    rd.RDKFingerprint, rd.RDKFingerprint
              ],
            [
                {'radius':1, 'useFeatures':True, 'nBits':32}, {'radius':1, 'useFeatures':True, 'nBits':64},
                    {'radius':1, 'useFeatures':True, 'nBits':128},  {'radius':1, 'useFeatures':True, 'nBits':256},
                    {'radius':1, 'useFeatures':True, 'nBits':1024}, {'radius':1, 'useFeatures':True, 'nBits':2048},
                {'radius':2, 'useFeatures':True, 'nBits':32}, {'radius':2, 'useFeatures':True, 'nBits':64},
                    {'radius':2, 'useFeatures':True, 'nBits':128},  {'radius':2, 'useFeatures':True, 'nBits':256},
                    {'radius':2, 'useFeatures':True, 'nBits':1024}, {'radius':2, 'useFeatures':True, 'nBits':2048},
                {'radius':3, 'useFeatures':True, 'nBits':32}, {'radius':3, 'useFeatures':True, 'nBits':64},
                    {'radius':3, 'useFeatures':True, 'nBits':128},  {'radius':3, 'useFeatures':True, 'nBits':256},
                    {'radius':3, 'useFeatures':True, 'nBits':1024}, {'radius':3, 'useFeatures':True, 'nBits':2048},
                {'radius':1, 'nBits':32}, {'radius':1, 'nBits':64},
                    {'radius':1, 'nBits':128},  {'radius':1, 'nBits':256},
                    {'radius':1, 'nBits':1024}, {'radius':1, 'nBits':2048},
                {'radius':2, 'nBits':32}, {'radius':2, 'nBits':64},
                    {'radius':2, 'nBits':128},  {'radius':2, 'nBits':256},
                    {'radius':2, 'nBits':1024}, {'radius':2, 'nBits':2048},
                {'radius':3, 'nBits':32}, {'radius':3, 'nBits':64},
                    {'radius':3, 'nBits':128},  {'radius':3, 'nBits':256},
                    {'radius':3, 'nBits':1024}, {'radius':3, 'nBits':2048},
                {}, 
                {'fpSize':32}, {'fpSize':64},
                    {'fpSize':128}, {'fpSize':256},
                    {'fpSize':1024}, {'fpSize':2048}
              ]
            ):
        
        info['{}_raw'.format(fp)] = defaultdict()
        info['{}_raw'.format(fp)]['dir'] = 'fp_descriptors/{}/raw'.format(fp)
        info['{}_raw'.format(fp)]['X_type'] = 'fps'
        info['{}_raw'.format(fp)]['model_names'] = ['SVR - Precomputed Kernel']
        info['{}_raw'.format(fp)]['kwargs'] = None
        
        for i in ['concat', ]:#'sum']:
            info['{}_{}'.format(fp, i)] = defaultdict()
            info['{}_{}'.format(fp, i)]['dir'] = 'fp_descriptors/{}/{}'.format(fp,i)
            info['{}_{}'.format(fp, i)]['X_type'] = 'fps'
            info['{}_{}'.format(fp, i)]['model_names'] = [
            'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
            'SVR - Sigmoid Kernel', 'Random Forest',
            'Linear Regression', 'k-Nearest Neighbours', 
            'Bayes Generalised Linear Model',
            'Gradient Boosting', 'Decision Tree'
            ]
            info['{}_{}'.format(fp, i)]['kwargs'] = None
    
        info['{}_raw'.format(fp)]['X'], \
            info['{}_concat'.format(fp)]['X'] = \
                    assemble_fingerprint_descriptors(
                        rxn_components, reactions, rxn_smiles, fp_type, fps_kw,
                        return_raw=True,  return_concat=True, return_sum=False
                        )

    dir_setup(
        descriptor_names=[info[k]['dir'] for k in info.keys()],
        test_types=['cv'],
        test_names={
            'cv': None,
            }
        )
    
    for d, info_d in info.items():
        
        # Split into descriptors (X) and targets (y).
        X = info_d['X'].reset_index('yield_exp', drop=True)
        y = info_d['X'].reset_index('yield_exp').yield_exp
        
        # IN-SAMPLE
        test_name='cv'
        ml_tests.predict(
            X_train=X, 
            y_train=y, 
            X_test=None,
            models={k: ml_tests.models[k] for k in info_d['model_names']},
            param_grid={k: ml_tests.param_grid[k] 
                            for k in info_d['model_names']},
            X_type=info_d['X_type'],
            saveas='./results/{}/{}'.format(
                    info_d['dir'], test_name),
            save_table=True, 
            save_model=False,
            kwargs=info_d['kwargs']
            )
            
if __name__ == '__main__':
    main()
