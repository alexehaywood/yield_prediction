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
        'SVR - Sigmoid Kernel', 'Random Forest'
        ]
    info['quantum']['X'] = pd.read_excel(
        'data/original/quantum_descriptors/quantum_descriptors.xlsx',
        index_col=[0,1,2,3,4]
        )
    
    info['one-hot'] = defaultdict()
    info['one-hot']['dir'] = 'one_hot_encodings'
    info['one-hot']['X_type'] = 'one-hot'
    info['one-hot']['model_names'] = [
        'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'Random Forest'
        ]
    info['one-hot']['X'] = assemble_one_hot_encondings(
        rxn_components, reactions
        )

    info['graphs'] = defaultdict()
    info['graphs']['dir'] = 'graph_descriptors'
    info['graphs']['X_type'] = 'graphs'
    info['graphs']['model_names'] = ['SVR - Precomputed Kernel']
    info['graphs']['X'] = assemble_graph_descriptors(
        rxn_components, reactions, rxn_smiles
        )

    for fp, fp_type, fps_kw in zip(
            [
             'FMorgan1_1024', 'FMorgan2_1024', 'FMorgan3_1024',
             'FMorgan1_2048', 'FMorgan2_2048', 'FMorgan3_2048',
             'Morgan1_1024', 'Morgan2_1024', 'Morgan3_1024',
             'Morgan1_2048', 'Morgan2_2048', 'Morgan3_2048',
             'MACCS', 
             'RDK_1024', 'RDK_2048'
             ],
            [
             rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
             rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
             rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
             rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect, rd.GetMorganFingerprintAsBitVect,
             rd.GenMACCSKeys, 
             rd.RDKFingerprint, rd.RDKFingerprint
             ],
            [
             {'radius':1, 'useFeatures':True, 'nBits':1024}, {'radius':2, 'useFeatures':True, 'nBits':1024}, {'radius':3, 'useFeatures':True, 'nBits':1024},
             {'radius':1, 'useFeatures':True, 'nBits':2048}, {'radius':2, 'useFeatures':True, 'nBits':2048}, {'radius':3, 'useFeatures':True, 'nBits':2048},
             {'radius':1, 'nBits':1024}, {'radius':2, 'nBits':1024}, {'radius':3, 'nBits':1024},
             {'radius':1, 'nBits':2048}, {'radius':2, 'nBits':2048}, {'radius':3, 'nBits':2048},
             {}, 
             {'fpSize':1024}, {'fpSize':2048}
             ]
            ):
        
        info['{}_raw'.format(fp)] = defaultdict()
        info['{}_raw'.format(fp)]['dir'] = 'fp_descriptors/{}/raw'.format(fp)
        info['{}_raw'.format(fp)]['X_type'] = 'fps'
        info['{}_raw'.format(fp)]['model_names'] = ['SVR - Precomputed Kernel']
        
        for i in ['concat', 'sum']:
            info['{}_{}'.format(fp, i)] = defaultdict()
            info['{}_{}'.format(fp, i)]['dir'] = 'fp_descriptors/{}/{}'.format(fp,i)
            info['{}_{}'.format(fp, i)]['X_type'] = 'fps'
            info['{}_{}'.format(fp, i)]['model_names'] = [
            'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
            'SVR - Sigmoid Kernel', 'Random Forest'
            ]
    
        info['{}_raw'.format(fp)]['X'], \
            info['{}_concat'.format(fp)]['X'], \
                info['{}_sum'.format(fp)]['X'] = \
                    assemble_fingerprint_descriptors(
                        rxn_components, reactions, rxn_smiles, fp_type, fps_kw
                        )

    dir_setup(
        descriptor_names=[info[k]['dir'] for k in info.keys()],
        test_types=['in_sample', 'out_of_sample', 'validation'],
        test_names={
            'out_of_sample': ['additive', 'aryl_halide', 'base', 'ligand'],
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

        # Additive Tests.
        test_name = 'out_of_sample'
        rxn_component = 'additive'
        
        mols_plate3 = [16, 17, 18, 19, 20, 21, 22, 23]
        mols_plate2 = [8, 9, 10, 11, 12, 13, 14, 15]
        mols_plate1 = [1, 2, 3, 4, 5, 6]
        
        additive_ranking = ranking.make_test_sets(
            ranked_mols=ranked_mols[rxn_component]
            )
        
        mol_sets = [mols_plate1, mols_plate2, mols_plate3]
        mol_sets.extend(additive_ranking.values())
        
        names = ['plate_1', 'plate_2', 'plate_3']
        names.extend(
            ['ranking_test{}'.format(i) for i in additive_ranking.keys()]
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
            
        # Aryl Halide Tests.
        test_name = 'out_of_sample'
        rxn_component = 'aryl_halide'
        
        mols_Cl = [1, 4, 7, 10, 13]
        mols_Br = [2, 5, 8, 11, 14]
        mols_I = [3, 6, 9, 12, 15]
        
        mols_phenyl = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        mols_pyridyl = [10, 11, 12, 13, 14, 15]
        
        aryl_halide_ranking = ranking.make_test_sets(
            ranked_mols=ranked_mols[rxn_component],
            n_sets=3
            )
        
        mol_sets = [mols_Cl, mols_Br, mols_I, mols_phenyl, mols_pyridyl]
        mol_sets.extend(aryl_halide_ranking.values())
        
        names = ['halide_test_Cl', 'halide_test_Br', 'halide_test_I', 
                 'aryl_test_phenyl', 'aryl_test_pyridyl']
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
        
        # Leave-one-out Tests.
        test_name = 'out_of_sample'
        
        for rxn_component in ['base', 'ligand']:
            mols = X.index.get_level_values(rxn_component).drop_duplicates()
            
            for mol in mols:
                molecule_test_list = [mol]
                
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
                        info_d['dir'], test_name, rxn_component, 'LOO_{}'.format(mol))
                    )
        
            
if __name__ == '__main__':
    main()
