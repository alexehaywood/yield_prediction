#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import os
from collections import defaultdict
import numpy as np

os.sys.path.append(os.getcwd())

from generate_descriptors import \
    assemble_graph_descriptors, \
    assemble_fingerprint_descriptors, \
    assemble_one_hot_encondings, \
    assemble_quantum_descriptors
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
    
    reactions_org = pd.read_excel(
        './data/original/reactions/rxns_subset_missing_additive.xlsx'
        )
    reactions_val = pd.read_excel(
        './data/validation/reactions/rxns_all.xlsx'
        )
    
    rxn_smiles_org = pd.read_excel(
        './data/original/reactions/rxns_missing_additive_smi.xlsx',
        ).set_index(['additive', 'aryl_halide', 'base', 'ligand', 'yield_exp'])
    rxn_smiles_val = pd.read_excel(
        './data/validation/reactions/rxns_smi.xlsx',
        ).set_index(['additive', 'aryl_halide', 'base', 'ligand'])
    
    subset_mols_to_be_removed = {
        'aryl_halide': [
            'OCc1ccccc1I', 'Ic1ccc(cc1)C#N', 'CC(C)c1ccccc1I', 
            'Fc1cccc(I)c1C#N', 'Ic1cccc(CC#N)c1', 'CC(=O)c1cncc(Br)c1', 
            'FC(F)(F)c1ccc(I)cc1', 'CC1=NN=C(I)C=C1', 'CCC1=CC=CC(I)=C1'
            ],
        'ligand': ['BrettPhos']
        }
    
    reactions_org_subset = reactions_org.copy()
    rxn_smiles_org_subset = rxn_smiles_org.copy()
    
    reactions_org_subset = reactions_org_subset[
        ~reactions_org_subset['aryl_halide'].str.contains('iodo')
        ]
    rxn_smiles_org_subset = rxn_smiles_org_subset[
        ~rxn_smiles_org_subset.index.get_level_values(
            'aryl_halide').str.contains('iodo')
        ]
    
    reactions_val_subset = reactions_val.copy()
    rxn_smiles_val_subset = rxn_smiles_val.copy()
    
    for rxn_component, mols in subset_mols_to_be_removed.items():
        reactions_val_subset = reactions_val_subset[
            ~reactions_val_subset[rxn_component].isin(mols)
            ]
        rxn_smiles_val_subset = rxn_smiles_val_subset[
            ~rxn_smiles_val_subset.index.isin(mols, level=rxn_component)
            ]      
   
    info = defaultdict()
    
    info['quantum'] = defaultdict()
    info['quantum']['dir'] = 'quantum_descriptors_missing_additive'
    info['quantum']['X_type'] = 'quantum'
    info['quantum']['model_names'] = [
        'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'Random Forest'
        ]
    info['quantum']['X_validation'] = defaultdict()
    info['quantum']['X_validation']['subset_mols_train'] \
        = assemble_quantum_descriptors(
            rxn_components, reactions_org, 
            'data/original/quantum_descriptors_missing_additive'
            )
    info['quantum']['X_validation']['subset_mols_test'] \
        = assemble_quantum_descriptors(
            rxn_components, reactions_val, 
            'data/validation/quantum_descriptors_missing_additive'
            )
    info['quantum']['X_validation']['subset_mols_train'] \
        = info['quantum']['X_validation']['subset_mols_train'][
            ~info['quantum']['X_validation']['subset_mols_train'].index.get_level_values(
                'aryl_halide').str.contains('iodo')
            ]
    
    info['one-hot'] = defaultdict()
    info['one-hot']['dir'] = 'one_hot_encodings'
    info['one-hot']['X_type'] = 'one-hot'
    info['one-hot']['model_names'] = [
        'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
        'SVR - Sigmoid Kernel', 'Random Forest'
        ]
    info['one-hot']['X_validation'] = defaultdict()
    info['one-hot']['X_validation']['all_mols_train'], \
        info['one-hot']['X_validation']['all_mols_test'] \
            = assemble_one_hot_encondings(
                rxn_components, 
                reactions_org,
                reactions_val
                )
    info['one-hot']['X_validation']['subset_mols_train'], \
        info['one-hot']['X_validation']['subset_mols_test'] \
            = assemble_one_hot_encondings(
                rxn_components, 
                reactions_org_subset,
                reactions_val_subset
                )
    
    info['graphs'] = defaultdict()
    info['graphs']['dir'] = 'graph_descriptors'
    info['graphs']['X_type'] = 'graphs'
    info['graphs']['model_names'] = ['SVR - Precomputed Kernel']
    
    info['graphs']['X_validation'] = defaultdict()        
    info['graphs']['X_validation']['all_mols_train'] \
        = assemble_graph_descriptors(
            rxn_components, reactions_org, rxn_smiles_org
            )
    info['graphs']['X_validation']['all_mols_test'] \
        = assemble_graph_descriptors(
        rxn_components, reactions_val, rxn_smiles_val
        )
    info['graphs']['X_validation']['subset_mols_train'] \
        = assemble_graph_descriptors(
            rxn_components, reactions_org_subset, rxn_smiles_org_subset
            )
    info['graphs']['X_validation']['subset_mols_test'] \
        = assemble_graph_descriptors(
            rxn_components, reactions_val_subset, rxn_smiles_val_subset
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
        
        info['{}_concat'.format(fp)] = defaultdict()
        info['{}_concat'.format(fp)]['dir'] = 'fp_descriptors/{}/concat'.format(fp)
        info['{}_concat'.format(fp)]['X_type'] = 'fps'
        info['{}_concat'.format(fp)]['model_names'] = [
            'SVR - Linear Kernel', 'SVR - Poly Kernel', 'SVR - RBF Kernel',
            'SVR - Sigmoid Kernel', 'Random Forest'
            ]
        info['{}_concat'.format(fp)]['X_validation'] = defaultdict()

        info['{}_concat'.format(fp)]['X_validation']['all_mols_train'] \
            = assemble_fingerprint_descriptors(
                rxn_components, reactions_org, rxn_smiles_org,
                fp_type, fps_kw, return_raw=False, return_concat=True, return_sum=False
                )
        info['{}_concat'.format(fp)]['X_validation']['all_mols_test'] \
            = assemble_fingerprint_descriptors(
                rxn_components, reactions_val, rxn_smiles_val,
                fp_type, fps_kw, return_raw=False, return_concat=True, return_sum=False
                )
        info['{}_concat'.format(fp)]['X_validation']['subset_mols_train'] \
            = assemble_fingerprint_descriptors(
                rxn_components,
                reactions_org_subset, rxn_smiles_org_subset,
                fp_type, fps_kw, return_raw=False, return_concat=True, return_sum=False
                )
        info['{}_concat'.format(fp)]['X_validation']['subset_mols_test'] \
            = assemble_fingerprint_descriptors(
                rxn_components,
                reactions_val_subset, rxn_smiles_val_subset,
                fp_type, fps_kw, return_raw=False, return_concat=True, return_sum=False
                )
                
    dir_setup(
        descriptor_names=[info[k]['dir'] for k in info.keys()],
        test_types=['validation'],
        test_names={
            'validation': ['all_mols', 'subset_mols']
            }
        )
    
    for d, info_d in info.items():
        if info_d['X_type'] == 'quantum':
            test_names = ['subset_mols']
        else:
            test_names = ['all_mols', 'subset_mols']
        
        # Validation Test
        test_type = 'validation'
        
        for test_name in test_names:
        
            X_train = info_d['X_validation']['{}_train'.format(test_name)]
            y_train = info_d['X_validation']['{}_train'.format(test_name)].reset_index('yield_exp').yield_exp
            X_test = info_d['X_validation']['{}_test'.format(test_name)]
            
            ml_tests.predict(
                X_train=X_train, 
                y_train=y_train,
                X_test=X_test,
                models={k: ml_tests.models[k] for k in info_d['model_names']},
                param_grid={k: ml_tests.param_grid[k] 
                            for k in info_d['model_names']},
                X_type=info_d['X_type'],
                saveas='./results/{}/{}/{}'.format(
                    info_d['dir'], test_type, test_name)
                )
            
if __name__ == '__main__':
    main()
