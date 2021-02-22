# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:30:57 2020

@author: alexe
"""
import pandas as pd
from collections import defaultdict
import numpy as np
import os
from itertools import groupby

def group_list(test_list): 
    # Sort list.
    test_list.sort() 
    
    # Group similar substrings.
    grouped_dict = {j: list(i) for j, i in groupby(test_list, 
                      lambda a: a.split('_')[0])}
    
    return grouped_dict

def get_results(descriptor_names, test_types, test_names, sheet_name='scores',
                index_col=0, dir_results='.'):
    results = defaultdict()
    for test_type in test_types:
        results[test_type] = defaultdict()
        
        if test_names[test_type] is not None:
            for test_name in test_names[test_type]:
                results[test_type][test_name] = []
                
                for descriptor_name in descriptor_names:
                    
                    for test_set in os.listdir('{}/results/{}/{}/{}'.format(
                        dir_results, descriptor_name, test_type, test_name
                        )):
                        if os.path.isdir('{}/results/{}/{}/{}/{}'.format(
                            dir_results, descriptor_name, test_type, test_name,
                            test_set
                            )):
                        
                            info = defaultdict()
                            if '/' in descriptor_name:
                                info.update({'Descriptor': descriptor_name.split('/')[1]})
                            else:
                                info.update({'Descriptor': descriptor_name})
                            test = ''.join(
                                [j for j, i in groupby([test_set], lambda a: a.split('_')[0])]
                                )
                            info.update({
                                'Test': test,
                                'Test Set': test_set,
                                })
                            df = pd.read_excel(
                                '{}/results/{}/{}/{}/{}/results.xlsx'.format(
                                    dir_results, descriptor_name, 
                                    test_type, test_name, test_set), 
                                sheet_name=sheet_name, 
                                index_col=index_col
                                )
                            if not df.index.name:
                                df.index.name = sheet_name
                            names = list(info.keys())
                            df = pd.concat(
                                [df], 
                                keys=[tuple(info.values())], 
                                names=names
                                )
                            results[test_type][test_name].append(df)
                            
                            if 'yield_exp' in df.columns:
                                df.drop(columns='yield_exp', inplace=True)
                            
                            df.columns = df.columns.str.lstrip('yield_pred')
                            
                            if descriptor_name == 'graph_descriptors':
                                df.rename(
                                    columns={'SVR - Precomputed Kernel': 'SVR - WL Kernel'}, 
                                    inplace=True
                                    )
                            elif 'SVR - Precomputed Kernel' in df.columns:
                                df.rename(
                                    columns={'SVR - Precomputed Kernel': 'SVR - Tanimoto Kernel'}, 
                                    inplace=True
                                    )
                                
                        elif test_set=='results.xlsx':
                            
                            info = defaultdict()
                            if '/' in descriptor_name:
                                info.update({'Descriptor': descriptor_name.split('/')[1]})
                            else:
                                info.update({'Descriptor': descriptor_name})
                            df = pd.read_excel(
                                '{}/results/{}/{}/{}/results.xlsx'.format(
                                    dir_results, descriptor_name, 
                                    test_type, test_name), 
                                sheet_name=sheet_name, 
                                # index_col=index_col
                                )
                            df.replace(np.nan, 'none', inplace=True)
                            df = df.set_index(['additive', 'aryl_halide', 'base', 'ligand'])
                            if not df.index.name:
                                df.index.name = sheet_name
                            names = list(info.keys())
                            df = pd.concat(
                                [df], 
                                keys=[tuple(info.values())], 
                                names=names
                                )
                            results[test_type][test_name].append(df)
                            
                            if 'yield_exp' in df.columns:
                                df.drop(columns='yield_exp', inplace=True)
                            
                            df.columns = df.columns.str.lstrip('yield_pred')
                            
                            if descriptor_name == 'graph_descriptors':
                                df.rename(
                                    columns={'SVR - Precomputed Kernel': 'SVR - WL Kernel'}, 
                                    inplace=True
                                    )
                            elif 'SVR - Precomputed Kernel' in df.columns:
                                df.rename(
                                    columns={'SVR - Precomputed Kernel': 'SVR - Tanimoto Kernel'}, 
                                    inplace=True
                                    )
                            
                if results[test_type][test_name]:
                    results[test_type][test_name] = pd.concat(
                        results[test_type][test_name]
                        )
                    
                    if any(results[test_type][test_name].index.duplicated()):
                        results[test_type][test_name] = \
                            results[test_type][test_name].sum(
                                axis='index',
                                level=results[test_type][test_name].index.names,
                                min_count=1
                                )
            # else:
            #     info = {
            #         'Descriptor': descriptor_name,
            #         'Test Type': test_type,
            #         }
            #     df = pd.read_excel(
            #         '{}/results/{}/{}/results.xlsx'.format(
            #             dir_results, descriptor_name, test_type), 
            #         sheet_name=sheet_name, 
            #         index_col=index_col
            #         ).T
            #     names = list(info.keys())
            #     names.append('Model')
            #     df = pd.concat(
            #         [df], 
            #         keys=[tuple(info.values())], 
            #         names=names
            #         )
            #     results.append(df)
                
    return results

def get_fp_bit_length_results(scores_mean, metric):
    fps_bit_length = scores_mean.loc[
        (scores_mean.index.isin(['ranking'], level='Test')) 
        & 
        (scores_mean.index.get_level_values(
            'Descriptor').str.contains('Morgan|RDK'))
        ][metric]['Mean']

    fps_bit_length.index = fps_bit_length.index.droplevel('Test')
    # fps_bit_length.columns = fps_bit_length.columns.droplevel(0)
    fps_bit_length = pd.DataFrame(fps_bit_length)
    
    fps_bit_length['Descriptor'] = fps_bit_length.index.get_level_values(
        'Descriptor').str.split('_').str[0].values
    fps_bit_length['Bit Length'] = pd.to_numeric(
        fps_bit_length.index.get_level_values(
            'Descriptor').str.split('_').str[1].values
        )
        
    fps_bit_length.index = fps_bit_length.index.droplevel('Descriptor')
    fps_bit_length = fps_bit_length.reset_index().set_index(
        ['Descriptor', 'Bit Length']
        )
    
    fps_bit_length = fps_bit_length.pivot(columns='Model')
    
    fps_bit_length.columns = fps_bit_length.columns.droplevel(0)
    
    fps_bit_length['Mean'] = fps_bit_length.mean(axis='columns')
    
    return fps_bit_length
                        
fps =  [
        'Morgan1_32', 'Morgan1_64', 'Morgan1_128', 'Morgan1_256', 
            'Morgan1_512', 'Morgan1_1024', 'Morgan1_2048',
        'Morgan2_32', 'Morgan2_64', 'Morgan2_128', 'Morgan2_256', 
            'Morgan2_512', 'Morgan2_1024', 'Morgan2_2048',
        'Morgan3_32', 'Morgan3_64', 'Morgan3_128', 'Morgan3_256', 
            'Morgan3_512', 'Morgan3_1024', 'Morgan3_2048',
        'FMorgan1_32', 'FMorgan1_64', 'FMorgan1_128', 'FMorgan1_256',  
            'FMorgan1_512', 'FMorgan1_1024', 'FMorgan1_2048',
        'FMorgan2_32', 'FMorgan2_64', 'FMorgan2_128', 'FMorgan2_256', 
            'FMorgan2_512', 'FMorgan2_1024', 'FMorgan2_2048',
        'FMorgan3_32', 'FMorgan3_64', 'FMorgan3_128', 'FMorgan3_256', 
            'FMorgan3_512', 'FMorgan3_1024', 'FMorgan3_2048',
        'RDK_32', 'RDK_64', 'RDK_128', 'RDK_256', 
            'RDK_512', 'RDK_1024', 'RDK_2048',
        'MACCS', 
        ]
                        
dirs = defaultdict()
dirs['quantum'] = 'quantum_descriptors'
dirs['quantum_noI'] = 'quantum_descriptors_noI'
dirs['one-hot'] = 'one_hot_encodings'
dirs['graphs'] = 'graph_descriptors'
for fp in fps:
    dirs['{}_raw'.format(fp)] = 'fp_descriptors/{}/raw'.format(fp)
    dirs['{}_concat'.format(fp)] = 'fp_descriptors/{}/concat'.format(fp)

descriptor_names=[dirs[k] for k in dirs.keys()]
test_types=['out_of_sample']
test_names={
    'out_of_sample': ['additive', 'aryl_halide', 'base', 'ligand'],
    }

scores = get_results(
    descriptor_names=[dirs[k] for k in dirs.keys()],
    test_types=['out_of_sample'],
    test_names={
        'out_of_sample': ['additive', 'aryl_halide', 'base', 'ligand'],
        }
    )

for test_type in scores.keys():
    if test_type == 'out_of_sample':
        scores_mean = defaultdict()
        for test_name in scores[test_type].keys():
            if isinstance(scores[test_type][test_name], pd.DataFrame):
                scores_mean[test_name] = pd.DataFrame()
                
                scores_test_set = scores[test_type][test_name].unstack('Test Set').stack(0)
                
                scores_mean[test_name]['Mean'] = scores_test_set.mean(1)
                scores_mean[test_name]['Std'] = scores_test_set.std(1)
                
                scores_mean[test_name] = scores_mean[test_name].unstack(2)
                scores_mean[test_name] = scores_mean[test_name].reorder_levels(
                    [1,0], axis='columns'
                    ).sort_index(
                        axis='columns', level=[0,1]
                        )
                scores_mean[test_name].index = scores_mean[test_name].index.rename('Model', -1)
                scores_mean[test_name] = scores_mean[test_name].reset_index(
                    ).set_index(['Test', 'Descriptor', 'Model']
                                ).sort_values(['Test', 'Descriptor', 'Model'])
                
                scores[test_type][test_name] = scores[test_type][test_name].unstack().stack(0)
                scores[test_type][test_name].index = scores[test_type][test_name].index.rename('Model', -1)
                scores[test_type][test_name] = scores[test_type][test_name].reset_index(
                    ).set_index(['Test', 'Descriptor', 'Model', 'Test Set']
                                ).sort_values(['Test', 'Descriptor', 'Model', 'Test Set'])

        
        writer = pd.ExcelWriter('results/out_of_sample_results.xlsx')
        for test_name, results in scores[test_type].items():
            results.to_excel(writer, sheet_name=test_name)
        for test_name, results in scores_mean.items():
            results.to_excel(writer, sheet_name='{}_mean'.format(test_name))
        writer.save()


fps_bit_length_results = defaultdict()
fps_bit_length_results['Additive Mean R2'] = get_fp_bit_length_results(
    scores_mean['additive'], 
    'R-squared'
    )
fps_bit_length_results['Additive Mean RMSE'] = get_fp_bit_length_results(
    scores_mean['additive'], 
    'RMSE'
    )
fps_bit_length_results['Aryl Halide Mean R2'] = get_fp_bit_length_results(
    scores_mean['aryl_halide'], 
    'R-squared'
    )
fps_bit_length_results['Aryl Halide Mean RMSE'] = get_fp_bit_length_results(
    scores_mean['aryl_halide'], 
    'RMSE'
    )

writer = pd.ExcelWriter('results/fp_bit_length_results.xlsx', engine='xlsxwriter')
for name, results in fps_bit_length_results.items():
    results.to_excel(writer, sheet_name=name)
writer.save()
 

descriptor_names=[
    'quantum_descriptors', 
    'fp_descriptors/Morgan1_1024/raw', 'fp_descriptors/Morgan1_1024/concat',
    'fp_descriptors/Morgan2_1024/raw', 'fp_descriptors/Morgan2_1024/concat',
    'fp_descriptors/Morgan3_1024/raw', 'fp_descriptors/Morgan3_1024/concat',
    'fp_descriptors/MACCS/raw', 'fp_descriptors/MACCS/concat',
    'fp_descriptors/RDK_1024/raw', 'fp_descriptors/RDK_1024/concat',
    'graph_descriptors', 'one_hot_encodings'
    ]

test_types=['out_of_sample']
test_names={
    'out_of_sample': ['additive', 'aryl_halide', 'base', 'ligand'],
    }
test_names={
        'out_of_sample': ['additive', 'aryl_halide'],
        }

y_pred = get_results(
        descriptor_names=descriptor_names,
        test_types=['out_of_sample'],
        test_names={
            'out_of_sample': ['additive', 'aryl_halide', 'base', 'ligand'],
            },
        sheet_name='y_pred',
        index_col=[0,1,2,3]
       )

for test_type in y_pred.keys():
    if test_type == 'out_of_sample':
        y_pred_to_save = defaultdict(dict)

        for test_name in y_pred[test_type].keys():
            descriptors = y_pred[test_type][test_name].index.get_level_values(
                'Descriptor').drop_duplicates()
            
            for descriptor in descriptors:
                if 'ranking' in y_pred[test_type][test_name].index.get_level_values(
                        'Test'):
                    y_pred_to_save[test_name][descriptor] = y_pred[test_type][test_name][
                        (y_pred[test_type][test_name].index.get_level_values(
                            'Descriptor') == descriptor)
                        &
                        (y_pred[test_type][test_name].index.get_level_values(
                            'Test') == 'ranking')
                        ].dropna(axis='columns', how='all')
                elif 'LOO' in y_pred[test_type][test_name].index.get_level_values(
                        'Test'):
                    y_pred_to_save[test_name][descriptor] = y_pred[test_type][test_name][
                        (y_pred[test_type][test_name].index.get_level_values(
                            'Descriptor') == descriptor)
                        &
                        (y_pred[test_type][test_name].index.get_level_values(
                            'Test') == 'LOO')
                        ].dropna(axis='columns', how='all')
                
                y_pred_to_save[test_name][descriptor].index = \
                    y_pred_to_save[test_name][descriptor].index.droplevel(
                        ['Descriptor', 'Test'])                

for name, fps in y_pred_to_save.items():
    writer = pd.ExcelWriter('results/main_text_{}_y_pred.xlsx'.format(name),)# engine='xlsxwriter')
    for k, df in fps.items():
        df.to_excel(writer, sheet_name='{}'.format(k))
    writer.save()
           
     

for test_type in scores.keys():
    if test_type == 'out_of_sample':
        descriptors = []
        for descriptor in descriptor_names:
            if '/' in descriptor:
                descriptors.append(descriptor.split('/')[1])
            else:
                descriptors.append(descriptor)
        
        scores_subset = defaultdict()
        for test_name in test_names[test_type]:
            
            scores_subset[test_name] = pd.concat([
                 scores[test_type][test_name][
                    (scores[test_type][test_name].index.isin(
                        descriptors, level='Descriptor'))
                    &
                    (scores[test_type][test_name].index.get_level_values(
                        'Test') == 'ranking')
                    ].unstack('Test Set'),
                scores_mean[test_name][
                    (scores_mean[test_name].index.isin(
                        descriptors, level='Descriptor'))
                    &
                    (scores_mean[test_name].index.get_level_values(
                        'Test') == 'ranking')
                    ]
                ], axis=1)
            scores_subset[test_name] = scores_subset[test_name].sort_values(
                ['scores'], 
                axis='columns'
                )            
    
        writer = pd.ExcelWriter('results/main_text_scores.xlsx')
        for test_name, results in scores_subset.items():
            results.to_excel(writer, sheet_name=test_name)
        writer.save()

                
descriptor_names=[
    'quantum_descriptors_missing_additive', 
    'fp_descriptors/Morgan1_1024/raw', 'fp_descriptors/Morgan1_1024/concat',
    'graph_descriptors', 'one_hot_encodings'
    ]

test_types=['validation']
test_names={
    'validation': ['subset_mols'],
    }

y_pred_prospective = get_results(
        descriptor_names=descriptor_names,
        test_types=test_types,
        test_names=test_names,
        sheet_name='y_pred',
        index_col=[0,1,2,3]
       )

y_pred_to_save = defaultdict(dict)

for test_type in test_types:
    for test_name in y_pred_prospective[test_type].keys():
        descriptors = y_pred_prospective[test_type][test_name].index.get_level_values(
            'Descriptor').drop_duplicates()
        
        for descriptor in descriptors:
            descriptor_name = '_'.join(descriptor.split('_')[0:2])
            y_pred_to_save[test_name][descriptor_name] = y_pred_prospective[test_type][test_name][
                (y_pred_prospective[test_type][test_name].index.get_level_values(
                    'Descriptor') == descriptor)
                ].dropna(axis='columns', how='all')
            
            y_pred_to_save[test_name][descriptor_name].index = \
                y_pred_to_save[test_name][descriptor_name].index.droplevel(
                    ['Descriptor'])

        writer = pd.ExcelWriter('results/prospective_ypred_subsetmols.xlsx')
        for descriptor, results in y_pred_to_save[test_name].items():
            results.to_excel(writer, sheet_name=descriptor, merge_cells=False)
        writer.save()

descriptor_names=[
    'fp_descriptors/Morgan1_1024/raw', 'fp_descriptors/Morgan1_1024/concat',
    'graph_descriptors', 'one_hot_encodings'
    ]
test_types=['validation']
test_names={
    'validation': ['all_mols'],
    }

y_pred_prospective = get_results(
        descriptor_names=descriptor_names,
        test_types=test_types,
        test_names=test_names,
        sheet_name='y_pred',
        index_col=[0,1,2,3]
       )

y_pred_to_save = defaultdict(dict)

for test_type in test_types:
    for test_name in y_pred_prospective[test_type].keys():
        descriptors = y_pred_prospective[test_type][test_name].index.get_level_values(
            'Descriptor').drop_duplicates()
        
        for descriptor in descriptors:
            descriptor_name = '_'.join(descriptor.split('_')[0:2])
            y_pred_to_save[test_name][descriptor_name] = y_pred_prospective[test_type][test_name][
                (y_pred_prospective[test_type][test_name].index.get_level_values(
                    'Descriptor') == descriptor)
                ].dropna(axis='columns', how='all')
            
            y_pred_to_save[test_name][descriptor_name].index = \
                y_pred_to_save[test_name][descriptor_name].index.droplevel(
                    ['Descriptor'])
        
        writer = pd.ExcelWriter('results/prospective_ypred_allmols.xlsx')
        for descriptor, results in y_pred_to_save[test_name].items():
            results.to_excel(writer, sheet_name=descriptor, merge_cells=False)
        writer.save()