#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather reactions and corresponding yield data. Generate quantum-chemical and
graph-based descriptors.

@author: pcxah5
"""

from collections import defaultdict
import pandas as pd
import os
from itertools import  product

from tools.data.plate import Plate
import tools.data.load_plate_from_excel as load_plate_from_excel


def merge_yields(directory, plate_name):
    yield_1 = pd.read_csv(directory + '/yields/{}.1.csv'.format(plate_name))
    yield_2 = pd.read_csv(directory + '/yields/{}.2.csv'.format(plate_name))
    yield_3 = pd.read_csv(directory + '/yields/{}.3.csv'.format(plate_name))
    yield_4 = pd.read_csv(directory + '/yields/{}.4.csv'.format(plate_name))
    yield_all = pd.concat([yield_1, yield_2, yield_3, yield_4], 
                           axis=0, 
                           join='outer', 
                           ignore_index=True, 
                           sort=False)
    return yield_all
            

def assemble_reactions_and_yields(dir_data='.'):
    if 'reactions' not in os.listdir(dir_data):
        os.mkdir(dir_data + 'reactions')
        
    plate_names = ['plate1', 'plate2', 'plate3']
    
    # Define molecule keys.
    key = load_plate_from_excel.define_molecule_keys(
        dir_data, 
        'molecule_keys.xlsx', 
        sheet_name=None, 
        index_col=0
        )

    # Load in plate rows, columns and codes.
    plate_setup = load_plate_from_excel.setup_plate(
        plate_names, 
        dir_data+'plates_setup.xlsx', 
        dir_data+'plates_codes.xlsx')
    
    # Load in plate yields.
    yields = defaultdict()
    for plate_name in plate_names:
        yields[plate_name] = merge_yields(dir_data, plate_name)
        
        yields[plate_name] = pd.concat(
            [yields[plate_name], plate_setup[plate_name]['ID']],
            axis=1
            )
        
    # Define the dimensions of each plate.
    plates = defaultdict()
    for plate_name in plate_names:
        plates[plate_name] = Plate(32, 48)
    
    # Fill plates. 
    for plate_name, plate in plates.items():
        plate.fillPlate(
            rows=plate_setup[plate_name]['rows'],
            columns=plate_setup[plate_name]['columns'],
            key=key,
            codes=plate_setup[plate_name]['codes'])
    
        yields[plate_name] = yields[plate_name][
                yields[plate_name].ID.isin(plate.matrix_code.values())
                ]
        
    # Combine all plate data together.
    all_rxns = []
    for plate_name, plate in plates.items():
        for k, df in yields[plate_name][
                ['product_scaled', 'ID', 'corr_factor', 'internal_standard',
                 'product'
                 ]].iterrows():
            info = {'ID': df.ID,
                    **plate.getConditions(df.ID),
                    'yield_exp': df.product_scaled,
                    'corr_factor': df.corr_factor,
                    'area_abs_std': df.internal_standard,
                    'area_abs_prod': df.product,
                    }
            
            all_rxns.append(info)
            
    # Save all reactions.
    all_rxns = pd.DataFrame.from_records(all_rxns)
    all_rxns.to_excel(dir_data + 'reactions/rxns_all.xlsx', index=False)
    
    # Remove 'nan' yields.
    all_rxns_no_nan_yields = all_rxns[all_rxns.yield_exp != '#VALUE!']
    all_rxns_no_nan_yields = all_rxns_no_nan_yields.dropna(
        subset=['yield_exp']
        )
    all_rxns_no_nan_yields = all_rxns_no_nan_yields[
        ['additive', 'aryl_halide', 'base', 'ligand', 'yield_exp']
        ]
    all_rxns_no_nan_yields = all_rxns_no_nan_yields.reindex(
        sorted(all_rxns_no_nan_yields.columns), 
        axis=1
        )
    all_rxns_no_nan_yields = all_rxns_no_nan_yields.sort_values(
        sorted(all_rxns_no_nan_yields.columns)
        )
    all_rxns_no_nan_yields.to_excel(
        dir_data + 'reactions/rxns_all_no_nan_yields.xlsx', 
        index=False
        )
    
    # Remove reactions with no aryl halides and additive 7.
    subset_rxns_missing_additive = all_rxns_no_nan_yields.dropna(
        subset=['aryl_halide']
        )
    subset_rxns_missing_additive = subset_rxns_missing_additive[
        subset_rxns_missing_additive.additive != key['additive'][7]
        ]
    subset_rxns_missing_additive.to_excel(
        dir_data + 'reactions/rxns_subset_missing_additive.xlsx', 
        index=False
        )
    
    # Remove reactions with no aryl halides, no additives and additive 7.
    subset_rxns = all_rxns.dropna(subset=['aryl_halide', 'additive'])
    subset_rxns = subset_rxns[subset_rxns.additive != key['additive'][7]]
    
    # Save subset of reactions with 'nan' yields.
    subset_rxns = subset_rxns[
        ['additive', 'aryl_halide', 'base', 'ligand', 'yield_exp']
        ]
    subset_rxns.reset_index(drop=True, inplace=True) 
    subset_rxns.to_excel(dir_data + 'reactions/rxns_subset_nan_yields.xlsx')
    
    # Remove 'nan' yields.
    subset_rxns_no_nan_yields = subset_rxns[subset_rxns.yield_exp != '#VALUE!']
    subset_rxns_no_nan_yields = subset_rxns_no_nan_yields.dropna(
        subset=['yield_exp']
        )
    
    # Save subset of reactions without 'nan' yields.
    subset_rxns_no_nan_yields.reset_index(drop=True, inplace=True)
    subset_rxns_no_nan_yields.to_excel(
        dir_data + 'reactions/rxns_subset_no_nan_yields.xlsx')

def assemble_reactions(tuple_of_lists, column_headers, saveas_dir=None):
    rxns = pd.DataFrame(list(product(*tuple_of_lists)), columns=column_headers)
    rxns = rxns.reindex(sorted(rxns.columns), axis=1)
    rxns = rxns.sort_values(sorted(rxns.columns))
    
    if saveas_dir is not None:
        if not os.path.exists(saveas_dir):
            os.makedirs(saveas_dir)
        rxns.to_excel('{}/rxns_all.xlsx'.format(saveas_dir), index=False)
    
    return rxns

    
if __name__ == '__main__':
    dir_data = 'C:/Users/alexe/OneDrive/Documents/PhD/Year 3/Work/yield_prediction/data/original/'
    assemble_reactions_and_yields(dir_data)
    validation_molecule_keys = pd.read_excel(
        'data/validation/molecule_keys.xlsx',
        sheet_name=None
        )
    validation_rxns = assemble_reactions(
        tuple_of_lists=(col['name'].to_list() 
                        for col in validation_molecule_keys.values()),
        column_headers=[k for k in validation_molecule_keys.keys()],
        saveas_dir='data/validation/reactions'
        )
