#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather reactions and corresponding yield data. Generate quantum-chemical and
graph-based descriptors.

@author: pcxah5
"""

from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import scale
import numpy as np
import os 

os.sys.path.append(os.getcwd())

import tools.data.rdkit_tools as rd


def combine_descriptors_to_reactions(dict_df, rxns):
    """
    Create descriptors for each reaction from individual descriptors of 
    molecules.

    Parameters
    ----------
    dict_df : dict of DataFrame
        The keys are the reaction components and the values are the DataFrames
        are the descriptors for each molecule.
    rxns : DataFrame
        Reactions in the original paper.

    Returns
    -------
    descriptors: Dataframe
        Reactions and descriptors.
        
    """
    descriptors = pd.DataFrame()
    # For each reaction component.
    for k, df in dict_df.items():
        all_mol_single_component = pd.DataFrame()
        
        # For each molecule within the reaction component.
        for mol in df.name:
            if pd.isnull(mol):
                 org = rxns[rxns[k].isna()]
                 org = org.reset_index(drop=True)
                 des = df[df.name.isna()]
                 
            else:
                # Get original reactions containig the molecule.
                org=rxns[rxns[k] == mol].reset_index(drop=True)
                # Get descriptors for the molecule.
                des=df[df.name == mol]
            
            # Copy the descriptors row so the dataframe has the same length as 
            # org.
            des_rep = pd.concat(
                [des]*len(org), 
                ignore_index=True
                ).drop(columns=['name'])
            
            # Add the repeated descriptors dataframe to the original reactions.
            single_mol_single_component = pd.concat(
                [org, des_rep],
                axis=1
                )
            
            # Add the reactions and descriptors to a dataframe containing all 
            # molecules.
            all_mol_single_component = pd.concat(
                [all_mol_single_component, single_mol_single_component], 
                axis=0
                )
        
        # Add all molecules and descriptors for a single molecule type.
        if descriptors.empty:
            descriptors = descriptors.append(all_mol_single_component)
        else:
            # ERROR MESSAGE PRINTED HERE.
            descriptors = pd.merge(
                descriptors, 
                all_mol_single_component, 
                on=list(set(descriptors.columns).intersection(rxns.columns))
                )

    descriptors.set_index(
        sorted(list(set(descriptors.columns).intersection(rxns.columns))), 
        inplace=True
        )
        
    return descriptors
            

def assemble_quantum_descriptors(rxn_components, reactions, dir_descriptors,
                                 saveas=None):    
    """
    Assemble quantum chemical descriptors for the combinatorial reactions from
    the quantum descriptors of each reaction component.
    
    Parameters
    ----------
    rxn_components : list
        The names of the reaction components that also correspond to the names
        of the csv files that contain the quantum descriptors. 
        E.g. 'aryl_halide', 'base' etc.
    reactions : DataFrame
        Molecules present in each reaction.
    dir_descriptors : str
        Path to the csv files that contain the quantum descriptors. 
    saveas: str or ExcelWriter object, optional
        File path or existing ExcelWriter. The default is None.
    Returns
    -------
    None.

    """
    
    quantum_data = defaultdict()
    for rxn_component in rxn_components:
        quantum_data[rxn_component]= pd.read_csv(
            '{}/{}.csv'.format(dir_descriptors, rxn_component))
    
    quantum_descriptors = combine_descriptors_to_reactions(
        quantum_data, reactions)
        
    if saveas is not None:
        quantum_descriptors.to_excel(saveas, merge_cells=False)
    
    return quantum_descriptors

def get_smiles_data(rxn_components, reactions, hand_coded_smi, saveas=None):
    
    smiles = defaultdict(list)
    for rxn_component in rxn_components:
        print(rxn_component)
        
        for mol in reactions[rxn_component].drop_duplicates():
            print(mol)

            if pd.isnull(mol):
                smiles[rxn_component].append(
                    {'name': mol,
                      '{}_smiles'.format(rxn_component): np.nan
                      })
                
            else:
                smi = rd.name_to_smiles(mol)

                if pd.isnull(smi):
                    try:
                        smi = hand_coded_smi.loc[mol].smi
                        
                    except KeyError:
                        print('{} is not in hand-coded smiles, please update \
                              hand_coded_smi.xlsx'.format(mol))
                    
                    smiles[rxn_component].append(
                            {'name': mol,
                             '{}_smiles'.format(rxn_component): smi
                             })
                
                else:
                    smiles[rxn_component].append(
                        {'name': mol,
                          '{}_smiles'.format(rxn_component): smi
                          })
        
        smiles[rxn_component] = pd.DataFrame.from_records(
            smiles[rxn_component])
    
    rxn_smiles = combine_descriptors_to_reactions(smiles, reactions)
    
    if saveas is not None:
        rxn_smiles.to_excel(saveas, merge_cells=False)
    return rxn_smiles

def assemble_graph_descriptors(rxn_components, reactions, rxn_smiles):
    
    graphs_data = defaultdict(list)
    for rxn_component in rxn_components:
        for mol in reactions[rxn_component].drop_duplicates():
            
            if pd.isnull(mol):
                graphs_data[rxn_component].append(
                    {'name': mol,
                     '{}_molg'.format(rxn_component): np.nan
                     })
            else:
                smi = rxn_smiles.loc[
                    rxn_smiles.index.isin([mol], rxn_component)
                    ]['{}_smiles'.format(rxn_component)].drop_duplicates().item()
                
                graphs_data[rxn_component].append(
                    {'name': mol,
                     '{}_molg'.format(rxn_component): rd.molg_from_smi(smi)
                     })
            
        graphs_data[rxn_component] = pd.DataFrame.from_records(
            graphs_data[rxn_component])
            
    graph_descriptors = combine_descriptors_to_reactions(graphs_data, reactions)
    
    return graph_descriptors


def assemble_fingerprint_descriptors(rxn_components, reactions, rxn_smiles,  
                                     fp_type=rd.RDKFingerprint, fps_kw={},
                                     return_raw=True,  return_concat=True,
                                     return_sum=True):
    
    fps_data = defaultdict(list)
    for rxn_component in rxn_components:
        for mol in reactions[rxn_component].drop_duplicates():

            if pd.isnull(mol):
                fps_data[rxn_component].append(
                    {'name': mol,
                     '{}_fps'.format(rxn_component): np.nan
                     })
            else:
                smi = rxn_smiles.loc[
                    rxn_smiles.index.isin([mol], rxn_component)
                    ]['{}_smiles'.format(rxn_component)].drop_duplicates().item()
                
                fps_data[rxn_component].append(
                    {'name': mol,
                     '{}_fps'.format(rxn_component): rd.fps_from_smi(
                         smi, fp_type, fps_kw=fps_kw)
                     })
            
        fps_data[rxn_component] = pd.DataFrame.from_records(
            fps_data[rxn_component])
            
    fp_descriptors_raw = combine_descriptors_to_reactions(fps_data, reactions)
    
    
    bit_size = fp_descriptors_raw.iloc[0][0].GetNumBits()
    fp_descriptors_concat = []
    fp_descriptors_sum = []
    for ind, row in fp_descriptors_raw.iterrows():
        fps=[]
        for i in row:
            if pd.isnull(i):
                fps.append(np.array([0 for i in np.arange(0,bit_size,1)]))
            else:
                fps.append(np.array(list(map(int, i.ToBitString()))))
    
        fp_descriptors_concat.append(np.array([i for i in fps]).flatten())
        fp_descriptors_sum.append(sum(fps))
        
    fp_descriptors_concat = pd.DataFrame(
        data=np.array(fp_descriptors_concat), 
        index=fp_descriptors_raw.index
        )
    fp_descriptors_sum = pd.DataFrame(
        data=np.array(fp_descriptors_sum), 
        index=fp_descriptors_raw.index
        )
        
    return (fp_descriptors_raw,fp_descriptors_concat,fp_descriptors_sum) \
        if (return_raw is True) and (return_concat is True) and (return_sum is True) \
    else (fp_descriptors_raw,fp_descriptors_concat) \
        if (return_raw is True) and (return_concat is True) and (return_sum is False) \
    else (fp_descriptors_raw,fp_descriptors_sum) \
        if (return_raw is True) and (return_concat is False) and (return_sum is True) \
    else (fp_descriptors_concat,fp_descriptors_sum) \
        if (return_raw is False) and (return_concat is True) and (return_sum is True) \
    else fp_descriptors_raw \
        if (return_raw is True) and (return_concat is False) and (return_sum is False) \
    else fp_descriptors_concat \
        if (return_raw is False) and (return_concat is True) and (return_sum is False) \
    else fp_descriptors_sum \
        if (return_raw is False) and (return_concat is False) and (return_sum is True) \
    else None

    
def assemble_one_hot_encondings(rxn_components, reactions, reactions_test=None):
    
    if reactions_test is None:
        one_hot_encodings = pd.concat(
            [pd.get_dummies(reactions[rxn_component], dtype=int)
             for rxn_component in rxn_components
             ],
            axis=1
            ).set_index(pd.MultiIndex.from_frame(reactions))
        
        return one_hot_encodings
    
    else:
        reactions_both = pd.concat([reactions, reactions_test])
        
        one_hot_encodings = pd.concat(
            [pd.get_dummies(reactions_both[rxn_component], dtype=int)
             for rxn_component in rxn_components
             ],
            axis=1
            ).set_index(pd.MultiIndex.from_frame(reactions_both))

        one_hot_encodings_reactions = one_hot_encodings.loc[
            pd.MultiIndex.from_frame(reactions)
            ]
        
        index_diff = one_hot_encodings.index.names.difference(
            pd.MultiIndex.from_frame(reactions_test).names
            )
        if index_diff:
            one_hot_encodings.index = one_hot_encodings.index.droplevel(
                index_diff
                )
        index_diff = pd.MultiIndex.from_frame(reactions_test).names.difference(
            one_hot_encodings.index.names
            )
        if index_diff:
            reactions_test.index = reactions_test.index.droplevel(
                index_diff
                )
        
        one_hot_encodings_reactions_test = one_hot_encodings.loc[
            pd.MultiIndex.from_frame(reactions_test)
            ]
        
        return one_hot_encodings_reactions, one_hot_encodings_reactions_test
        
        
    

    
if __name__ == '__main__':
    rxn_components = ['additive', 'aryl_halide', 'base', 'ligand']
    
    reactions = pd.read_excel(
        './data/original/reactions/rxns_subset_no_nan_yields.xlsx',
        index_col=0
        )
    dir_descriptors = './data/original/quantum_descriptors'
    quantum_descriptors = assemble_quantum_descriptors(
        rxn_components, reactions, dir_descriptors, 
        './data/original/quantum_descriptors/quantum_descriptors.xlsx'
        )
    
    reactions_missing_additive = pd.read_excel(
        './data/original/reactions/rxns_subset_missing_additive.xlsx',
        )
    dir_descriptors = './data/original/quantum_descriptors_missing_additive'
    quantum_descriptors = assemble_quantum_descriptors(
        rxn_components, reactions_missing_additive, dir_descriptors, 
        './data/original/quantum_descriptors_missing_additive/quantum_descriptors.xlsx'
        )
    
    hand_coded_smi = pd.read_excel(
        './data/original/hand_coded_smi.xlsx', 
        index_col=0
        )
    
    get_smiles_data(
        rxn_components, reactions_missing_additive, hand_coded_smi, 
        './data/original/reactions/rxns_missing_additive_smi.xlsx')
    
    
    # smiles = get_smiles_data(
    #     rxn_components, reactions, hand_coded_smi, 
    #     './data/original/reactions/rxns_smi.xlsx')
    rxn_smiles = pd.read_excel(
        './data/original/reactions/rxns_smi.xlsx',
        index_col=[0, 1, 2, 3, 4]
        )

    graph_descriptors = assemble_graph_descriptors(
        rxn_components, reactions, rxn_smiles)
    
    fp_descriptors_raw, fp_descriptors_concat, fp_descriptors_sum = \
        assemble_fingerprint_descriptors(rxn_components, reactions, rxn_smiles)
        

    validation_rxns =  pd.read_excel(
        './data/validation/reactions/rxns_all.xlsx',
        )
    validation_smiles = pd.read_excel(
        './data/validation/smiles/molecule_smiles.xlsx',
        sheet_name=None
        )
    validation_rxn_smiles = combine_descriptors_to_reactions(
        validation_smiles, 
        validation_rxns
        )
    validation_rxn_smiles.set_index(validation_rxns.columns.to_list(), inplace=True)
    validation_rxn_smiles.to_excel(
        './data/validation/reactions/rxns_smi.xlsx',
        merge_cells=False
        )
    
    
    
