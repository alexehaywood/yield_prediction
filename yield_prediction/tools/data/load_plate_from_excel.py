#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from collections import defaultdict
import pandas as pd


def define_molecule_keys(directory, file_name, *args, **kwargs):
    if file_name.endswith('.xlsx'):
        key = pd.read_excel(directory + file_name, *args, **kwargs)
        
    elif file_name.endswith('csv'):
        key = pd.read_csv(directory + file_name, *args, **kwargs)

    for k, v in key.items():
        try:
            key[k] = v['Name'].to_dict()
        except:
            key[k] = v['Retention Time'].to_dict()
            
    return key


def setup_plate(plate_names, io_setup, io_codes):
    """
    Load in plate rows, columns and codes.

    Parameters
    ----------
    plate_names : list
        Names of the plates in the file_name_setup and file_name_codes.
    io_setup : str
        A valid string path for .xlsx or .csv file that contains plate row and 
        column data.
    io_codes : str
        A valid string path for .xlsx or .csv file that contains plate codes.

    Returns
    -------
    plate_setup : dict
        A dict containing plate molecule rows, columns and plate-well codes.

    """

    plate_setup = defaultdict(dict)
    for plate_name in plate_names:

        if io_setup.endswith('.xlsx'):
            for i in ['rows', 'columns']:
                plate_setup[plate_name][i] = pd.read_excel(
                    io_setup,
                    sheet_name='{}_{}'.format(plate_name, i)
                    )

        elif io_setup.endswith('.csv'):
            for i in ['rows', 'columns']:
                plate_setup[plate_name][i] = pd.read_csv(
                    io_setup,
                    sheet_name='{}_{}'.format(plate_name, i)
                    )

        if io_codes.endswith('.xlsx'):
            plate_setup[plate_name]['codes'] = pd.read_excel(
                    io_codes,
                    sheet_name=plate_name,
                    header=None
                    )

        elif io_setup.endswith('.csv'):
            plate_setup[plate_name]['codes'] = pd.read_csv(
                    io_codes,
                    sheet_name=plate_name,
                    header=None
                    )

        plate_setup[plate_name]['ID'] = pd.Series()
        
        for n, i in plate_setup[plate_name]['codes'].iloc[0:16, 0:24].iterrows():
            plate_setup[plate_name]['ID'] = plate_setup[plate_name]['ID'].append(
                    i, ignore_index=True)
            
        for n, i in plate_setup[plate_name]['codes'].iloc[0:16, 24:].iterrows():
            plate_setup[plate_name]['ID'] = plate_setup[plate_name]['ID'].append(
                    i, ignore_index=True)
            
        for n, i in plate_setup[plate_name]['codes'].iloc[16:, 0:24].iterrows():
            plate_setup[plate_name]['ID'] = plate_setup[plate_name]['ID'].append(
                    i, ignore_index=True)
            
        for n, i in plate_setup[plate_name]['codes'].iloc[16:, 24:].iterrows():
            plate_setup[plate_name]['ID'] = plate_setup[plate_name]['ID'].append(
                    i, ignore_index=True)
        
        plate_setup[plate_name]['ID'].rename('ID', inplace=True)
        
    return plate_setup