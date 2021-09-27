# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:09:32 2021

@author: alexe
"""

import pandas as pd


def get_data(fpath, **kwargs):
    if any(fpath.endswith(i)
           for i in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']
           ):
        if 'index_col' in kwargs:
            new_kwargs = {k: v for k, v in kwargs.items() if k != 'index_col'}
            index_col = kwargs['index_col']
            data = pd.read_excel(fpath, **new_kwargs).set_index(index_col)
        else:
            data = pd.read_excel(fpath, **kwargs)

    elif fpath.endswith('.csv'):
        data = pd.read_csv(fpath, **kwargs)

    return data
