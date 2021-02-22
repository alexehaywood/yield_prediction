#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:19:47 2019

@author: pcxah5

Create a plate defining the reaction components in each well and 
the corresponding reaction code.
"""
from collections import defaultdict
import pandas as pd

class Plate:
    
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        
        matrix_rxn = defaultdict()
        for row in range(rows):
            for col in range(cols):
                matrix_rxn[(row, col)] = defaultdict()
        self.matrix_rxn = matrix_rxn
        
        matrix_code = defaultdict()
        for row in range(rows):
            for col in range(cols):
                matrix_code[(row, col)] = str
        self.matrix_code = matrix_code
        
    def fillBlock(self, row, col, var, value):
        for r in row:
            for c in col:
                self.matrix_rxn[(r, c)].update({var: value})
    
    def fillColumn(self, col, var, value):
        row = [i for i in range(self.rows)]
        self.fillBlock(row, col, var, value)

    def fillRow(self, row, var, value):
        col = [i for i in range(self.cols)]
        self.fillBlock(row, col, var, value)
        
    def fillCodeMatrix(self, plate_codes):
        for r in plate_codes.index:
            for c in plate_codes.columns:
                self.matrix_code[(r, c)] = plate_codes.loc[r, c]
                
    def fillPlate(self, rows, columns, key, codes=pd.DataFrame()):
        for name in columns:
            for n, i in columns[name].iteritems():
                self.fillColumn([n], name, key[name][i])
    
        for name in rows:
            for n, i in rows[name].iteritems():
                self.fillRow([n], name, key[name][i])
                
        self.fillCodeMatrix(codes)
    
    def getConditions(self, code):
        # Find key (row and column numbers) corresponding to specified value (code).
        r_c = list(self.matrix_code.keys())[list(self.matrix_code.values()).index(code)]
        # Use row and column numbers to find reaction conditions from matrix_rxn.
        conditions = self.matrix_rxn[r_c]
        return conditions
    
    def removeBlock(self, row, col):
        for r in row:
            for c in col:
                self.matrix_rxn.pop((r, c), None)
                self.matrix_code.pop((r, c), None)
                
    def removeColumn(self, col):
        row = [i for i in range(self.rows)]
        self.removeBlock(row, col)
        
    def removeRow(self, row):
        col = [i for i in range(self.cols)]
        self.removeBlock(row, col)
        
    def removeValue(self, var, value):
        r_c_list = []
        for k, v in self.matrix_rxn.items():
            if v[var] is value:
                r_c_list.append(k)
        for r_c in r_c_list:
            self.matrix_rxn.pop(r_c, None)
            self.matrix_code.pop(r_c, None)
