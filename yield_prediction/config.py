# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 18:37:10 2021

@author: alexe
"""

from configparser import ConfigParser
import ast

class settings_loader():
    def __init__(self, config_fpath):
        self.config_fpath = config_fpath

        self.config = ConfigParser()
        self.config.read(config_fpath)
        
    def load(self):
        self.input_reactions_fpath = self.config['FileSettings']['input_reactions_fpath']
        self.output_log_fpath = self.config['FileSettings']['output_log_fpath']
        self.output_table_fpaths = self.config['FileSettings']['output_table_fpaths'].replace(' ', '').split(',')
        self.output_model_fpaths = self.config['FileSettings']['output_model_fpaths'].replace(' ', '').split(',')
        self.n_jobs = int(self.config['FileSettings']['n_jobs'])

        self.descriptor_type = self.config['DescriptorSettings']['descriptor_type']
        self.descriptor_settings = ast.literal_eval(self.config['DescriptorSettings']['descriptor_settings'])
        self.descriptor_cols = self.config['DescriptorSettings']['descriptor_cols'].replace(' ', '').split(',')
        self.descriptor_index = ast.literal_eval(self.config['DescriptorSettings']['descriptor_index'])
        self.target_col = self.config['DescriptorSettings']['target_col']

        self.models = self.config['MachineLearningSettings']['models'].replace(' ', '').split(',')

        self.splitter = self.config['InternalSplitterSettings']['splitter'].replace(' ', '').split(',')
        if self.config['InternalSplitterSettings']['splitter_settings']:
            self.splitter_settings = ast.literal_eval(self.config['InternalSplitterSettings']['splitter_settings'])

        self.validation_reactions_fpath = self.config['ValidationSettings']['validation_reactions_fpath']
        self.validation_output_table_fpath = self.config['ValidationSettings']['validation_output_table_fpath']
        self.validation_output_model_fpath = self.config['ValidationSettings']['validation_output_model_fpath']
        self.pretrained_model_fpath = self.config['ValidationSettings']['pretrained_model_fpath']
