# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:32:00 2021

@author: alexe
"""

import logging, os, argparse
from datetime import datetime

from config import settings_loader
from tools.utils import data_importer
from tools.module_manager import module_loader
from tools.machine_learning.machine_learning import machine_learning_grid_search


class program():

    def __init__(self, working_dir, settings):
        self.working_dir = working_dir
        self.settings = settings
        self.modules = module_loader(n_jobs=self.settings.n_jobs)
        
    def run_preparation(self):
        logging.info('STEP 1: Load in data\n')

        input_fpath = os.path.join(
            self.working_dir, self.settings.input_reactions_fpath
            )

        data = data_importer.get_data(
            input_fpath,
            index_col=self.settings.descriptor_index
            )

        data_descriptors = data[self.settings.descriptor_cols]
        data_target = data[self.settings.target_col]

        logging.info('STEP 2: Generate descriptors\n')

        descriptors = self.modules.get_descriptors(
            self.settings.descriptor_type,
            data_descriptors,
            **self.settings.descriptor_settings
            )

        model_names = self.settings.models

        preprocessing = [
            self.modules.get_preprocessing(
                self.settings.descriptor_type,
                data_descriptors,
                model
                )
            for model in model_names
            ]
        models = [
            self.modules.get_model(model)
            for model in model_names
            ]

        if any(i for i in preprocessing):
            pipelines = [
                self.modules.get_pipeline(preprocess, model, model_name)
                for preprocess, model, model_name
                in zip(preprocessing, models, model_names)
                ]
        else:
            pipelines = models

        param_grids = [
            self.modules.get_param_grid(model)
            for model in model_names
            ]

        self.descriptors, self.data_target, self.model_names, \
        self.preprocessing, self.pipelines, self.param_grids \
            = descriptors, data_target, model_names, preprocessing, \
            pipelines, param_grids

    def start(self):
        self.run_preparation()

        logging.info('STEP 3: Split data and run machine learning\n')

        if self.settings.splitter:
            for n, splitter in enumerate(self.settings.splitter):
                splitter_settings = self.settings.splitter_settings[n]
                output_dir = self.settings.output_table_fpaths[n]

                logging.info('\tSplit:\t\t{}'.format(splitter))
                logging.info('\tOutput Folder:\t{}'.format(
                    os.path.join(output_dir)
                    ))

                test_sets = self.modules.get_split(
                    self.descriptors, self.data_target,
                    splitter, splitter_settings
                    )

                for test_set_name, test_set in test_sets.items():
                    logging.info('\n\tTest Set:\t{}'.format(
                        os.path.join(test_set_name)
                        ))
                    ml = machine_learning_grid_search(
                        self.model_names, self.pipelines, self.param_grids,
                        n_jobs=self.settings.n_jobs,
                        **test_set
                        )
                    ml.run()
                    ml.save_results(
                        saveas_dir='{}/{}'.format(
                            output_dir,
                            test_set_name
                            )
                        )
                    if any(i for i in self.settings.output_model_fpaths):
                        ml.save_models(
                            saveas_dir='{}/{}'.format(
                                self.settings.output_model_fpaths[n],
                                test_set_name
                                )
                            )
        return ml

    def start_validation(self):
        self.run_preparation()

        logging.info('STEP 3: Generate validation descriptors\n')

        output_dir = self.settings.validation_output_table_fpath

        logging.info('\tValidation')
        logging.info('\tOutput Folder:\t{}'.format(
            os.path.join(output_dir)
            ))

        data_test = data_importer.get_data(
            self.settings.validation_reactions_fpath,
            index_col=self.settings.descriptor_index
            )
        data_descriptors_test = data_test[
            self.settings.descriptor_cols
            ]
        descriptors_test = self.modules.get_descriptors(
            self.settings.descriptor_type,
            data_descriptors_test,
            **self.settings.descriptor_settings
            )

        logging.info('STEP 4: Run machine learning\n')

        test_sets = {
            'validation': {
                'X_train': self.descriptors,
                'y_train': self.data_target,
                'X_test': descriptors_test
                }
            }

        for test_set_name, test_set in test_sets.items():
            logging.info('\n\tTest Set:\t{}'.format(
                os.path.join(test_set_name)
                ))
            ml = machine_learning_grid_search(
                self.model_names, self.pipelines, self.param_grids,
                n_jobs=self.settings.n_jobs,
                **test_set
                )
            ml.run()
            ml.save_results(
                saveas_dir='{}/{}'.format(
                    output_dir,
                    test_set_name
                    )
                )
            if self.settings.validation_output_model_fpath:
                ml.save_models(
                    saveas_dir='{}/{}'.format(
                        self.settings.validation_output_model_fpath,
                        test_set_name
                        )
                    )
        return ml


def main(args):
    ini_dir = 'settings.ini'

    print(args.__dict__)

    if args.settings_fpath:
        ini_dir = args.settings_fpath
        (working_dir, filename) = os.path.split(ini_dir)

        if not filename.endswith('ini'):
            print('No settings.ini in --settings_fpath')
            return

        if not os.path.exists(ini_dir):
            print('File not found: settings.ini')

        print('Load settings.ini: ' + working_dir)

    else:
        filename = ini_dir
        working_dir = os.path.dirname(os.path.realpath(__file__))

    settings = settings_loader(os.path.join(working_dir, filename))
    settings.load()

    if settings.output_log_fpath:
        output_log_dir, output_log_fname = os.path.split(
            settings.output_log_fpath)

    else:
        output_log_dir, output_log_fname = 'output', 'output.log'

    if not os.path.exists(output_log_dir):
        os.makedirs(output_log_dir)

    logging.basicConfig(
        format='%(message)s',
        filename=os.path.join(output_log_dir, output_log_fname),
        level=logging.INFO
        )

    logging.info('PROGRAM STARTED {}\n'.format(datetime.now()))

    if any(settings.splitter):
        ml = program(working_dir, settings).start()
    elif settings.validation_reactions_fpath:
        ml = program(working_dir, settings).start_validation()

    logging.info('\nPROGRAM ENDED {}\n'.format(datetime.now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--settings_fpath', '-f',
        help='Path to settings.ini in your working directory. \
            Default is current working directotry'
        )
    args = parser.parse_args()

    main(args)
