# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:13:58 2021

@author: alexe
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
import math
from sklearn.metrics import r2_score, mean_squared_error
import pickle

from config import settings_loader
from tools.module_manager import module_loader
from tools.utils import data_importer
from tools.utils.plotting import plotting
from tools.descriptors.tanimoto_kernels import tanimoto_kernel, tanimoto_kernel_with_missing_mols


def calc_max_tan(settings, splitter_names, saveas=None):
    modules = module_loader()

    input_fpath = settings.input_reactions_fpath

    data = data_importer.get_data(
            input_fpath,
            index_col=settings.descriptor_index
            )

    data_target = data[settings.target_col]
    data_descriptors = data[settings.descriptor_cols]

    descriptors = modules.get_descriptors(
            settings.descriptor_type,
            data_descriptors,
            **settings.descriptor_settings
            )

    max_tan = defaultdict()
    for n, splitter in enumerate(settings.splitter):
        print(splitter)
        name = splitter_names[n]
        splitter_settings = settings.splitter_settings[n]

        test_sets = modules.get_split(
            descriptors, data_target,
            splitter, splitter_settings
            )

        max_tan[name] = pd.DataFrame()
        for test_set_name, test_set in test_sets.items():
            print(test_set_name)
            tk = tanimoto_kernel()
            tk.fit(test_set['X_train'])
            tanimoto_scores = tk.transform(test_set['X_test'])

            max_tan_subset = pd.DataFrame(
                np.amax(tanimoto_scores, axis=1),
                index=test_set['X_test'].index,
                columns=['max_tanimoto']
                )

            max_tan[name] = max_tan[name].append(max_tan_subset)

    if saveas is not None:
        pickle.dump(
            max_tan,
            open('graphs_for_manuscript/max_tan_{}.pkl'.format(saveas), 'wb')
            )

    return max_tan


def calc_validation_max_tan(settings, saveas=None):
    modules = module_loader()

    input_fpath = settings.input_reactions_fpath

    data = data_importer.get_data(
            input_fpath,
            index_col=settings.descriptor_index
            )

    data_descriptors = data[settings.descriptor_cols]

    descriptors = modules.get_descriptors(
            settings.descriptor_type,
            data_descriptors,
            **settings.descriptor_settings
            )

    data_test = data_importer.get_data(
        settings.validation_reactions_fpath,
        index_col=settings.descriptor_index
        )
    data_descriptors_test = data_test[
        settings.descriptor_cols
        ]
    descriptors_test = modules.get_descriptors(
        settings.descriptor_type,
        data_descriptors_test,
        **settings.descriptor_settings
        )

    test_sets = {
        'validation': {
            'X_train': descriptors,
            'X_test': descriptors_test
            }
        }

    max_tan = pd.DataFrame()
    for test_set_name, test_set in test_sets.items():
        print(test_set_name)
        tk = tanimoto_kernel()
        tk.fit(test_set['X_train'])
        tanimoto_scores = tk.transform(test_set['X_test'])

        max_tan_subset = pd.DataFrame(
            np.amax(tanimoto_scores, axis=1),
            index=test_set['X_test'].index,
            columns=['max_tanimoto']
            )

        max_tan = max_tan.append(max_tan_subset)

    if saveas is not None:
        pickle.dump(
            max_tan,
            open('graphs_for_manuscript/max_tan_{}.pkl'.format(saveas), 'wb')
            )

    return max_tan


def plot_max_sim_all(max_tan, saveas_dir=None, saveas_name=None):
    if saveas_dir is not None:
        if not os.path.exists('{}/distributions_all_rxns'.format(saveas_dir)):
            os.makedirs('{}/distributions_all_rxns'.format(saveas_dir))

    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (3.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600,
                'sharey': True}
        )
    plotter.add_plot(
        x=[df.max_tanimoto for df in max_tan.values()],
        kind='hist',
        plot_kw={
            'bins': np.arange(0, 1.05, 0.05),
            'color': custom_cmap(np.linspace(0, 1, len(max_tan))),
            'edgecolor': 'white',
            'linewidth': 0.25,
            'rwidth': 0.95,
            'weights': [
                np.ones(len(df.max_tanimoto))/len(df.max_tanimoto) * 100
                for df in max_tan.values()
                ],
            # 'label': [k.replace('_', '\n').title() for k in max_tan.keys()]
            },
        xlim=(0, 1),
        ylim=(0, 100),
        xlabel='Maximum Similarity to Training',
        ylabel='Proportion of Reactions (%)',
        tick_params={'labelsize': 8, 'pad': 1},
        # lgd_kw={
        #     'markerscale':1.5, 'scatteryoffsets':[1], 'fontsize':8,
        #     'loc':'upper right',
        #     'ncol':1,
        #     'bbox_to_anchor':(1, 1),
        #     'frameon':False,
        #     'columnspacing':1.25,
        #     },
        )

    if saveas_dir is None:
        plotter.save_plot()
    else:
        plotter.save_plot('{}/distributions_all_rxns/max_sim_{}'.format(
            saveas_dir, saveas_name))


def plot_max_sim_all_val(max_tan, saveas_dir=None, saveas_name=None):
    if saveas_dir is not None:
        if not os.path.exists('{}/distributions_all_rxns'.format(saveas_dir)):
            os.makedirs('{}/distributions_all_rxns'.format(saveas_dir))

    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (3.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600,
                'sharey': True}
        )
    plotter.add_plot(
        x=[df.max_tanimoto for df in max_tan.values()],
        kind='hist',
        plot_kw={
            'bins': np.arange(0, 1.05, 0.05),
            'color': grey,
            'edgecolor': 'white',
            'linewidth': 0.25,
            'rwidth': 0.95,
            # 'weights': [
            #     np.ones(len(df.max_tanimoto))/len(df.max_tanimoto) * 100
            #     for df in max_tan.values()
            #     ],
            # 'label': [k.replace('_', '\n').title() for k in max_tan.keys()]
            },
        xlim=(0, 1),
        # ylim=(0, 100),
        xlabel='Maximum Similarity to Training',
        ylabel='Number of Aryl Halides',
        tick_params={'labelsize': 8, 'pad': 1},
        # lgd_kw={
        #     'markerscale':1.5, 'scatteryoffsets':[1], 'fontsize':8,
        #     'loc':'upper right',
        #     'ncol':1,
        #     'bbox_to_anchor':(1, 1),
        #     'frameon':False,
        #     'columnspacing':1.25,
        #     },
        )

    if saveas_dir is None:
        plotter.save_plot()
    else:
        plotter.save_plot('{}/distributions_all_rxns/max_sim_{}'.format(
            saveas_dir, saveas_name))


def analyse_max_tan(y_pred_max_tan, model_names, name=None, saveas_dir=None):
    if saveas_dir is not None:
        if not os.path.exists('{}/performance_max_sim'.format(saveas_dir)):
            os.makedirs('{}/performance_max_sim'.format(saveas_dir))

    y_pred_max_tan['bins'] = pd.cut(
        y_pred_max_tan.max_tanimoto,
        bins=np.arange(
            math.floor(min(y_pred_max_tan.max_tanimoto) / 0.05) * 0.05,
            math.ceil(max(y_pred_max_tan.max_tanimoto) / 0.05) * 0.05 + 0.05,
            0.05
            ),
        right=False
        )

    max_tan_analysis = defaultdict()
    max_tan_analysis['rmse'] = defaultdict(dict)
    max_tan_analysis['r2'] = defaultdict(dict)
    for i in y_pred_max_tan['bins'].cat.categories:
        for model in model_names:
            if len(y_pred_max_tan[y_pred_max_tan['bins'] == i]) < 2:
                max_tan_analysis['rmse'][
                    '{}-{}'.format(i.left, i.right)
                    ].update({model: np.nan})
                max_tan_analysis['r2'][
                    '{}-{}'.format(i.left, i.right)
                    ].update({model: np.nan})
            else:
                max_tan_analysis['rmse'][
                    '{}-{}'.format(i.left, i.right)
                    ].update({model: mean_squared_error(
                        y_pred_max_tan[
                            y_pred_max_tan['bins'] == i
                            ].index.get_level_values('yield_exp'),
                        y_pred_max_tan[
                            y_pred_max_tan['bins'] == i
                            ][model],
                        squared=False
                        )})
                max_tan_analysis['r2'][
                    '{}-{}'.format(i.left, i.right)].update({model: r2_score(
                        y_pred_max_tan[
                            y_pred_max_tan['bins'] == i
                            ].index.get_level_values('yield_exp'),
                        y_pred_max_tan[
                            y_pred_max_tan['bins'] == i
                            ][model],
                        )})

    for metric, v in max_tan_analysis.items():
        max_tan_analysis[metric] = pd.DataFrame(v).T
        max_tan_analysis[metric] = max_tan_analysis[metric].dropna()

    if math.floor(np.nanmin(
        max_tan_analysis['r2'].values
            )*10)/10 > -1:
        y_min = math.floor(np.nanmin(
            max_tan_analysis['r2'].values
            )*10)/10
    else:
        y_min = -1

    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (7.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600,
                'sharey': True}
        )
    plotter.add_plot(
        x=max_tan_analysis['r2'],
        kind='bar',
        plot_kw={
            'color': custom_cmap(np.linspace(0, 1, len(model_names))),
            'legend': False,
            'edgecolor': 'white',
            'linewidth': 0.25,
            'rot': 0,
            },
        xlabel='Maximum Similarity to Training',
        ylabel='$R^2$',
        ylim=(y_min, 1),
        tick_params={'labelsize': 8, 'pad': 1},
        lgd_kw={
            'bbox_to_anchor': (0.5, -0.3),
            'ncol': 3,  # 4,
            'loc': 'center',
            'fontsize': 8,
            'frameon': False,
            'columnspacing': 1.25
            },
        grid_kw={
            'b': True,
            'which': 'major',
            'axis': 'y',
            'color': grey,
            'linestyle': '-',
            'linewidth': 0.5,
            }
        )
    if saveas_dir and name is not None:
        plotter.save_plot(
            '{}/performance_max_sim/r2_sim_{}'.format(saveas_dir, name)
            )
    else:
        plotter.save_plot()

    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (7.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600,
                'sharey': True}
        )
    plotter.add_plot(
        x=max_tan_analysis['rmse'],
        kind='bar',
        plot_kw={
            'color': custom_cmap(np.linspace(0, 1, len(model_names))),
            'legend': False,
            'edgecolor': 'white',
            'linewidth': 0.25,
            'rot': 0,
            },
        xlabel='Maximum Similarity to Training',
        ylabel='RMSE (%)',
        ylim=(0, math.ceil(np.nanmax(
            max_tan_analysis['rmse'].values
            )/5)*5),
        tick_params={'labelsize': 8, 'pad': 1},
        lgd_kw={
            'bbox_to_anchor': (0.5, -0.3),
            'ncol': 3,  # 4,
            'loc': 'center',
            'fontsize': 8,
            'frameon': False,
            'columnspacing': 1.25
            },
        grid_kw={
            'b': True,
            'which': 'major',
            'axis': 'y',
            'color': grey,
            'linestyle': '-',
            'linewidth': 0.5,
            }
        )
    if saveas_dir and name is not None:
        plotter.save_plot(
            '{}/performance_max_sim/rmse_sim_{}'.format(saveas_dir, name)
            )
    else:
        plotter.save_plot()

    return max_tan_analysis


if 'graphs_for_manuscript' not in os.listdir('.'):
    os.mkdir('graphs_for_manuscript')
if 'SI' not in os.listdir('graphs_for_manuscript'):
    os.mkdir('graphs_for_manuscript/SI')

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap',
    [(0, '#00586a'),
     (0.15, '#326784'),
     (0.29, '#5a7499'),
     (0.43, '#8182aa'),
     (0.57, '#a68fb5'),
     (0.71, '#c99ebc'),
     (0.85, '#e7afc1'),
     (1, '#ffc3c5')
     ])

custom_cmap_heatmap = LinearSegmentedColormap.from_list(
    'custom_cmap',
    [(0, '#003742'),
     (0.15, '#00586a'),
     (0.25, '#326784'),
     (0.35, '#5a7499'),
     (0.45, '#8182aa'),
     (0.55, '#a68fb5'),
     (0.65, '#c99ebc'),
     (0.75, '#e7afc1'),
     (0.85, '#ffc3c5'),
     (1, '#f1f1f1')
     ])

grey = '#666666'

reactions = pd.read_excel(
    'input/Doyle/reactions/rxns_subset_no_nan_yields.xlsx',
    index_col=0
    ).set_index(['additive', 'aryl_halide', 'base', 'ligand'])

descriptor_names_dict = {
    'wl_kernel_wl5': 'WL',
    '_quantum': 'Quantum',
    '_one-hot_encodings': 'One-hot',
    'fingerprints_morgan1_512': 'Fps: Morgan1',
    'fingerprints_rdk_512': 'Fps: RDK',
    'fingerprints_maccs': 'Fps: MACCS',
    'tanimoto_kernel_morgan1_512': 'Tan: Morgan1',
    'tanimoto_kernel_rdk_512': 'Tan: RDK',
    'tanimoto_kernel_maccs': 'Tan: MACCS',
            }

y_pred_best = defaultdict(dict)

for file, name, descriptors, models in zip(
        ['output/additive_ranking_ypred.xlsx',
         'output/aryl_halide_ranking_ypred.xlsx'],
        ['additive', 'aryl_halide'],
        [['_one-hot_encodings', '_quantum', 'fingerprints_morgan1_512',
          'tanimoto_kernel_morgan1_512', 'wl_kernel_wl5'],
         ['_one-hot_encodings', '_quantum', 'fingerprints_morgan1_512',
          'tanimoto_kernel_morgan1_512', 'wl_kernel_wl5']],
        [['SVR - Poly Kernel', 'SVR - RBF Kernel', 'SVR - Poly Kernel',
          'SVR - Precomputed Kernel', 'SVR - Precomputed Kernel'],
         ['SVR - Poly Kernel', 'SVR - RBF Kernel', 'SVR - Poly Kernel',
          'SVR - Precomputed Kernel', 'SVR - Precomputed Kernel']]
        ):
    for descriptor, model in zip(descriptors, models):
        descriptor_name = descriptor_names_dict[descriptor]
        model_name = model.split(' ')[2]
        descriptor_model_name = '{} - {}'.format(
            descriptor_name, model_name
            )
        y_pred_best[name][descriptor_model_name] = pd.read_excel(
            file,
            sheet_name=descriptor,
            index_col=[0, 1, 2, 3],
            usecols=['additive', 'aryl_halide', 'base', 'ligand', model]
            )

    df = pd.concat(y_pred_best[name], axis=1)
    df.columns = df.columns.droplevel(1)
    df = pd.concat([df, reactions], axis=1, join='inner')
    df = df.set_index('yield_exp', append=True)

    y_pred_best[name] = df


path_head = 'settings_examples/tanimoto_kernel'
max_tan = defaultdict(dict)
for settings_file, saveas in zip(
        ['settings_graphs_max_tan_maccs.ini',
         'settings_graphs_max_tan_morgan1.ini',
         'settings_graphs_max_tan_morgan2.ini',
         'settings_graphs_max_tan_morgan3.ini',
         'settings_graphs_max_tan_rdk.ini'
         ],
        ['maccs', 'morgan1', 'morgan2', 'morgan3', 'rdk'
         ]
        ):
    path = 'graphs_for_manuscript/max_tan_{}.pkl'.format(saveas)
    if os.path.exists(path):
        max_tan[saveas] = pickle.load(open(path, 'rb'))
    else:
        settings = settings_loader(os.path.join(path_head, settings_file))
        settings.load()
        max_tan[saveas] = calc_max_tan(
            settings,
            splitter_names=['additive_ranking', 'aryl_halide_ranking'],
            saveas=saveas
            )
    plot_max_sim_all(max_tan[saveas], 'graphs_for_manuscript', saveas)

for rxn_component in ['additive', 'aryl_halide']:
    for fp in ['maccs', 'morgan1', 'morgan2', 'morgan3', 'rdk']:
        df = y_pred_best[rxn_component].copy()
        model_names = df.columns
        df.index = df.index.droplevel('yield_exp')
        df = pd.concat(
            [df, max_tan[fp]['{}_ranking'.format(rxn_component)]], axis=1
            )
        df = pd.concat([df, reactions], axis=1, join='inner')
        df = df.set_index('yield_exp', append=True)
        analyse_max_tan(
            df,
            model_names,
            name='{}_{}'.format(rxn_component, fp),
            saveas_dir='graphs_for_manuscript'
            )


path_head = 'settings_examples/tanimoto_kernel'
max_tan = defaultdict(dict)
for settings_file, saveas in zip(
        ['settings_graphs_max_tan_ah_morgan2.ini'
         ],
        ['validation_morgan2'
         ]
        ):
    path = 'graphs_for_manuscript/max_tan_{}.pkl'.format(saveas)
    if os.path.exists(path):
        max_tan[saveas] = pickle.load(open(path, 'rb'))
    else:
        settings = settings_loader(os.path.join(path_head, settings_file))
        settings.load()
        max_tan[saveas] = calc_validation_max_tan(
            settings,
            saveas=saveas
            )
plot_max_sim_all_val(max_tan, 'graphs_for_manuscript', saveas)
