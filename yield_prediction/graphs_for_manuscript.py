# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:16:49 2020

@author: alexe
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from scipy.stats import chi2_contingency, pearsonr
import math
from sklearn.metrics import r2_score, mean_squared_error
import pickle

from config import settings_loader
from tools.module_manager import module_loader
from tools.utils import data_importer
from tools.utils.plotting import plotting
from tools.descriptors.tanimoto_kernels import tanimoto_kernel

class ranking():

    def rank_mols(mols, molecule_keys, rxn_component, yields):
        ranked_mols = []
        for mol in mols:
            ranked_mols.append({
                rxn_component: mol,
                'key': molecule_keys[rxn_component][
                    molecule_keys[rxn_component] == mol].index[0],
                'mean_yield': np.mean(
                    yields.loc[yields[rxn_component] == mol].yield_exp),
                'std_yield': np.std(
                    yields.loc[yields[rxn_component] == mol].yield_exp)
                })

        ranked_mols = pd.DataFrame(ranked_mols)
        ranked_mols.sort_values(by=['mean_yield'], inplace=True)
        ranked_mols.reset_index(inplace=True, drop=True)
        ranked_mols.index = ranked_mols.index.set_names(['order'])

        return ranked_mols

    def make_test_sets(ranked_mols, n_sets=4):
        test_sets = defaultdict()
        n = 1
        for test_n in np.arange(1, n_sets+1):
            test_sets[test_n] = ranked_mols.key[n:-1:n_sets].tolist()
            n = n+1

        return test_sets


def get_yields(rxn_df, mols, rxn_component, molecule_keys=None):
    if molecule_keys is not None:
        mols = [molecule_keys[i] for i in mols]

    rxn_subset = rxn_df.iloc[rxn_df.index.isin(mols, level=rxn_component)]

    yields = rxn_subset.reset_index('yield_exp').yield_exp.values

    return yields


def plot_yield_distributions_all(rxn_smiles, saveas_dir='.'):
    if 'distributions_all_rxns' not in os.listdir(saveas_dir):
        os.mkdir('{}/distributions_all_rxns'.format(saveas_dir))

    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize':(3.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600}
        )
    plotter.add_plot(
        x=rxn_smiles.reset_index('yield_exp').yield_exp.values,
        kind='hist',
        plot_kw={
            'bins': np.arange(0, 110, 10),
            'color': grey,
            'edgecolor': 'white',
            'linewidth': 0.25,
            'rwidth': 0.95,
            'weights': np.ones(
                len(rxn_smiles.index.get_level_values('yield_exp'))
                ) / len(rxn_smiles.index.get_level_values('yield_exp')) * 100
            },
        xlabel='Experimental Yield (%)',
        ylabel='Proportion of Reactions (%)',
        xlim=(0, 100),
        ylim=(0, 30),
        tick_params={'labelsize': 8, 'pad': 1}
        )
    plotter.save_plot(
        '{}/distributions_all_rxns/yield_distribution'.format(saveas_dir)
        )


def plot_yield_distributions_test_sets(
        rxn_smiles, rxn_component, test_sets, molecule_keys, name, saveas_dir
        ):
    if not os.path.exists('{}/SI/yield_distributions'.format(saveas_dir)):
        os.makedirs('{}/SI/yield_distributions'.format(saveas_dir))

    for test_name, mols in test_sets.items():

        y_train = rxn_smiles[
            ~rxn_smiles.index.get_level_values(rxn_component).isin(
                [molecule_keys[rxn_component].loc[i]
                 for i in mols]
                )
            ].index.get_level_values('yield_exp').values
        y_test = rxn_smiles[
            rxn_smiles.index.get_level_values(rxn_component).isin(
                [molecule_keys[rxn_component].loc[i]
                 for i in mols]
                )
            ].index.get_level_values('yield_exp').values
        y = rxn_smiles.reset_index('yield_exp').yield_exp.values

        plotter = plotting(
            rcParams={'font.size': 10, 'axes.titlesize': 10},
            fig_kw={'figsize': (3.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600}
            )
        plotter.add_plot(
            x=[y_train, y_test],
            kind='hist',
            plot_kw={
                'bins': np.arange(0, 110, 10),
                'histtype': 'bar',
                'color': custom_cmap(np.linspace(0, 1, 2)),
                # 'label':['Training Set', 'Test Set'],
                'edgecolor': 'white',
                'linewidth': 0.25,
                'rwidth': 0.95,
                'weights': [np.ones(len(y_train)) / len(y) * 100,
                            np.ones(len(y_test)) / len(y) * 100]
                },
            xlim=(0, 100),
            ylim=(0, 25),
            xlabel='Experimental Yield (%)',
            ylabel='Proportion of Reactions (%)',
            title=test_name,
            tick_params={'labelsize': 8, 'pad': 1},
            # lgd_kw={
            #     'markerscale': 1.5, 'scatteryoffsets': [1], 'fontsize': 8,
            #     'loc': 'upper right',
            #     'ncol': 1,
            #     'bbox_to_anchor': (1, 1),
            #     'frameon': False,
            #     'columnspacing': 1.25,
            #     }
            )
        plotter.save_plot('{}/SI/yield_distributions/{}_{}_{}'.format(
            saveas_dir, rxn_component, name,
            test_name.lower().replace(' ', ''))
            )

    y_train = [
        rxn_smiles[
            ~rxn_smiles.index.get_level_values(rxn_component).isin(
                [molecule_keys[rxn_component].loc[i]
                 for i in mols]
                )
            ].index.get_level_values('yield_exp').values
        for test_name, mols in test_sets.items()
        ]
    y_test = [
        rxn_smiles[
            rxn_smiles.index.get_level_values(rxn_component).isin(
                [molecule_keys[rxn_component].loc[i]
                 for i in mols]
                )
            ].index.get_level_values('yield_exp').values
        for test_name, mols in test_sets.items()
        ]
    y = rxn_smiles.reset_index('yield_exp').yield_exp.values
    data = [[y_train[n], y_test[n]] for n in np.arange(0, len(y_train))]
    if len(data) < 4:
        difference = 4 - len(data)
        for n in np.arange(0, difference):
            data.append(None)

    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5, 2), 'ncols': 4, 'nrows': 1, 'dpi': 600,
                'sharey': True}
        )
    plotter.add_plot(
        x=data,
        kind=['hist' for i in data],
        plot_kw=[{
            'bins': np.arange(0, 110, 10),
            'histtype': 'bar',
            'color': custom_cmap.reversed()(np.linspace(0, 1, 2)),
            # 'label': ['Training Set', 'Test Set'],
            'edgecolor': 'white',
            'linewidth': 0.25,
            'rwidth': 0.95,
            'weights': [np.ones(len(y_train[n])) / len(y) * 100,
                        np.ones(len(y_test[n])) / len(y) * 100]
            } for n in np.arange(0, len(y_train), 1)],
        xlim=(0, 100),
        ylim=(0, 25),
        title=[test_name for test_name in test_sets.keys()],
        tick_params={'labelsize': 8, 'pad': 1},
        )
    plotter.add_common_axes(xlabel='Experimental Yield (%)',
                            ylabel='Proportion of Reactions (%)')
    plotter.save_plot('{}/SI/yield_distributions/{}_{}'.format(
            saveas_dir, rxn_component, name))


def plot_ranked_order(ranked_mols, rxn_component, saveas_dir):
    if saveas_dir is not None:
        if not os.path.exists('{}/SI/ranked_order'.format(saveas_dir)):
            os.mkdir('{}/SI/ranked_order'.format(saveas_dir))

    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (7.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600}
        )
    plotter.add_plot(
            x=ranked_mols[rxn_component].index,
            y=ranked_mols[rxn_component]['mean_yield'],
            kind='errorbar',
            plot_kw={
                'yerr': ranked_mols[rxn_component]['std_yield'],
                'color': grey,
                'marker': 'o',
                'capsize': 3,
                },
            xlabel=rxn_component.replace('_', ' ').title(),
            ylabel='Mean Experimental Yield (%)',
            xticks={'ticks': ranked_mols[rxn_component].index},
            xticklabels={'labels': ranked_mols[rxn_component]['key'],
                         'horizontalalignment': 'center'}
            )
    if saveas_dir is None:
        plotter.save_plot()
    else:
        plotter.save_plot(
            '{}/SI/ranked_order/{}'.format(saveas_dir, rxn_component)
            )


def plot_kernel_descriptor_scores(scores, saveas_dir=None):
    if saveas_dir is not None:
        if 'kernel_desc_heatmap' not in os.listdir(saveas_dir):
            os.mkdir('{}/kernel_desc_heatmap'.format(saveas_dir))

    ylabels = [
        df['Mean'].index.get_level_values(
            'Descriptor'
            ).unique()
        for df in scores.values()
        ]
    xlabels = [
        df['Mean'].index.get_level_values(
            'Kernel'
            ).unique()
        for df in scores.values()
        ]

    x_to_num = [{n[1]:n[0] for n in enumerate(xlabel)} for xlabel in xlabels]
    y_to_num = [{n[1]:n[0] for n in enumerate(ylabel)} for ylabel in ylabels]

    y = [
        df['Mean'].index.get_level_values(
            'Descriptor'
            ).map(y_to_num[n])
        for n, df in enumerate(scores.values())
        ]
    x = [
        df['Mean'].index.get_level_values(
            'Kernel'
            ).map(x_to_num[n])
        for n, df in enumerate(scores.values())
        ]

    size = [df['Mean'] for df in scores.values()]

    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (4, 2.5), 'ncols': 2, 'nrows': 1,
                'sharey': True, 'dpi': 600}
        )

    s = []
    for ax in plotter.axes:
        length = plotter.fig.bbox_inches.width * ax.get_position().width
        value_range = np.diff(ax.get_xlim())
        length *= 72
        s.append(1 * (length / value_range))

    plotter.add_plot(
            x=x,
            y=y,
            kind='scatter',
            plot_kw=[
                {'s': s[n]*size[n],
                 'c': size[n],
                 'cmap': custom_cmap_heatmap.reversed(),
                 'vmin': 0,
                 'vmax': 1,
                 'marker': 'o'}
                for n in np.arange(0, len(x))
                ],
            xlim=[(-0.5, max(i.values()) + 0.5) for i in x_to_num],
            ylim=[(-0.5, max(i.values()) + 0.5) for i in y_to_num],
            title=[
                '{}\nRanked Test'.format(
                    rxn_component.replace('_', ' ').title()
                    )
                for rxn_component in scores.keys()
                ],
            tick_params={'labelsize': 8, 'pad': 1},
            xticks=[
                [{'ticks': [x_to_num_n[xlabel] for xlabel in xlabels_n]},
                 {'ticks': [x_to_num_n[xlabel]+0.5 for xlabel in xlabels_n],
                  'minor':True}]
                for xlabels_n, x_to_num_n in zip(xlabels, x_to_num)
                ],
            yticks=[
                [{'ticks': [y_to_num_n[ylabel] for ylabel in ylabels_n]},
                 {'ticks': [y_to_num_n[ylabel]+0.5 for ylabel in ylabels_n],
                  'minor': True}
                 ]
                for ylabels_n, y_to_num_n in zip(ylabels, y_to_num)
                ],
            xticklabels=[
                {'labels': xlabel_n, 'rotation': 45,
                 'horizontalalignment': 'right'}  # 'center'}
                for xlabel_n in xlabels
                ],
            yticklabels=[
                {'labels': ylabel_n}
                for ylabel_n in ylabels
                ],
            aspect=['equal', 'equal'],
            grid_kw={'which': 'minor', 'axis': 'both', 'color': grey}
            )
    plotter.adjust_fig(subplots_adjust_kw={'wspace': 0.15})
    plotter.add_cbar(tick_params={'labelsize': 8, 'pad': 1}, label='$R^2$')
    plotter.add_common_axes(
        ylabel='Descriptor', xlabel='Kernel', tick_params={'pad': 35}
        )
    if saveas_dir is not None:
        plotter.save_plot(
            '{}/kernel_desc_heatmap/r2_heatmap'.format(saveas_dir)
            )
    else:
        plotter.save_plot()


def calculate_chi2_and_plot_kde(y_pred, name=None, saveas_dir=None):
    if saveas_dir is not None:
        if not os.path.exists('{}/SI/residuals_kde'.format(saveas_dir)):
            os.makedirs('{}/SI/residuals_kde'.format(saveas_dir))

    residuals = -1*y_pred.sub(
        y_pred.index.get_level_values('yield_exp'),
        axis=0
        )

    residuals_bins = pd.DataFrame(
        [plt.hist(residuals[name], np.arange(-100, 105, 5))[0]
         for name in residuals.columns],
        index=residuals.columns
        )

    residuals_chi2 = defaultdict()
    for name1 in residuals.columns:
        for name2 in residuals.columns:
            if not any(i in residuals_chi2.keys()
                       for i in [(name1, name2), (name2, name1)]
                       ):
                residuals_chi2[(name1, name2)] = chi2_contingency(
                    residuals_bins.loc[
                        [name1, name2]].loc[
                            :, (residuals_bins.loc[[name1, name2]] != 0
                                ).any(axis=0)].loc[
                                    [name1, name2]]
                    )[1]

    residuals_chi2 = pd.Series(residuals_chi2)

    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (7.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600}
        )
    plotter.add_plot(
        x=residuals,
        kind='kde',
        plot_kw={
            'legend': True,
            'color': custom_cmap(np.linspace(0, 1, len(y_pred.columns))),
            'alpha': 1
            },
        xlabel='Residual Yield (%)',
        ylabel='Kernel Density Estimate',
        xlim=(-100, 100),
        tick_params={'labelsize': 8, 'pad': 1},
        lgd_kw={
            # 'markerscale': 1.5, 'scatteryoffsets': [1],
            'fontsize': 8,
            'loc': 'center',
            'ncol': 3,  # len(residuals.columns),
            'bbox_to_anchor': (0.5, -0.3),  #(0.5, -0.25),
            'frameon': False,
            'columnspacing': 1.25
            }
        )
    if saveas_dir and name is not None:
        plotter.save_plot('{}/SI/residuals_kde/{}'.format(saveas_dir, name))
    else:
        plotter.save_plot()

    return residuals_chi2


def analyse_yield_exp(y_pred, colours=None, name=None, saveas_dir=None):
    if saveas_dir is not None:
        if not os.path.exists('{}/SI/performance_yield_exp'.format(saveas_dir)):
            os.makedirs('{}/SI/performance_yield_exp'.format(saveas_dir))

    if colours is None:
        colours = custom_cmap(np.linspace(0, 1, len(y_pred.columns)))

    y_pred['bins'] = pd.cut(
        y_pred.index.get_level_values('yield_exp'),
        bins=np.arange(0, 110, 10),
        right=False
        )

    analysis = defaultdict()
    analysis['rmse'] = defaultdict(dict)
    for i in y_pred['bins'].cat.categories:
        for model in y_pred.columns.drop('bins'):
            if len(y_pred[y_pred['bins'] == i]) < 2:
                analysis['rmse']['{}-{}'.format(i.left, i.right)].update({
                    model: np.nan
                    })
            else:
                analysis['rmse']['{}-{}'.format(i.left, i.right)].update({
                    model: mean_squared_error(
                        y_pred[
                            y_pred['bins'] == i
                            ].index.get_level_values('yield_exp'),
                        y_pred[
                            y_pred['bins'] == i
                            ][model],
                        squared=False
                        )
                    })

    for metric, v in analysis.items():
        analysis[metric] = pd.DataFrame(v).T
        analysis[metric] = analysis[metric].dropna()

    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (7.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600,
                'sharey': True}
        )
    plotter.add_plot(
        x=analysis['rmse'],
        kind='bar',
        plot_kw={'color': colours, 'legend': False, 'edgecolor': 'white',
                 'linewidth': 0.25, 'rot': 0},
        xlabel='Experimental Yield (%)',
        ylabel='RMSE (%)',
        ylim=(0, 45),
        # ylim=(0, math.ceil(np.nanmax(
        #     analysis['rmse'].values
        #     )/5)*5),
        tick_params={'labelsize': 8, 'pad': 1},
        lgd_kw={'bbox_to_anchor': (0.5, -0.3), 'loc': 'center',
                'ncol': 3,  # 4,
                'fontsize': 8, 'frameon': False, 'columnspacing': 1.25},
        grid_kw={'b': True, 'which': 'major', 'axis': 'y', 'color': grey,
                 'linestyle': '-', 'linewidth': 0.5}
        )

    if saveas_dir and name is not None:
        plotter.save_plot(
            '{}/SI//performance_yield_exp/rmse_{}'.format(saveas_dir, name)
            )
    else:
        plotter.save_plot()

    return analysis


def get_halide_type(halide):
    if 'I' in halide:
        halide_type = 'Iodides'
    elif 'Br' in halide:
        halide_type = 'Bromides'
    elif 'Cl' in halide:
        halide_type = 'Chlorides'
    return halide_type


def plot_yield_heatmap(data, additive, name, saveas_dir=None):
    # Sort Data.
    data = pd.pivot_table(
        data,
        index=['additive', 'aryl_halide'],
        columns=['ligand', 'base'],
        dropna=False
        )

    data['halide'] = data.index.get_level_values(
        'aryl_halide').map(get_halide_type)
    data.set_index('halide', append=True, inplace=True)

    data['aryl_halide_key'] = data.index.get_level_values(
        'aryl_halide').map(
            lambda x: prospective_key['aryl_halide'][
                prospective_key['aryl_halide']['aryl_halide'] == x
                ].index[0]
            )
    data.set_index('aryl_halide_key', append=True, inplace=True)

    for k in ['aryl_halide']:
        data = data.reindex(
            prospective_key[k]['aryl_halide'].values,
            level=k,
            axis='index')

    for k in ['base', 'ligand', 'model']:
        data = data.reindex(prospective_key[k].values, level=k, axis='columns')

    # x axis.
    n_bases = len(data.columns.get_level_values('base').drop_duplicates())
    n_ligands = len(data.columns.get_level_values('ligand').drop_duplicates())

    text = []

    xticks = []
    for model in data.columns.get_level_values(0).drop_duplicates():
        start = data.columns.get_loc_level(model, level=0)[0].start
        stop = data.columns.get_loc_level(model, level=0)[0].stop
        xticks.append(stop-0.5)
        text.append(
            {'x': (start + (stop - start) / 2) / len(data.columns), 'y': 1.05,
             's': model, 'ha': 'center', 'fontsize': 8}
            )
    xticks = xticks[:-1]

    for ligand in data.columns.get_level_values('ligand').drop_duplicates():
        loc = [
            i for i, x in enumerate(
                data.columns.get_loc_level(ligand, level='ligand')[0]
                ) if x
            ]
        start = loc[0]
        text.extend(
            {'x': n / len(data.columns), 'y': 1.03,
             's': '{}'.format(
                 prospective_key['ligand'][
                     prospective_key['ligand'] == ligand
                     ].index[0]
                 ),
             'ha': 'center', 'fontsize': 8}
            for n in np.arange(
                    start+(n_bases/2), len(data.columns), n_bases*n_ligands)
            )

    for base in data.columns.get_level_values('base').drop_duplicates():
        loc = [
            i for i, x in enumerate(
                data.columns.get_loc_level(base, level='base')[0]
                ) if x
            ]
        start = loc[0]
        text.extend(
            {'x': n/len(data.columns), 'y': 1.01,
             's': '{}'.format(
                 prospective_key['base'][
                     prospective_key['base'] == base
                     ].index[0]
                 ),
             'ha': 'center', 'fontsize': 8}
            for n in np.arange((start+0.5), len(data.columns), n_bases)
            )

        if base == data.columns.get_level_values('base').drop_duplicates()[-1]:
            xticks_minor = [
                i + 0.5 for i, x in enumerate(
                    data.columns.get_loc_level(base, 'base')[0]
                    ) if x
                ]
    xticks_minor = xticks_minor[:-1]

    # y axis
    text.append(
        {'x': -0.08, 'y': 0.5,
         's': 'Additive: {}'.format(additive.capitalize()),
         'ha': 'center', 'fontsize': 8, 'rotation': 90}
        )

    yticks = []
    for halide in data.index.get_level_values('halide').drop_duplicates():
        loc = [
            i for i, x in enumerate(
                data.index.get_loc_level(halide, 'halide')[0]
                ) if x
            ]
        start = min(loc)
        stop = max(loc)
        text.append(
            {'x': -0.06, 'y': 1-(start + (stop - start)/2)/len(data.index),
             's': halide, 'ha': 'center', 'va': 'center', 'fontsize': 8,
             'rotation': 90}
            )
        yticks.extend([start, stop])

    yticks = [
        (yticks[n+1] + yticks[n])/2
        for n in np.arange(0, len(yticks))
        if n != len(yticks)-1
        if yticks[n+1] - yticks[n] == 1
        ]

    yticks_minor = np.arange(0, len(data.index), 1)

    yticklabels = data.index.get_level_values('aryl_halide_key')

    # Plot
    plotter = plotting(
        rcParams={'font.size': 8, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5/36*len(data.columns)+0.5, 10/49*len(data.index)),  # (7.5,12),
                'ncols': 1, 'nrows': 1, 'sharey': True, 'dpi': 600}
        )
    plotter.add_plot(
            x=data.values,
            kind='imshow',
            plot_kw={'cmap': custom_cmap_heatmap.reversed(),
                     'clim': (0, 100)},
            tick_params=[
                {'labelsize': 8, 'pad': 1},
                {'axis': 'x', 'which': 'major', 'top': True, 'bottom': False,
                 'labelbottom': False, 'labeltop': False, 'length': 37.5},
                {'axis': 'x', 'which': 'minor', 'top': True, 'bottom': False,
                 'labelbottom': False, 'labeltop': False, 'length': 25,
                 'color': 'black'},
                {'axis': 'y', 'which': 'major', 'left': True, 'right': False,
                 'labelleft': False, 'labelright': False, 'length': 27.5},
                {'axis': 'y', 'which': 'minor', 'left': False, 'right': False,
                 'labelleft': True, 'labelright': False},
                ],
            xticks=[{'ticks': xticks},
                    {'ticks': xticks_minor, 'minor': True}],
            yticks=[{'ticks': yticks},
                    {'ticks': yticks_minor, 'minor': True}],
            yticklabels={'labels': yticklabels, 'minor': True},
            annotate_heatmap={'valfmt': '{x:.0f}', 'threshold': 50,
                              'textkw': {'fontsize': 6.5}},
            text=text,
            grid_kw={'b': True, 'which': 'major', 'axis': 'both',
                     'color': 'black', 'linestyle': '-', 'linewidth': 1},
            )
    if saveas_dir is None:
        plotter.save_plot()
    else:
        plotter.save_plot(
            '{}/SI/validation/heatmap_ypred_{}_{}'.format(
                saveas_dir, additive, name
                )
            )


#%%
if __name__ == '__main__':
    
    # %% Setup
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

    molecule_keys = pd.read_excel(
            'input/Doyle/molecule_keys.xlsx',
            index_col=0,
            sheet_name=None,
            squeeze=True
            )

    rxn_components = ['additive', 'aryl_halide', 'base', 'ligand']

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

    # %% Yield Distributions
    original = pd.read_excel(
        'input/Doyle/reactions/rxns_subset_no_nan_yields.xlsx',
        index_col=0
        )

    rxn_smiles = pd.read_excel(
        'input/Doyle/reactions/rxns_smi.xlsx',
        index_col=[0, 1, 2, 3, 4]
        )

    ranked_mols = defaultdict()
    for rxn_component in rxn_components:
        ranked_mols[rxn_component] = ranking.rank_mols(
            mols=original[rxn_component].drop_duplicates().dropna(),
            molecule_keys=molecule_keys,
            rxn_component=rxn_component,
            yields=original
            )

    additive_ranking = ranking.make_test_sets(
        ranked_mols=ranked_mols['additive'],
        n_sets=4
        )

    aryl_halide_ranking = ranking.make_test_sets(
        ranked_mols=ranked_mols['aryl_halide'],
        n_sets=3
        )

    additive_test_sets = defaultdict()
    additive_test_sets['plate'] = defaultdict()
    additive_test_sets['plate']['Plate 1'] = [16, 17, 18, 19, 20, 21, 22, 23]
    additive_test_sets['plate']['Plate 2'] = [8, 9, 10, 11, 12, 13, 14, 15]
    additive_test_sets['plate']['Plate 3'] = [1, 2, 3, 4, 5, 6]
    additive_test_sets['ranking'] = defaultdict()
    additive_test_sets['ranking'] = {
        'Set {}'.format(k): v for k, v in additive_ranking.items()
        }

    aryl_halide_test_sets = defaultdict()
    aryl_halide_test_sets['halide'] = defaultdict()
    aryl_halide_test_sets['halide']['Chlorides'] = [1, 4, 7, 10, 13]
    aryl_halide_test_sets['halide']['Bromides'] = [2, 5, 8, 11, 14]
    aryl_halide_test_sets['halide']['Iodides'] = [3, 6, 9, 12, 15]
    aryl_halide_test_sets['aryl'] = defaultdict()
    aryl_halide_test_sets['aryl']['Phenyl'] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    aryl_halide_test_sets['aryl']['Pyridyl'] = [10, 11, 12, 13, 14, 15]
    aryl_halide_test_sets['ranking'] = defaultdict()
    aryl_halide_test_sets['ranking'] = {
        'Set {}'.format(k): v for k, v in aryl_halide_ranking.items()
        }

    base_test_sets = defaultdict()
    for i in molecule_keys['base'].keys():
        base_test_sets[i] = [i]

    ligand_test_sets = defaultdict()
    for i in molecule_keys['ligand'].keys():
        ligand_test_sets[i] = [i]

    plot_yield_distributions_all(rxn_smiles, 'graphs_for_manuscript')
    for rxn_component, test_sets, name in zip(
            ['additive', 'aryl_halide',
             'base', 'ligand', 'additive',
             'aryl_halide', 'aryl_halide'],
            [additive_test_sets['ranking'], aryl_halide_test_sets['ranking'],
             base_test_sets, ligand_test_sets, additive_test_sets['plate'],
             aryl_halide_test_sets['aryl'], aryl_halide_test_sets['halide']],
            ['ranking', 'ranking',
             'LOO', 'LOO', 'plate',
             'ring_type', 'halide']
            ):
        plot_yield_distributions_test_sets(
            rxn_smiles, rxn_component, test_sets, molecule_keys, name,
            saveas_dir='graphs_for_manuscript'
            )

    for rxn_component in ['additive', 'aryl_halide']:
        plot_ranked_order(
            ranked_mols, rxn_component, saveas_dir='graphs_for_manuscript'
            )

    # %% Cross-Validation

    if not os.path.exists('graphs_for_manuscript/SI/cv'):
        os.makedirs('graphs_for_manuscript/SI/cv')

    # Fingerprints and Tanimoto Kernel
    data = defaultdict(dict)

    for path, name in zip(
            ['output/cv_fps_svr.xlsx', 'output/cv_fps_linear.xlsx',
             'output/cv_fps_tree.xlsx'],
            ['SVR Models', 'Linear Models', 'Tree-Based Models']
            ):
        for sheet_name, metric in zip(
                ['best_mean_cv_scores_R-squared', 'best_mean_cv_scores_RMSE'],
                ['Mean $R^2$', 'Mean RMSE (%)'],
                ):
            data[metric][name] = pd.read_excel(
                path, sheet_name=sheet_name,
                usecols=['Descriptor', 'Mean']
                )
            data[metric][name]['Bit Length'] = [
                int(i[-1])
                for i in data[metric][name].Descriptor.str.split('_')
                ]
            data[metric][name]['Fingerprint Type'] = [
                i[-2].title()
                for i in data[metric][name].Descriptor.str.split('_')
                ]
            data[metric][name]['Descriptor'] = [
                ' '.join(i[:-2]).title()
                for i in data[metric][name].Descriptor.str.split('_')
                ]

    ylim = {'Mean $R^2$': (0, 1), 'Mean RMSE (%)': (0, 25)}
    for k, v in data.items():
        data[k] = pd.concat(v)
        data[k].index = data[k].index.droplevel(1)
        data[k].index = [
            '{}: {} '.format(i, j)
            for i, j in zip(data[k]['Descriptor'], data[k].index)
            ]
        data[k].index.name = 'Models'
        data[k].drop('Descriptor', axis=1, inplace=True)

        for fp_list, name in zip(
                [['Morgan1', 'Morgan2', 'Morgan3'],
                 ['Rdk']],
                ['morgan', 'rdk']
                ):
            dfs = [
                data[k][data[k]['Fingerprint Type'] == fp
                        ].drop('Fingerprint Type', axis=1
                               ).reset_index()
                for fp in fp_list
                ]
            dfs = [
                df.pivot_table(index=['Bit Length'], columns=['Models']
                               ).droplevel(0, axis=1)
                for df in dfs
                ]

            plotter = plotting(
                rcParams={'font.size': 10, 'axes.titlesize': 10},
                fig_kw={'figsize': (7.5, 2.5), 'ncols': 3, 'nrows': 1,
                        'dpi': 600, 'sharey': True}
                )

            if len(fp_list) == 3:
                plotter.add_plot(
                    x=[df.filter(regex='Fingerprints') for df in dfs],
                    plot_kw={'legend': False, 'marker': '.',
                             'color': custom_cmap(np.linspace(0, 1, 3))},
                    )
                plotter.add_plot(
                    x=[df.filter(regex='Tanimoto') for df in dfs],
                    plot_kw={'legend': False, 'marker': 's', 'markersize': 2.5,
                             'linestyle': '--',
                             'color': custom_cmap(np.linspace(0, 1, 3))},
                    title=fp_list,
                    ylim=ylim[k]
                    )
                plotter.add_common_axes(xlabel='Bit Length', ylabel=k)

            elif len(fp_list) == 1:
                plotter.add_plot(
                    x=[dfs[0].filter(regex='Fingerprints'), None, None],
                    plot_kw={'legend': True, 'marker': '.',
                             'color': custom_cmap(np.linspace(0, 1, 3))}
                    )
                plotter.add_plot(
                    x=[dfs[0].filter(regex='Tanimoto'), None, None],
                    plot_kw={'legend': True, 'marker': 's', 'markersize': 2.5,
                             'linestyle': '--',
                             'color': custom_cmap(np.linspace(0, 1, 3))},
                    title='RDK',
                    ylim=ylim[k],
                    xlabel='Bit Length',
                    ylabel=k,
                    lgd_kw={'bbox_to_anchor': (1, 0.5), 'ncol': 1,
                            'loc': 'center left', 'fontsize': 10,
                            'frameon': False}
                    )

            plotter.save_plot('graphs_for_manuscript/SI/cv/fps_{}_{}'.format(
                k.replace('$', '').replace('^', '').split(' ')[1], name
                ))

        df = data[k][
            (data[k]['Bit Length'] == 512) &
            (data[k]['Fingerprint Type'].str.contains('Morgan'))
            ].drop('Bit Length', axis=1).reset_index()
        df['Morgan Radius'] = [i for i in df['Fingerprint Type'].str[-1]]
        df.drop('Fingerprint Type', axis=1, inplace=True)
        df = df.pivot_table(index=['Morgan Radius'], columns=['Models']
                            ).droplevel(0, axis=1)

        plotter = plotting(
            rcParams={'font.size': 10, 'axes.titlesize': 10},
            fig_kw={'figsize': (2.5, 2.5), 'ncols': 1, 'nrows': 1,
                    'dpi': 600, 'sharey': True}
            )
        plotter.add_plot(
            x=df.filter(regex='Fingerprints'),
            plot_kw={'legend': False, 'marker': '.',
                     'color': custom_cmap(np.linspace(0, 1, 3))},
            )
        plotter.add_plot(
            x=df.filter(regex='Tanimoto'),
            plot_kw={'legend': False, 'marker': 's', 'markersize': 2.5,
                     'linestyle': '--',
                     'color': custom_cmap(np.linspace(0, 1, 3))},
            ylim=ylim[k],
            xlabel='Morgan Radius',
            ylabel=k,
            # lgd_kw={'bbox_to_anchor': (1, 0.5), 'ncol': 1,
            #         'loc': 'center left', 'fontsize': 10,
            #         'frameon': False}
            )
        plotter.save_plot('graphs_for_manuscript/SI/cv/fps_{}_{}'.format(
            k.replace('$', '').replace('^', '').split(' ')[1], 'morgan_radius'
            ))

    # WL Kernel
    for sheet_name, metric, ylim in zip(
            ['best_mean_cv_scores_R-squared', 'best_mean_cv_scores_RMSE'],
            ['Mean $R^2$', 'Mean RMSE (%)'],
            [(0, 1), (0, 25)]
            ):
        results = defaultdict()
        for path, name in zip(
                ['output/cv_wl_svr.xlsx', 'output/cv_wl_linear.xlsx',
                 'output/cv_wl_tree.xlsx'],
                ['SVR Models', 'Linear Models', 'Tree-Based Models']
                ):
            results[name] = pd.read_excel(
                path, sheet_name=sheet_name, usecols=['Descriptor', 'Mean']
                )
        results = pd.concat(results)
        results.index = results.index.droplevel(1)
        results.index.name = 'Models'
        results['WL Depth'] = [
            int(i[1]) for i in results.Descriptor.str.split('wl_kernel_wl')
            ]
        results = results.drop('Descriptor', axis=1).reset_index()
        results = results.pivot_table(index=['WL Depth'], columns=['Models']
                                      ).droplevel(0, axis=1)

        plotter = plotting(
            rcParams={'font.size': 10, 'axes.titlesize': 10},
            fig_kw={'figsize': (2.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600,
                    'sharey': True}
            )
        plotter.add_plot(
            x=results,
            plot_kw={'legend': False, 'marker': '.',
                     'color': custom_cmap(np.linspace(0, 1, 3))},
            ylim=ylim
            )
        plotter.add_common_axes(xlabel='WL Depth', ylabel=metric)
        plotter.save_plot('graphs_for_manuscript/SI/cv/wl_{}'.format(
            metric.replace('$', '').replace('^', '').split(' ')[1]
            ))

    # %% Kernel-descriptor r2 plot
    scores = defaultdict()
    for file, name in zip(
        ['output/additive_ranking.xlsx', 'output/aryl_halide_ranking.xlsx'],
        ['additive', 'aryl_halide']
            ):
        scores[name] = pd.read_excel(
            file,
            sheet_name='scores_R-squared',
            index_col=[0, 1],
            usecols=['Descriptor', 'Model', 'Mean']
            )

        scores[name] = scores[name][
            scores[name].index.get_level_values(
                'Model').str.contains('SVR')
                ]

        scores[name].rename(
            index=descriptor_names_dict,
            inplace=True
            )

        scores[name]['Kernel'] = scores[name].index.get_level_values(
                'Model').str.split(' ').str[2]
        scores[name].set_index('Kernel', append=True, inplace=True)

    plot_kernel_descriptor_scores(
        {name: scores[name][scores[name]['Mean'] >= 0]
         for name in ['additive', 'aryl_halide']},
        saveas_dir='graphs_for_manuscript'
        )

    # %% Single Predicted vs. Observed yields
    def plot_pred_obs_single(df, model, colour=grey, saveas_dir=None,
                             saveas_name=None):
        if saveas_dir is not None:
            if not os.path.exists('{}/SI/pred_vs_obs'.format(saveas_dir)):
                os.makedirs('{}/SI/pred_vs_obs'.format(saveas_dir))

        plotter = plotting(
            rcParams={'font.size': 10, 'axes.titlesize': 10},
            fig_kw={'figsize': (1.75, 2.5), 'ncols': 1, 'nrows': 1,
                    'dpi': 300, 'subplot_kw': {'aspect':'equal'}}
            )

        plotter.add_plot(
            x=df,
            kind='scatter',
            plot_kw={'x': 'yield_exp', 'y': model, 'color': colour,
                     'marker': '.', 's': 2.5, 'alpha': 0.75},
            )

        plotter.add_plot(
            x=[0, 100],
            y=[0, 100],
            plot_kw={'linestyle': 'dashed', 'color': 'black',
                     # 'alpha': 0.95,
                     'linewidth': 0.75}
            )

        m, c = np.polyfit(df['yield_exp'], df[model], 1)
        plotter.add_plot(
            x=df['yield_exp'],
            y=m*df['yield_exp'] + c,
            kind='line',
            plot_kw={'linewidth': 1.25, 'alpha': 1,  # 'label': model
                     'color': colour},
            xlim=(0, 100),
            ylim=(-20, 110),
            tick_params={'labelsize': 8, 'pad': 1},
            xlabel='Experimental Yield (%)',
            ylabel='Predicted Yield (%)'
            )

        if saveas_dir is None:
            plotter.save_plot()
        else:
            plotter.save_plot(
                '{}/SI/pred_vs_obs/{}.png'.format(saveas_dir, saveas_name)
                )

    y_pred = defaultdict(dict)
    for file, name in zip(
            ['output/aryl_halide_ranking_ypred.xlsx'],
            ['aryl_halide']
            ):
        n = 0
        for descriptor, model in zip(
                ['_one-hot_encodings', '_quantum',  'wl_kernel_wl5', 'fingerprints_morgan1'],
                ['SVR - Poly Kernel', 'SVR - RBF Kernel', 'SVR - Precomputed Kernel', 'SVR - Poly Kernel']
                ):
            descriptor_name = descriptor_names_dict[descriptor]
            model_name = model.split(' ')[2]
            descriptor_model_name = u'{} \u2014 {}'.format(
                descriptor_name, model_name
                )
            y_pred[name][descriptor_model_name] = pd.read_excel(
                file,
                sheet_name=descriptor,
                index_col=[0, 1, 2, 3],
                usecols=['additive', 'aryl_halide', 'base', 'ligand', model]
                )

            df = pd.concat(
                [y_pred[name][descriptor_model_name], reactions],
                axis=1,
                join='inner'
                )

            plot_pred_obs_single(
                df, model,
                rgb2hex(custom_cmap(np.linspace(0, 1, 5))[n]),
                saveas_dir='graphs_for_manuscript',
                saveas_name=descriptor_model_name.replace(
                    ' \u2014 ', '_').replace(':', '-')
                )

            n = n+1

    # %% Residual distribution plots, chi2, rmse vs. yield

    y_pred_best = defaultdict(dict)
    residuals_chi2_best = defaultdict(dict)

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

        residuals_chi2_best[name] = calculate_chi2_and_plot_kde(
            df,
            'best_{}'.format(name),
            'graphs_for_manuscript'
            )
        residuals_chi2_best[name].name = 'p-value'

        analyse_yield_exp(
            df,
            name='{}_ranking'.format(name),
            saveas_dir='graphs_for_manuscript'
            )

    # %% Scores
    # Requires y_pred_best to be defined

    # scores = defaultdict(list)

    # for splitter, splitter_settings in zip(
    #         ['activity_ranking', 'activity_ranking'],
    #         [{'rxn_component': 'additive', 'n_splits': 4}, {'rxn_component': 'aryl_halide', 'n_splits': 3}]
    #         ):
    #     modules = module_loader()

    #     descriptors = reactions.index.to_frame()
    #     descriptors.index = reactions.index
    #     test_sets = modules.get_split(
    #         descriptors, reactions['yield_exp'],
    #         splitter, splitter_settings
    #         )
    #     rxn_component = splitter_settings['rxn_component']

    #     for test_set_name, test_set in test_sets.items():
    #         molecule_test_list = test_set['X_train'][
    #             rxn_component].drop_duplicates().values

    #         y_true = y_pred_best[rxn_component][
    #             y_pred_best[rxn_component].index.isin(
    #                 molecule_test_list, level=rxn_component
    #                 )
    #             ].index.get_level_values('yield_exp').values

    #         for model in y_pred_best[rxn_component].columns:
    #             y_pred = y_pred_best[rxn_component][
    #                 y_pred_best[rxn_component].index.isin(
    #                     molecule_test_list, level=rxn_component
    #                     )
    #                 ][model].values

    #             scores[rxn_component].extend([{
    #                 'Test Set': test_set_name,
    #                 'Model': model,
    #                 'R-squared': r2_score(y_true, y_pred),
    #                 'RMSE': mean_squared_error(y_true, y_pred, squared=False),
    #                 'pearsonr': pearsonr(y_true, y_pred)[0]
    #                 }])

    #     scores[rxn_component] = pd.DataFrame(scores[rxn_component])
    #     scores[rxn_component] = scores[rxn_component].pivot_table(
    #         columns='Test Set', index='Model'
    #         )
    #     scores[rxn_component][('R-squared', 'Mean')] = \
    #         scores[rxn_component]['R-squared'].mean(axis=1)
    #     scores[rxn_component][('RMSE', 'Mean')] = \
    #         scores[rxn_component]['RMSE'].mean(axis=1)

    # scores_without_out_of_scope_mols = defaultdict(list)
    # out_of_scope_mols = defaultdict()
    # out_of_scope_y_pred = defaultdict()
    # for rxn_component, test_sets in zip(
    #         ['additive', 'aryl_halide'],
    #         [additive_ranking, aryl_halide_ranking] 
    #         ):
    #     out_of_scope_mols[rxn_component] = \
    #         max_tan[rxn_component]['morgan2'] == max_tan[rxn_component]['morgan2'].min()
    #     out_of_scope_mols[rxn_component] = out_of_scope_mols[rxn_component][
    #         out_of_scope_mols[rxn_component].max_tanimoto == True
    #         ].index.get_level_values(rxn_component).drop_duplicates().to_list()
        
    #     out_of_scope_y_pred[rxn_component] = y_pred_best[rxn_component][
    #         ~y_pred_best[rxn_component].index.isin(
    #             out_of_scope_mols[rxn_component], level=rxn_component
    #             )
    #         ]
        
    #     for test_set in test_sets.keys():
    #         molecule_test_list = [
    #             molecule_keys[rxn_component].loc[key] 
    #             for key in test_sets[test_set]
    #             ]

    #         y_true = out_of_scope_y_pred[rxn_component][
    #             out_of_scope_y_pred[rxn_component].index.isin(
    #                 molecule_test_list, level=rxn_component
    #                 )
    #             ].index.get_level_values('yield_exp').values

    #         for model in out_of_scope_y_pred[rxn_component].columns:
    #             y_pred = out_of_scope_y_pred[rxn_component][
    #                 out_of_scope_y_pred[rxn_component].index.isin(
    #                     molecule_test_list, level=rxn_component
    #                     )
    #                 ][model].values
    #             scores_without_out_of_scope_mols[rxn_component].extend([{
    #                 'Test Set': 'Set{}'.format(test_set),
    #                 'Model': model,
    #                 'R-squared': r2_score(y_true, y_pred),
    #                 'RMSE': mean_squared_error(y_true, y_pred, squared=False),
    #                 'pearsonr': pearsonr(y_true, y_pred)[0]
    #                 }])

    #     scores_without_out_of_scope_mols[rxn_component] \
    #         = pd.DataFrame(scores_without_out_of_scope_mols[rxn_component])
    #     scores_without_out_of_scope_mols[rxn_component] \
    #         = scores_without_out_of_scope_mols[rxn_component].pivot_table(
    #             columns='Test Set', index='Model'
    #             )
    #     scores_without_out_of_scope_mols[rxn_component][('R-squared', 'Mean')] = \
    #         scores_without_out_of_scope_mols[rxn_component]['R-squared'].mean(axis=1)
    #     scores_without_out_of_scope_mols[rxn_component][('RMSE', 'Mean')] = \
    #         scores_without_out_of_scope_mols[rxn_component]['RMSE'].mean(axis=1)

    # %% Similarity Heatmaps
    if 'similarity_heatmaps' not in os.listdir('graphs_for_manuscript/SI'):
        os.mkdir('graphs_for_manuscript/SI/similarity_heatmaps')
    
    for rxn_component, test in zip(
            ['additive', 'aryl_halide'],
            [additive_ranking, aryl_halide_ranking]
            ):
        tanimoto = defaultdict()
        index=[]
        for test_set_k, test_set_v in test.items():
            molecule_test_list = [molecule_keys[rxn_component].loc[key] 
                                  for key in test_set_v]
            index.extend(molecule_test_list)
            
            mols_train = pd.DataFrame(
                rxn_smiles[
                    ~rxn_smiles.index.isin(molecule_test_list, rxn_component)
                    ]['{}_smiles'.format(rxn_component)].drop_duplicates()
                )
            mols_train.index = mols_train.index.get_level_values(rxn_component)
            mols_train['mols'] = mols_train['{}_smiles'.format(rxn_component)].map(
                lambda x: rd.Chem.MolFromSmiles(x)
                )
            mols_train['fps'] = mols_train['mols'].map(
                lambda x: rd.Chem.AllChem.GetMorganFingerprintAsBitVect(x,2,1024)
                )
            
            mols_test = pd.DataFrame(
                rxn_smiles[
                    rxn_smiles.index.isin(molecule_test_list, rxn_component)
                    ]['{}_smiles'.format(rxn_component)].drop_duplicates()
                )
            mols_test.index = mols_test.index.get_level_values(rxn_component)
            mols_test['mols'] = mols_test['{}_smiles'.format(rxn_component)].map(
                lambda x: rd.Chem.MolFromSmiles(x)
                )
            mols_test['fps'] = mols_test['mols'].map(
                lambda x: rd.Chem.AllChem.GetMorganFingerprintAsBitVect(x,2,1024)
                )
            
            tanimoto[test_set_k] = pd.DataFrame(columns=mols_test.index)
            for ind, mol in mols_train.iterrows():
                tanimoto[test_set_k].loc[ind] \
                    = rd.DataStructs.BulkTanimotoSimilarity(
                        mol['fps'], mols_test['fps']
                        )
        
        tanimoto = pd.concat(tanimoto, axis=1, sort=False)
        tanimoto = tanimoto.droplevel(axis='columns', level=0)
        # tanimoto = tanimoto.loc[index]
        tanimoto = tanimoto.loc[ranked_mols[rxn_component][rxn_component]]
        
        xlabels = tanimoto.index.map(
            lambda x: int(molecule_keys[rxn_component][molecule_keys[rxn_component] == x].index.values))
        ylabels = tanimoto.columns.map(
            lambda x: int(molecule_keys[rxn_component][molecule_keys[rxn_component] == x].index.values))
            
        x_to_num = {n[1]:n[0] for n in enumerate(xlabels)}
        y_to_num = {n[1]:n[0] for n in enumerate(ylabels)}
            
        plotter = plotting(
            rcParams={'font.size':10, 'axes.titlesize':10},
            fig_kw={'figsize':(7.5,4), 'ncols':1, 'nrows':1, 
                    'sharey':True, 'dpi':600}
            )
        plotter.add_plot(
                x=tanimoto.T.values,
                kind='imshow', 
                plot_kw={'cmap':custom_cmap_heatmap.reversed(), 
                         'clim':(0,1)},
                xlabel='Training Set',
                ylabel='Test Set',
                tick_params={'labelsize':8, 'pad':1},
                xticks={'ticks': [x_to_num[xlabel] for xlabel in xlabels]},
                yticks={'ticks': [y_to_num[ylabel] for ylabel in ylabels]},
                xticklabels={'labels':xlabels, 'rotation':45, 
                              'horizontalalignment':'center'},
                yticklabels={'labels':ylabels},
                # # aspect=['equal', 'equal'],
                )
        plotter.add_cbar(tick_params={'labelsize':8, 'pad':1}, label='Tanimoto Score')
        plotter.save_plot('graphs_for_manuscript/SI/similarity_heatmaps/{}'.format(rxn_component))


    # %% Predicted vs. experimental yield scatter plots

    n_subplots = defaultdict(dict)
    n_subplots[3] = {'ncols': 3, 'nrows': 1, 'bbox_to_anchor': (0.5, -0.1)}
    n_subplots[4] = {'ncols': 3, 'nrows': 2, 'bbox_to_anchor': (0.5, 0)}
    n_subplots[5] = {'ncols': 3, 'nrows': 2, 'bbox_to_anchor': (0.5, 0)}
    n_subplots[6] = {'ncols': 3, 'nrows': 2, 'bbox_to_anchor': (0.5, 0)}

    def plot_pred_obs_multiple(dict_df, rxn_component, colours=None, saveas_dir=None, saveas_name=None):
        if saveas_dir is not None:
            if not os.path.exists('{}/SI/pred_vs_obs'.format(saveas_dir)):
                os.makedirs('{}/SI/pred_vs_obs'.format(saveas_dir))

        scores = defaultdict(dict)
        for k, df in dict_df.items():
            mols = df.index.get_level_values(
                rxn_component).drop_duplicates()

            ncols, nrows, bbox_to_anchor = n_subplots[len(mols)].values()

            models = df.drop(columns='yield_exp').columns
            if colours is None:
                colours = custom_cmap(np.linspace(0, 1, len(models)))
                colours = [rgb2hex(c) for c in colours]

            x_line = [[0, 100] for mol in mols]
            x = [df[df.index.isin([mol], level=rxn_component)] for mol in mols]

            x_best_fit = []
            y_best_fit = []
            for x_mol in x:
                x_best_fit_df = []
                y_best_fit_df = []
                for model in models:
                    m, c = np.polyfit(x_mol['yield_exp'], x_mol[model], 1)
                    x_best_fit_df.append(x_mol['yield_exp'])
                    y_best_fit_df.append((m * x_mol['yield_exp']) + c)
                x_best_fit.append(x_best_fit_df)
                y_best_fit.append(y_best_fit_df)

            for n in np.arange(0, (ncols * nrows) - len(x_line)):
                x_line.append(None)
                x.append(None)
                x_best_fit.append(None)

            plotter = plotting(
                rcParams={'font.size': 10, 'axes.titlesize': 10},
                fig_kw={'figsize': (7.5, 5), 'ncols': ncols, 'nrows': nrows,
                        'dpi': 600,  # 'subplot_kw': {'aspect':'equal'}
                        }
                )

            plotter.add_plot(
                x=x_line,
                y=[[0, 100] for mol in mols],
                plot_kw={'linestyle': 'dashed', 'color': 'black',
                         # 'alpha': 0.95,
                         'linewidth': 0.75}
                )

            plotter.add_plot(
                x=x,
                kind='scatter',
                plot_kw=[[{'x': 'yield_exp', 'y': model, 'color': colour,
                          'marker': '.', 's': 2.5, 'alpha': 0.75}
                         for model, colour in zip(models, colours)]
                         for mol in mols],
                text=[
                    {'x': 0.95, 'y': 0.05, 'fontsize': 6, 'ha': 'right',
                     's': '$R^2$\n{}'.format(
                          '\n'.join('{}: {}'.format(
                              model,
                              '{:.2f}'.format(
                                  r2_score(x[n]['yield_exp'], x[n][model])
                                  )
                              ) for model in models)
                          )}
                    for n, mol in enumerate(mols)]
                )

            plotter.add_plot(
                x=x_best_fit,
                y=y_best_fit,
                plot_kw=[[{'linestyle': 'solid', 'linewidth': 1.25,
                           'color': colour, 'alpha': 1, 'label': model}
                         for model, colour in zip(models, colours)]
                         for mol in mols],
                xlim=[(0, 100) for mol in mols],
                ylim=[(-20, 110) for mol in mols],
                title=[
                    '{} {}'.format(
                        ' '.join(rxn_component.split('_')).title(),
                        int(molecule_keys[rxn_component][
                            molecule_keys[rxn_component] == mol].index.values
                            )
                        )
                    for mol in mols
                    ],
                tick_params={'labelsize': 8, 'pad': 1},
                )

            plotter.add_common_axes(
                xlabel='Experimental Yield (%)',
                ylabel='Predicted Yield (%)',
                # lgd_kw={
                #     'bbox_to_anchor': (0.5, -0.015), 'ncol': 3,
                #     'loc': 'center', 'fontsize': 8, 'frameon': False,
                #     'columnspacing': 1.25
                #     }
                )
            plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})

            for n, mol in enumerate(mols):
                scores[k][mol] = defaultdict(dict)
                for model in models:
                    scores[k][mol]['R2'][model] = r2_score(
                        x[n]['yield_exp'],
                        x[n][model]
                        )
                    scores[k][mol]['RMSE'][model] = mean_squared_error(
                        x[n]['yield_exp'],
                        x[n][model],
                        squared=False
                        )

        if saveas_dir is None:
            plotter.save_plot()
        else:
            plotter.save_plot(
                '{}/SI/pred_vs_obs/{}.png'.format(saveas_dir, saveas_name)
                )
        return scores

    y_pred = defaultdict(dict)

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
            y_pred[name][descriptor_model_name] = pd.read_excel(
                file,
                sheet_name=descriptor,
                index_col=[0, 1, 2, 3, 4],
                usecols=['Test Set', 'additive', 'aryl_halide', 'base',
                         'ligand', model]
                )

        df = pd.concat(y_pred[name], axis=1)
        df.columns = df.columns.droplevel(1)
        df = pd.merge(
            df, reactions, on=['additive', 'aryl_halide', 'base', 'ligand'],
            right_index=True
            )

        y_pred[name] = df

    scores = defaultdict()
    for rxn_component in ['additive', 'aryl_halide']:
        test_set_names = y_pred[rxn_component].index.get_level_values(
            'Test Set').drop_duplicates()
        dict_df = defaultdict()
        for test_set_name in test_set_names:
            dict_df[test_set_name] = y_pred[rxn_component][
                y_pred[rxn_component].index.get_level_values(
                    'Test Set') == test_set_name
                ]
        scores[rxn_component] = plot_pred_obs_multiple(
            dict_df, rxn_component, saveas_dir='graphs_for_manuscript',
            saveas_name='{}_{}'.format(rxn_component, test_set_name)
            )

    # %% Distributions of Predicted Yields
    # x=y_pred_best['aryl_halide'].copy()
    # x=x.reset_index('yield_exp')
    # x=x.rename({'yield_exp': 'Experimental Yield'}, axis='columns')
    # plotter = plotting(
    #     rcParams={'font.size':10, 'axes.titlesize':10},
    #     fig_kw={'figsize':(7.5,7.5), 'ncols':1, 'nrows':3, 'dpi':600}
    #     )
    # plotter.add_plot(
    #         x=[x[x.index.get_level_values('aryl_halide').str.contains('chloro')],
    #            x[x.index.get_level_values('aryl_halide').str.contains('bromo')],
    #            x[x.index.get_level_values('aryl_halide').str.contains('iodo')]],
    #         kind='kde',
    #         plot_kw={
    #             'legend':False,
    #             'color':np.append([[0,0,0,0]], custom_cmap(np.linspace(0,1,4)), axis=0), 
    #             'alpha':1
    #             },
    #         title=['Aryl Chlorides', 'Aryl Bromides', 'Aryl Iodides'],
    #         xlim=(-10,110),
    #         tick_params={'labelsize':8, 'pad':1},
    #         )
    # plotter.add_common_axes(
    #         xlabel='Predicted Yield (%)',
    #         ylabel='Kernel Density Estimate',
    #         lgd_kw={
    #             'fontsize':8,
    #             'loc':'center',
    #             'ncol':len(x.columns),
    #             'bbox_to_anchor':(0.5, 0.035),
    #             'frameon':False,
    #             'columnspacing':1.25
    #             },
    #         )
    # plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})
    # plotter.save_plot()

    # %% Prospective reactions: ALL
    if 'validation' not in os.listdir('graphs_for_manuscript'):
        os.mkdir('graphs_for_manuscript/validation')
    if 'validation' not in os.listdir('graphs_for_manuscript/SI'):
        os.mkdir('graphs_for_manuscript/SI/validation')

    prospective_y_pred = defaultdict()
    for descriptor, model in zip(
            ['_one-hot_encodings', 'fingerprints_morgan1_512',
             'tanimoto_kernel_morgan1_512', 'wl_kernel_wl5'],
            ['SVR - Poly Kernel', 'SVR - Poly Kernel',
             'SVR - Precomputed Kernel', 'SVR - Precomputed Kernel']
            ):
        descriptor_name = descriptor_names_dict[descriptor]
        model_name = model.split(' ')[2]
        descriptor_model_name = '{} - {}'.format(
            descriptor_name, model_name
            )
        prospective_y_pred[descriptor_model_name] = pd.read_excel(
                'output/validation_all_ypred.xlsx',
                sheet_name=descriptor,
                index_col=[0, 1, 2, 3],
                usecols=['additive', 'aryl_halide', 'base', 'ligand', model]
                )

    df = pd.concat(prospective_y_pred, axis=1)
    df.columns = df.columns.droplevel(1)
    prospective_y_pred = df

    colours = [
        rgb2hex(c) for c in [
            custom_cmap(np.linspace(0, 1, 5))[0],
            custom_cmap(np.linspace(0, 1, 5))[2],
            custom_cmap(np.linspace(0, 1, 5))[3],
            custom_cmap(np.linspace(0, 1, 5))[4]
            ]
        ]

    # All
    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (7.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600}
        )
    plotter.add_plot(
        x=prospective_y_pred,
        kind='kde',
        plot_kw={
            'legend': True,
            'color': colours,
            'alpha': 1
            },
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        xlim=(-10, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        lgd_kw={
            'fontsize': 8,
            'loc': 'center',
            'ncol': 2,
            'bbox_to_anchor': (0.5, -0.3),
            'frameon': False,
            'columnspacing': 1.25
            }
        )
    plotter.save_plot(
        'graphs_for_manuscript/validation/ypred_distribution_all')

    # Bases
    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5, 7.5), 'ncols': 1, 'nrows': 3, 'dpi': 600}
        )
    plotter.add_plot(
        x=[prospective_y_pred[prospective_y_pred.index.isin(['MTBD'], 'base')],
           prospective_y_pred[prospective_y_pred.index.isin(['BTMG'], 'base')],
           prospective_y_pred[prospective_y_pred.index.isin(['DBU'], 'base')]],
        kind='kde',
        plot_kw={
            'legend': False,
            'color': colours,
            'alpha': 1
            },
        title=['MTBD', 'BTMG', 'DBU'],
        xlim=(-10, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        # grid_kw={
        #         'b':True,
        #         'which':'major',
        #         'axis':'x',
        #         'color':grey,
        #         'linestyle':'-',
        #         'linewidth':0.25,
        #         }
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        lgd_kw={
            'fontsize': 8,
            'loc': 'center',
            'ncol': 2,
            'bbox_to_anchor': (0.5, 0.03),
            'frameon': False,
            'columnspacing': 1.25
            },
        )
    plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})
    plotter.save_plot('graphs_for_manuscript/SI/validation/ypred_bases_all')

    # Aryl halides
    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5, 7.5), 'ncols': 1, 'nrows': 3, 'dpi': 600}
        )
    plotter.add_plot(
        x=[prospective_y_pred[prospective_y_pred.index.get_level_values('aryl_halide').str.contains('Cl')],
           prospective_y_pred[prospective_y_pred.index.get_level_values('aryl_halide').str.contains('Br')],
           prospective_y_pred[prospective_y_pred.index.get_level_values('aryl_halide').str.contains('I')]],
        # x=[x[(x.index.get_level_values('ligand') != 'none') &
        #     (x.index.get_level_values('aryl_halide').str.contains('Cl'))],
        #    x[(x.index.get_level_values('ligand') != 'none') &
        #     (x.index.get_level_values('aryl_halide').str.contains('Br'))],
        #    x[(x.index.get_level_values('ligand') != 'none') &
        #     (x.index.get_level_values('aryl_halide').str.contains('I'))],
        #    ],
        kind='kde',
        plot_kw={
            'legend': False,
            'color': colours,
            'alpha': 1
            },
        title=['Aryl Chlorides', 'Aryl Bromides', 'Aryl Iodides'],
        xlim=(-10, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        lgd_kw={
            'fontsize': 8,
            'loc': 'center',
            'ncol': 2,
            'bbox_to_anchor': (0.5, 0.03),
            'frameon': False,
            'columnspacing': 1.25
            },
        )
    plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})
    plotter.save_plot(
        'graphs_for_manuscript/SI/validation/ypred_aryl_halides_all')

    # Ligands
    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5, 10), 'ncols': 1, 'nrows': 4, 'dpi': 600}
        )
    plotter.add_plot(
        x=[prospective_y_pred[prospective_y_pred.index.isin(['none'], 'ligand')],
           prospective_y_pred[prospective_y_pred.index.isin(['t-BuXPhos'], 'ligand')],
           prospective_y_pred[prospective_y_pred.index.isin(['t-BuBrettPhos'], 'ligand')],
           prospective_y_pred[prospective_y_pred.index.isin(['BrettPhos'], 'ligand')],
           ],
        kind='kde',
        plot_kw={
            'legend': False,
            'color': colours,
            'alpha': 1
            },
        title=['No Ligand', '$t$-BuXPhos', '$t$-BuBrettPhos', 'BrettPhos'],
        xlim=(-10, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        lgd_kw={
            'fontsize': 8,
            'loc': 'center',
            'ncol': 2,
            'bbox_to_anchor': (0.5, 0.035),
            'frameon': False,
            'columnspacing': 1.25
            },
        )
    plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})
    plotter.save_plot('graphs_for_manuscript/SI/validation/ypred_ligands_all')

    # Ligands and aryl halides
    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5, 7.5), 'ncols': 3, 'nrows': 3, 'dpi': 600,
                'sharex': True, 'sharey': True}
        )
    plotter.add_plot(
        x=[prospective_y_pred[
            (prospective_y_pred.index.get_level_values('ligand') != 'none') &
            (prospective_y_pred.index.get_level_values('aryl_halide').str.contains('Cl')) &
            (prospective_y_pred.index.isin(['BrettPhos'], 'ligand'))],
           prospective_y_pred[
            (prospective_y_pred.index.get_level_values('ligand') != 'none') &
            (prospective_y_pred.index.get_level_values('aryl_halide').str.contains('Cl')) &
            (prospective_y_pred.index.isin(['t-BuXPhos'], 'ligand'))],
           prospective_y_pred[
            (prospective_y_pred.index.get_level_values('ligand') != 'none') &
            (prospective_y_pred.index.get_level_values('aryl_halide').str.contains('Cl')) &
            (prospective_y_pred.index.isin(['t-BuBrettPhos'], 'ligand'))],
           prospective_y_pred[
            (prospective_y_pred.index.get_level_values('ligand') != 'none') &
            (prospective_y_pred.index.get_level_values('aryl_halide').str.contains('Br')) &
            (prospective_y_pred.index.isin(['BrettPhos'], 'ligand'))],
           prospective_y_pred[
            (prospective_y_pred.index.get_level_values('ligand') != 'none') &
            (prospective_y_pred.index.get_level_values('aryl_halide').str.contains('Br')) &
            (prospective_y_pred.index.isin(['t-BuXPhos'], 'ligand'))],
           prospective_y_pred[
            (prospective_y_pred.index.get_level_values('ligand') != 'none') &
            (prospective_y_pred.index.get_level_values('aryl_halide').str.contains('Br')) &
            (prospective_y_pred.index.isin(['t-BuBrettPhos'], 'ligand'))],
           prospective_y_pred[
            (prospective_y_pred.index.get_level_values('ligand') != 'none') &
            (prospective_y_pred.index.get_level_values('aryl_halide').str.contains('I')) &
            (prospective_y_pred.index.isin(['BrettPhos'], 'ligand'))],
           prospective_y_pred[
            (prospective_y_pred.index.get_level_values('ligand') != 'none') &
            (prospective_y_pred.index.get_level_values('aryl_halide').str.contains('I')) &
            (prospective_y_pred.index.isin(['t-BuXPhos'], 'ligand'))],
           prospective_y_pred[
            (prospective_y_pred.index.get_level_values('ligand') != 'none') &
            (prospective_y_pred.index.get_level_values('aryl_halide').str.contains('I')) &
            (prospective_y_pred.index.isin(['t-BuBrettPhos'], 'ligand'))],
           ],
        kind='kde',
        plot_kw={
            'legend': False,
            'color': colours,
            'alpha': 1
            },
        title=['Chlorides, BrettPhos', 'Chlorides, t-BuXPhos',
               'Chlorides, t-BuBrettPhos',
               'Bromides, BrettPhos', 'Bromides, t-BuXPhos',
               'Bromides, t-BuBrettPhos',
               'Iodides, BrettPhos', 'Iodides, t-BuXPhos',
               'Iodides, t-BuBrettPhos'],
        xlim=(-10, 110),
        ylim=(0, 0.1),
        tick_params={'labelsize': 8, 'pad': 1},
        grid_kw={
                'b': True,
                'which': 'major',
                'axis': 'x',
                'color': grey,
                'linestyle': '-',
                'linewidth': 0.25,
                }
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        lgd_kw={
            'fontsize': 8,
            'loc': 'center',
            'ncol': 2,
            'bbox_to_anchor': (0.5, 0.03),
            'frameon': False,
            'columnspacing': 1.25
            },
        )
    plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})
    plotter.save_plot()

    # Scatter structure vs. structure
    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (7.5, 2.5), 'ncols': 3, 'nrows': 1, 'dpi': 600,
                'subplot_kw': {'aspect': 'equal'}}
        )
    plotter.add_plot(
        x=[prospective_y_pred['WL - Precomputed'],
           prospective_y_pred['WL - Precomputed'],
           prospective_y_pred['Fps: Morgan1 - Poly']],
        y=[prospective_y_pred['Fps: Morgan1 - Poly'],
           prospective_y_pred['Tan: Morgan1 - Precomputed'],
           prospective_y_pred['Tan: Morgan1 - Precomputed']],
        kind='scatter',
        plot_kw={'color': grey, 'marker': '.', 's': 2.5, 'alpha': 0.75}
        )
    y_best_fit = []
    text = []
    for x, y in zip(
            [prospective_y_pred['WL - Precomputed'],
             prospective_y_pred['WL - Precomputed'],
             prospective_y_pred['Fps: Morgan1 - Poly']],
            [prospective_y_pred['Fps: Morgan1 - Poly'],
             prospective_y_pred['Tan: Morgan1 - Precomputed'],
             prospective_y_pred['Tan: Morgan1 - Precomputed']]
                ):
        m, c = np.polyfit(x, y, 1)
        y_best_fit.append((m * x) + c)
        text.append({
            'x': 0.95, 'y': 0.05, 'fontsize': 6, 'ha': 'right',
            's': 'Pearson r: {:.2f}'.format(pearsonr(x, y)[0])
            })
    plotter.add_plot(
        x=[prospective_y_pred['WL - Precomputed'],
           prospective_y_pred['WL - Precomputed'],
           prospective_y_pred['Fps: Morgan1 - Poly']],
        y=y_best_fit,
        kind='line',
        plot_kw={'linewidth': 1.25, 'alpha': 1, 'color': grey},
        text=text,
        xlim=(-10, 110),
        ylim=(-10, 110),
        xlabel=['WL - Precomputed', 'WL - Precomputed', 'Fps: Morgan1 - Poly'],
        ylabel=['Fps: Morgan1 - Poly', 'Tan: Morgan1 - Precomputed',
                'Tan: Morgan1 - Precomputed'],
        tick_params={'labelsize': 8, 'pad': 0.75},
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Predicted Yield (%)',
        tick_params={'axis': 'y', 'pad': 20},
        )
    plotter.adjust_fig(subplots_adjust_kw={'wspace': 0.4})
    plotter.save_plot(
        'graphs_for_manuscript/SI/validation/morgan_graphs_correlation'
        )

    x = prospective_y_pred.copy()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC(C)C1=CC=CC=C1Cl'], 'aryl_halide'))].mean()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC(C)C1=CC=CC=C1Cl'], 'aryl_halide'))].std()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC(C)C1=CC=CC=C1Br'], 'aryl_halide'))].mean()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC(C)C1=CC=CC=C1Br'], 'aryl_halide'))].std()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC(C)c1ccccc1I'], 'aryl_halide'))].mean()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC(C)c1ccccc1I'], 'aryl_halide'))].std()
    
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC1=NN=C(Cl)C=C1'], 'aryl_halide'))].mean()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC1=NN=C(Cl)C=C1'], 'aryl_halide'))].std()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC1=NN=C(Br)C=C1'], 'aryl_halide'))].mean()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC1=NN=C(Br)C=C1'], 'aryl_halide'))].std()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC1=NN=C(I)C=C1'], 'aryl_halide'))].mean()
    x[(~x.index.isin(['none'], 'ligand')) & (x.index.isin(['CC1=NN=C(I)C=C1'], 'aryl_halide'))].std()
    
    # np.mean(rxn_smiles_org[rxn_smiles_org.index.isin(['2-chloropyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.std(rxn_smiles_org[rxn_smiles_org.index.isin(['2-chloropyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.mean(rxn_smiles_org[rxn_smiles_org.index.isin(['2-bromopyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.std(rxn_smiles_org[rxn_smiles_org.index.isin(['2-bromopyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.mean(rxn_smiles_org[rxn_smiles_org.index.isin(['2-iodopyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.std(rxn_smiles_org[rxn_smiles_org.index.isin(['2-iodopyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    
    # np.mean(rxn_smiles_org[rxn_smiles_org.index.isin(['3-chloropyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.std(rxn_smiles_org[rxn_smiles_org.index.isin(['3-chloropyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.mean(rxn_smiles_org[rxn_smiles_org.index.isin(['3-bromopyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.std(rxn_smiles_org[rxn_smiles_org.index.isin(['3-bromopyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.mean(rxn_smiles_org[rxn_smiles_org.index.isin(['3-iodopyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))
    # np.std(rxn_smiles_org[rxn_smiles_org.index.isin(['3-iodopyridine'], 'aryl_halide')].index.get_level_values('yield_exp'))

    # %% Prospective Reactions: Yield Distributions
    prospective_key = defaultdict()
    prospective_key['ligand'] = {
        'L$_0$': 'none',
        'L$_1$': 't-BuXPhos',
        'L$_2$': 't-BuBrettPhos',
        'L$_3$': 'BrettPhos',
        }
    prospective_key['base'] = {
        'B$_1$': 'BTMG',
        'B$_2$': 'MTBD',
        'B$_3$': 'DBU',
        }
    prospective_key['aryl_halide'] = {
        'H$_{1}$': 'CC(=O)Cc1ccccc1Cl', 'H$_{2}$': 'CC1=C(Cl)C=CC(F)=C1',
        'H$_{3}$': 'CC(C)C1=CC=CC=C1Cl', 'H$_{4}$': 'OCc1ccccc1Cl',
        'H$_{5}$': 'ClC1=C(C=CC=C1)C#N', 'H$_{6}$': 'ClC1=CC=CC=C1C2=CC=CC=C2',
        'H$_{7}$': 'CC1=NN=C(Cl)C=C1', 'H$_{8}$': 'CN(C)C1=CC=CC(Cl)=C1',
        'H$_{9}$': 'Clc1cccc(c1)C#N', 'H$_{10}$': 'Clc1ccc(cc1)N(=O)=O',
        'H$_{11}$': 'CCOc1cccc(Cl)c1', 'H$_{12}$': 'Clc1ccc(cc1)C#N',
        'H$_{13}$': 'CC(O)c1ccc(Cl)cc1', 'H$_{14}$': 'CC(=O)c1ccc(Cl)cc1',
        'H$_{15}$': 'Clc1cccc2cccnc12', 'H$_{16}$': 'Clc1ccc(CC#N)cc1',
        'H$_{17}$': 'FC(F)(F)C1=CC=C(Cl)C=N1', 'H$_{18}$': 'CCC1=CC(Cl)=CC=C1',
        'H$_{19}$': 'OCc1ccc(Cl)cc1', 'H$_{20}$': 'FC(F)(F)c1ccc(Cl)cc1',
        'H$_{21}$': 'Clc1ccccn1', 'H$_{22}$': 'Clc1cccnc1',
        'H$_{23}$': 'FC1=CC=CC(Br)=C1C#N', 'H$_{24}$': 'Cc1c(Cl)cccc1Br',
        'H$_{25}$': 'CNC(=O)c1ccccc1Br', 'H$_{26}$': 'CC(=O)Cc1ccccc1Br',
        'H$_{27}$': 'C[C@@H](O)c1ccccc1Br', 'H$_{28}$': 'Oc1ccnc2ccc(Br)cc12',
        'H$_{29}$': 'Cc1cc(F)ccc1Br', 'H$_{30}$': 'Oc1ccc(Br)c2ccccc12',
        'H$_{31}$': 'BrC1=C(C=CC=C1)C#N', 'H$_{32}$': 'CC(C)C1=CC=CC=C1Br',
        'H$_{33}$': 'CN(C)C1=CC=CC=C1Br', 'H$_{34}$': 'CC(=O)c1cccc(Br)c1',
        'H$_{35}$': 'Brc1ccccc1-c1ccccc1', 'H$_{36}$': 'CC1=NN=C(Br)C=C1',
        'H$_{37}$': 'CCOc1ccccc1Br', 'H$_{38}$': 'CN(C)c1cccc(Br)c1',
        'H$_{39}$': 'Brc1cccc(c1)C#N', 'H$_{40}$': 'CC(C)Oc1cccc(Br)c1',
        'H$_{41}$': 'NC(=O)c1ccc(Br)nc1', 'H$_{42}$': 'CC(=O)c1cncc(Br)c1',
        'H$_{43}$': 'CCOc1cccc(Br)c1', 'H$_{44}$': 'OC1=CC(Br)=CC=C1',
        'H$_{45}$': 'BrC1=CC=C(C=C1)N(=O)=O',
        'H$_{46}$': 'FC(F)(F)C1=CC=CC=C1Br', 'H$_{47}$': 'OC1=CC=C(Br)C=C1',
        'H$_{48}$': 'CN(C)c1ccc(Br)cc1', 'H$_{49}$': 'FC(F)(F)C1=CC=C(Br)C=N1',
        'H$_{50}$': 'FC(F)(F)c1cccc(Br)c1', 'H$_{51}$': 'CCc1ccc(Br)cc1',
        'H$_{52}$': 'Fc1cccc(I)c1C#N', 'H$_{53}$': 'OCc1ccccc1I',
        'H$_{54}$': 'CC(C)c1ccccc1I', 'H$_{55}$': 'CC1=NN=C(I)C=C1',
        'H$_{56}$': 'Ic1cccc(CC#N)c1', 'H$_{57}$': 'Ic1ccc(cc1)C#N',
        'H$_{58}$': 'CCC1=CC=CC(I)=C1', 'H$_{59}$': 'FC(F)(F)c1ccc(I)cc1',
        }
    prospective_key['model'] = {
        0: 'One-hot - Poly',
        1: 'Quantum - RBF',
        2: 'Fps: Morgan1 - Poly',
        3: 'Tan: Morgan1 - Precomputed',
        4: 'WL - Precomputed'
        }

    for k, v in prospective_key.items():
        prospective_key[k] = pd.Series(v, name=k)

    prospective_key['aryl_halide'] = pd.DataFrame(
        prospective_key['aryl_halide'], columns=['aryl_halide'])
    prospective_key['aryl_halide']['halide'] \
        = prospective_key['aryl_halide'].aryl_halide.map(get_halide_type)
    # prospective_key['aryl_halide'] \
    #     = prospective_key['aryl_halide'].reset_index().set_index(
    #         'halide').loc[
    #             ['Chlorides', 'Bromides', 'Iodides']
    #             ].reset_index()
    # prospective_key['aryl_halide'].index \
    #     = prospective_key['aryl_halide'].index.map(
    #         lambda x: 'H$_{{{}}}$'.format(x+1)
    #         )
    # prospective_key['aryl_halide']['name'] \
    #     = prospective_key['aryl_halide']['aryl_halide'].map(
    #         lambda x: rd.name_from_smiles(x)
    #         )
    
    original_aryl_halides = {
        'FC(F)(F)c1ccc(I)cc1': '1-iodo-4-(trifluoromethyl)benzene',
        'FC(F)(F)c1ccc(Cl)cc1': '1-chloro-4-(trifluoromethyl)benzene',
        'CCc1ccc(Br)cc1': '1-bromo-4-ethylbenzene',
        'Clc1ccccn1': '2-chloropyridine',
        'Clc1cccnc1': '3-chloropyridine'
        }

    for additive in prospective_y_pred.index.get_level_values(
            'additive').drop_duplicates():
        data = pd.DataFrame(
            prospective_y_pred[
                prospective_y_pred.index.isin([additive], 'additive')
                ]
            )
        data.columns.rename('model', inplace=True)
        plot_yield_heatmap(
            data,
            additive,
            name='all',
            saveas_dir='graphs_for_manuscript'
            )

    temp = prospective_y_pred.copy()
    temp['aryl_halide'] = prospective_y_pred.index.get_level_values(
        'aryl_halide').map(
            lambda x: original_aryl_halides[x]
            if x in original_aryl_halides.keys()
            else np.nan
            )
    temp.index = temp.index.droplevel('aryl_halide')
    temp.set_index('aryl_halide', inplace=True, append=True)
    temp = temp.reorder_levels(
        ['additive', 'aryl_halide', 'base', 'ligand'], 'index')
    temp = temp.loc[temp.index.dropna()]
    temp = reactions.merge(temp, left_index=True, right_index=True)

    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (3.5, 3.5), 'ncols': 1, 'nrows': 1, 'dpi': 600}
        )

    plotter.add_plot(
        x=[0, 100], y=[0, 100],
        plot_kw={'linestyle': 'dashed', 'color': 'black', 'linewidth': 0.75}
        )

    plotter.add_plot(
        x=temp,
        kind='scatter',
        plot_kw=[
            {'x': 'yield_exp', 'y': model,
             'color': colours[n],
             'label': model, 'marker': 'o', 's': 10, 'alpha': 0.75,
             }
            for n, model in enumerate(prospective_y_pred.columns)
            ],
        xlim=(0, 100),
        ylim=(-20, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        xlabel='Experimental Yield (%)',
        ylabel='Predicted Yield (%)',
        text=[
            {'x': 0.8, 'y': 0.05, 'fontsize': 8, 'ha': 'right',
             's': '$R^2$\n{}'.format('\n'.join(
                 '{}:  {}'.format(
                    model,
                    '{:.2f}'.format(r2_score(temp['yield_exp'], temp[model])),
                    )
                 for model in prospective_y_pred.columns
                 ))},
            {'x': 0.95, 'y': 0.05, 'fontsize': 8, 'ha': 'right',
             's': 'RMSE\n{}'.format('\n'.join(
                 '{}'.format(
                    '{:.1f}'.format(
                        mean_squared_error(temp['yield_exp'],
                                           temp[model],
                                           squared=False))
                    )
                 for model in prospective_y_pred.columns
                 ))}
            ],
        lgd_kw={
            'bbox_to_anchor': (0.5, -0.12),
            'ncol': 2,
            'loc': 'upper center',
            'fontsize': 8,
            'frameon': False,
            'columnspacing': 1.25
            }
        )

    plotter.save_plot('graphs_for_manuscript/SI/validation/predVobs_all')

    # %% Prospective reactions: SUBSET
    prospective_y_pred_subset = defaultdict()
    for descriptor, model in zip(
            ['_one-hot_encodings', '_quantum', 'fingerprints_morgan1_512',
             'tanimoto_kernel_morgan1_512', 'wl_kernel_wl5'],
            ['SVR - Poly Kernel', 'SVR - RBF Kernel', 'SVR - Poly Kernel',
             'SVR - Precomputed Kernel', 'SVR - Precomputed Kernel']
            ):
        descriptor_name = descriptor_names_dict[descriptor]
        model_name = model.split(' ')[2]
        descriptor_model_name = '{} - {}'.format(
            descriptor_name, model_name
            )
        prospective_y_pred_subset[descriptor_model_name] = pd.read_excel(
                'output/validation_subset_ypred.xlsx',
                sheet_name=descriptor,
                index_col=[0, 1, 2, 3],
                usecols=['additive', 'aryl_halide', 'base', 'ligand', model]
                )

    df = pd.concat(prospective_y_pred_subset, axis=1)
    df.columns = df.columns.droplevel(1)
    prospective_y_pred_subset = df

    colours = [rgb2hex(c) for c in custom_cmap(np.linspace(0, 1, 5))]

    # Subset
    x_subset = prospective_y_pred_subset.copy()

    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (7.5, 2.5), 'ncols': 1, 'nrows': 1, 'dpi': 600}
        )
    plotter.add_plot(
        x=x_subset,
        kind='kde',
        plot_kw={
            'legend': True,
            'color': colours,
            'alpha': 1
            },
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        xlim=(-10, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        lgd_kw={
            # 'markerscale':1.5, 'scatteryoffsets':[1],
            'fontsize': 8,
            'loc': 'center',
            'ncol': 3,
            'bbox_to_anchor': (0.5, -0.3),
            'frameon': False,
            'columnspacing': 1.25
            }
        )
    plotter.save_plot(
        'graphs_for_manuscript/validation/ypred_distribution_subset')

    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5, 5.5), 'ncols': 1, 'nrows': 2, 'dpi': 600}
        )
    plotter.add_plot(
        x=[x_subset[
            (x_subset.index.get_level_values('ligand') != 'none') &
            (x_subset.index.get_level_values('aryl_halide').str.contains('Cl'))
            ],
           x_subset[
            (x_subset.index.get_level_values('ligand') != 'none') &
            (x_subset.index.get_level_values('aryl_halide').str.contains('Br'))
            ]
           ],
        kind='kde',
        plot_kw={
            'legend': False,
            'color': colours,
            'alpha': 1
            },
        title=['Aryl Chlorides', 'Aryl Bromides'],
        xlim=(-10, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        lgd_kw={
            'fontsize': 8,
            'loc': 'center',
            'ncol': 3,
            'bbox_to_anchor': (0.5, 0.025),
            'frameon': False,
            'columnspacing': 1.25
            },
        )
    plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})
    plotter.save_plot(
        'graphs_for_manuscript/SI/validation/ypred_aryl_halides_subset')

    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5, 7.5), 'ncols': 1, 'nrows': 3, 'dpi': 600}
        )
    plotter.add_plot(
        x=[x_subset[x_subset.index.isin(['none'], 'ligand')][
            ['One-hot - Poly', 'Fps: Morgan1 - Poly',
             'Tan: Morgan1 - Precomputed', 'WL - Precomputed']],
           x_subset[x_subset.index.isin(['t-BuXPhos'], 'ligand')],
           x_subset[x_subset.index.isin(['t-BuBrettPhos'], 'ligand')]],
        kind='kde',
        plot_kw=[{'legend': False, 'color': [
            colours[0], colours[2], colours[3], colours[4]
            ], 'alpha': 1},
                 {'legend': False, 'color': colours, 'alpha': 1},
                 {'legend': False, 'color': colours, 'alpha': 1}],
        title=['No Ligand', '$t$-BuXPhos', '$t$-BuBrettPhos'],
        xlim=(-10, 110),
        tick_params={'labelsize': 8, 'pad': 1}
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        lgd_kw={
            'fontsize': 8,
            'loc': 'center',
            'ncol': 3,
            'bbox_to_anchor': (0.5, 0.030),
            'frameon': False,
            'columnspacing': 1.25
            },
        )
    plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})
    plotter.save_plot(
        'graphs_for_manuscript/SI/validation/ypred_ligands_subset')

    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5, 7.5), 'ncols': 1, 'nrows': 3, 'dpi': 600}
        )
    plotter.add_plot(
        x=[x_subset[x_subset.index.isin(['MTBD'], 'base')],
           x_subset[x_subset.index.isin(['BTMG'], 'base')],
           x_subset[x_subset.index.isin(['DBU'], 'base')]],
        kind='kde',
        plot_kw={'legend': False, 'color': colours, 'alpha': 1},
        title=['MTBD', 'BTMG', 'DBU'],
        xlim=(-10, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        # grid_kw={
        #         'b': True,
        #         'which': 'major',
        #         'axis': 'x',
        #         'color': grey,
        #         'linestyle': '-',
        #         'linewidth': 0.25,
        #         }
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        lgd_kw={
            'fontsize': 8,
            'loc': 'center',
            'ncol': 3,
            'bbox_to_anchor': (0.5, 0.03),
            'frameon': False,
            'columnspacing': 1.25
            },
        )
    plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})
    plotter.save_plot('graphs_for_manuscript/SI/validation/ypred_bases_subset')

    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (7.5, 5.5), 'ncols': 1, 'nrows': 2, 'dpi': 600}
        )
    plotter.add_plot(
        x=[x_subset[x_subset.index.isin(['3-methylisoxazole'], 'additive')],
           x_subset[x_subset.index.isin(['none'], 'additive')]],
        kind='kde',
        plot_kw={'legend': False, 'color': colours, 'alpha': 1},
        title=['3-methylisoxazole', 'No Additive'],
        xlim=(-10, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        # grid_kw={
        #         'b': True,
        #         'which': 'major',
        #         'axis': 'x',
        #         'color': grey,
        #         'linestyle': '-',
        #         'linewidth': 0.25,
        #         }
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Kernel Density Estimate',
        lgd_kw={
            'fontsize': 8,
            'loc': 'center',
            'ncol': 3,
            'bbox_to_anchor': (0.5, 0.025),
            'frameon': False,
            'columnspacing': 1.25
            },
        )
    plotter.adjust_fig(subplots_adjust_kw={'hspace': 0.25})
    plotter.save_plot(
        'graphs_for_manuscript/SI/validation/ypred_additives_subset')

    # Scatter structure vs. structure
    plotter = plotting(
        rcParams={'font.size': 10},
        fig_kw={'figsize': (7.5, 5), 'ncols': 3, 'nrows': 2, 'dpi': 600,
                'subplot_kw': {'aspect': 'equal'}}
        )
    x = [x_subset['Quantum - RBF'],
         x_subset['Quantum - RBF'],
         x_subset['Quantum - RBF'],
         x_subset['WL - Precomputed'],
         x_subset['WL - Precomputed'],
         x_subset['Fps: Morgan1 - Poly']]
    y = [x_subset['WL - Precomputed'],
         x_subset['Fps: Morgan1 - Poly'],
         x_subset['Tan: Morgan1 - Precomputed'],
         x_subset['Fps: Morgan1 - Poly'],
         x_subset['Tan: Morgan1 - Precomputed'],
         x_subset['Tan: Morgan1 - Precomputed']]
    y_best_fit = []
    text = []
    for xx, yy in zip(x, y):
        m, c = np.polyfit(xx, yy, 1)
        y_best_fit.append((m * xx) + c)
        text.append({
            'x': 0.95, 'y': 0.05, 'fontsize': 6, 'ha': 'right',
            's': 'Pearson r: {:.2f}'.format(pearsonr(xx, yy)[0])
            })
    plotter.add_plot(
        x=x,
        y=y,
        kind='scatter',
        plot_kw={'color': grey, 'marker': '.', 's': 2.5, 'alpha': 0.75},
        # )
    # plotter.add_plot(
    #     x=x,
    #     y=y_best_fit,
    #     kind='line',
    #     plot_kw={'linewidth': 1.25, 'alpha': 1, 'color': grey},
        text=text,
        xlim=(-10, 110),
        ylim=(-10, 110),
        xlabel=['Quantum - RBF', 'Quantum - RBF', 'Quantum - RBF',
                'WL - Precomputed', 'WL - Precomputed', 'Fps: Morgan1 - Poly'],
        ylabel=['WL - Precomputed', 'Fps: Morgan1 - Poly',
                'Tan: Morgan1 - Precomputed', 'Fps: Morgan1 - Poly',
                'Tan: Morgan1 - Precomputed', 'Tan: Morgan1 - Precomputed'],
        tick_params={'labelsize': 8, 'pad': 0.75},
        )
    plotter.add_common_axes(
        xlabel='Predicted Yield (%)',
        ylabel='Predicted Yield (%)',
        tick_params={'axis': 'y', 'pad': 20},
        )
    plotter.adjust_fig(subplots_adjust_kw={'wspace': 0.4})
    plotter.save_plot(
        'graphs_for_manuscript/SI/validation/quantum_structure_correlation'
        )

    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC(C)C1=CC=CC=C1Cl'], 'aryl_halide'))].mean()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC(C)C1=CC=CC=C1Cl'], 'aryl_halide'))].std()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC(C)C1=CC=CC=C1Br'], 'aryl_halide'))].mean()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC(C)C1=CC=CC=C1Br'], 'aryl_halide'))].std()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC(C)c1ccccc1I'], 'aryl_halide'))].mean()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC(C)c1ccccc1I'], 'aryl_halide'))].std()
    
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC1=NN=C(Cl)C=C1'], 'aryl_halide'))].mean()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC1=NN=C(Cl)C=C1'], 'aryl_halide'))].std()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC1=NN=C(Br)C=C1'], 'aryl_halide'))].mean()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC1=NN=C(Br)C=C1'], 'aryl_halide'))].std()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC1=NN=C(I)C=C1'], 'aryl_halide'))].mean()
    x_subset[(~x_subset.index.isin(['none'], 'ligand')) & (x_subset.index.isin(['CC1=NN=C(I)C=C1'], 'aryl_halide'))].std()

    x_subset = x_subset.reindex(
        x_subset.index.to_list() +
        [ind for ind in prospective_y_pred.index if ind not in x_subset.index]
        )

    prospective_key['model'][3] = 'Tan:\nMorgan1 - Precomputed'
    for additive in x_subset.index.get_level_values('additive').drop_duplicates():
        data = pd.DataFrame(x_subset[x_subset.index.isin([additive], 'additive')])
        data.columns.rename('model', inplace=True)
        data = data.rename(columns={'Tan: Morgan1 - Precomputed': 'Tan:\nMorgan1 - Precomputed'})
        data=test.dropna()
        plot_yield_heatmap(
            data,
            additive,
            name='subset',
            saveas_dir='graphs_for_manuscript'
            )

    temp = x_subset.copy()
    temp['aryl_halide'] = x_subset.index.get_level_values('aryl_halide').map(
        lambda x: original_aryl_halides[x]
        if x in original_aryl_halides.keys()
        else np.nan
        )
    temp.index = temp.index.droplevel('aryl_halide')
    temp.set_index('aryl_halide', inplace=True, append=True)
    temp = temp.reorder_levels(['additive', 'aryl_halide', 'base', 'ligand'], 'index')
    temp = temp.loc[temp.index.dropna()]
    temp = reactions.merge(temp, left_index=True, right_index=True)
    temp = temp.dropna()

    plotter = plotting(
        rcParams={'font.size': 10, 'axes.titlesize': 10},
        fig_kw={'figsize': (3.5, 3.5), 'ncols': 1, 'nrows': 1, 'dpi': 600}
        )

    plotter.add_plot(
        x=[0, 100], y=[0, 100],
        plot_kw={'linestyle': 'dashed', 'color': 'black', 'linewidth': 0.75}
        )

    plotter.add_plot(
        x=temp,
        kind='scatter',
        plot_kw=[
            {'x': 'yield_exp', 'y': model,
             'color': colours[n],
             'label': model, 'marker': 'o', 's': 10, 'alpha': 0.75,
             }
            for n, model in enumerate(x_subset.columns)
            ],
        xlim=(0, 100),
        ylim=(-20, 110),
        tick_params={'labelsize': 8, 'pad': 1},
        xlabel='Experimental Yield (%)',
        ylabel='Predicted Yield (%)',
        text=[
            {'x': 0.8, 'y': 0.05, 'fontsize': 8, 'ha': 'right',
             's': '$R^2$\n{}'.format('\n'.join(
                 '{}:  {}'.format(
                    model,
                    '{:.2f}'.format(r2_score(temp['yield_exp'], temp[model])),
                    )
                 for model in x_subset.columns
                 ))},
            {'x': 0.95, 'y': 0.05, 'fontsize': 8, 'ha': 'right',
             's': 'RMSE\n{}'.format('\n'.join(
                 '{}'.format(
                    '{:.1f}'.format(
                        mean_squared_error(
                            temp['yield_exp'],
                            temp[model],
                            squared=False))
                    )
                 for model in x_subset.columns
                 ))}
            ],
        lgd_kw={
            'bbox_to_anchor': (0.5, -0.12),
            'ncol': 3,
            'loc': 'upper center',
            'fontsize': 8,
            'frameon': False,
            'columnspacing': 1.25
            }
        )

    plotter.save_plot('graphs_for_manuscript/SI/validation/predVobs_subset')
