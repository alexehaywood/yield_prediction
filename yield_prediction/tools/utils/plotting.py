# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:54:56 2020

@author: alexe
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from collections.abc import Iterable
import pandas as pd

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

class plotting:
    
    def __init__(self, rcParams={}, fig_kw={}):
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update(rcParams)
        
        self.fig, self.axes = plt.subplots(**fig_kw)
        
        self.lines = []
                
    def plot(self, ax, x, y=None, kind='line', plot_kw={}):
        
        if isinstance(x, pd.DataFrame):
            if (kind == 'line') or (kind == 'scatter'):
                if type(plot_kw) is list:
                    for n in np.arange(0, len(plot_kw)):
                        self.graph = x.plot(kind=kind, ax=ax,  **plot_kw[n])
                else:
                    self.graph = x.plot(kind=kind, ax=ax,  **plot_kw)
            elif kind == 'hist' or kind == 'bar' or kind == 'kde':
                self.graph = x.plot(kind=kind, ax=ax,  **plot_kw)
            
        elif any(isinstance(i, Iterable) for i in x) and (kind not in ['hist', 'imshow']):
            if type(plot_kw) is dict:
                plot_kw=[plot_kw for i in x]
                
            for n in np.arange(0, len(x)):
                if kind == 'line':
                    self.graph = ax.plot(x[n], y[n], **plot_kw[n])
                elif kind == 'scatter':
                    self.graph = ax.scatter(x[n], y[n], **plot_kw[n])
                elif kind == 'bar':
                    self.graph = ax.bar(x[n], y[n], **plot_kw[n])
                elif kind == 'errorbar':
                    self.graph = ax.errorbar(x[n], y[n], **plot_kw[n])

        else:
            if kind == 'line':
                self.graph = ax.plot(x, y, **plot_kw)
            elif kind == 'scatter':
                self.graph = ax.scatter(x, y, **plot_kw)
            elif kind == 'bar':
                self.graph = ax.bar(x, y, **plot_kw)
            elif kind == 'errorbar':
                self.graph = ax.errorbar(x, y, **plot_kw)
            elif kind == 'hist':
                self.graph = ax.hist(x, **plot_kw)
            elif kind == 'imshow':
                self.graph = ax.imshow(x, **plot_kw)
        
        return ax
    
    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                         textcolors=["black", "white"],
                         threshold=None, textkw={}):
        """
        A function to annotate a heatmap.
    
        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A list or array of two color specifications.  The first is used for
            values below a threshold, the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """
    
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()
    
        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.
    
        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(**textkw)
    
        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = ticker.StrMethodFormatter(valfmt)
    
        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
        
    def setup_axes(
            self, ax,
            xlabel=None, ylabel=None, title=None,
            xlim=(None, None), ylim=(None, None),
            tick_params={}, 
            xticks=None, yticks=None,
            xticklabels=None, yticklabels=None, 
            aspect=None, text=None, annotate_heatmap=False,
            grid_kw={}, lgd_kw=None
            ):
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        ax.set_title(title)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        if type(xticks) is dict:
            ax.set_xticks(**xticks)
        elif xticks is not None:
            for i in xticks:
                ax.set_xticks(**i)
        if type(yticks) is dict:
            ax.set_yticks(**yticks)
        elif yticks is not None:
            for i in yticks:
                ax.set_yticks(**i)
            
        if (xticklabels is not None) and (type(xticklabels) is dict):
            ax.set_xticklabels(**xticklabels)
        if (yticklabels is not None) and (type(yticklabels) is dict):
            ax.set_yticklabels(**yticklabels)
            
        if type(tick_params) is dict:
            ax.tick_params(**tick_params)
        else:
            for i in tick_params:
                ax.tick_params(**i)
        
        if aspect is not None:
            ax.set_aspect(aspect)
            
        if type(text) is dict:
            ax.text(transform=ax.transAxes, **text)
        elif text is not None:
            for i in text:
                ax.text(transform=ax.transAxes, **i)
        
        if annotate_heatmap is True:
            self.annotate_heatmap(self.graph)
        elif type(annotate_heatmap) is dict:
            self.annotate_heatmap(self.graph, **annotate_heatmap)
        
        if type(grid_kw) is dict:
            if  grid_kw:
                ax.grid(**grid_kw)
        else:
            for i in grid_kw:
                ax.grid(**i)
        
        if (lgd_kw is not None) and (type(lgd_kw) is dict):
            self.lgd = ax.legend(**lgd_kw)
        else:
            self.lgd = None
            
        return ax
    
    def add_plot(
            self, x, y=None, kind='line', plot_kw={},
            xlabel=None, ylabel=None, title=None, 
            xlim=None, ylim=None, 
            tick_params={}, 
            xticks=None, yticks=None,
            xticklabels=None, yticklabels=None, 
            aspect=None, text=None, annotate_heatmap=False,
            grid_kw={}, lgd_kw=None
            ):
        
        if type(self.axes) is not np.ndarray:
            self.axes = self.plot(
                self.axes, x, y, kind, plot_kw
                )
            self.axes = self.setup_axes(
                self.axes, xlabel, ylabel, title, xlim, ylim, 
                tick_params, xticks, yticks, xticklabels, yticklabels,
                aspect, text, annotate_heatmap, grid_kw, lgd_kw
                )
        
        else:
            if y is None:
                y = [y for i in self.axes.flatten()]
            if type(plot_kw) is dict:
                plot_kw = [plot_kw for i in self.axes.flatten()]
            if type(kind) is str:
                kind = [kind for i in self.axes.flatten()]
            
            if xlabel is None:
               xlabel = [None for i in self.axes.flatten()]
            if ylabel is None:
                ylabel = [None for i in self.axes.flatten()]
            if type(xlabel) is str:
                xlabel = [xlabel for i in self.axes.flatten()]
            if type(ylabel) is str:
                ylabel = [ylabel for i in self.axes.flatten()]
                
            if title is None:
                title = [None for i in self.axes.flatten()]
            if type(title) is str:
                title = [title for i in self.axes.flatten()]
            
            if xlim is None:
                xlim = [None for i in self.axes.flatten()]
            if ylim is None:
                ylim = [None for i in self.axes.flatten()]
            if type(xlim) is not list:
                xlim = [xlim for i in self.axes.flatten()]
            if type(ylim) is not list:
                ylim = [ylim for i in self.axes.flatten()]
                
            if type(tick_params) is dict:
                tick_params = [tick_params for i in self.axes.flatten()]
            if xticks is None:
                xticks = [xticks for i in self.axes.flatten()]
            if yticks is None:
                yticks = [yticks for i in self.axes.flatten()]
            if xticklabels is None:
                xticklabels = [xticklabels for i in self.axes.flatten()]
            if yticklabels is None:
                yticklabels = [yticklabels for i in self.axes.flatten()]
                
            if aspect is None:
                aspect = [aspect for i in self.axes.flatten()]
                
            if (text is None) or (type(text) is dict):
                text = [text for i in self.axes.flatten()]
            
            if type(annotate_heatmap) is not list:
                annotate_heatmap = [annotate_heatmap for i in self.axes.flatten()]
                
            
            if type(grid_kw) is dict:
                grid_kw = [grid_kw for i in self.axes.flatten()]
                                
            if lgd_kw is None:
                lgd_kw = [None for i in self.axes.flatten()]
            if type(lgd_kw) is dict:
                lgd_kw = [lgd_kw for i in self.axes.flatten()]
                
            n=0
            for index, ax in np.ndenumerate(self.axes):
                if x[n] is not None:
                    print(n,x[n])
                    np.put(
                        self.axes, 
                        index, 
                        self.plot(ax, x[n], y[n], kind[n], plot_kw[n])
                        )
                    np.put(
                        self.axes, 
                        index, 
                        self.setup_axes(
                            ax, xlabel[n], ylabel[n], title[n], xlim[n], ylim[n],
                            tick_params[n], xticks[n], yticks[n], 
                            xticklabels[n], yticklabels[n], 
                            aspect[n], text[n], annotate_heatmap[n],
                            grid_kw[n], lgd_kw[n]
                            )
                        )
                n=n+1        
        
            for ax in self.axes.flatten():
                if not ax.has_data():
                    ax.remove()
                    self.axes = np.array(self.fig.axes)
                    
    def add_cbar(self, tick_params={}, label=None):
        if type(self.axes) is not np.ndarray:
            cbar = self.fig.colorbar(self.graph, ax=self.axes)
        else:
            cbar = self.fig.colorbar(self.graph, ax=self.axes.ravel().tolist())
        cbar.ax.tick_params(**tick_params)
        cbar.set_label(label)
        
    def add_common_axes(self, xlabel=None, ylabel=None, title=None, 
                        tick_params={}, lgd_kw=None):
        
        if (lgd_kw is not None) and (type(lgd_kw) is dict):
            
            handles = []
            labels = []
            for ax in self.fig.axes:
                handle, label = ax.get_legend_handles_labels()
                # for i in label:
                #     if i not in labels:
                #         handles.extend(handle)
                #         labels.extend(label)
                for l, h in zip(label, handle):
                    if l not in labels:
                        handles.append(h)
                        labels.append(l)
            
            self.lgd = self.fig.legend(handles=handles, labels=labels, **lgd_kw)
            
        else:
            self.lgd = None
        
        self.common_axes = self.fig.add_subplot(111, frame_on=False)
        self.common_axes.tick_params(labelcolor='none', bottom=False, 
                                     left=False)
        if type(tick_params) is dict:
            self.common_axes.tick_params(**tick_params)
        else:
            for i in tick_params:
                self.common_axes.tick_params(**i)
        
        self.common_axes.set_xlabel(xlabel)
        self.common_axes.set_ylabel(ylabel)
        self.common_axes.set_title(title)       
        
    def adjust_fig(self, title=None, subplots_adjust_kw={}):
        self.fig.suptitle(title)
        self.fig.subplots_adjust(**subplots_adjust_kw)
        
    
    def save_plot(self, saveas=None, saveas_kw={'bbox_inches':'tight'}):
        if saveas is None:
            plt.show()
        elif self.lgd is not None:
            self.fig.savefig(
                '{}.png'.format(saveas), 
                bbox_extra_artists=(self.lgd,), 
                **saveas_kw
                )
        else:
            self.fig.savefig('{}.png'.format(saveas), **saveas_kw)
        
        plt.close()

# plotter = plotting()
# plotter.add_plot(
#     x=np.arange(0,1,0.25), 
#     y=np.arange(0,1,0.25), 
#     kind='scatter', 
#     plot_kw={'color':'red', 'label':'red line'}
#     ) 
# plotter.add_plot(
#     x=np.arange(0,1,0.5), 
#     y=np.arange(0,1,0.5), 
#     kind='scatter', 
#     plot_kw={'color':'blue', 'label':'blue line'}, 
#     lgd_kw={}
#     ) 
# plotter.save_plot()

# plotter = plotting(fig_kw={'ncols':2})
# plotter.add_plot(
#     x=[np.arange(0,1,0.25), np.arange(0,1,0.5)], 
#     y=[np.arange(0,1,0.25),  np.arange(0,1,0.5)], 
#     kind=['scatter', 'line'], 
#     plot_kw=[{'color':'red'}, {'color':'blue'}]
#     ) 
# plotter.save_plot()

# plotter = plotting(fig_kw={'ncols':2})
# plotter.add_plot(
#     x=[[np.arange(0,1,0.25), np.arange(0,1,0.5)], np.arange(0,1,0.5)], 
#     y=[[np.arange(0,1,0.25), np.arange(0,1,0.5)], np.arange(0,1,0.5)], 
#     kind=['scatter', 'line'], 
#     plot_kw=[[{'color':'red', 'label':'red dot'}, {'color':'green', 'label':'green dot'}], {'color':'blue', 'label':'blue line'}]
#     ) 
# plotter.add_common_axes('x', 'y', 'title', lgd_kw={'bbox_to_anchor':(0.5, 0.5)})
# plotter.save_plot()
