#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:37:51 2024

@author: amounier
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import random as rd


def plot_timeserie(data,figsize=(5,5),dpi=300,labels=None,figs_folder=None,
                   save_fig=None,show=True,xlim=None,ylim_bottom=None,ylim_top=None,ylabel=None,
                   legend_loc=None,figax=None,colors=None,linestyles=None,**kwargs):
    """
    Fonction d'affichage de séries temporelles

    """
    if figax is None:
        fig,ax = plt.subplots(dpi=dpi,figsize=figsize)
    else:
        fig,ax = figax
    
    data_plot = data.copy()
    if xlim is not None:
        ax.set_xlim(xlim)
        data_plot = data_plot[(data_plot.index >= xlim[0])&(data_plot.index <= xlim[1])]
    
    if labels is None:
        labels = data_plot.columns
        
    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    for i,c in enumerate(data_plot.columns):
        if colors is not None:
            kwargs['color'] = colors[i]
        if linestyles is not None:
            kwargs['ls'] = linestyles[i]
            
        ax.plot(data_plot[c],label=labels[i],**kwargs)
    ax.legend(loc=legend_loc)
        
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    #formatter = mdates.DateFormatter('%b')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    if ylim_bottom is not None or ylim_top is not None:
        ax.set_ylim(bottom=ylim_bottom,top=ylim_top)
    
    if save_fig is not None:
        if figs_folder is None:
            figs_folder = input()
        plt.savefig(os.path.join(figs_folder,'{}.png'.format(save_fig)),bbox_inches='tight')
    if show:
        plt.show()
        plt.close()
        return 
    return fig,ax


def get_extent():
    extent = [-5, 9.8, 41.3, 51.3]
    return extent


def blank_national_map():
    fig = plt.figure(figsize=(7,7), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent(get_extent())
    
    ax.add_feature(cfeature.OCEAN, color='lightgrey',zorder=2)
    ax.add_feature(cfeature.LAND, color='w',zorder=1)
    ax.add_feature(cfeature.COASTLINE,zorder=5)
    ax.add_feature(cfeature.BORDERS,zorder=3)
    
    return fig,ax


def plot_pie_chart(random_seed=1):
    rd.seed(random_seed)
    
    # TODO en faire une fonction appelable
    fig, ax = plt.subplots(figsize=(5,5),dpi=300)

    size = 0.4
    vals = [[60.], [37., 40.], [29., 10.,5.]]
    labels_cat = list()
    vals_flat = [x for xs in vals for x in xs]
    percentage_vals_flat = [x/sum(xs)*100 for xs in vals for x in xs]
    unique_value = [float(len(xs)!=1) for xs in vals for x in xs]            
    labels_flat = ['oui']*len(vals_flat)
    labels_flat = ['{} ({:.0f}%)'.format(l,p) if bool(f) else '' for l,f,p in zip(labels_flat, unique_value,percentage_vals_flat)]
    
    cmap = plt.colormaps["tab20c"]
    inner_colors = cmap(np.arange(len(vals))*4)
    outer_colors = [x for xs in [[cmap(i*4)]*len(l) for i,l in enumerate(vals)] for x in xs]
    
    intensities = [rd.random()*2/3 + 0.33 for e in outer_colors]
    outer_colors = [(r,v,b,i*f) for (r,v,b,_),i,f in zip(outer_colors,intensities,unique_value)]
    
    ax.pie([sum(l) for l in vals], radius=1, colors=inner_colors,
           labels=labels_cat,labeldistance=None,
           wedgeprops=dict(width=size, edgecolor='w',))
    
    ax.pie(vals_flat, radius=1+size, colors=outer_colors,
           wedgeprops=dict(width=size, edgecolor='w'),labels=labels_flat)
    
    ax.set(aspect="equal")
    ax.legend(labels=labels_cat, bbox_to_anchor=(1.1, 1.))
    plt.show()


#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    blank_national_map()

    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    return 

if __name__ == '__main__':
    main()