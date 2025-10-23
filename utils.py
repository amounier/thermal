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
import matplotlib.colors as colors
from scipy.ndimage import zoom
import pandas as pd
from datetime import date


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


def custom_xycmap(corner_colors=('#de4fa6','#2a1a8a','#dedce8','#4fadd0'), n=(3,3)):
    """
    from : https://github.com/rbjansen/xycmap
    """
    xn, yn = n
    if xn < 2 or yn < 2:
        raise ValueError("Expected n >= 2 categories.")
    
    if corner_colors[0].startswith('#'):
        color_array = np.array(
            [[list(int(corner_colors[0].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),
              list(int(corner_colors[1].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),],
             [list(int(corner_colors[2].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),
              list(int(corner_colors[3].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),],],)
        
    else:
        color_array = np.array(
            [[list(colors.to_rgba(corner_colors[0])),
              list(colors.to_rgba(corner_colors[1])),],
             [list(colors.to_rgba(corner_colors[2])),
              list(colors.to_rgba(corner_colors[3])),],],)
    
    zoom_factor_x = xn / 2  # Divide by the original two categories.
    zoom_factor_y = yn / 2
    zcolors = zoom(color_array, (zoom_factor_y, zoom_factor_x, 1), order=1)

    return zcolors


def get_scenarios_color():
    cmap = custom_xycmap()
    
    dict_scenarios = {'ACP_NOF':cmap[0,0]/255,
                      'ACP_REF':cmap[0,1]/255,
                      'ACP_SOF':cmap[0,2]/255,
                      'REF_NOF':cmap[1,0]/255,
                      'REF_REF':cmap[1,1]/255,
                      'REF_SOF':cmap[1,2]/255,
                      'ACM_NOF':cmap[2,0]/255,
                      'ACM_REF':cmap[2,1]/255,
                      'ACM_SOF':cmap[2,2]/255}
    return dict_scenarios


def get_zcl_colors():
    cmap = plt.get_cmap('viridis')
    
    zcl_dict = {e:None for e in ['H1a', 'H1b', 'H1c', 'H2a', 'H2b', 'H2c', 'H2d', 'H3']}
    for idx,(k,v) in enumerate(zcl_dict.items()):
        zcl_dict[k] = cmap(idx/len(zcl_dict.keys()))
        
    return zcl_dict

#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    
    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_utils'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    
    #%% blank map
    if False:
        blank_national_map()
    
    #%% bivariate cmap
    if True:
        cmap = custom_xycmap()
        
        fig,ax = plt.subplots(figsize=(3,3),dpi=300)
        
        ax.plot([-1,4],[1.5]*2,color='w')
        ax.plot([-1,4],[0.5]*2,color='w')
        ax.plot([1.5]*2,[-1,3],color='w')
        ax.plot([0.5]*2,[-1,3],color='w')

        ax.imshow(cmap)
        
        ax.text(-0.65,0,'More AC',va='center',ha='right')
        ax.text(-0.65,1,'REF',va='center',ha='right',style='italic')
        ax.text(-0.65,2,'Less AC',va='center',ha='right')
        ax.text(0,2.8,'Cold CLZ',ha='center',va='bottom')
        ax.text(1,2.8,'REF',ha='center',va='bottom',style='italic')
        ax.text(2,2.8,'Hot CLZ',ha='center',va='bottom')
        
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(figs_folder,'scenarios.png'), bbox_inches='tight')
        plt.show()
    

    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    return 

if __name__ == '__main__':
    main()