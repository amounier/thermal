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


def plot_timeserie(data,figsize=(5,5),dpi=300,labels=None,figs_folder=None,
                   save_fig=None,show=True,xlim=None,ylim_bottom=None,
                   figax=None,**kwargs):
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
        
    for i,c in enumerate(data_plot.columns):
        ax.plot(data_plot[c],label=labels[i],**kwargs)
    ax.legend()
        
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    #formatter = mdates.DateFormatter('%b')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    
    if save_fig is not None:
        if figs_folder is None:
            figs_folder = input()
        plt.savefig(os.path.join(figs_folder,'{}.png'.format(save_fig)),bbox_inches='tight')
    if show:
        plt.show()
        plt.close()
        return 
    return fig,ax


def main():
    tic = time.time()
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    return 

if __name__ == '__main__':
    main()