#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:57:30 2025

@author: amounier
"""

import time
from datetime import date
import os
import pandas as pd
import xarray as xr
import matplotlib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import mapping
import seaborn as sns
import tqdm
import numpy as np

from administrative import France, Climat, Climat_summer, Climat_winter
from utils import blank_national_map, get_extent

def map_xarray(data, cmap='viridis',cbar_label=None, value_max=None, 
               value_min=None, add_zcl=None, cbar_extend='both',
               figs_folder=None, save=None):
    
    fig,ax = blank_national_map()
        
    cmap = matplotlib.colormaps.get_cmap(cmap)
    
    plotted_data = data.copy()
    extent = get_extent()
    plotted_data = plotted_data.sel(lat=slice(extent[2],extent[3]), lon=slice(extent[0],extent[1]))
    
    if value_min is None:
        value_min = float(plotted_data.quantile(0.01))
    if value_max is None:
        value_max = float(plotted_data.quantile(0.99))
    
    img = plotted_data.plot(ax=ax, transform=ccrs.PlateCarree(),add_colorbar=False,
                            cmap=cmap, vmax=value_min, vmin=value_max)
    
    ax.set_title('')
    
    ax_cb = fig.add_axes([0,0,0.1,0.1])
    posn = ax.get_position()
    ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
    fig.add_axes(ax_cb)
    cbar = plt.colorbar(img,cax=ax_cb,extend=cbar_extend,extendfrac=0.02)
    cbar.set_label(cbar_label)
    
    if add_zcl == 'winter':
        france = France()
        climats_winter = [Climat_winter(e) for e in france.climats_winter]
        climats_winter = pd.DataFrame(index=climats_winter)
        climats_winter['geometry'] = [d.geometry for d in climats_winter.index]
        climats_winter = gpd.GeoDataFrame(climats_winter, geometry=climats_winter.geometry)
        climats_winter.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k',lw=0.7)
        
    if add_zcl == 'summer':
        france = France()
        climats_summer = [Climat_summer(e) for e in france.climats_summer]
        climats_summer = pd.DataFrame(index=climats_summer)
        climats_summer['geometry'] = [d.geometry for d in climats_summer.index]
        climats_summer = gpd.GeoDataFrame(climats_summer, geometry=climats_summer.geometry)
        climats_summer.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k',lw=0.7)
        
    if save is not None:
        plt.savefig(os.path.join(figs_folder,'{}.png'.format(save)),bbox_inches='tight')
    return fig,ax
    

def get_departement_temperature(period='DJF'):
    
    if period == 'DJF':
        temperature = xr.open_dataset(os.path.join('data','CORDEX','CORDEX Europe - Mean temperature (T) deg C - 1981-2010 - December to February (49 models)','map.nc'))
        
    elif period == 'JJA':
        temperature = xr.open_dataset(os.path.join('data','CORDEX','CORDEX Europe - Mean temperature (T) deg C - 1981-2010 - June to August (49 models)','map.nc'))
        
    temperature = temperature.tas
    tempertaure = temperature.rio.write_crs("epsg:4326")
    
    france = France()
    departements = france.departements
    data_departements = {'departement':[],'temperature':[]}
    
    for dep in tqdm.tqdm(departements):
        geom = dep.geometry
        centroid = geom.centroid
        geom = pd.Series(geom).apply(mapping)
        
        clipped = tempertaure.rio.clip(geom, 'epsg:4326', drop=False)
        mean_clipped = float(clipped.mean())
        
        if np.isnan(mean_clipped):
            mean_clipped = float(tempertaure.sel(lon=centroid.x, lat=centroid.y, method='nearest').values)
        
        data_departements['departement'].append(dep)
        data_departements['temperature'].append(mean_clipped)
    
    data_departements = pd.DataFrame().from_dict(data_departements)
    return data_departements
        
#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_climate_zone'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
    
    
    #%% Caractérisation des zones climatiques et de leur pertinence par rapport aux données météo
    if False:
        france = France()
        
        temperature_DJF = xr.open_dataset(os.path.join('data','CORDEX','CORDEX Europe - Mean temperature (T) deg C - 1981-2010 - December to February (49 models)','map.nc'))
        # temperature_DJF = xr.open_dataset(os.path.join('data','CORDEX','E-OBS (Europe) - Mean temperature (T) deg C - 1980-2015 Observations - December to February ','map.nc'))
        temperature_DJF = temperature_DJF.tas
        temperature_DJF = temperature_DJF.rio.write_crs("epsg:4326")
    
        temperature_JJA = xr.open_dataset(os.path.join('data','CORDEX','CORDEX Europe - Mean temperature (T) deg C - 1981-2010 - June to August (49 models)','map.nc'))
        temperature_JJA = temperature_JJA.tas
        temperature_JJA = temperature_JJA.rio.write_crs("epsg:4326")
        
        # affichage des cartes de température 
        if False:
            map_xarray(temperature_DJF, add_zcl='winter', 
                       cbar_label='Mean temperature (°C) - EURO-CORDEX 1981-2010',
                       figs_folder=figs_folder, save='DJF_temperature',)
            map_xarray(temperature_JJA, add_zcl='summer', 
                       cbar_label='Mean temperature (°C) - EURO-CORDEX 1981-2010',
                       figs_folder=figs_folder, save='JJA_temperature',)
            
        # temperatures moyennes des departements 
        departements = france.departements
        
        data_departements = {'departement':[],'temperature_DJF':[], 'temperature_JJA':[]}
    
        for dep in tqdm.tqdm(departements):
            geom = dep.geometry
            centroid = geom.centroid
            geom = pd.Series(geom).apply(mapping)
            
            clipped_DJF = temperature_DJF.rio.clip(geom, 'epsg:4326', drop=False)
            clipped_JJA = temperature_JJA.rio.clip(geom, 'epsg:4326', drop=False)
            
            mean_clipped_DJF = float(clipped_DJF.mean())
            mean_clipped_JJA = float(clipped_JJA.mean())
            
            if np.isnan(mean_clipped_DJF):
                mean_clipped_DJF = float(temperature_DJF.sel(lon=centroid.x, lat=centroid.y, method='nearest').values)
                mean_clipped_JJA = float(temperature_JJA.sel(lon=centroid.x, lat=centroid.y, method='nearest').values)
            
            data_departements['departement'].append(dep)
            data_departements['temperature_DJF'].append(mean_clipped_DJF)
            data_departements['temperature_JJA'].append(mean_clipped_JJA)
            # data_departements['geometry'].append(dep.geometry)
            
        data_departements = pd.DataFrame().from_dict(data_departements)
        # data_departements = gpd.GeoDataFrame(data_departements, geometry=data_departements.geometry)
        data_departements['zcl'] = [dep.climat for dep in data_departements.departement]
        
        if False:
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            hue_order = sorted(list(set(data_departements['zcl'])))
            sns.scatterplot(data=data_departements, ax=ax,hue_order=hue_order,
                            palette=sns.color_palette("hls", 8),
                            x='temperature_DJF',y='temperature_JJA',hue='zcl')
            ax.set_xlim([0,8])
            ax.set_ylim([14,24])
            ax.legend(ncols=4)
            plt.show()
            
        if False:
            def scatter_hist(data, ax, ax_histx, ax_histy, **kwargs):
                hue_order = sorted(list(set(data_departements['zcl'])))
                ax_histx.tick_params(axis="x", labelbottom=False)
                ax_histy.tick_params(axis="y", labelleft=False)
    
                sns.scatterplot(data=data, ax=ax,hue_order=hue_order,
                                palette=sns.color_palette("hls", 8),
                                x='temperature_DJF',y='temperature_JJA',hue='zcl')
            
                # # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
                # binwidth_x = 2*(np.quantile(x,0.75)-np.quantile(x, 0.25))/(len(x)**(1/3))
                binwidth_x = 1
                # binwidth_y = 2*(np.quantile(y,0.75)-np.quantile(y, 0.25))/(len(y)**(1/3))
                binwidth_y = 1
    
                # bins_x = np.arange(np.min(x), np.max(x) + binwidth_x, binwidth_x)
                # bins_y = np.arange(np.min(y), np.max(y) + binwidth_y, binwidth_y)
                # ax_histx.hist(x, bins=bins_x, color=kwargs.get('color'),alpha=kwargs.get('alpha'))
                sns.kdeplot(data=data,#binwidth=binwidth_x,
                             ax=ax_histx, hue_order=hue_order,
                             palette=sns.color_palette("hls", 8),
                             x='temperature_DJF',hue='zcl',legend=False,)#kde=True)
                
                sns.kdeplot(data=data,#binwidth=binwidth_y,
                             ax=ax_histy, hue_order=hue_order,
                             palette=sns.color_palette("hls", 8),
                             y='temperature_JJA',hue='zcl',legend=False,)#kde=True)
                
                # sns.histplot(x, bins=bins_x, color=kwargs.get('color'),alpha=kwargs.get('alpha'),ax=ax_histx)
                # ax_histy.hist(y, orientation='horizontal', bins=bins_y, color=kwargs.get('color'),alpha=kwargs.get('alpha'))
                
                ylim_histx = ax_histx.get_ylim()
                ylim_histx = [0,ylim_histx[1]]
                
                ax_histx.set_ylim(ylim_histx)
                return ax
            
            fig = plt.figure(layout='constrained',dpi=300,figsize=(5,5))
            ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
            ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax,)
            ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax,)
            
            ax.set_xlim([-6,8])
            ax.set_ylim([10,24])
            ax_histx.set_ylim([0,0.07])
            ax_histy.set_xlim([0,0.07])
                
            # for i,zcl in enumerate(sorted(list(set(data_departements['zcl'])))):
            #     # plotter = data_departements[data_departements.zcl==zcl]
            palette = sns.color_palette("hls", 8)
            
            
            ax = scatter_hist(data_departements, 
                              ax, ax_histx, ax_histy,
                              alpha=0.4)
            ax.legend()
            plt.show()
            # # seuils présentés ici : https://learnche.org/pid/least-squares-modelling/outliers-discrepancy-leverage-and-influence-of-the-observations
            # ax.plot([df_influence.hat_diag.mean()]*2, [df_influence.standard_resid.min(), df_influence.standard_resid.max()], color='k', ls=':')
            # ax.plot([df_influence.hat_diag.mean()*2]*2, [df_influence.standard_resid.min(), df_influence.standard_resid.max()], color='grey', ls=':')
            # ax.plot([df_influence.hat_diag.mean()*3]*2, [df_influence.standard_resid.min(), df_influence.standard_resid.max()], color='lightgrey', ls=':')
                    
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()