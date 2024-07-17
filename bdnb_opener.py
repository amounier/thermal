#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:06:58 2024

@author: amounier
"""

import time 
import os
import pandas as pd
import geopandas as gpd
import dask_geopandas 
import fiona
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from collections import Counter
import dask.dataframe as dd
from matplotlib.ticker import MaxNLocator
from datetime import date
import matplotlib.dates as mdates
import seaborn as sns
import re
import requests
import io
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import cartopy.geodesic as cgeo
from urllib.request import urlopen, Request
from PIL import Image
import xmltodict
import json
from urllib.error import HTTPError

pd.set_option('future.no_silent_downcasting', True)

# définition de la date du jour
today = pd.Timestamp(date.today()).strftime('%Y%m%d')

# définition du fichier de sortie
output_folder = os.path.join('output')
folder = '{}_DPE_successifs'.format(today)
if folder not in os.listdir(output_folder):
    os.mkdir(os.path.join(output_folder,folder))
if 'figs' not in os.listdir(os.path.join(output_folder, folder)):
    os.mkdir(os.path.join(output_folder,folder,'figs'))


# =============================================================================
# Fonctions relatives à l'ouverture (optimale) de la BDNB
# =============================================================================

def get_layer_names():
    # tested file
    file = os.path.join('data','BDNB','open_data_millesime_2023-11-a_dep75_gpkg','gpkg','bdnb.gpkg')
    
    # list of layers in file
    layers_list = fiona.listlayers(file)
    
    return layers_list
    

def speed_test_opening(dask_only=False, plot=False):
    # tested file
    file = os.path.join('data','BDNB','open_data_millesime_2023-11-a_dep75_gpkg','gpkg','bdnb.gpkg')
    
    if dask_only:
        test_npartitions = False
        test_chunksize = False
        
        if test_npartitions:
            npartitions_list = list(range(1,21))
            opening_speed_list = []
            for npartitions in npartitions_list:
                tic = time.time()
                bdnb = dask_geopandas.read_file(file, npartitions=npartitions, layer='dpe_logement')
                tac = time.time()
                opening_speed_list.append(tac-tic)
                
            if plot:
                fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                ax.plot(npartitions_list, opening_speed_list)
                plt.show()
                
        if test_chunksize:
            chunksize_list = list(np.linspace(1e2,1e8,1000))
            opening_speed_list = []
            for cs in tqdm.tqdm(chunksize_list):
                tic = time.time()
                bdnb = dask_geopandas.read_file(file, chunksize=cs, layer='dpe_logement')
                tac = time.time()
                opening_speed_list.append(tac-tic)
                
            if plot:
                fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                ax.plot(chunksize_list, opening_speed_list)
                plt.show()
        return 
    
    else:
        # summary dict of opening speed by method
        methods_speed_dict = dict()
        
        # geopandas vanilla
        tic = time.time()
        bdnb = gpd.read_file(file, layer='adresse_compile')
        tac = time.time()
        methods_speed_dict['gpd_vanilla'] = tac-tic
        
        # geopandas pyogrio
        tic = time.time()
        bdnb = gpd.read_file(file, engine='pyogrio', layer='adresse_compile')
        tac = time.time()
        methods_speed_dict['gpd_pyogrio'] = tac-tic
        
        # dask-geopandas
        tic = time.time()
        bdnb = dask_geopandas.read_file(file, npartitions=4, layer='adresse_compile')
        tac = time.time()
        methods_speed_dict['dask_gpd'] = tac-tic
    
        return methods_speed_dict


def get_bdnb(chunksize=5e4):
    file = os.path.join('data','BDNB','open_data_millesime_2023-11-a_dep75_gpkg','gpkg','bdnb.gpkg')
    bdnb_dpe_logement = dask_geopandas.read_file(file, chunksize=chunksize, layer='dpe_logement')
    bdnb_rel_batiment_groupe_dpe_logement = dask_geopandas.read_file(file, chunksize=chunksize, layer='rel_batiment_groupe_dpe_logement')
    bdnb_batiment_groupe_compile = dask_geopandas.read_file(file, chunksize=chunksize, layer='batiment_groupe_compile')
    # batiment_groupe_id
    return bdnb_dpe_logement, bdnb_rel_batiment_groupe_dpe_logement, bdnb_batiment_groupe_compile
    

def cull_empty_partitions(df):
    ll = list(df.map_partitions(len).compute())
    df_delayed = df.to_delayed()
    df_delayed_new = list()
    pempty = None
    for ix, n in enumerate(ll):
        if 0 == n:
            pempty = df.get_partition(ix)
        else:
            df_delayed_new.append(df_delayed[ix])
    if pempty is not None:
        df = dd.from_delayed(df_delayed_new, meta=pempty)
    return df


# =============================================================================
# fonctions relatives à l'étude des DPE
# =============================================================================


def suspect_identification(plot=False, force=False, number_batiment_groupe=1000):
    """
    étude des DPE successifs sur des batiments (pas d'informations sur le logement)

    Parameters
    ----------
    plot : boolean, optional
        DESCRIPTION. The default is False.
    force : boolean, optional
        DESCRIPTION. The default is False.
    number_batiment_groupe : int, optional
        à maximiser in fine (:). The default is 1000.

    Returns
    -------
    None.

    """
    
    # ouverture des données sous dask
    print('Opening BDNB...')
    bdnb_dpe_logement, bdnb_rel_batiment_groupe_dpe_logement, bdnb_batiment_groupe_compile = get_bdnb()
    
    bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile[['batiment_groupe_id','ffo_bat_nb_log']]
    bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile.compute()
    
    bdnb_rel_batiment_groupe_dpe_logement = bdnb_rel_batiment_groupe_dpe_logement[['batiment_groupe_id','identifiant_dpe']]
    bdnb_rel_batiment_groupe_dpe_logement = bdnb_rel_batiment_groupe_dpe_logement.compute()
    
    bdnb_dpe_logement = bdnb_dpe_logement[['identifiant_dpe','date_etablissement_dpe','classe_bilan_dpe','type_dpe','surface_habitable_logement']]
    bdnb_dpe_logement = bdnb_dpe_logement.set_index('identifiant_dpe')
    bdnb_dpe_logement = bdnb_dpe_logement.compute()
    
    multiple_dpe_batiment_groupe = dict(Counter(bdnb_rel_batiment_groupe_dpe_logement.batiment_groupe_id))
    
    multiple_dpe_batiment_groupe_df = pd.DataFrame().from_dict({'batiment_groupe_id':multiple_dpe_batiment_groupe.keys(),'nb_dpe':multiple_dpe_batiment_groupe.values()})
    multiple_dpe_batiment_groupe_df = multiple_dpe_batiment_groupe_df[multiple_dpe_batiment_groupe_df.nb_dpe>1]
    
    base_multiple_dpe_batiment_groupe_df = multiple_dpe_batiment_groupe_df[multiple_dpe_batiment_groupe_df.nb_dpe<multiple_dpe_batiment_groupe_df.nb_dpe.quantile(0.99)]
    
    if plot:
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        sns.histplot(base_multiple_dpe_batiment_groupe_df,x='nb_dpe',stat='percent',ax=ax,binwidth=5,)
        ax.set_xlabel('Nombre de DPE par bâtiment ({} bâtiments)'.format(len(base_multiple_dpe_batiment_groupe_df)))
        plt.show()
        
    
    batiment_group_list = base_multiple_dpe_batiment_groupe_df.batiment_groupe_id.to_list()
    
    if isinstance(number_batiment_groupe,int):
        batiment_group_list = batiment_group_list[:number_batiment_groupe]
    
    results = dict()
    pbar = tqdm.tqdm(enumerate(batiment_group_list), total=len(batiment_group_list))
    for i,bg_id in pbar:
        pbar.set_description(bg_id)
        pbar.refresh()

        results[bg_id] = {'identifiant_dpe':[],
                          'date_etablissement_dpe':[],
                          'classe_bilan_dpe':[],
                          'surface_habitable_logement':[],
                          'type_dpe':[],
                          'nb_log_batiment_groupe':[]}
        
        dpe_bg = bdnb_rel_batiment_groupe_dpe_logement[bdnb_rel_batiment_groupe_dpe_logement.batiment_groupe_id==bg_id].identifiant_dpe
        # dpe_bg = dpe_bg.compute()
        
        nb_logement = bdnb_batiment_groupe_compile[bdnb_batiment_groupe_compile.batiment_groupe_id==bg_id].ffo_bat_nb_log
        
        try: 
            nb_logement = int(nb_logement.values[0])
        except ValueError:
            continue
        
        for dpe_id in dpe_bg:
            bdnb_dpe_logement_filtered = bdnb_dpe_logement.loc[dpe_id]#.compute()
            date_etablissement_dpe, classe_bilan_dpe, type_dpe, surface_habitable_logement = bdnb_dpe_logement_filtered.values
            
            if np.isnan(surface_habitable_logement):
                continue
            
            results[bg_id]['identifiant_dpe'].append(dpe_id)
            results[bg_id]['date_etablissement_dpe'].append(date_etablissement_dpe)
            results[bg_id]['classe_bilan_dpe'].append(classe_bilan_dpe)
            results[bg_id]['type_dpe'].append(type_dpe)
            results[bg_id]['surface_habitable_logement'].append(int(surface_habitable_logement))
            results[bg_id]['nb_log_batiment_groupe'].append(nb_logement)
    
    letter_to_number_dict = {chr(ord('@')+n):n for n in range(1,10)}
    
    for bg_id in batiment_group_list:
        df_dpe_bg_id = pd.DataFrame().from_dict(results.get(bg_id))
        df_dpe_bg_id = df_dpe_bg_id.dropna(axis=0)
        df_dpe_bg_id = df_dpe_bg_id[df_dpe_bg_id.type_dpe=='dpe arrêté 2021 3cl logement']
        df_dpe_bg_id['classe_bilan_dpe_number'] = [letter_to_number_dict.get(e) for e in df_dpe_bg_id.classe_bilan_dpe]
        df_dpe_bg_id = df_dpe_bg_id.sort_values('date_etablissement_dpe')
        
        if df_dpe_bg_id.empty or len(df_dpe_bg_id)<2:
            continue
    
        # filtre pour ne garder que les surface de logements identiques d'un DPE au suivant 
        filter_same_surface = np.asarray([df_dpe_bg_id.surface_habitable_logement==df_dpe_bg_id.surface_habitable_logement.shift(1),
                                          df_dpe_bg_id.surface_habitable_logement==df_dpe_bg_id.surface_habitable_logement.shift(-1)]).any(0)
        df_dpe_bg_id = df_dpe_bg_id[filter_same_surface]
        
        # filtre pour ne garder que les changements d'étiquette d'un DPE au suivant
        filter_same_etiquette = np.asarray([df_dpe_bg_id.classe_bilan_dpe_number==df_dpe_bg_id.classe_bilan_dpe_number.shift(1),
                                            df_dpe_bg_id.classe_bilan_dpe_number==df_dpe_bg_id.classe_bilan_dpe_number.shift(-1)]).any(0)
        df_dpe_bg_id = df_dpe_bg_id[~filter_same_etiquette]
        
        # filtre pour ne garder que les DPE distants de moins de 30 jours
        df_dpe_bg_id['days_difference'] = df_dpe_bg_id.date_etablissement_dpe - df_dpe_bg_id.date_etablissement_dpe.shift(1)
        df_dpe_bg_id['days_difference'] = df_dpe_bg_id.days_difference.dt.days
        if not any(df_dpe_bg_id.days_difference < 30.):
            continue
        
        # filtre pour ne garder que les bâtiments ayant au moins 2 DPE successifs
        if len(df_dpe_bg_id)<2:
            continue
        
        # sortie des fichiers csv pour les bâtiments 'suspects'
        suspect_folder = 'raw_suspicious_DPE'
        save_path = os.path.join(output_folder,folder,suspect_folder)
        if suspect_folder not in os.listdir(os.path.join(output_folder,folder)):
            os.mkdir(save_path)
        df_dpe_bg_id.to_csv(os.path.join(save_path,'{}.csv'.format(bg_id)),index=False)
        
    return len(batiment_group_list)


def plot_raw_suspects():
    """
    Visualisation de la première sélection brute de suspects potentiels

    Returns
    -------
    None.

    """
    folder = '{}_DPE_successifs'.format(today)
    
    if folder not in os.listdir(output_folder):
        suspect_identification()
    
    done_batiment_group_list = os.listdir(os.path.join(output_folder, folder))
    done_batiment_group_list = [s.replace('.csv','') for s in done_batiment_group_list if s.endswith('.csv')]
    
    if len(done_batiment_group_list) == 0:
        suspect_identification()
        done_batiment_group_list = os.listdir(os.path.join(output_folder, folder))
        done_batiment_group_list = [s.replace('.csv','') for s in done_batiment_group_list if s.endswith('.csv')]
        
    for bg_id in tqdm.tqdm(done_batiment_group_list):
        df_dpe_bg_id = pd.read_csv(os.path.join(output_folder, folder,'{}.csv'.format(bg_id)))
        df_dpe_bg_id['date_etablissement_dpe'] = [pd.to_datetime(t) for t in df_dpe_bg_id.date_etablissement_dpe]
        
        if 'figs' not in os.listdir(os.path.join(output_folder, folder)):
            os.mkdir(os.path.join(output_folder,folder,'figs'))
            
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(df_dpe_bg_id.date_etablissement_dpe, df_dpe_bg_id.classe_bilan_dpe_number,ls=':',marker='o')
        ax.set_title('{} ({} log)'.format(bg_id,df_dpe_bg_id.nb_log_batiment_groupe.values[0]))
        ax.set_yticks(ticks:=list(range(1,8)),labels=[chr(ord('@')+n) for n in ticks])
        locator = mdates.AutoDateLocator(minticks=1, maxticks=4)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        plt.show()
        plt.close()
    return


def download_dpe_details(dpe_id, force=False):
    """
    Téléchargement des fichiers de sorties des DPE (au format XLSX)

    Parameters
    ----------
    dpe_id : str
        DESCRIPTION.
    force : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    if '{}.xlsx'.format(dpe_id) in os.listdir(os.path.join('data','DPE','XLS')) or force:
        return
    
    try:
        dls = "https://observatoire-dpe-audit.ademe.fr/pub/dpe/{}/xml-excel".format(dpe_id)
        req = Request(dls) 
        req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0') 
        
        content = urlopen(req)
        
        with open(os.path.join('data','DPE','XLS','{}.xlsx'.format(dpe_id)), 'wb') as output:
            output.write(content.read())
    except HTTPError:
        return 
    return 



def open_dpe_details(dpe_id, sheet_name='logement'):
    """
    Ouverture et formatage des données DPE XLSX

    Parameters
    ----------
    dpe_id : str
        DESCRIPTION.
    sheet_name : str, optional
        DESCRIPTION. The default is 'logement'.

    Returns
    -------
    dpe_data : TYPE
        DESCRIPTION.

    """
    try:
        dpe_xls_path = os.path.join('data','DPE','XLS','{}.xlsx'.format(dpe_id))
        dpe_data = pd.read_excel(dpe_xls_path, sheet_name=sheet_name)
    except FileNotFoundError:
        print('Downloading XLS details of DPE {} from observatoire-dpe-audit.ademe.fr...'.format(dpe_id))
        
        download_dpe_details(dpe_id, force=False)
        
        try:
            dpe_xls_path = os.path.join('data','DPE','XLS','{}.xlsx'.format(dpe_id))
            dpe_data = pd.read_excel(dpe_xls_path, sheet_name=sheet_name)
        except FileNotFoundError:
            print('Error: {} is not available from observatoire-dpe-audit.ademe.fr :('.format(dpe_id))
            return pd.DataFrame()
        
    group_name = {'logement':'caracteristique_generale','administratif':'administratif'}.get(sheet_name)
    
    dpe_data = dpe_data.rename(columns={'Unnamed: 0':'variables',group_name:'dpe_values'})
    
    new_col = []
    
    for val,group in zip(dpe_data.variables,dpe_data.dpe_values):
        if pd.isnull(val) and not pd.isnull(group):
            group_name = group
        if pd.isnull(val):
            new_val = np.nan
        else:
            new_val = '{}--{}'.format(group_name,val).replace('_0--','--')
        new_col.append(new_val)
    dpe_data['variables'] = new_col
    
    new_val_col = []
    for cols in zip(*[dpe_data[c] for c in dpe_data.columns]):
        var = cols[0]
        if pd.isnull(var):
            group_name = np.nan
        else:
            group_name = var.split('--')[0]
        all_values = list(cols[1:])
        new_val = []
        for i,v in enumerate(all_values):
            if i == 0 and pd.isnull(v):
                new_val.append(np.nan)
                continue
                
            if not pd.isnull(v):
                new_val.append(v)
                
        if len(new_val) == 1:
            new_val = new_val[0]
        new_val_col.append(new_val)
    dpe_data['dpe_values'] = new_val_col
    
    dpe_data = dpe_data[['variables','dpe_values']]
        
    dpe_data = dpe_data[~dpe_data.variables.isna()]
    dpe_data = dpe_data.fillna('nan')
    return dpe_data


def intersection_dpe_details(dpe_data_1,dpe_data_2):
    """
    Sélection des variables des données DPE communes entre deux jeux de données

    Parameters
    ----------
    dpe_data_1 : pandas DataFrame
        DESCRIPTION.
    dpe_data_2 : pandas DataFrame
        DESCRIPTION.

    Returns
    -------
    dpe_data_1 : TYPE
        DESCRIPTION.
    dpe_data_2 : TYPE
        DESCRIPTION.

    """
    minimal_variables = [e for e in dpe_data_1.variables.values if e in dpe_data_2.variables.values]
    minimal_variables = [e for e in minimal_variables if not 'reference' in e]
    dpe_data_1 = dpe_data_1[dpe_data_1.variables.isin(minimal_variables)].reset_index(drop=True)
    dpe_data_2 = dpe_data_2[dpe_data_2.variables.isin(minimal_variables)].reset_index(drop=True)
    return dpe_data_1, dpe_data_2
    

def difference_dpe_details(dpe_id_1,dpe_id_2):
    """
    Récupération des différences entre les jeux de données DPE de deux diagnostics

    Parameters
    ----------
    dpe_id_1 : str
        DESCRIPTION.
    dpe_id_2 : str
        DESCRIPTION.

    Returns
    -------
    difference_1 : TYPE
        DESCRIPTION.
    difference_2 : TYPE
        DESCRIPTION.

    """
    data_logement_1 = open_dpe_details(dpe_id_1,sheet_name='logement')
    data_logement_2 = open_dpe_details(dpe_id_2,sheet_name='logement')
    data_logement_1, data_logement_2 = intersection_dpe_details(data_logement_1, data_logement_2)
    
    data_admin_1 = open_dpe_details(dpe_id_1,sheet_name='administratif')
    data_admin_2 = open_dpe_details(dpe_id_2,sheet_name='administratif')
    data_admin_1, data_admin_2 = intersection_dpe_details(data_admin_1, data_admin_2)
    
    diff_logement = (data_logement_1[data_logement_1.columns.values[1:]]!=data_logement_2[data_logement_2.columns.values[1:]]).any(axis=1)
    difference_logement_1 = data_logement_1[diff_logement]
    difference_logement_1 = difference_logement_1.replace('nan',np.nan)
    difference_logement_2 = data_logement_2[diff_logement]
    difference_logement_2 = difference_logement_2.replace('nan',np.nan)
    
    diff_admin = (data_admin_1[data_admin_1.columns.values[1:]]!=data_admin_2[data_admin_2.columns.values[1:]]).any(axis=1)
    difference_admin_1 = data_admin_1[diff_admin]
    difference_admin_1 = difference_admin_1.replace('nan',np.nan)
    difference_admin_2 = data_admin_2[diff_admin]
    difference_admin_2 = difference_admin_2.replace('nan',np.nan)
    
    difference_1 = pd.concat([difference_admin_1, difference_logement_1])
    difference_2 = pd.concat([difference_admin_2, difference_logement_2])
    return difference_1, difference_2
        


def analysis_suspicious_DPE(plot=None,details=True,number_batiment_groupe=1000, force=False):
    """
    analyse des gains de DPE parmi les suspects potentiels  

    Parameters
    ----------
    plot : TYPE, optional
        DESCRIPTION. The default is None.
    details : TYPE, optional
        DESCRIPTION. The default is True.
    number_batiment_groupe : TYPE, optional
        DESCRIPTION. The default is 1000.
    force : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    suspicious_batiment_group_dict_dpe_id : TYPE
        DESCRIPTION.

    """
    if isinstance(plot,bool):
        if plot:
            show_plots = True
        else:
            show_plots = False
    elif isinstance(plot,list):
        bg_id_plot_list = plot
        show_plots = False
    else:
        show_plots = False
        bg_id_plot_list = []
    show_details = details
    
    suspect_folder = 'raw_suspicious_DPE'
    path = os.path.join(output_folder,folder,suspect_folder)
    if suspect_folder not in os.listdir(os.path.join(output_folder,folder)):
        os.mkdir(path)
        
    done_batiment_group_list = os.listdir(path)
    done_batiment_group_list = [s.replace('.csv','') for s in done_batiment_group_list if s.endswith('.csv')]
    
    if len(done_batiment_group_list) == 0 or force:
        suspect_identification(number_batiment_groupe=number_batiment_groupe)
        done_batiment_group_list = os.listdir(path)
        done_batiment_group_list = [s.replace('.csv','') for s in done_batiment_group_list if s.endswith('.csv')]

    suspicious_batiment_group_dict_dpe_number = dict()
    suspicious_batiment_group_dict_dpe_id = dict()
    for bg_id in tqdm.tqdm(done_batiment_group_list):
        df_dpe_bg_id = pd.read_csv(os.path.join(path,'{}.csv'.format(bg_id)))
        df_dpe_bg_id['date_etablissement_dpe'] = [pd.to_datetime(t) for t in df_dpe_bg_id.date_etablissement_dpe]
        
        # filtre pour une période de 30 jours
        filter_days_difference = df_dpe_bg_id.days_difference<30
        filter_days_difference = np.asarray([filter_days_difference.values, filter_days_difference.shift(-1).fillna(False).values])
        filter_days_difference = filter_days_difference.any(0)
        df_suspicious_dpe_bg_id = df_dpe_bg_id[filter_days_difference]
        
        # filtre pour DPE à la baisse
        filter_dpe_gains = [False]*len(df_suspicious_dpe_bg_id)
        for i in range(1,len(filter_dpe_gains)):
            old_dpe, new_dpe = df_suspicious_dpe_bg_id.classe_bilan_dpe_number.values[i-1:i+1]
            if old_dpe>new_dpe:
                filter_dpe_gains[i] = True
                filter_dpe_gains[i-1] = True
        df_suspicious_dpe_bg_id = df_suspicious_dpe_bg_id[np.asarray(filter_dpe_gains)]
                
        # affichage des trajectoires de DPE
        if show_plots and 'figs' not in os.listdir(os.path.join(output_folder, folder)):
            os.mkdir(os.path.join(output_folder,folder,'figs'))
        
        if df_suspicious_dpe_bg_id.empty:
            continue
        
        if show_plots or bg_id in bg_id_plot_list:
            fig,ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.plot(df_dpe_bg_id.date_etablissement_dpe, df_dpe_bg_id.classe_bilan_dpe_number,ls=':',marker='o')
            
        for i in range(1,len(df_suspicious_dpe_bg_id)):
            if df_suspicious_dpe_bg_id.days_difference.values[i-1:i+1][-1]<30 and df_suspicious_dpe_bg_id.classe_bilan_dpe_number.values[i-1]>df_suspicious_dpe_bg_id.classe_bilan_dpe_number.values[i]:
                
                if show_plots or bg_id in bg_id_plot_list:
                    ax.plot(df_suspicious_dpe_bg_id.date_etablissement_dpe.values[i-1:i+1], df_suspicious_dpe_bg_id.classe_bilan_dpe_number.values[i-1:i+1],ls=':',marker='o',color='red')
                
                if bg_id not in suspicious_batiment_group_dict_dpe_number.keys():
                    suspicious_batiment_group_dict_dpe_number[bg_id] = []
                    suspicious_batiment_group_dict_dpe_id[bg_id] = []
                suspicious_batiment_group_dict_dpe_number[bg_id].append(list(df_suspicious_dpe_bg_id.classe_bilan_dpe_number.values[i-1:i+1]))
                suspicious_batiment_group_dict_dpe_id[bg_id].append(list(df_suspicious_dpe_bg_id.identifiant_dpe.values[i-1:i+1]))
        
        if show_plots or bg_id in bg_id_plot_list:
            ax.set_title('{} ({} log)'.format(bg_id,df_dpe_bg_id.nb_log_batiment_groupe.values[0]))
            ax.set_yticks(ticks:=list(range(1,8)),labels=[chr(ord('@')+n) for n in ticks])
            locator = mdates.AutoDateLocator(minticks=1, maxticks=4)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
            save_path = os.path.join('output',folder,'figs','{}_dpe.png'.format(bg_id))
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()
            plt.close()
    
    # calcul du nombre de batiments concernés par des DPE suspects
    number_suspicious_bg = len(suspicious_batiment_group_dict_dpe_number)
    
    # calcul du nombre de gains d'étiquettes moyens et des principaux changements 
    mean_dpe_gains = 0
    counter = 0
    dpe_gain_counter_dict = dict()
    for k,v in suspicious_batiment_group_dict_dpe_number.items():
        for dpe_ini,dpe_redo in v:
            dpe_gain = dpe_ini-dpe_redo
            mean_dpe_gains += dpe_gain
            counter += 1
            if (dpe_ini,dpe_redo) not in dpe_gain_counter_dict.keys():
                dpe_gain_counter_dict[(dpe_ini,dpe_redo)] = 1
            else:
                dpe_gain_counter_dict[(dpe_ini,dpe_redo)] += 1 
    mean_dpe_gains = mean_dpe_gains/counter
    
    # tri par ordre décroissant de valeur 
    dpe_gain_counter_dict = {k: v for k, v in sorted(dpe_gain_counter_dict.items(), key=lambda item: item[1],reverse=True)}
    number_to_letter_dict = {n:chr(ord('@')+n) for n in range(1,10)}
    
    if show_details:
        print('\nPourcentage de bâtiments présentants des DPE suspicieux :')
        print('\t- {:.1f}% ({}/{})'.format(number_suspicious_bg/number_batiment_groupe*100,number_suspicious_bg, number_batiment_groupe))
        
        print("\nGains moyens d'étiquettes :")
        print('\t- {:.1f} ({} obs.)'.format(mean_dpe_gains,number_suspicious_bg))
        
        print("Détails des gains d'étiquettes :")
        for (dpe_ini, dpe_redo), nb_chg in dpe_gain_counter_dict.items():
            letter_ini, letter_redo = number_to_letter_dict.get(dpe_ini), number_to_letter_dict.get(dpe_redo)
            print('\t- {} -> {} : {:>4.1f}% ({}/{})'.format(letter_ini, letter_redo, nb_chg/number_suspicious_bg*100, nb_chg, number_suspicious_bg))
        print('\n')
        
    return suspicious_batiment_group_dict_dpe_id, suspicious_batiment_group_dict_dpe_number


def draw_local_map(geometry,style='map',figsize=12, radius=370, grey_background=True, save_path=None, include_OSM_copyright=True):
    """
    based on https://www.theurbanist.com.au/2021/03/plotting-openstreetmap-images-with-cartopy/

    """
    
    def image_spoof(self, tile):
        """Reformat for cartopy"""
        url = self._image_url(tile)                
        req = Request(url)                         
        req.add_header('User-agent','Anaconda 3')  
        fh = urlopen(req) 
        im_data = io.BytesIO(fh.read())            
        fh.close()                                 
        img = Image.open(im_data)  
        if grey_background:
            img = img.convert("L")             
        img = img.convert(self.desired_tile_form)  
        return img, self.tileextent(tile), 'lower' 
    
    # reformat web request for street map spoofing
    cimgt.OSM.get_image = image_spoof 
    img = cimgt.OSM()
    
    fig = plt.figure(figsize=(figsize,figsize)) 
    
    # project using coordinate reference system (CRS) of street map
    ax = plt.axes(projection=img.crs) 
    data_crs = ccrs.PlateCarree()
    
    # compute OSM scale
    scale = int(100/np.log(radius))
    scale = (scale<20) and scale or 19
    
    # compute extent of map
    lon,lat = geometry.centroid.x, geometry.centroid.y
    dist = radius*1.1
    dist_cnr = np.sqrt(2*dist**2)
    top_left = cgeo.Geodesic().direct(points=(lon,lat),azimuths=-45,distances=dist_cnr)[:,0:2][0]
    bot_right = cgeo.Geodesic().direct(points=(lon,lat),azimuths=135,distances=dist_cnr)[:,0:2][0]
    extent = [top_left[0], bot_right[0], bot_right[1], top_left[1]]
    ax.set_extent(extent)
    
    # add OSM with zoom specification
    ax.add_image(img, int(scale)) 
    
    # add building on map
    ax.add_geometries(geometry, crs=data_crs, color='tab:blue')
    
    # add OSM copyright
    if include_OSM_copyright:
        ax.text(0.5, -0.035, '\xa9 OpenStreetMap contributors', fontsize='x-large', horizontalalignment='center',verticalalignment='bottom', transform=ax.transAxes)
        
    # sauvegarde de l'image
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        
    return fig,ax



def neighbourhood_map(batiment_groupe_id,save=True):
    """
    carte des alentours d'un bâtiment de la BDNB

    Parameters
    ----------
    batiment_groupe_id : TYPE
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    # requête à la BDNB
    r = requests.get('https://api.bdnb.io/v1/bdnb/donnees/batiment_groupe_complet/adresse',
                     params={'batiment_groupe_id': 'eq.'+batiment_groupe_id,#},
                             'select': 'batiment_groupe_id, s_geom_groupe, contient_fictive_geom_groupe, geom_groupe'},
                     headers = {"Accept": "application/geo+json"},)

    # print(r.text)
    
    # lecture des données d'API
    gdf = gpd.read_file(io.StringIO(r.text))
    gdf = gdf[['geometry']]
    gdf = gdf.set_crs(epsg=2154, allow_override=True)
    gdf = gdf[gdf.columns[~gdf.isnull().all()]]
    
    # reprojection en longitude latitude
    gdf = gdf.to_crs(epsg=4326) 
    
    # sauvegarde de la carte
    if save:
        save_path = os.path.join('output',folder,'figs','{}_map.png'.format(batiment_groupe_id))
    else:
        save_path = None
    fig,ax = draw_local_map(gdf.iloc[0].geometry, save_path=save_path)
    plt.show()
    plt.close()
    return


def get_batiment_groupe_infos(batiment_groupe_id,variables=None):
    """
    requete à l'API de la BDNB pour récupérer les informations d'un bâtiment
    (par l'usage de l'identifiant batiment_groupe_id)

    Parameters
    ----------
    batiment_groupe_id : str
        DESCRIPTION.
    variables : list, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """
    # requête à la BDNB
    r = requests.get('https://api.bdnb.io/v1/bdnb/donnees/batiment_groupe_complet/adresse',
                     params={'batiment_groupe_id': 'eq.'+batiment_groupe_id},
                     headers = {"Accept": "application/geo+json"},)

    data = json.loads(r.text)
    
    if isinstance(variables, list):
        res = dict()
        for key in variables:
            res[key] = data.get('features')[0].get('properties').get(key)
    elif isinstance(variables, str):
        res = dict()
        res[variables] = data.get('features')[0].get('properties').get(variables)
    else:
        res = data
    return res
    
    
    
# =============================================================================
# main script
# =============================================================================
def main():
    tic = time.time()
    
    # get layers name
    if False:
        layers = get_layer_names()
        
        
    # benchmark opening 
    if False:
        print(speed_test_opening()) 
        print(speed_test_opening(dask_only=True, plot=True)) 
    
    
    # plot des diagnostics suspects
    if False:
        plot_raw_suspects()
    
    # neighbourhood map
    if False:
        bg_id_list = ['bdnb-bg-RGSM-7GV4-4QBK', 'bdnb-bg-FHEF-WAAZ-S5XC', 'bdnb-bg-9CBX-DZ3C-1DYC','bdnb-bg-243J-HGRU-FVA5']
        # bg_id_list = ['bdnb-bg-C1W3-KNUP-R5E2']
        for bg_id in bg_id_list:
            neighbourhood_map(batiment_groupe_id=bg_id)
            
            
    # analyse des champs modifier pour obtenir des gains de DPE
    if True:
        number_batiment_groupe = 49304 #'all' 
        # suspect_identification(number_batiment_groupe=number_batiment_groupe, plot=False)
        # suspicious_batiment_group_dict_dpe_id, suspicious_batiment_group_dict_dpe_number = analysis_suspicious_DPE(number_batiment_groupe=number_batiment_groupe, plot=['bdnb-bg-C1W3-KNUP-R5E2'])
        
        # for k,v in suspiciou^
                    
        # test = 'bdnb-bg-RGSM-7GV4-4QBK' # appartement avec un echangement d'isolation et de chauffage 
        # test = 'bdnb-bg-FHEF-WAAZ-S5XC' # appartement avec un changement de chaudière collective
        # test = 'bdnb-bg-9CBX-DZ3C-1DYC' # maison, aucun doute pratiquement
        # test = 'bdnb-bg-243J-HGRU-FVA5' # incertitude sur les compléments d'adresse
        test = 'bdnb-bg-98CZ-MLWR-CH3E' # de G à C # faux positif, 1er etage et RDC
        # test = 'bdnb-bg-C1W3-KNUP-R5E2' # de G à C # tout change
        # test = 'bdnb-bg-G9F7-7KST-V2YN' # logement collectif classique, passage de F à E
        
        infos_test = get_batiment_groupe_infos(test,variables=['l_libelle_adr','nb_log','annee_construction'])
        
        suspicious_batiment_group_dict_dpe_id, suspicious_batiment_group_dict_dpe_number = analysis_suspicious_DPE(number_batiment_groupe=number_batiment_groupe, plot=[test])
        neighbourhood_map(batiment_groupe_id=test)
        
        test_dpe_ids = suspicious_batiment_group_dict_dpe_id.get(test)[0]
        gains_test_dpe = suspicious_batiment_group_dict_dpe_number.get(test)[0]
        
        print('Adresse du bâtiment : {} ({} logements)'.format(infos_test.get('l_libelle_adr'),infos_test.get('nb_log')))
        print('Passage de {} à {}'.format(*[{n:chr(ord('@')+n) for n in range(1,10)}.get(int(e)) for e in gains_test_dpe]))
        
        suspect_folder = 'raw_suspicious_DPE'
        path = os.path.join(output_folder,folder,suspect_folder)
        dpe_dates_data_test = pd.read_csv(os.path.join(path,'{}.csv'.format(test)))
        dpe_dates_data_test = dpe_dates_data_test[dpe_dates_data_test.identifiant_dpe.isin(test_dpe_ids)]
        print("Date d'établissement des DPE : {}, {}".format(*dpe_dates_data_test.date_etablissement_dpe.values))
        
        print(test_dpe_ids)
        
        difference_1, difference_2 = difference_dpe_details(test_dpe_ids[0],test_dpe_ids[1])
        
        if any(['adresse_bien' in e for e in difference_1.variables.values]):
            print('Caution! Different address element')

        difference_1.to_csv(os.path.join(output_folder,folder,'diff_{}.csv'.format(test_dpe_ids[0])))
        difference_2.to_csv(os.path.join(output_folder,folder,'diff_{}.csv'.format(test_dpe_ids[1])))
    
        print(difference_1.set_index('variables').index.to_list())
        print(difference_2.set_index('variables').index.to_list())
        # print()
        # print(difference_1.set_index('variables').loc['mur--description'].values)
        # print(difference_2.set_index('variables').loc['mur--description'].values)
        # print()
        # print(difference_1.set_index('variables').loc['plancher_bas--methode_saisie_u'].values)
        # print(difference_2.set_index('variables').loc['plancher_bas--methode_saisie_u'].values)
        # print()
        # print(difference_1.set_index('variables').loc['baie_vitree--methode_saisie_perf_vitrage'].values)
        # print(difference_2.set_index('variables').loc['baie_vitree--methode_saisie_perf_vitrage'].values)
        
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()