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
# from matplotlib.ticker import MaxNLocator
from datetime import date
import matplotlib.dates as mdates
import seaborn as sns
# import re
import requests
import io
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import cartopy.geodesic as cgeo
from urllib.request import urlopen, Request
from PIL import Image
# import xmltodict
import json
from urllib.error import HTTPError
from unidecode import unidecode
from pyogrio.errors import DataSourceError
# from pyproj import Transformer






# =============================================================================
# Fonctions relatives à l'ouverture (optimale) de la BDNB
# =============================================================================

def get_layer_names(dep='75'):
    # tested file
    file = os.path.join('data','BDNB','open_data_millesime_2023-11-a_dep{}_gpkg'.format(dep),'gpkg','bdnb.gpkg')
    
    # list of layers in file
    layers_list = fiona.listlayers(file)
    
    return layers_list
    

def speed_test_opening(dep='75',dask_only=False, plot=False):
    # tested file
    file = os.path.join('data','BDNB','open_data_millesime_2023-11-a_dep{}_gpkg'.format(dep),'gpkg','bdnb.gpkg')
    
    if dask_only:
        test_npartitions = False
        test_chunksize = False
        
        if test_npartitions:
            npartitions_list = list(range(1,21))
            opening_speed_list = []
            for npartitions in npartitions_list:
                tic = time.time()
                _ = dask_geopandas.read_file(file, npartitions=npartitions, layer='dpe_logement')
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
                _ = dask_geopandas.read_file(file, chunksize=cs, layer='dpe_logement')
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
        _ = gpd.read_file(file, layer='adresse_compile')
        tac = time.time()
        methods_speed_dict['gpd_vanilla'] = tac-tic
        
        # geopandas pyogrio
        tic = time.time()
        _ = gpd.read_file(file, engine='pyogrio', layer='adresse_compile')
        tac = time.time()
        methods_speed_dict['gpd_pyogrio'] = tac-tic
        
        # dask-geopandas
        tic = time.time()
        _ = dask_geopandas.read_file(file, npartitions=4, layer='adresse_compile')
        tac = time.time()
        methods_speed_dict['dask_gpd'] = tac-tic
    
        return methods_speed_dict


def download_bdnb(dep):
    # TODO: télécharger le fichier manquant
    url = 'https://open-data.s3.fr-par.scw.cloud/bdnb_millesime_2023-11-a/millesime_2023-11-a_dep{}/open_data_millesime_2023-11-a_dep{}_gpkg.zip'.format(dep,dep)
    print(url)
    return 
    

def get_bdnb(dep='75',chunksize=5e4):
    """
    Ouvre de manière non compilée les données de la BDNB d'un département, selon 3 tables:
        - dpe_logement
        - rel_batiment_groupe_dpe_logement
        - batiment_groupe_compile

    Parameters
    ----------
    dep : str, optional
        code du département. The default is '75'.
    chunksize : float, optional
        taille des chunk dask. The default is 5e4.

    Raises
    ------
    DataSourceError
        Si le fichier demandé n'est pas disponible.

    Returns
    -------
    bdnb_dpe_logement : TYPE
        DESCRIPTION.
    bdnb_rel_batiment_groupe_dpe_logement : TYPE
        DESCRIPTION.
    bdnb_batiment_groupe_compile : TYPE
        DESCRIPTION.

    """
    # TODO : à modifier quand j'aurais les données complètes
    file = os.path.join('data','BDNB','open_data_millesime_2023-11-a_dep{}_gpkg'.format(dep),'gpkg','bdnb.gpkg')
    try:
        bdnb_dpe_logement = dask_geopandas.read_file(file, chunksize=chunksize, layer='dpe_logement')
        bdnb_rel_batiment_groupe_dpe_logement = dask_geopandas.read_file(file, chunksize=chunksize, layer='rel_batiment_groupe_dpe_logement')
        bdnb_batiment_groupe_compile = dask_geopandas.read_file(file, chunksize=chunksize, layer='batiment_groupe_compile')
    except DataSourceError:
        # TODO: télécharger le fichier manquant
        download_bdnb(dep=dep)
        raise DataSourceError('Fichier {} indisponible.'.format(file))
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


def suspect_identification(path,plot=False, force=False, number_batiment_groupe=1000):
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
        save_path = os.path.join(path,suspect_folder)
        if suspect_folder not in os.listdir(os.path.join(path)):
            os.mkdir(save_path)
        df_dpe_bg_id.to_csv(os.path.join(save_path,'{}.csv'.format(bg_id)),index=False)
        
    return len(batiment_group_list)


def plot_raw_suspects(folder,output_folder,suspect_folder):
    """
    Visualisation de la première sélection brute de suspects potentiels

    Returns
    -------
    None.

    """
    if folder not in os.listdir(output_folder):
        suspect_identification(path=os.path.join(output_folder,folder))
    
    done_batiment_group_list = os.listdir(os.path.join(output_folder, folder,suspect_folder))
    done_batiment_group_list = [s.replace('.csv','') for s in done_batiment_group_list if s.endswith('.csv')]
    
    if len(done_batiment_group_list) == 0:
        suspect_identification(path=os.path.join(output_folder,folder))
        done_batiment_group_list = os.listdir(os.path.join(output_folder, folder))
        done_batiment_group_list = [s.replace('.csv','') for s in done_batiment_group_list if s.endswith('.csv')]
        
    for bg_id in tqdm.tqdm(done_batiment_group_list):
        df_dpe_bg_id = pd.read_csv(os.path.join(output_folder, folder, suspect_folder,'{}.csv'.format(bg_id)))
        # print(bg_id, df_dpe_bg_id.columns)
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



def open_dpe_details(dpe_id, sheet_name='logement', retry=True):
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
        if not retry:
            return pd.DataFrame()
        print('Downloading XLS details of DPE {} from observatoire-dpe-audit.ademe.fr...'.format(dpe_id))
        
        download_dpe_details(dpe_id, force=False)
        
        try:
            dpe_xls_path = os.path.join('data','DPE','XLS','{}.xlsx'.format(dpe_id))
            dpe_data = pd.read_excel(dpe_xls_path, sheet_name=sheet_name)
        except FileNotFoundError:
            print('Error: {} is not available from observatoire-dpe-audit.ademe.fr :('.format(dpe_id))
            return pd.DataFrame()
        
    group_name = {'logement':'caracteristique_generale','administratif':'administratif', 'lexique':'lexique'}.get(sheet_name)
    
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


def intersection_dpe_details(dpe_data_1,dpe_data_2,duplicates_keep='first'):
    """
    Sélection des variables des données DPE communes entre deux jeux de données

    Parameters
    ----------
    dpe_data_1 : TYPE
        DESCRIPTION.
    dpe_data_2 : TYPE
        DESCRIPTION.
    duplicates_keep : {‘first’, ‘last’, False}, optional
        DESCRIPTION. The default is 'first'.

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
    
    dpe_data_1 = dpe_data_1.drop_duplicates(subset=['variables'], keep=duplicates_keep).reset_index(drop=True)
    dpe_data_2 = dpe_data_2.drop_duplicates(subset=['variables'], keep=duplicates_keep).reset_index(drop=True)
    
    return dpe_data_1, dpe_data_2
    

def difference_dpe_details(dpe_id_1,dpe_id_2,download_retry=True):
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
    data_logement_1 = open_dpe_details(dpe_id_1,sheet_name='logement',retry=download_retry)
    data_logement_2 = open_dpe_details(dpe_id_2,sheet_name='logement',retry=download_retry)
    
    if data_logement_1.empty or data_logement_2.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    data_logement_1, data_logement_2 = intersection_dpe_details(data_logement_1, data_logement_2, duplicates_keep='first')
    
    data_admin_1 = open_dpe_details(dpe_id_1,sheet_name='administratif',retry=download_retry)
    data_admin_2 = open_dpe_details(dpe_id_2,sheet_name='administratif',retry=download_retry)
    data_admin_1, data_admin_2 = intersection_dpe_details(data_admin_1, data_admin_2, duplicates_keep='first')
    
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
        


def analysis_suspicious_DPE(save_path, plot=None,details=True,number_batiment_groupe=1000, force_suspect_id=False, force=False):
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
            bg_id_plot_list = []
    elif isinstance(plot,list):
        bg_id_plot_list = plot
        show_plots = False
    else:
        show_plots = False
        bg_id_plot_list = []
    show_details = details
    
    # save_path = os.path.join(output_folder,folder)
    
    suspect_folder = 'raw_suspicious_DPE'
    path = os.path.join(save_path,suspect_folder)
    if suspect_folder not in os.listdir(os.path.join(save_path)):
        os.mkdir(path)
        
    done_batiment_group_list = os.listdir(path)
    done_batiment_group_list = [s.replace('.csv','') for s in done_batiment_group_list if s.endswith('.csv')]
    
    if len(done_batiment_group_list) == 0 or force_suspect_id:
        suspect_identification(save_path, number_batiment_groupe=number_batiment_groupe)
        done_batiment_group_list = os.listdir(path)
        done_batiment_group_list = [s.replace('.csv','') for s in done_batiment_group_list if s.endswith('.csv')]

    suspicious_batiment_group_dict_dpe_number = dict()
    suspicious_batiment_group_dict_dpe_id = dict()
    
    if 'suspicious_batiment_group_dict_dpe_id.json' not in os.listdir(os.path.join(save_path)) or force:
        
        pbar = tqdm.tqdm(done_batiment_group_list, total=len(done_batiment_group_list))
        for bg_id in pbar:
            pbar.set_description(bg_id)
            pbar.refresh()
            
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
            if show_plots and 'figs' not in os.listdir(save_path):
                os.mkdir(os.path.join(save_path,'figs'))
            
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
                save_path = os.path.join(save_path,'figs','{}_dpe.png'.format(bg_id))
                plt.savefig(save_path, bbox_inches='tight')
                plt.show()
                plt.close()
        
        with open(os.path.join(save_path,'suspicious_batiment_group_dict_dpe_id.json'), 'w') as fp:
            json.dump(suspicious_batiment_group_dict_dpe_id, fp)
            
        class numpy_Encoder(json.JSONEncoder):
            """
            Réencodage des nombres numpy pour le module json.
            """
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(numpy_Encoder, self).default(obj)
            
        with open(os.path.join(save_path,'suspicious_batiment_group_dict_dpe_number.json'), 'w') as fp:
            json.dump(suspicious_batiment_group_dict_dpe_number, fp, cls=numpy_Encoder)
    
    # ouverture des fichiers générés
    with open(os.path.join(save_path,'suspicious_batiment_group_dict_dpe_id.json')) as f:
        suspicious_batiment_group_dict_dpe_id = json.load(f)
    with open(os.path.join(save_path,'suspicious_batiment_group_dict_dpe_number.json')) as f:
        suspicious_batiment_group_dict_dpe_number = json.load(f)
        
    
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
        if number_batiment_groupe == 'all':
            number_batiment_groupe = 49304 # TODO: to deal with this hardcoded number (len of batiment_group_list in suspect_identification)
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



def get_filtered_suspicious_DPE(path, force=False, show_details=True):
    """
    recuperation des dictionnaires filtrés au maximum

    Parameters
    ----------
    force : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    number_batiment_groupe = 'all' 
    suspicious_batiment_group_dict_dpe_id, suspicious_batiment_group_dict_dpe_number = analysis_suspicious_DPE(save_path=path, number_batiment_groupe=number_batiment_groupe, plot=False, details=False)
    
    suspicious_batiment_group_dict_dpe_id_filtered = dict()
    suspicious_batiment_group_dict_dpe_number_filtered = dict()
    
    if 'suspicious_batiment_group_dict_dpe_id_filtered.json' not in os.listdir(os.path.join(path)) or force:
        pbar = tqdm.tqdm(suspicious_batiment_group_dict_dpe_id.keys(), total=len(suspicious_batiment_group_dict_dpe_id.keys()))
        for bg_id in pbar:
            pbar.set_description(bg_id)
            pbar.refresh()
            
            for i,(dpe_id_1, dpe_id_2) in enumerate(suspicious_batiment_group_dict_dpe_id.get(bg_id)):
                
                # deal with strange error
                manual_exception_bg_id = ['bdnb-bg-LHGG-Y4PK-1R3C']
                if bg_id in manual_exception_bg_id:
                    continue
                else:
                    difference_1, difference_2 = difference_dpe_details(dpe_id_1,dpe_id_2,download_retry=False)
                
                if difference_1.empty or difference_2.empty:
                    continue
                
                difference_1 = difference_1.set_index('variables')
                difference_2 = difference_2.set_index('variables')
                
                # # filtre sur au moins un élément de l'adresse qui ne coincide pas (très large)
                # filter_address_infos = not any(['adresse_bien' in e for e in difference_1.index.values])
                
                # # filtre sur le nombre de mur et leur orientation (filtre fin)
                # filter_walls_orientation = True
                # if 'mur--orientation' in difference_1.index.values:
                #     murs_1 = sorted(difference_1.loc['mur--orientation'].values[0])
                #     murs_2 = sorted(difference_2.loc['mur--orientation'].values[0])
                #     if murs_1 != murs_2:
                #         filter_walls_orientation = False
                
                # ajout d'un filtre sur l'étage du logement (si donnée disponible)
                filter_floor_number = True
                if 'adresse_bien--compl_ref_logement' in difference_1.index:
                    floor_number_1 = decode_floor_number(difference_1.loc['adresse_bien--compl_ref_logement'].values[0])
                    floor_number_2 = decode_floor_number(difference_2.loc['adresse_bien--compl_ref_logement'].values[0])
                    if floor_number_1 != floor_number_2 or floor_number_1==-1 or floor_number_2==-1:
                        filter_floor_number = False
                
                # test en ne regardant que la coincidence explicite de l'étage
                # tester plusieurs forces de filtres
                global_filter = filter_floor_number
                if global_filter:
                    if bg_id not in suspicious_batiment_group_dict_dpe_id_filtered.keys():
                        suspicious_batiment_group_dict_dpe_id_filtered[bg_id] = []
                        suspicious_batiment_group_dict_dpe_number_filtered[bg_id] = []
                    suspicious_batiment_group_dict_dpe_id_filtered[bg_id].append([dpe_id_1, dpe_id_2])
                    suspicious_batiment_group_dict_dpe_number_filtered[bg_id].append(suspicious_batiment_group_dict_dpe_number.get(bg_id)[i])
                    
        with open(os.path.join(path,'suspicious_batiment_group_dict_dpe_id_filtered.json'), 'w') as fp:
            json.dump(suspicious_batiment_group_dict_dpe_id_filtered, fp)
        
        class numpy_Encoder(json.JSONEncoder):
            """
            Réencodage des nombres numpy pour le module json.
            """
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(numpy_Encoder, self).default(obj)
            
        with open(os.path.join(path,'suspicious_batiment_group_dict_dpe_number_filtered.json'), 'w') as fp:
            json.dump(suspicious_batiment_group_dict_dpe_number_filtered, fp, cls=numpy_Encoder)
             
    
    # ouverture des fichiers générés
    with open(os.path.join(path,'suspicious_batiment_group_dict_dpe_id_filtered.json')) as f:
        suspicious_batiment_group_dict_dpe_id_filtered = json.load(f)
    with open(os.path.join(path,'suspicious_batiment_group_dict_dpe_number_filtered.json')) as f:
        suspicious_batiment_group_dict_dpe_number_filtered = json.load(f)
        
    # calcul du nombre de batiments concernés par des DPE suspects
    number_suspicious_bg = len(suspicious_batiment_group_dict_dpe_number_filtered)
    
    # calcul du nombre de gains d'étiquettes moyens et des principaux changements 
    mean_dpe_gains = 0
    counter = 0
    dpe_gain_counter_dict = dict()
    for k,v in suspicious_batiment_group_dict_dpe_number_filtered.items():
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
        if number_batiment_groupe == 'all':
            number_batiment_groupe = 49304
        print('\nPourcentage de bâtiments présentants des DPE suspects :')
        print('\t- {:.1f}% ({}/{})'.format(number_suspicious_bg/number_batiment_groupe*100,number_suspicious_bg, number_batiment_groupe))
        
        print("\nGains moyens d'étiquettes :")
        print('\t- {:.1f} ({} obs.)'.format(mean_dpe_gains,number_suspicious_bg))
        
        print("Détails des gains d'étiquettes :")
        for (dpe_ini, dpe_redo), nb_chg in dpe_gain_counter_dict.items():
            letter_ini, letter_redo = number_to_letter_dict.get(dpe_ini), number_to_letter_dict.get(dpe_redo)
            print('\t- {} -> {} : {:>4.1f}% ({}/{})'.format(letter_ini, letter_redo, nb_chg/number_suspicious_bg*100, nb_chg, number_suspicious_bg))
        print('\n')

    return suspicious_batiment_group_dict_dpe_id_filtered, suspicious_batiment_group_dict_dpe_number_filtered



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
    extent = [float(f) for f in [top_left[0], bot_right[0], bot_right[1], top_left[1]]]
    ax.set_extent(extent, crs=ccrs.PlateCarree()) 
    
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



def neighbourhood_map(batiment_groupe_id, path,save=True):
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
        save_path = os.path.join(path,'figs','{}_map.png'.format(batiment_groupe_id))
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


def draw_city_map(list_bg_id, city='Paris', style='map',figsize=20, grey_background=True, save_path=None, include_OSM_copyright=True):
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
        
        # define map parameters
        dict_city_map_parameters = {'Paris':{'scale':13,
                                             'extent':[2.2199,2.4723,48.8091,48.9065],
                                             }
                                    }
        
        scale = dict_city_map_parameters.get(city).get('scale')
        extent = dict_city_map_parameters.get(city).get('extent')
        
        # retrive geometry info from bdnb
        _, _, bdnb_batiment_groupe_compile = get_bdnb()
        bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile[['batiment_groupe_id','geometry']]
        bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile.compute()
        bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile[bdnb_batiment_groupe_compile.batiment_groupe_id.isin(list_bg_id)]
        bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile.set_crs(epsg=2154, allow_override=True)
        
        # reprojection en longitude latitude
        bdnb_batiment_groupe_compile = bdnb_batiment_groupe_compile.to_crs(epsg=4326) 
        
        ax.set_extent(extent)
        
        # add OSM with zoom specification
        ax.add_image(img, int(scale)) 
        
        # add building on map
        ax.add_geometries(bdnb_batiment_groupe_compile.geometry, crs=data_crs, color='tab:blue')
        
        # add OSM copyright
        if include_OSM_copyright:
            ax.text(0.5, -0.035, '\xa9 OpenStreetMap contributors', fontsize='x-large', horizontalalignment='center',verticalalignment='bottom', transform=ax.transAxes)
            
        # sauvegarde de l'image
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            
        return fig,ax
    
    
def plot_dpe_distribution(path, dep='75', save=True, max_xlim=600):
    """
    graphe de la distribution des DPE, en indiquant les limites entre catégories

    Parameters
    ----------
    save : TYPE, optional
        DESCRIPTION. The default is True.
    max_xlim : TYPE, optional
        DESCRIPTION. The default is 600.

    Returns
    -------
    dpe_data : TYPE
        DESCRIPTION.

    """
    def get_dpe_conso(dep):
        dpe_data, _ , _ = get_bdnb(dep)
        dpe_data = dpe_data[dpe_data.type_dpe=='dpe arrêté 2021 3cl logement'][['conso_5_usages_ep_m2','conso_5_usages_ef_m2']].compute() 
        return dpe_data
    
    dpe_data = get_dpe_conso(dep=dep)
    dpe_data = dpe_data.dropna()
    dpe_data = dpe_data.map(int)
    counter_dict = dict(Counter(dpe_data.conso_5_usages_ep_m2))
    counter_dict_sorted = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[0])}
    
    etiquette_colors_dict = {'A':(0, 156, 109),'B':(82, 177, 83),'C':(120, 189, 118),'D':(244, 231, 15),'E':(240, 181, 15),'F':(235, 130, 53),'G':(215, 34, 31)}
    etiquette_colors_dict = {k: tuple(map(lambda x: x/255, v)) for k,v in etiquette_colors_dict.items()}
    etiquette_ep_dict = {'A':[0,70],'B':[70,110],'C':[110,180],'D':[180,250],'E':[250,330],'F':[330,420],'G':[420,np.inf]}
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=300,)
    for eti in etiquette_colors_dict.keys():
        inf_ep, sup_ep = etiquette_ep_dict.get(eti)
        color = etiquette_colors_dict.get(eti)
        counter_dict_eti = {k:v for k,v in counter_dict_sorted.items() if k > inf_ep and k <= sup_ep}
        ax.bar(list(counter_dict_eti.keys()), list(counter_dict_eti.values()), width=1., color=color, label=eti)
    
    ax.set_xlim([0,max_xlim])
    ax.set_ylabel("Nombre d'observations (département {})".format(dep))
    ax.legend()
    ax.set_xlabel("Consommation annuelle en énergie primaire (kWh.m$^{-2}$)")
    ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
    if save:
        save_path = os.path.join(path,'figs','distribution_dpe_{}.png'.format(dep))
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()
    return 


def plot_var_distribution(var, path, save=True, max_xlim=None,min_xlim=None,var_label=None,alpha=0.7,rounder=1,percentage=True,show=True):
    """
    graphe de la distribution des DPE, en indiquant les limites entre catégories

    Parameters
    ----------
    save : TYPE, optional
        DESCRIPTION. The default is True.
    max_xlim : TYPE, optional
        DESCRIPTION. The default is 600.

    Returns
    -------
    dpe_data : TYPE
        DESCRIPTION.

    """
    def get_bdnb_var(var=var):
        _, _ , bdnb = get_bdnb()
        # dpe_data = dpe_data[dpe_data.type_dpe=='dpe arrêté 2021 3cl logement'][['conso_5_usages_ep_m2','conso_5_usages_ef_m2']].compute() 
        return bdnb[var].compute()
    
    save = True
    
    bdnb_var_data = get_bdnb_var(var).dropna()
    bdnb_var_data = round(bdnb_var_data/rounder)*rounder
    counter_var = dict(Counter(bdnb_var_data))
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=300,)
    if percentage:
        ax.bar(list(counter_var.keys()), np.asarray(list(counter_var.values()))/sum(counter_var.values())*100, width=rounder, color='k',alpha=alpha,label='BDNB ({})'.format(sum(counter_var.values())))
        ylabel = "Pourcentage d'observations"
    else:
        ax.bar(list(counter_var.keys()), list(counter_var.values()), width=rounder, color='k',alpha=alpha,label='BDNB ({})'.format(sum(counter_var.values())))
        ylabel = "Nombre d'observations"
    
    if max_xlim is not None:
        ax.set_xlim(right=max_xlim)
    if min_xlim is not None:
        ax.set_xlim(left=min_xlim)
        
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_xlabel(var_label)
    # ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [max_xlim])
    if save:
        save_path = os.path.join(path,'figs','distribution_{}.png'.format(var))
    else:
        save_path = None
    plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
        plt.close()
    return fig,ax


def get_dpe_change_details(path,force=False,add_bdnb_data=True):
    """
    données de changement de champs d'un DPE à un autre

    Parameters
    ----------
    force : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    data_details : TYPE
        DESCRIPTION.

    """
    sbgd_filtered_dpe_id, sbgd_filtered_dpe_number = get_filtered_suspicious_DPE(path=path, show_details=False, force=force)
    
    data_details = {'bg_id':[],'dpe_id_1':[],'dpe_id_2':[],'dpe_letter_1':[],'dpe_letter_2':[]}
    
    if 'dpe_change_details.csv' not in os.listdir(os.path.join(path)) or force:
        pbar = tqdm.tqdm(sbgd_filtered_dpe_id.keys(), total=len(sbgd_filtered_dpe_id.keys()))
        for bg_id in pbar:
            pbar.set_description(bg_id)
            pbar.refresh()
            
            dpe_id_list_list = sbgd_filtered_dpe_id.get(bg_id)
            dpe_number_list_list = sbgd_filtered_dpe_number.get(bg_id)
            
            # parcours des dpe pour récupérer et stocker les données de différence
            for dpe_id_list, dpe_number_list in zip(dpe_id_list_list, dpe_number_list_list):
                dpe_letter_list = [{n:chr(ord('@')+n) for n in range(1,10)}.get(int(dpe)) for dpe in dpe_number_list]
                
                data_details['bg_id'].append(bg_id)
                data_details['dpe_id_1'].append(dpe_id_list[0])
                data_details['dpe_id_2'].append(dpe_id_list[1])
                data_details['dpe_letter_1'].append(dpe_letter_list[0])
                data_details['dpe_letter_2'].append(dpe_letter_list[1])
                
                # récupération des cahmps différents 
                difference_1, difference_2 = difference_dpe_details(*dpe_id_list)
                
                # récupération des informations administratives
                admin_data_1 = open_dpe_details(dpe_id_list[0],sheet_name='administratif')
                admin_data_2 = open_dpe_details(dpe_id_list[1],sheet_name='administratif')
                
                # sélection des variables concernant le diagnostiqueur
                diagnostiqueur_variables_1 = [diag for diag in admin_data_1.variables if 'diagnostiqueur--' in diag]
                diagnostiqueur_variables_2 = [diag for diag in admin_data_2.variables if 'diagnostiqueur--' in diag]
                
                # ajout des données diagnostiqueurs dans le dictionnaire de résultat
                diagnostiqueur_variables = [(admin_data_1, diagnostiqueur_variables_1), (admin_data_2, diagnostiqueur_variables_2)]
                
                for i,(admin_data, diag_vars) in enumerate(diagnostiqueur_variables):
                    for variable, value in zip(admin_data.variables, admin_data.dpe_values):
                        
                        if variable not in diag_vars:
                            continue
                        
                        # ajout du suffixe du dpe (1 ou 2)
                        variable = variable + '--{}'.format(i+1)
                        
                        # ajout de l'entrée dans le distionnaire de résultat si nouveau
                        if variable not in data_details.keys():
                            data_details[variable] = [np.nan]*(len(data_details.get('dpe_id_1'))-1)
                        
                        # print(variable)
                        data_details[variable].append(value)
                
                
                for variable_1, value_1, variable_2, value_2 in zip(difference_1.variables, difference_1.dpe_values, difference_2.variables, difference_2.dpe_values):
                    
                    if any([e in variable_1 for e in ['administratif','diagnostiqueur']]):
                        continue
                    
                    # ajout du suffixe du dpe (1 ou 2)
                    variable_1 = variable_1 + '--1'
                    variable_2 = variable_2 + '--2'
                    
                    # ajout de l'entrée dans le distionnaire de résultat si nouveau
                    if variable_1 not in data_details.keys():
                        data_details[variable_1] = [np.nan]*(len(data_details.get('dpe_id_1'))-1)
                    if variable_2 not in data_details.keys():
                        data_details[variable_2] = [np.nan]*(len(data_details.get('dpe_id_1'))-1)
                    
                    # suppresion des virgules pour enregistrement en csv
                    if isinstance(value_1, str):
                        value_1 = value_1.replace(',',';')
                    elif isinstance(value_1, list):
                        value_1 = [e.replace(',',';') for e in value_1 if isinstance(e, str)]
                    if isinstance(value_2, str):
                        value_2 = value_2.replace(',',';')
                    elif isinstance(value_2, list):
                        value_2 = [e.replace(',',';') for e in value_2 if isinstance(e, str)]
                    
                    data_details[variable_1].append(value_1)
                    data_details[variable_2].append(value_2)
                
                data_details_len = len(data_details.get('dpe_id_1'))
                for k,v in data_details.items():
                    if len(v) < data_details_len:
                        data_details[k] = data_details.get(k) + [np.nan]*(data_details_len-len(v))
                    
        data_details = pd.DataFrame().from_dict(data_details)
        
        # ajout des données aux bâtiments pour étude
        def get_bg_infos():
            _, _ , bg_data = get_bdnb()
            bg_data = bg_data[['batiment_groupe_id','bdtopo_bat_hauteur_mean','ffo_bat_annee_construction']].compute()
            return bg_data
        
        bg_infos = get_bg_infos()
        bg_infos = bg_infos.set_index('batiment_groupe_id')
        bg_infos = bg_infos.loc[data_details.bg_id]
        bg_infos = bg_infos.reset_index(drop=True)
        
        data_details = data_details.join(bg_infos)
        data_details.to_csv(os.path.join(path,'dpe_change_details.csv'), index=False)
        
    data_details = pd.read_csv(os.path.join(path,'dpe_change_details.csv'), low_memory=False, sep=',')
    return data_details


def decode_floor_number(s):
    """
    Récupération de l'étage du logement du DPE

    Parameters
    ----------
    s : str
        Chaine de caractère correspondant au champ 'adresse_bien--compl_ref_logement' dans les données XLS DPE.

    Returns
    -------
    floor_number : TYPE
        DESCRIPTION.

    """
    s = unidecode(s.lower()).replace('-eme','').replace('eme','').replace(':','').replace('rdc','0').replace(' ',',').replace(';',',').replace('er','').replace('iem','')
    s_list = s.split(',')
    s_list = [e for e in s_list if len(e)>0]
    try:
        floor_idx = s_list.index("etage")
    except ValueError:
        try:
            floor_idx = s_list.index("etag")
        except ValueError:
            floor_idx = None
            floor_number = -1
    
    if floor_idx is not None:
        prev_idx, next_idx = max(floor_idx-1,0), min(floor_idx+1,len(s_list)-1)
        if prev_idx != floor_idx:
            try:
                prev_floor_number = int(s_list[prev_idx])
            except ValueError:
                prev_floor_number = -1
        else:
            prev_floor_number = -1
            
        if next_idx != floor_idx:
            try:
                next_floor_number = int(s_list[next_idx])
            except ValueError:
                next_floor_number = -1
        else:
            next_floor_number = -1
        floor_number = max(prev_floor_number, next_floor_number)
    return floor_number


def get_lexique_DPE():
    random_dpe_id = '2175E0016996P'
    lexique = open_dpe_details(random_dpe_id,sheet_name='lexique',retry=True)
    res_dict = dict()
    for k,v in zip(lexique.variables,lexique.dpe_values):
        res_dict[k.replace('lexique--','')] = v
    return res_dict
    
#%% ===========================================================================
# main script
# =============================================================================
def main():
    tic = time.time()
    
    # définition de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # définition du fichier de sortie
    output_folder = os.path.join('output')
    folder = '{}_DPE_successifs'.format(today)
    if folder not in os.listdir(output_folder):
        os.mkdir(os.path.join(output_folder,folder))
    if 'figs' not in os.listdir(os.path.join(output_folder, folder)):
        os.mkdir(os.path.join(output_folder,folder,'figs'))
    
    output_path = os.path.join(output_folder,folder)
    
    pd.set_option('future.no_silent_downcasting', True)
    
    departement = '75' # pas encore prévu pour que ça puisse être différent
    # departement = '24'
    
    #%% get layers name
    if False:
        layers = get_layer_names()
        print(layers)
        
    #%% benchmark opening 
    if False:
        print(speed_test_opening()) 
        print(speed_test_opening(dask_only=True, plot=True)) 
    

    
    #%% graphe des distribution des DPE présents dans la BDNB (paris pour l'instant)
    if False:
        # uniquement cette fonction a été mise à jour pour le changement le département 
        plot_dpe_distribution(dep=departement, path=output_path,max_xlim=600)
    
    #%% plot des diagnostics suspects
    if False:
        plot_raw_suspects(folder,output_folder,'raw_suspicious_DPE')
    
    #%% neighbourhood map
    if False:
        bg_id_list = ['bdnb-bg-RGSM-7GV4-4QBK', 'bdnb-bg-FHEF-WAAZ-S5XC', 'bdnb-bg-9CBX-DZ3C-1DYC','bdnb-bg-243J-HGRU-FVA5']
        # bg_id_list = ['bdnb-bg-C1W3-KNUP-R5E2']
        for bg_id in bg_id_list:
            neighbourhood_map(path=output_path, batiment_groupe_id=bg_id)
    
    if False:
        number_batiment_groupe = 'all' 
        suspicious_batiment_group_dict_dpe_id, _ = analysis_suspicious_DPE(save_path=output_path,number_batiment_groupe=number_batiment_groupe, details=False)
        bg_id_list = list(suspicious_batiment_group_dict_dpe_id.keys())
        draw_city_map(list_bg_id=bg_id_list)
            
            
    #%% analyse des champs modifier pour obtenir des gains de DPE pour un bâtiment individuel (tests)
    if False:
        number_batiment_groupe = 'all' 
                    
        # test = 'bdnb-bg-RGSM-7GV4-4QBK' # appartement avec un echangement d'isolation et de chauffage 
        # test = 'bdnb-bg-FHEF-WAAZ-S5XC' # appartement avec un changement de chaudière collective
        # test = 'bdnb-bg-9CBX-DZ3C-1DYC' # maison, aucun doute pratiquement
        # test = 'bdnb-bg-243J-HGRU-FVA5' # incertitude sur les compléments d'adresse
        # test = 'bdnb-bg-98CZ-MLWR-CH3E' # de G à C # faux positif, 1er etage et RDC
        # test = 'bdnb-bg-C1W3-KNUP-R5E2' # de G à C # tout change
        # test = 'bdnb-bg-G9F7-7KST-V2YN' # logement collectif classique, passage de F à E
        
        # tests pour le filtre général qui marche pas 
        # test = 'bdnb-bg-8GU6-RUCE-MAGP'
        # test = 'bdnb-bg-JP7N-2Q3J-2J8P'
        # test = 'bdnb-bg-NFA2-348V-RQZF'
        # test = 'bdnb-bg-LHGG-Y4PK-1R3C'
        # test = 'bdnb-bg-ZFYJ-L3CU-BEPX'
        # test = 'bdnb-bg-6R5M-AXNQ-N3YN'
        test = 'bdnb-bg-7YS1-GYPX-XJYY'

        infos_test = get_batiment_groupe_infos(test,variables=['l_libelle_adr','nb_log','annee_construction'])
        
        suspicious_batiment_group_dict_dpe_id, suspicious_batiment_group_dict_dpe_number = analysis_suspicious_DPE(save_path=output_path, number_batiment_groupe=number_batiment_groupe, plot=[test], details=False)
        neighbourhood_map(path=output_path, batiment_groupe_id=test)
        
        test_dpe_ids = suspicious_batiment_group_dict_dpe_id.get(test)[0]
        gains_test_dpe = suspicious_batiment_group_dict_dpe_number.get(test)[0]
        
        print()
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
    
        # print(difference_1.set_index('variables').index.to_list())
        # print(difference_2.set_index('variables').index.to_list())
        # print()
        # print(difference_1.set_index('variables').loc['inertie--classe_inertie'].values)
        # print(difference_2.set_index('variables').loc['inertie--classe_inertie'].values)
        # # print()
        # print(difference_1.set_index('variables').loc['generateur_ecs--description'].values)
        # print(difference_2.set_index('variables').loc['generateur_ecs--description'].values)
        # # print()
        # print(difference_1.set_index('variables').loc['emetteur_chauffage--type_emission_distribution'].values)
        # print(difference_2.set_index('variables').loc['emetteur_chauffage--type_emission_distribution'].values)
        
        
    #%% téléchargement des XLS des DPE
    if False:
        number_batiment_groupe = 'all' 
        suspicious_batiment_group_dict_dpe_id, suspicious_batiment_group_dict_dpe_number = analysis_suspicious_DPE(save_path=output_path,number_batiment_groupe=number_batiment_groupe, plot=False, details=False)
        list_dpe_ids = [dpe for dpe_list_list in list(suspicious_batiment_group_dict_dpe_id.values()) for dpe_list in dpe_list_list for dpe in dpe_list]
        
        pbar = tqdm.tqdm(enumerate(list_dpe_ids), total=len(list_dpe_ids))
        for i,dpe_id in pbar:
            pbar.set_description(dpe_id)
            pbar.refresh()
            download_dpe_details(dpe_id)
    
    
    #%% test d'un nouveau filtre entre les appartements
    if False:
        dpe_change = get_dpe_change_details(path=output_path,force=False)
        
        s = 0
        for i,(rc1, rc2) in enumerate(zip(dpe_change['adresse_bien--compl_ref_logement--1'], dpe_change['adresse_bien--compl_ref_logement--2'])):
            if i < 1e9:
                if pd.isnull(rc1):
                    continue
                
                rc1_floor_number = decode_floor_number(rc1)
                rc2_floor_number = decode_floor_number(rc2)
                
                if rc1_floor_number == rc2_floor_number and rc1_floor_number!=-1:
                    print(rc1,'(',rc1_floor_number,')',' ET ',rc2,'(',rc2_floor_number,')')
                    # print()
                else:
                    s+=1
        print(len(dpe_change)-s,'/',len(dpe_change))
        
            
    #%% filtre des couples de DPE faux positifs 
    if False:
        # TODO : à vérifier 
        sbgd_filtered_dpe_id, sbgd_filtered_dpe_number = get_filtered_suspicious_DPE(path=output_path,show_details=True, force=False)
        
    #%% statistiques sur les données DPE de différence
    if False:
        import ast
        
        dpe_change = get_dpe_change_details(path=output_path,force=False)
        
        # TODO: il faut trier les données pour avoir les mêmes orientations des murs etc
        dpe_change_mur = dpe_change[[c for c in dpe_change.columns if 'mur--' in c]]
        tot_n = 0
        eglenmur_n = 0
        noneglenmur_n = 0
        for mur_or_1, mur_or_2 in zip(dpe_change_mur['mur--orientation--1'],dpe_change_mur['mur--orientation--2']):
            tot_n += 1
            if isinstance(mur_or_1, str):
                try:
                    mur_or_1 = ast.literal_eval(mur_or_1)
                except ValueError:
                    mur_or_1 = [mur_or_1]
                try:
                    mur_or_2 = ast.literal_eval(mur_or_2)
                except ValueError:
                    mur_or_2 = [mur_or_2]
                
                if len(mur_or_1)==len(mur_or_2):
                    eglenmur_n += 1
                else:
                    noneglenmur_n += 1
        
        # print(eglenmur_n, noneglenmur_n, tot_n)
        
        # exclusion des variables de consommations (resultats de calculs finaux)
        conso_var_list = ['installation_chauffage--besoin_ch--1',
                          'installation_chauffage--besoin_ch--2',
                          'installation_chauffage--besoin_ch_depensier--1',
                          'installation_chauffage--besoin_ch_depensier--2',
                          'installation_chauffage--conso_ch--1',
                          'installation_chauffage--conso_ch--2',
                          'installation_chauffage--conso_ch_depensier--1',
                          'installation_chauffage--conso_ch_depensier--2',
                          'generateur_chauffage--conso_ch--1',
                          'generateur_chauffage--conso_ch--2',
                          'generateur_chauffage--conso_ch_depensier--1',
                          'generateur_chauffage--conso_ch_depensier--2',
                          'installation_ecs--conso_ecs--1',
                          'installation_ecs--conso_ecs--2',
                          'installation_ecs--conso_ecs_depensier--1',
                          'installation_ecs--conso_ecs_depensier--2',
                          'installation_ecs--besoin_ecs--1', 
                          'installation_ecs--besoin_ecs--2',
                          'installation_ecs--besoin_ecs_depensier--1',
                          'installation_ecs--besoin_ecs_depensier--2',
                          'generateur_ecs--conso_ecs--1',
                          'generateur_ecs--conso_ecs--2',
                          'generateur_ecs--conso_ecs_depensier--1',
                          'generateur_ecs--conso_ecs_depensier--2',
                          'climatisation--conso_fr--1',
                          'climatisation--conso_fr--2',
                          'climatisation--besoin_fr--1',
                          'climatisation--besoin_fr--2',
                          'climatisation--conso_fr_depensier--1',
                          'climatisation--conso_fr_depensier--2']
        
        # exclusion des variables génériques
        generic_var_list = ['bg_id', 
                            'bdtopo_bat_hauteur_mean',
                            'ffo_bat_annee_construction', 
                            'dpe_letter_1', 
                            'dpe_letter_2',
                            'dpe_id_1',
                            'dpe_id_2',
                            'batiment_groupe_id',
                            'caracteristique_generale--surface_habitable_logement--1',
                            'caracteristique_generale--surface_habitable_logement--2',
                            'caracteristique_generale--hsp--1',
                            'caracteristique_generale--hsp--2',
                            'caracteristique_generale--surface_habitable_immeuble--1',
                            'caracteristique_generale--surface_habitable_immeuble--2']
        
        # exclusison des variables étant des résultats de calculs intermédiaires
        computation_results_var_list = ['ventilation--hvent--1',
                                        'ventilation--hvent--2',
                                        'ventilation--hperm--1',
                                        'ventilation--hperm--2',
                                        'ventilation--conso_auxiliaire_ventilation--1',
                                        'ventilation--conso_auxiliaire_ventilation--2']
        
        # exclusion des données diagnostiqueurs et adresse
        diag_var_list = [c for c in dpe_change.columns if c.startswith('diagnostiqueur')]
        adresse_var_list = [c for c in dpe_change.columns if c.startswith('adresse_bien')]
        
        excluded_cols = generic_var_list+conso_var_list+diag_var_list+computation_results_var_list+adresse_var_list
        
        res = {c:len(dpe_change[c].dropna()) for c in dpe_change.columns if c not in excluded_cols}
        sorted_res = dict(sorted(res.items(), key=lambda kv: kv[1], reverse=False))
        
        res_group_list = list(set([elem.split('--')[0] for elem in res.keys()]))
        res_group = dict()
        for g in res_group_list:
            res_g_dict = {k:v for k,v in sorted_res.items() if g in k}
            if g=='installation_ecs':
                # print(res_g_dict)
                pass
            res_group[g] = max(res_g_dict.values())
            
        sorted_res_group = dict(sorted(res_group.items(), key=lambda kv: kv[1], reverse=False))
        sorted_res_group = {k:v for k,v in sorted_res_group.items() if v > 10}
        
        
        fig, ax = plt.subplots(dpi=300,figsize=(6,len(sorted_res_group)//2))
        ax.barh(range(len(sorted_res_group)), list(sorted_res_group.values()), align='center')
        ax.set_yticks(range(len(sorted_res_group)), list(sorted_res_group.keys()))
        ax.set_xlim(right=len(dpe_change))
        ax.set_xlabel("Nombre d'observations (N={})".format(len(dpe_change)))
        ax.set_ylabel('Catégories des différentes variables renseignées')
        plt.show()
        
        
        specific_group = 'mur'
        
        res_specific = {c:len(dpe_change[c].dropna()) for c in dpe_change.columns if c.startswith('{}--'.format(specific_group)) and c.endswith('--1') and c not in excluded_cols}
        sorted_res_specific = dict(sorted(res_specific.items(), key=lambda kv: kv[1], reverse=False))
        
        fig, ax = plt.subplots(dpi=300,figsize=(6,len(sorted_res_specific)//2))
        ax.barh(range(len(sorted_res_specific)), list(sorted_res_specific.values()), align='center')
        ax.set_yticks(range(len(sorted_res_specific)), [e[len(specific_group)+2:-3] for e in list(sorted_res_specific.keys())])
        ax.set_xlim(right=len(dpe_change))
        ax.set_title(specific_group)
        plt.show()
        
        lexique = get_lexique_DPE()
        var = 'umur'
        print(lexique.get(var))
        
        # print(dict(Counter(dpe_change['{}--{}--1'.format(specific_group,var)].dropna())))
        print(sum(dict(Counter(dpe_change['{}--{}--1'.format(specific_group,var)].dropna())).values()))
        
        change_var_dict = {'no_change':0}
        for version_1, version_2 in zip(dpe_change['{}--{}--1'.format(specific_group,var)],dpe_change['{}--{}--2'.format(specific_group,var)]):
            if isinstance(version_1, float):
                if np.isnan(version_1) or np.isnan(version_2):
                    continue
            
            if version_1 != version_2:
                key = '{} -> {}'.format(version_1, version_2)
                if key not in change_var_dict.keys():
                    change_var_dict[key] = 0
                change_var_dict[key] += 1
            else:
                change_var_dict['no_change'] += 1
        
        change_var_dict = dict(sorted(change_var_dict.items(), key=lambda item: item[1], reverse=True))
        
        print()
        print('Changement de valeur')
        for k,v in change_var_dict.items():
            if v > 1:
                print(k,v)
        
    
    #%% affichage de la distribution d'une variable dans la bdnb parisienne
    if False:
        plot_var_distribution(var='ffo_bat_annee_construction', path=output_path,min_xlim=1600,rounder=20,percentage=True,max_xlim=2020)
        
    #%% statistiques sur les liens entre période de construction et gains dans la manipulation des DPE
    if False:
        dpe_change = get_dpe_change_details(path=output_path,force=False)
        
        letter_to_number_dict = {chr(ord('@')+n):n for n in range(1,10)}
        gains_etiquette = []
        
        for eti1, eti2 in zip(dpe_change.dpe_letter_1, dpe_change.dpe_letter_2):
            eti1_number = letter_to_number_dict.get(eti1)
            eti2_number = letter_to_number_dict.get(eti2)
            
            gain = eti1_number-eti2_number
            gains_etiquette.append(gain)
        dpe_change['dpe_gain'] = gains_etiquette
        
        if True:
            rounder = 10
            alpha = 0.5
            
            dpe_change_plot = dpe_change.copy()
            # dpe_change_plot = dpe_change[dpe_change.dpe_gain>1]

            counter_var = dict(Counter(round(dpe_change_plot.ffo_bat_annee_construction.dropna()/rounder)*rounder))
            
            fig,ax = plot_var_distribution(var='ffo_bat_annee_construction', path=output_path,min_xlim=1600,rounder=rounder,percentage=True,max_xlim=2020,show=False,alpha=alpha,save=False)
            ax.bar(np.asarray(list(counter_var.keys())), np.asarray(list(counter_var.values()))/sum(counter_var.values())*100, width=rounder, color='tab:blue',alpha=alpha,label='Modification DPE ({})'.format(sum(counter_var.values())))
            ax.legend()
            ax.set_xlabel('Année de construction (arrondie à 10 ans)')
            save_path = os.path.join(output_path,'figs','distribution_periode_constrcution.png')
            plt.savefig(save_path, bbox_inches='tight')


    #%% étude du logiciel de calcul
    if False:
        dpe_change = get_dpe_change_details(path=output_path,force=False)
        # dpe_change_logiciel = dpe_change[[c for c in dpe_change.columns if 'diagnostiqueur' in c]+['dpe_letter_1','dpe_letter_2']]
        
        # 'usr_logiciel_id--1',
        # 'version_logiciel--1',
        # 'version_moteur_calcul--1'
               
        change_logiciel_version_dict = {'no_change':0}
        for version_1, version_2 in zip(dpe_change['diagnostiqueur--version_moteur_calcul--1'],dpe_change['diagnostiqueur--version_moteur_calcul--2']):
            if version_1 != version_2:
                key = '{} -> {}'.format(version_1, version_2)
                if key not in change_logiciel_version_dict.keys():
                    change_logiciel_version_dict[key] = 0
                change_logiciel_version_dict[key] += 1
            else:
                change_logiciel_version_dict['no_change'] += 1
        
        change_logiciel_version_dict = dict(sorted(change_logiciel_version_dict.items(), key=lambda item: item[1], reverse=True))
        
        for k,v in change_logiciel_version_dict.items():
            print(k,v)
        # print(change_logiciel_version_dict)
                    
            
            
    
   
        
        
        
        
                        
        
        
        
        
        
        
        


        
        
        
        
        
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()