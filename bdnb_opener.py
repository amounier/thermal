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


today = pd.Timestamp(date.today()).strftime('%Y%m%d')


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
# # à mettre dans main plus tard 
# =============================================================================

# étude des DPE successifs sur des batiments (pas d'informations sur le logement)
if True:
    tic = time.time()
    
    # ouverture des données sous dask
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
    
    if True:
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        sns.histplot(base_multiple_dpe_batiment_groupe_df,x='nb_dpe',stat='percent',ax=ax,binwidth=5,)
        ax.set_xlabel('Nombre de DPE par bâtiment ({} bâtiments)'.format(len(base_multiple_dpe_batiment_groupe_df)))
        plt.show()
        
    
    batiment_group_list = base_multiple_dpe_batiment_groupe_df.batiment_groupe_id.to_list()
    
    number_batiment_groupe = 100
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
        
        # print('{}/{}: {} ({} DPE toutes méthodes)'.format(i+1,number_batiment_groupe,bg_id, len(dpe_bg)))
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
            
    # print(results)
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
    letter_to_number_dict = {chr(ord('@')+n):n for n in range(1,10)}
    
    for bg_id in batiment_group_list:
    # for bg_id in ['bdnb-bg-2GSS-QG39-2CAH']:
    # for bg_id in ['bdnb-bg-65F1-XL3D-CUER']:
    # for bg_id in ['bdnb-bg-1HX6-5V4H-1L6W']:
    # for bg_id in ['bdnb-bg-KP9N-M5LM-X8CB']:
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
        # filter_same_etiquette = np.asarray([df_dpe_bg_id.classe_bilan_dpe_number>df_dpe_bg_id.classe_bilan_dpe_number.shift(1),
        #                                     df_dpe_bg_id.classe_bilan_dpe_number<df_dpe_bg_id.classe_bilan_dpe_number.shift(-1)]).any(0)
        filter_same_etiquette = np.asarray([df_dpe_bg_id.classe_bilan_dpe_number==df_dpe_bg_id.classe_bilan_dpe_number.shift(1),
                                            df_dpe_bg_id.classe_bilan_dpe_number==df_dpe_bg_id.classe_bilan_dpe_number.shift(-1)]).any(0)
        df_dpe_bg_id = df_dpe_bg_id[~filter_same_etiquette]
        
        # filtre pour ne garder que les DPE distants de moins de 45 jours
        df_dpe_bg_id['time_difference'] = df_dpe_bg_id.date_etablissement_dpe - df_dpe_bg_id.date_etablissement_dpe.shift(1)
        if not any(df_dpe_bg_id.time_difference < pd.Timedelta(45, unit='d')):
            continue
        
        # filtre pour ne garder que les bâtiments ayant au moins 2 DPE successifs
        if len(df_dpe_bg_id)<2:
            continue
        
        # sortie des fichiers csv pour les bâtiments 'suspects'
        output_folder = os.path.join('output')
        folder = '{}_DPE_successifs'.format(today)
        if folder not in os.listdir(output_folder):
            os.mkdir(os.path.join(output_folder,folder))
        df_dpe_bg_id.to_csv(os.path.join(output_folder,folder,'{}.csv'.format(bg_id)),index=False)
        
        # print(df_dpe_bg_id.identifiant_dpe)
        
    if True:
        output_folder = os.path.join('output')
        folder = '{}_DPE_successifs'.format(today)
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
            locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
            # formatter = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_locator(locator)
            # ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
            plt.show()
            plt.close()
    
    # prochaine étape : regarder le fichier xml https://observatoire-dpe-audit.ademe.fr/pub/dpe/2375E4547621M/xml
    
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
    
    
            
    
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()