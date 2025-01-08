#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:53:00 2025

@author: amounier
"""

import time
from datetime import date, datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from administrative import France, Climat, Departement, draw_departement_map, draw_climat_map
from climate_zone_characterisation import get_departement_temperature


def get_cee_statistics(force=False):
    """
    Récuparation des données de délivrance des CEE par semestre
    Données brutes disponibles ici : https://www.ecologie.gouv.fr/politiques-publiques/comites-pilotage-lettres-dinformation-statistiques-du-dispositif-certificats

    Parameters
    ----------
    force : boolean, optional
        Force la réouverture et le formatage du fichier. The default is False.

    Returns
    -------
    cee_data : pandas DataFrame
        données CEE.

    """
    save_name = 'statistiques_delivrance_cee.csv'
    
    if save_name not in os.listdir(os.path.join('data','CEE')) or force:
        
        france = France()
        departements = france.departements
        
        fiches = ['BAR-TH-104','BAR-TH-129','BAR-TH-150','BAR-TH-159',]
        
        cee_folder = os.path.join('data','CEE',"SDB délivrances-date d'engagement-Standards-par semestre d'engagement")
        files = [e for e in sorted(os.listdir(cee_folder)) if not e.startswith('.')]
        
        files_date_dict = {f:pd.to_datetime(f.split('.')[0].split(' ')[-1]) for f in files}
        
        cee_data = {'date':[],'departement':[],'fiche':[],'volume_cee':[]}
        
        for file in tqdm.tqdm(files):
            cee_temp = pd.read_excel(os.path.join(cee_folder,file), sheet_name='Classique',usecols='A:DA')
            cee_temp = cee_temp.set_index('Numéro département')
            cee_temp = cee_temp.rename(columns={c:Departement(c).code for c in cee_temp.columns})
            
            for dep in departements:
                cee_dep_temp = cee_temp[[dep.code]]
                for f in fiches:
                    volume_cee = cee_dep_temp.loc[f].values[0]
            
                    cee_data['date'].append(files_date_dict.get(file))
                    cee_data['departement'].append(dep.code)
                    cee_data['fiche'].append(f)
                    cee_data['volume_cee'].append(volume_cee)
                
        cee_data = pd.DataFrame().from_dict(cee_data)
        cee_data['zcl'] = [Departement(d).climat for d in cee_data.departement]
        
        cee_data.to_csv(os.path.join('data','CEE',save_name),index=False)
        
    cee_data = pd.read_csv(os.path.join('data','CEE',save_name))
    cee_data['date'] = [pd.to_datetime(d) for d in cee_data.date]
    # cee_data = cee_data.set_index('date')
    
    return cee_data
    
    

#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_insulation_characterisation'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
    
    
    #%% Caractérisation des analyses géographiques présentées dans ONRE oct 2024
    # https://www.statistiques.developpement-durable.gouv.fr/les-renovations-energetiques-aidees-du-secteur-residentiel-entre-2016-et-2021?rubrique=&dossier=843982
    
    # moins de rénovations car climat doux dans le sud : vraiment ?
    if True:
        data = pd.read_excel(os.path.join('data','SDES','graphiques_onre_2016_2021.xlsx'), sheet_name='Carte1', skiprows=2)
        data['Département'] = [Departement(dep) for dep in data.Département]
        
        data['zcl'] = [dep.climat for dep in data.Département]
        data['zcl'] = pd.Categorical(data.zcl, France().climats)
        
        data['housing_surface'] = [np.nan]*len(data)
        data['households'] = [np.nan]*len(data)
        data['temperature_DJF'] = get_departement_temperature(period='DJF').temperature
        data['temperature_JJA'] = get_departement_temperature(period='JJA').temperature
        
        # carte des données étudiées
        if True:
            france = France()
            climats = [Climat(e) for e in france.climats]
            draw_climat_map({c:None for c in climats},zcl_label=True, 
                            figs_folder=figs_folder, save='zcl',
                            add_legend=False,lw=0.7)
            
            # data_dict = {d:v for d,v in zip(data.Département,data['Intensité energétique'])}
            # draw_departement_map(data_dict, figs_folder, automatic_cbar_values=True)
            
        # graphes des interactions avec le climat
        if False:
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.boxplot(data=data,x="zcl", y='Intensité energétique',ax=ax)
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.scatterplot(data=data,x="temperature_DJF", y='Intensité energétique',ax=ax,alpha=0.6)
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.scatterplot(data=data,x="temperature_JJA", y='Intensité energétique',ax=ax,alpha=0.6)
            plt.show()
            
            # TODO : ajouter d'autres variables tq : DPE, économies réelles
            
        
    #%% TREMI
    if False:
        pass
    s
    
    #%% CEE
    if False:
        # Pour les PAC air-air : BAR-TH-129 ( = clim fixe)
        # Pour les PAC air-eau ou eau-eau : BAR-TH-104
        # PAC hybride individuelle (air-eau + gaz) : BAR-TH-159
        # PAC collectives à absorption air/eau : BAR-TH-150
        
        fiches = ['BAR-TH-104','BAR-TH-129','BAR-TH-150','BAR-TH-159',]
        cee_data = get_cee_statistics()
        
        # données de 2023 par encore consolidées
        cee_data = cee_data[cee_data.date.dt.year<2023]
        
        print(cee_data)
        
        # affichage de l'évolution des 4 fiches à l'échelle nationale
        if False:
            cee_data_nat = cee_data.groupby(by=['date','fiche'])['volume_cee'].sum().reset_index()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.lineplot(data=cee_data_nat,x='date',y='volume_cee',hue='fiche',ax=ax)
            ax.set_ylim(bottom=0.)
            plt.show()
            
        if True:
            cee_data_nat = cee_data.groupby(by=['date','fiche','zcl'])['volume_cee'].sum().reset_index()
            
            for f in fiches:
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                sns.lineplot(data=cee_data_nat[cee_data_nat.fiche==f],x='date',y='volume_cee',hue='zcl',ax=ax)
                ax.set_ylim(bottom=0.)
                ax.set_title(f)
                plt.show()
        
        # TODO : passer des volumes_CEE en nombre de systèms (estimés)
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()