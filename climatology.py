#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 14:54:36 2026

@author: amounier
"""

import os 
import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt


#%% ===========================================================================
# script
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_climatology'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie TODO
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)

    #%% Coefficient de rigueur hivernal
    if False:
        # https://www.statistiques.developpement-durable.gouv.fr/indice-de-rigueur-degres-jours-unifies-aux-niveaux-national-regional-et-departemental
        sdes_rigueur_17 = pd.read_excel(os.path.join('data','SDES','dju_donnees_nationales_1970_2024_v4.xlsx'),sheet_name='DJU17').set_index('year')
        sdes_rigueur_15 = pd.read_excel(os.path.join('data','SDES','dju_donnees_nationales_1970_2024_v4.xlsx'),sheet_name='DJU15').set_index('year')
    
        iea_rigueur_16 = pd.read_csv(os.path.join('data','IEA','IEA_CMCC_HDD16monthlyworldbypopallmonths.csv'),header=9,encoding='latin-1')
        iea_rigueur_16 = iea_rigueur_16[iea_rigueur_16.ISO3=='FRA']
        iea_rigueur_16['Date'] = iea_rigueur_16.Date.map(pd.to_datetime) 
        iea_rigueur_16 = iea_rigueur_16.set_index('Date')
        iea_rigueur_16 = iea_rigueur_16[iea_rigueur_16.index.year<2025]
        iea_rigueur_16 = iea_rigueur_16[['HDD16']]
        iea_rigueur_16 = iea_rigueur_16.groupby(pd.Grouper(freq="YS")).sum()
        iea_rigueur_16.index = iea_rigueur_16.index.year
        iea_rigueur_16['reference'] = [iea_rigueur_16.loc[list(range(1991,2021))].mean().values[0]]*len(iea_rigueur_16)
        iea_rigueur_16['Indice de rigueur'] = iea_rigueur_16.HDD16/iea_rigueur_16.reference
        # color = plt.get_cmap('viridis')(0.5)
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(sdes_rigueur_17.index,sdes_rigueur_17['Indice de rigueur'],label='SDES 17')
        ax.plot(sdes_rigueur_15.index,sdes_rigueur_15['Indice de rigueur'],label='SDES 15')
        ax.plot(iea_rigueur_16.index,iea_rigueur_16['Indice de rigueur'],label='IEA 16')
        ax.set_ylim(bottom=0.)
        ax.legend()
        ax.set_xlim(left=1990,right=2025)
        plt.show()
        
    #%% Coefficient de rigueur estival
    if False:
        iea_rigueur_18 = pd.read_csv(os.path.join('data','IEA','IEA_CMCC_CDD18monthlyworldbypopallmonths.csv'),header=9,encoding='latin-1')
        iea_rigueur_18 = iea_rigueur_18[iea_rigueur_18.ISO3=='FRA']
        iea_rigueur_18['Date'] = iea_rigueur_18.Date.map(pd.to_datetime) 
        iea_rigueur_18 = iea_rigueur_18.set_index('Date')
        iea_rigueur_18 = iea_rigueur_18[iea_rigueur_18.index.year<2025]
        iea_rigueur_18 = iea_rigueur_18[['CDD18']]
        iea_rigueur_18 = iea_rigueur_18.groupby(pd.Grouper(freq="YS")).sum()
        iea_rigueur_18.index = iea_rigueur_18.index.year
        iea_rigueur_18['reference'] = [iea_rigueur_18.loc[list(range(1991,2021))].mean().values[0]]*len(iea_rigueur_18)
        iea_rigueur_18['Indice de rigueur'] = iea_rigueur_18.CDD18/iea_rigueur_18.reference
        # color = plt.get_cmap('viridis')(0.5)
        
        iea_rigueur_18.to_csv(os.path.join('data','IEA','iea_rigueur_18.csv'))
        
        iea_rigueur_23 = pd.read_csv(os.path.join('data','IEA','IEA_CMCC_CDD23monthlyworldbypopallmonths.csv'),header=9,encoding='latin-1')
        iea_rigueur_23 = iea_rigueur_23[iea_rigueur_23.ISO3=='FRA']
        iea_rigueur_23['Date'] = iea_rigueur_23.Date.map(pd.to_datetime) 
        iea_rigueur_23 = iea_rigueur_23.set_index('Date')
        iea_rigueur_23 = iea_rigueur_23[iea_rigueur_23.index.year<2025]
        iea_rigueur_23 = iea_rigueur_23[['CDD23']]
        iea_rigueur_23 = iea_rigueur_23.groupby(pd.Grouper(freq="YS")).sum()
        iea_rigueur_23.index = iea_rigueur_23.index.year
        iea_rigueur_23['reference'] = [iea_rigueur_23.loc[list(range(1991,2021))].mean().values[0]]*len(iea_rigueur_23)
        iea_rigueur_23['Indice de rigueur'] = iea_rigueur_23.CDD23/iea_rigueur_23.reference
        # color = plt.get_cmap('viridis')(0.5)
        
        iea_rigueur_23.to_csv(os.path.join('data','IEA','iea_rigueur_23.csv'))
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(iea_rigueur_18.index,iea_rigueur_18['Indice de rigueur'],label='IEA 18')
        ax.plot(iea_rigueur_23.index,iea_rigueur_23['Indice de rigueur'],label='IEA 23')
        ax.set_ylim(bottom=0.)
        ax.legend()
        ax.set_xlim(left=1990,right=2025)
        plt.show()
        
    #%% Classement des pays les plus froids/chauds
    if True:
        data = pd.read_csv(os.path.join('data','IEA','IEA_CMCC_Temperaturemonthlyworldbypopallmonths.csv'),header=9,encoding='latin-1')
        # data = data[data.ISO3=='FRA']
        data['Date'] = pd.to_datetime(data.Date)
        data = data.set_index('Date')
        data = data[data.index.year<2025]
        data = data[['Territory','Temperature']]
        
        data = data[data.index.year.isin(list(range(2000,2025)))]
        data = data.reset_index().groupby('Territory').mean()[['Temperature']]
        
        data = data.sort_values(by='Temperature')
        print(data.index.to_list())
        print(data.index.to_list().index('France'), len(data))
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()