#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:08:40 2024

@author: amounier
"""


import time 
import pandas as pd
from datetime import date
import os
import matplotlib.pyplot as plt
import numpy as np

from energy_consumption_RC import get_coordinates, get_meteo_data
from utils import plot_timeserie

# Définition des dictionnaires administratifs de régions
dict_region_code_region_name = {11:'Île-de-France',
                                24:'Centre-Val de Loire',
                                27:'Bourgogne-Franche-Comté',
                                28:'Normandie',
                                32:'Hauts-de-France',
                                44:'Grand-Est',
                                52:'Pays de la Loire',
                                53:'Bretagne',
                                75:'Nouvelle-Aquitaine',
                                76:'Occitanie',
                                84:'Auvergne-Rhône-Alpes',
                                93:"Provence-Alpes-Côte d'Azur",
                                94:'Corse'}
dict_region_name_region_code = {v:k for k,v in dict_region_code_region_name.items()}

dict_region_code_chef_lieu = {11:'Paris',
                              24:'Orléans',
                              27:'Dijon',
                              28:'Rouen',
                              32:'Lille',
                              44:'Strasbourg',
                              52:'Nantes',
                              53:'Rennes',
                              75:'Bordeaux',
                              76:'Toulouse',
                              84:'Lyon',
                              93:'Marseille',
                              94:'Ajaccio'}


def open_electricity_consumption(scale='national', force=False):
    enedis_data_folder = os.path.join('data','Enedis','202408_consommation_horaire')
    
    if scale == 'national':
        clean_data_file = 'national_electricity_consumption.csv'
        
        if clean_data_file not in os.listdir(enedis_data_folder) or force:
            # Données de consommation aux PDL de moins de 36 kVA (https://www.data.gouv.fr/fr/datasets/agregats-segmentes-de-consommation-electrique-au-pas-1-2-h-des-points-de-soutirage-36kva-maille-nationale-1/)
            # Il n'y a pas de résidentiel de plus de 36 kVA
            raw_data_file = 'conso-inf36.csv'
            raw_data = pd.read_csv(os.path.join(enedis_data_folder,raw_data_file), sep=';')
            
            raw_data = raw_data[['horodate', 'profil', 'plage_de_puissance_souscrite',
                                 'nb_points_soutirage', 'total_energie_soutiree_wh']]
                        
            # j'ai pas besoin des différences par puissances souscrites
            raw_data = raw_data[raw_data.plage_de_puissance_souscrite=='P0: Total <= 36 kVA']
            
            # je me focalise sur le résidentiel, et je ne m'intéresse pas aux différences de profils
            res_profiles = [c for c in set(raw_data.profil.values) if 'RES' in c]
            raw_data = raw_data[raw_data.profil.isin(res_profiles)]
            raw_data = raw_data.groupby(by='horodate')[['nb_points_soutirage', 'total_energie_soutiree_wh']].sum().reset_index()
            
            # enregistrement des données filtrées
            raw_data.to_csv(os.path.join(enedis_data_folder,clean_data_file),index=False)
        
        data = pd.read_csv(os.path.join(enedis_data_folder,clean_data_file))
        data = data.set_index('horodate')
        data.index = pd.to_datetime(data.index)
        data = data.groupby(pd.Grouper(freq='h')).sum()
        data.index = data.index.tz_localize(None)
    
    if scale == 'regional':
        clean_data_file  = 'regional_electricity_consumption.csv'
        
        if clean_data_file not in os.listdir(enedis_data_folder) or force:
            # Données de consommation aux PDL de moins de 36 kVA (https://www.data.gouv.fr/fr/datasets/agregats-segmentes-de-consommation-electrique-au-pas-1-2-h-des-points-de-soutirage-36kva-maille-nationale-1/)
            raw_data_file = 'conso-inf36-region.csv'
            raw_data = pd.read_csv(os.path.join(enedis_data_folder,raw_data_file), sep=';')
            
            raw_data = raw_data[['horodate', 'profil', 'plage_de_puissance_souscrite','region',
                                  'nb_points_soutirage', 'total_energie_soutiree_wh']]
                        
            # j'ai pas besoin des différences par puissances souscrites
            raw_data = raw_data[raw_data.plage_de_puissance_souscrite=='P0: Total <= 36 kVA']
            
            # je me focalise sur le résidentiel, et je ne m'intéresse pas aux différences de profils
            res_profiles = [c for c in set(raw_data.profil.values) if 'RES' in c]
            raw_data = raw_data[raw_data.profil.isin(res_profiles)]
            raw_data = raw_data.groupby(['horodate','region'])[['nb_points_soutirage', 'total_energie_soutiree_wh']].sum().reset_index()
            
            # enregistrement des données filtrées
            raw_data.to_csv(os.path.join(enedis_data_folder,clean_data_file),index=False)
        
        data = pd.read_csv(os.path.join(enedis_data_folder,clean_data_file))
        
        reformatted_data = {'horodate':sorted(list(set(data.horodate)))}
        
        for reg in list(set(data.region.to_list())):
            if reg == 'Nouvelle Aquitaine':
                regcode = dict_region_name_region_code.get(reg.replace(' ','-'))
            else:
                regcode = dict_region_name_region_code.get(reg)
            data_reg = data[data.region==reg]
            for c in ['nb_points_soutirage', 'total_energie_soutiree_wh']:
                c_reg = c + '_reg_{}'.format(regcode)
            reformatted_data[c_reg] = data_reg[c].values
        
        data = pd.DataFrame().from_dict(reformatted_data)
        data = data.set_index('horodate')
        data.index = pd.to_datetime(data.index)
        data = data.groupby(pd.Grouper(freq='h')).sum()
        data.index = data.index.tz_localize(None)
        
    return data


    
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_thermal_sensitivity'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    #--------------------------------------------------------------------------
        
    national_consumption_data = open_electricity_consumption('national')
    regional_consumption_data = open_electricity_consumption('regional')
    
    # Vérification de la somme des énergies consommées par région
    if False:
        sum_reg = regional_consumption_data.copy()
        sum_reg = sum_reg[[c for c in sum_reg.columns if c.startswith('total_energie_soutiree')]]
        sum_reg = pd.DataFrame(sum_reg.sum(axis=1)).rename(columns={0:'total_energie_soutiree_wh_sum_reg'})
        fig,ax = plot_timeserie(national_consumption_data[['total_energie_soutiree_wh']], figsize=(10,5),
                                figs_folder=figs_folder, save_fig='total_energie_soutiree_wh_national_enedis',
                                show=False, alpha=0.5)
        fig,ax = plot_timeserie(sum_reg, figax=(fig,ax),
                                figs_folder=figs_folder, save_fig='total_energie_soutiree_wh_national_enedis',
                                show=False, alpha=0.5)
        
        
    # plot_timeserie(national_data[['nb_points_soutirage']], figsize=(10,5),
    #                figs_folder=figs_folder, save_fig='nb_points_soutirage_national_enedis')
    # plot_timeserie(national_data[['total_energie_soutiree_wh']], figsize=(10,5),
    #                figs_folder=figs_folder, save_fig='total_energie_soutiree_wh_national_enedis')
    
    
    
    
    # Premier test de thermosensibilité
    if True:
        paris_meteo_data = get_meteo_data('Paris',[2020,2024])
        data = paris_meteo_data.join(national_consumption_data,how='outer')
        
        # data['weekday'] = data.index.weekday
        # weekday_data = dict()
        # for i in range(0,7):
        #     weekday_data[i] = (data[data.weekday==i]).total_energie_soutiree_wh.values
            
        # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        # ax.errorbar(list(weekday_data.keys()),[np.nanmean(weekday_data.get(i)) for i in range(0,7)], yerr = [np.nanstd(weekday_data.get(i)) for i in range(0,7)])
        
        def piecewise_linear(x, x0, x1, y0, k1, k2):
            return np.piecewise(x, [x <= x0, (x > x0)&(x<x1), x>=x1], [lambda x:k1*x + y0-k1*x0, lambda x: y0, lambda x:k2*x + y0-k2*x0])
        
        from scipy.optimize import curve_fit
        
        data = data.dropna(axis=0)#[:20000]
        x = data.temperature_2m.values
        y = data.total_energie_soutiree_wh.values/1e9
        p0 = (5, 20, 10, 1,1)
        p , e = curve_fit(piecewise_linear, x, y, maxfev=100000, p0=p0)
        xd = np.linspace(-5, 40, 10000)
        
        print(p)
    
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(x,y,alpha=0.1, ls='',marker='.')
        ax.plot(xd, piecewise_linear(xd, *p))
        plt.show()
        
    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
if __name__ == '__main__':
    main()
