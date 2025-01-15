#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 23:34:19 2025

@author: amounier
"""

import time
import os
import pandas as pd
from datetime import date
import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from meteorology import get_historical_weather_data
from thermal_model import (refine_resolution, 
                           aggregate_resolution, 
                           run_thermal_model, 
                           plot_timeserie)
from behaviour import Behaviour
from administrative import Climat
from typologies import Typology
from future_meteorology import get_projected_weather_data

#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_thermal_optimisation'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    
    
    #%% Caractérisation des temps de calcul
    if True:
        # Localisation
        city = 'Beauvais'
        # city = 'Nice'
        
        # Période de calcul
        period = [2010,2010]
        # period = [2003,2003]
        
        # Checkpoint weather data
        weather_data_checkfile = ".weather_data_{}_{}_{}_".format(city,period[0],period[1]) + today + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_historical_weather_data(city,period)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
            
        # Définition des habitudes
        conventionnel = Behaviour('conventionnel_th-bce_2020')
        conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
        conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
        
        typo_name = 'FR.N.SFH.03.Gen'
        typo = Typology(typo_name)
        
        # print(typo.modelled_Upb)
        
        t1 = time.time()
        simulation = run_thermal_model(typo, conventionnel, weather_data)
        simulation = aggregate_resolution(simulation, resolution='h')
        simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
        initial_Bch = simulation.heating_needs.values[0]
        initial_Bfr = simulation.cooling_needs.values[0]
        t2 = time.time()
        print('{} an(s) de simulation : {:.2f}s.'.format(len(list(range(*period)))+1,t2-t1))
        
        # Besoins et températures de consignes
        # TODO faire varier les météos et les périodes de construction
        if False:
            Tch_cons_list = np.linspace(15,23,20)
            Tfr_cons_list = np.linspace(22,30,20)
            Bch_list = np.asarray([0]*len(Tch_cons_list))
            Bfr_list = np.asarray([0]*len(Tfr_cons_list))
            
            for idx,tch in tqdm.tqdm(enumerate(Tch_cons_list),total=len(Tch_cons_list)):
                conventionnel.heating_rules = {i:[tch]*24 for i in range(1,8)}
                conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
                typo.heater_maximum_power = 10000*typo.households # W
                typo.cooler_maximum_power = 0 # W
                
                simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
                simulation = aggregate_resolution(simulation, resolution='h')
                simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
                Bch_list[idx] = simulation.heating_needs.mean()
            
            Bch_list = (Bch_list-initial_Bch)/initial_Bch*100
            
            a_ch,b_ch = np.polyfit(Tch_cons_list, Bch_list, deg=1)
            r2_ch = r2_score(Bch_list, a_ch*Tch_cons_list+b_ch)
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(Tch_cons_list, Bch_list, marker='o',ls='', color='tab:red', label='Model')
            ax.plot(Tch_cons_list, a_ch*Tch_cons_list+b_ch, color='k',label='linear fit (R$^2$ = {:.2f})\nslope : {:.1f}%.°C'.format(r2_ch, a_ch)+'$^{-1}$')
            ax.legend()
            plt.show()
            
            for idx,tfr in tqdm.tqdm(enumerate(Tfr_cons_list),total=len(Tfr_cons_list)):
                conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
                conventionnel.cooling_rules = {i:[tfr]*24 for i in range(1,8)}
                typo.heater_maximum_power = 0
                typo.cooler_maximum_power = 10000*typo.households # W
                
                simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
                simulation = aggregate_resolution(simulation, resolution='h')
                simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
                Bfr_list[idx] = simulation.cooling_needs.mean()
                
            Bfr_list = (Bfr_list-initial_Bfr)/initial_Bfr*100
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(Tfr_cons_list, Bfr_list)
            plt.show()
            
            
            
        # Évolution des besoins en chaud et froid selon les épaisseurs d'isolants 
        if True:
            # Localisation
            city = 'Beauvais'
            city = 'Brest'
            # city = 'Nice'
            
            # Période de calcul
            period = [2010,2010]
            # period = [2003,2003]
            
            # Checkpoint weather data
            weather_data_checkfile = ".weather_data_{}_{}_{}_".format(city,period[0],period[1]) + today + ".pickle"
            if weather_data_checkfile not in os.listdir():
                weather_data = get_historical_weather_data(city,period)
                weather_data = refine_resolution(weather_data, resolution='600s')
                pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
            else:
                weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
            
            # Définition des habitudes
            conventionnel = Behaviour('conventionnel_th-bce_2020')
            # valeurs moyennes (enquete ifop et coda)
            conventionnel.heating_rules = {i:[19.9]*24 for i in range(1,8)}
            conventionnel.cooling_rules = {i:[22.6]*24 for i in range(1,8)}
            
            typo_name = 'FR.N.SFH.03.Gen'
            typo_name = 'FR.N.SFH.09.Gen'
            typo = Typology(typo_name)
            
            thickness_list = np.linspace(0, 0.5, 30)
            Bch_list = np.asarray([0]*len(thickness_list))
            Bfr_list = np.asarray([0]*len(thickness_list))
            for idx,thickness in tqdm.tqdm(enumerate(thickness_list),total=len(thickness_list)):
                
                typo.w0_insulation_thickness = thickness
                typo.w1_insulation_thickness = thickness
                typo.w2_insulation_thickness = thickness
                typo.w3_insulation_thickness = thickness
            
                simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
                simulation = aggregate_resolution(simulation, resolution='h')
                simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
                Bfr_list[idx] = simulation.cooling_needs.mean()
                Bch_list[idx] = simulation.heating_needs.mean()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(thickness_list, Bfr_list, color='tab:blue', label='Bfr')
            ax.plot(thickness_list, Bch_list, color='tab:red', label='Bch')
            ax.plot(thickness_list, Bfr_list+Bch_list, color='k', label='B')
            plt.show()
            
            typo_name = 'FR.N.SFH.03.Gen'
            typo_name = 'FR.N.SFH.06.Gen'
            typo = Typology(typo_name)
            
            # thickness_list = np.linspace(0, 0.5, 30)
            # Bch_list = np.asarray([0]*len(thickness_list))
            # Bfr_list = np.asarray([0]*len(thickness_list))
            # for idx,thickness in tqdm.tqdm(enumerate(thickness_list),total=len(thickness_list)):
                
            #     typo.floor_insulation_thickness = thickness
            
            #     simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
            #     simulation = aggregate_resolution(simulation, resolution='h')
            #     simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
            #     Bfr_list[idx] = simulation.cooling_needs.mean()
            #     Bch_list[idx] = simulation.heating_needs.mean()
            
            # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot(thickness_list, Bfr_list, color='tab:blue', label='Bfr')
            # ax.plot(thickness_list, Bch_list, color='tab:red', label='Bch')
            # ax.plot(thickness_list, Bfr_list+Bch_list, color='k', label='B')
            # plt.show()
            
            # typo_name = 'FR.N.SFH.03.Gen'
            # typo_name = 'FR.N.SFH.06.Gen'
            # typo = Typology(typo_name)
            
            # efficiency_list = np.linspace(0.0001, 0.5, 30)
            # Bch_list = np.asarray([0]*len(efficiency_list))
            # Bfr_list = np.asarray([0]*len(efficiency_list))
            # for idx,eff in tqdm.tqdm(enumerate(efficiency_list),total=len(efficiency_list)):
                
            #     typo.ventilation_efficiency = eff
            
            #     simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
            #     simulation = aggregate_resolution(simulation, resolution='h')
            #     simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
            #     Bfr_list[idx] = simulation.cooling_needs.mean()
            #     Bch_list[idx] = simulation.heating_needs.mean()
            
            # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot(efficiency_list, Bfr_list, color='tab:blue', label='Bfr')
            # ax.plot(efficiency_list, Bch_list, color='tab:red', label='Bch')
            # ax.plot(efficiency_list, Bfr_list+Bch_list, color='k', label='B')
            # plt.show()
    
    
    tac = time.time()
    print("Done in {:.2f}s".format(tac-tic))
    
if __name__ == '__main__':
     main()