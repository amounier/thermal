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
import multiprocessing

from meteorology import get_historical_weather_data
from thermal_model import (refine_resolution, 
                           aggregate_resolution, 
                           run_thermal_model, 
                           plot_timeserie)
from behaviour import Behaviour
from administrative import Climat, France
from typologies import Typology
from future_meteorology import get_projected_weather_data



def compute_energy_gains(component,typo_code,zcl,output_path,
                         behaviour='conventionnel',parrallel_compute=False,
                         period=[2000,2020],plot=True,nb_intervals=10,show=False,
                         progressbar=False):
    
    city = zcl.center_prefecture
    typo = Typology(typo_code)
    
    # weather data
    weather_data_checkfile = ".weather_data_{}_{}_{}".format(city,period[0],period[1]) + ".pickle"
    if weather_data_checkfile not in os.listdir():
        weather_data = get_historical_weather_data(city,period)
        weather_data = refine_resolution(weather_data, resolution='600s')
        pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
    else:
        weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
    
    if behaviour == 'conventionnel':
        behaviour = Behaviour('conventionnel_th-bce_2020')
    
    
    dict_all_components = {'floor':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=nb_intervals)-0.05,
                                    'var_label':'Supplementary floor insulation thickness (m)',
                                    'var_saver':'{}_{}_{}_{}_{}-{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1])
                                    },
                           'walls':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=nb_intervals)-0.05,
                                    'var_label':'Supplementary walls insulation thickness (m)',
                                    'var_saver':'{}_{}_{}_{}_{}-{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1])
                                    },
                           'roof':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=nb_intervals)-0.05,
                                   'var_label':'Supplementary roof insulation thickness (m)',
                                   'var_saver':'{}_{}_{}_{}_{}-{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1])
                                    },
                           'albedo':{'var_space':['light','medium','dark','black'],
                                     'var_label':'External surface color',
                                     'var_saver':'{}_{}_{}_{}_{}-{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1])
                                     },
                           'ventilation':{'var_space':np.linspace(0, 0.5, nb_intervals),
                                          'var_label':'Ventilation efficiency',
                                          'var_saver':'{}_{}_{}_{}_{}-{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1])
                                          },
                           }
    
    dict_plot_top_value = {'SFH':250,
                           'TH':250,
                           'MFH':250,
                           'AB':250}
    
    dict_components = dict_all_components.get(component)
    
    var_space = dict_components.get('var_space')
    
    Bch_list = [0]*len(var_space)
    Bfr_list = [0]*len(var_space)
    
    if '{}.pickle'.format(dict_components.get('var_saver')) not in os.listdir(output_path):
        if progressbar:
            iterator = tqdm.tqdm(enumerate(var_space),total=len(var_space),desc=component)
        else:
            iterator = enumerate(var_space)
            
        for idx,var_value in iterator:
            typo = Typology(typo_code)
            
            if component == 'floor':
                typo.floor_insulation_thickness = typo.floor_insulation_thickness + var_value
            if component == 'walls':
                typo.w0_insulation_thickness = typo.w0_insulation_thickness + var_value
                typo.w1_insulation_thickness = typo.w0_insulation_thickness + var_value
                typo.w2_insulation_thickness = typo.w0_insulation_thickness + var_value
                typo.w3_insulation_thickness = typo.w0_insulation_thickness + var_value
            if component == 'roof':
                typo.ceiling_supplementary_insulation_thickness = var_value
            if component == 'albedo':
                typo.roof_color = var_value
                typo.w0_color = var_value
                typo.w1_color = var_value
                typo.w3_color = var_value
                typo.w2_color = var_value
            if component == 'ventilation':
                typo.ventilation_efficiency = var_value
                
            simulation = run_thermal_model(typo, behaviour, weather_data, pmax_warning=False)
            simulation = aggregate_resolution(simulation, resolution='h')
            simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
            Bfr_list[idx] = simulation.cooling_needs.to_list()
            Bch_list[idx] = simulation.heating_needs.to_list()
        
        
        pickle.dump((Bch_list,Bfr_list), open(os.path.join(os.path.join(output_path),'{}.pickle'.format(dict_components.get('var_saver'))), "wb"))
    
    Bch_list,Bfr_list = pickle.load(open(os.path.join(os.path.join(output_path),'{}.pickle'.format(dict_components.get('var_saver'))), 'rb'))
    
    Bch_list = np.asarray(Bch_list)/(1e3 * typo.surface)
    Bfr_list = np.asarray(Bfr_list)/(1e3 * typo.surface)
    Btot_list = Bch_list + Bfr_list
    
    
    if plot:
        Bch_mean = Bch_list.mean(axis=1)
        Bfr_mean = Bfr_list.mean(axis=1)
        Btot_mean = Btot_list.mean(axis=1)
        Bch_std = Bch_list.std(axis=1)
        Bfr_std = Bfr_list.std(axis=1)
        Btot_std = Btot_list.std(axis=1)
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        # ax.plot(var_space, Bch_mean,color='tab:red', label='Heating needs')
        # ax.fill_between(var_space, Bch_mean+Bch_std, Bch_mean-Bch_std, color='tab:red', alpha=0.3)
        ax.errorbar(var_space, Bch_mean,yerr=Bch_std,
                    color='tab:red', label='Heating needs',
                    ls=':',marker='o',mec='w',capsize=3)
        
        # ax.plot(var_space, Bfr_mean, color='tab:blue', label='Cooling needs')
        # ax.fill_between(var_space, Bfr_mean+Bfr_std, Bfr_mean-Bfr_std, color='tab:blue', alpha=0.3)
        ax.errorbar(var_space, Bfr_mean,yerr=Bfr_std,
                    color='tab:blue', label='Cooling needs',
                    ls=':',marker='o',mec='w',capsize=3)
        
        # ax.plot(var_space, Btot_mean, color='k', label='Total needs')
        # ax.fill_between(var_space, Btot_mean+Btot_std, Btot_mean-Btot_std, color='k', alpha=0.3)
        ax.errorbar(var_space, Btot_mean,yerr=Btot_std,
                    color='k', label='Total needs',
                    ls=':',marker='o',mec='w',capsize=3)
        
        ax.set_ylim(bottom=0.,top=dict_plot_top_value.get(typo.type))
        ax.set_xlabel(dict_components.get('var_label'))
        ax.set_ylabel('Energy needs (kWh.yr$^{-1}$.m$^{-2}$)')
        ax.set_title('{} - {}'.format(typo.code,zcl.code))
        ax.legend()
        plt.savefig(os.path.join(output_path,'figs','{}.png'.format(dict_components.get('var_saver'))),bbox_inches='tight')
        if show: 
            plt.show()
        plt.close()
    
    return 


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
        
    
    
    #%% Variation des paramètres d'isolation 
    if True:
        
        # Caractérisation du temps de calcul
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
            # conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
            # conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
            
            typo_name = 'FR.N.SFH.03.Gen'
            typo = Typology(typo_name)
            
            # print(typo.modelled_Upb)
            
            t1 = time.time()
            simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
            simulation = aggregate_resolution(simulation, resolution='h')
            simulation_year = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
            initial_Bch = simulation_year.heating_needs.values[0]
            initial_Bfr = simulation_year.cooling_needs.values[0]
            t2 = time.time()
            print('{} an(s) de simulation : {:.2f}s.'.format(len(list(range(*period)))+1,t2-t1))
            
            # plot_timeserie(simulation[['heating_needs','cooling_needs']],figsize=(15,5),ylabel='Energy needs (Wh)',
            #                figs_folder=figs_folder,show=True)
            
            
        
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
            
            
            
        # Évolution relatifs des besoins en chaud et froid selon les épaisseurs d'isolants (comparaiosn litt)
        if False:
            
            # Localisation
            city = 'Aalborg' # pour la comparaison avec Pomianowski
            
            # Période de calcul
            period = [2000,2010]
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
            # conventionnel.heating_rules = {i:[19.9]*24 for i in range(1,8)}
            # conventionnel.cooling_rules = {i:[22.6]*24 for i in range(1,8)}
            
            # typo_name = 'FR.N.SFH.03.Gen'
            typo_name = 'FR.N.MFH.03.Gen'
            typo = Typology(typo_name)
            
            thickness_list = np.linspace(0.05, 0.3, 30)
            Bch_list = []
            Bfr_list = []
            
            for idx,thickness in tqdm.tqdm(enumerate(thickness_list),total=len(thickness_list)):
                typo.ceiling_supplementary_insulation_thickness = 0.1
                typo.w0_insulation_thickness = thickness
                typo.w1_insulation_thickness = thickness
                typo.w2_insulation_thickness = thickness
                typo.w3_insulation_thickness = thickness
            
                simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
                simulation = aggregate_resolution(simulation, resolution='h')
                simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
                
                Bfr_list.append(simulation.cooling_needs.to_list())
                Bch_list.append(simulation.heating_needs.to_list())
            
            Bch_list = np.asarray(Bch_list)
            Bfr_list = np.asarray(Bfr_list)
            Bfr_list = Bfr_list/Bfr_list[0]
            Bch_list = Bch_list/Bch_list[0]
            
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot(thickness_list, Bfr_list, color='tab:blue', label='Bfr')
            ax.plot(thickness_list, Bch_list.mean(axis=1), color='tab:red', label='Modelled')
            ax.fill_between(thickness_list, 
                            # Bch_list.mean(axis=1)+Bch_list.std(axis=1), 
                            Bch_list.max(axis=1),
                            # Bch_list.mean(axis=1)-Bch_list.std(axis=1),
                            Bch_list.min(axis=1),
                            color='tab:red',alpha=0.2)
            # ax.plot(thickness_list, Btot_list, color='k', label='B')
            
            ref_data = pd.read_csv(os.path.join('data','Literature','3-Pomianowski_2023.csv')).set_index('insulation_thickness')
            for c in ref_data.columns:
                ref_data[c] = ref_data[c]/ref_data.loc[0.05][c]
                
            ax.errorbar(ref_data.index, ref_data.mean(axis=1), ref_data.max(axis=1)-ref_data.min(axis=1), 
                        color='k', label='Pomianowski 2023',ls='',marker='o',mfc='w')

            ax.legend()
            ax.set_xlim([0.05,0.3])
            ax.set_xlabel('Walls insulation thickness (m)')
            ax.set_ylabel('Heating needs ratio')
            ax.set_ylim(bottom=0.)
            plt.savefig(os.path.join(figs_folder,'effect_walls_insulation_heating_needs.png'),bbox_inches='tight')
            plt.show()
            
            
        # Évolution des monogestes
        if True:
            # Localisation
            zcl = Climat('H1a')
            zcl = Climat('H3')
            typo_code = 'FR.N.SFH.01.Gen'
            
            # premier test
            if False:
                # compute_energy_gains('roof',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True)
                
                # compute_energy_gains('walls',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True)
                
                # compute_energy_gains('floor',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True)
                
                # compute_energy_gains('albedo',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True)
                
                compute_energy_gains('ventilation',typo_code,zcl,
                                     output_path=os.path.join(output, folder),
                                     behaviour='conventionnel',
                                     period=[2000,2020],
                                     plot=True,show=True,
                                     progressbar=True)
            
            # test de parallelisation
            if True:
                zc_list = ['H1b','H2c','H3']
                typo_list = ['FR.N.SFH.01.Gen',
                             'FR.N.TH.01.Gen',
                             'FR.N.MFH.01.Gen',
                             'FR.N.AB.03.Gen']
                
                # TODO faire shading
                components = ['roof','walls','floor','albedo']#,'ventilation']#,'shading']
                
                nb_cpu = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(nb_cpu)
                
                run_list = []
                for zc in zc_list:
                    for typo in typo_list:
                        for comp in components:
                            run_list.append((comp,typo,Climat(zc),os.path.join(output, folder)))
                        
                pool.starmap(compute_energy_gains, run_list)
                                 
    
        # Effets de l'albedo de la maison 
        if False:
            # Localisation
            city = 'Beauvais'
            # city = 'Brest'
            city = 'Nice'
            
            # Période de calcul
            period = [2009,2010]
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
            # conventionnel.heating_rules = {i:[19.9]*24 for i in range(1,8)}
            # conventionnel.cooling_rules = {i:[22.6]*24 for i in range(1,8)}
            
            typo_name = 'FR.N.SFH.03.Gen'
            # typo_name = 'FR.N.SFH.06.Gen'
            typo = Typology(typo_name)
            
            color_list = ['light','medium','dark','black']
            
            Bch_list = np.asarray([0]*len(color_list))
            Bfr_list = np.asarray([0]*len(color_list))
            for idx,color in tqdm.tqdm(enumerate(color_list),total=len(color_list)):
                
                typo.roof_color = color
                typo.w0_color = color
                typo.w1_color = color
                typo.w3_color = color
                typo.w2_color = color
            
                simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
                simulation = aggregate_resolution(simulation, resolution='h')
                simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
                Bfr_list[idx] = simulation.cooling_needs.mean()
                Bch_list[idx] = simulation.heating_needs.mean()
            
            Btot_list = Bfr_list + Bch_list
            # Btot_list = Btot_list/Btot_list[0]
            # Bfr_list = Bfr_list/Bfr_list[0]
            # Bch_list = Bch_list/Bch_list[0]
    
                
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(color_list, Bfr_list, color='tab:blue', label='Bfr')
            ax.plot(color_list, Bch_list, color='tab:red', label='Bch')
            ax.plot(color_list, Btot_list, color='k', label='B')
            plt.show()
        
        
        # Effets de deux dates et 2 localisation
        if False:
            ls_list = ['solid','dotted','dashed','dashdot',]
            
            climate_zones = ['H1a','H3']
            
            years = [2090,2010] # chaud et froid
            
            for _,zcl in enumerate(climate_zones):
                city = Climat(zcl).center_prefecture
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                
                for i,year in enumerate(years):

                    period = [year]*2
            
                    weather_data_checkfile = ".weather_data_{}_{}_{}_".format(city,period[0],period[1]) + today + ".pickle"
                    if weather_data_checkfile not in os.listdir():
                        if year < 2024:
                            weather_data = get_historical_weather_data(city,period)
                        else:
                            weather_data = get_projected_weather_data(city,Climat(zcl).codint,nmod=0,rcp=85,future_period=period)
                        weather_data = refine_resolution(weather_data, resolution='600s')
                        pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
                    else:
                        weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
                        
                    agg_weather = aggregate_resolution(weather_data,'D',agg_method='mean')[['temperature_2m']]
                    
                    ax.plot(agg_weather.values, label='{} {}'.format(city,year),color=['tab:red','tab:blue'][i])
                    ax.plot([0,len(agg_weather.values)], [agg_weather.mean()]*2,ls=':',color=['tab:red','tab:blue'][i])
                ax.legend()
                plt.show()
            
            for _,zcl in enumerate(climate_zones):
                city = Climat(zcl).center_prefecture
                # if city == 'Beauvais':
                #     continue
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                
                for i,year in enumerate(years):

                    period = [year]*2
            
                    weather_data_checkfile = ".weather_data_{}_{}_{}_".format(city,period[0],period[1]) + today + ".pickle"
                    if weather_data_checkfile not in os.listdir():
                        weather_data = get_historical_weather_data(city,period)
                        weather_data = refine_resolution(weather_data, resolution='600s')
                        pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
                    else:
                        weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
            
                    # Définition des habitudes
                    conventionnel = Behaviour('conventionnel_th-bce_2020')
            
                    typo_name = 'FR.N.SFH.03.Gen'
                    # typo_name = 'FR.N.SFH.06.Gen'
                    # typo_name = 'FR.N.SFH.09.Gen'
                    typo = Typology(typo_name)
                    
                    thickness_list = np.linspace(0, 0.4, 30)
                    
                    Bch_checkfile = ".Bch_{}_{}_{}_{}_".format(typo_name,city,period[0],period[1]) + today + ".pickle"
                    Bfr_checkfile = ".Bfr_{}_{}_{}_{}_".format(typo_name,city,period[0],period[1]) + today + ".pickle"
                    
                    if Bch_checkfile not in os.listdir():
                        Bch_list = np.asarray([0]*len(thickness_list))
                        Bfr_list = np.asarray([0]*len(thickness_list))
                        
                    for idx,thickness in tqdm.tqdm(enumerate(thickness_list),total=len(thickness_list)):
                        
                        if Bch_checkfile not in os.listdir():
                            # print('oui')
                            
                            typo.w0_insulation_thickness = thickness
                            typo.w1_insulation_thickness = thickness
                            typo.w2_insulation_thickness = thickness
                            typo.w3_insulation_thickness = thickness
                        
                            simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
                            simulation = aggregate_resolution(simulation, resolution='h')
                            simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
                            Bfr_list[idx] = simulation.cooling_needs.mean()
                            Bch_list[idx] = simulation.heating_needs.mean()
                            
                    if Bch_checkfile not in os.listdir():   
                        pickle.dump(Bch_list, open(Bch_checkfile, "wb"))
                        pickle.dump(Bfr_list, open(Bfr_checkfile, "wb"))
                            
                    Bch_list = pickle.load(open(Bch_checkfile, 'rb'))
                    Bfr_list = pickle.load(open(Bfr_checkfile, 'rb'))
                    
                    Bch_list = Bch_list/typo.surface
                    Bfr_list = Bfr_list/typo.surface
                    
                    Btot_list = Bfr_list + Bch_list
                    
                    
                    
                    # relatif 
                    # Btot_list = Btot_list/Btot_list[0]
                    # Bfr_list = Bfr_list/Bfr_list[0]
                    # Bch_list = Bch_list/Bch_list[0]
                    
                    # difference
                    # Btot_list = Btot_list-Btot_list[0]
                    # Bfr_list = Bfr_list-Bfr_list[0]
                    # Bch_list = Bch_list-Bch_list[0]
                    
    
                    ax.plot(thickness_list, Bfr_list, label='Bfr ({} {})'.format(city,year), color='tab:blue', ls=ls_list[i%4])
                    ax.plot(thickness_list, Bch_list, label='Bch ({} {})'.format(city,year), color='tab:red', ls=ls_list[i%4])
                    ax.plot(thickness_list, Btot_list, label='B ({} {})'.format(city,year), color='k', ls=ls_list[i%4])
                        
                ax.legend()
                # ax.set_ylim(bottom=0.,top=3.5)
                # ax.set_ylim(bottom=0.,top=2.5e7)
                # ax.set_ylim(bottom=0.,top=2.5e7)
                # ax.set_ylim(bottom=0.)
                # ax.plot([thickness_list[0],thickness_list[-1]],[1]*2,color='k',zorder=-1)
                ax.set_xlim([thickness_list[0],thickness_list[-1]])
                ax.set_title(typo_name)
                plt.show()
            
        
        # Graphes de variables conjointes 
        if False:
            # Localisation
            city = 'Beauvais'
            # city = 'Brest'
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
            # typo = Typology(typo_name)
            
            thickness_list = np.linspace(0, 0.5, 15)
            thickness_list2 = np.linspace(0, 0.55, 15)
            Bch_list = np.meshgrid(thickness_list,thickness_list2)[0]*0
            Bfr_list = np.meshgrid(thickness_list,thickness_list2)[0]*0
            
            for idx,thw in tqdm.tqdm(enumerate(thickness_list),total=len(thickness_list)):
                for idy,thf in enumerate(thickness_list2):
                    typo = Typology(typo_name)
                    
                    typo.w0_insulation_thickness = thw
                    # typo.w1_insulation_thickness = thw
                    typo.w2_insulation_thickness = thf
                    # typo.w3_insulation_thickness = thw
                    
                    # typo.floor_insulation_thickness = thf
            
                    simulation = run_thermal_model(typo, conventionnel, weather_data, pmax_warning=False)
                    simulation = aggregate_resolution(simulation, resolution='h')
                    simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
                    Bfr_list[idx,idy] = simulation.cooling_needs.mean()
                    Bch_list[idx,idy] = simulation.heating_needs.mean()
            
            pickle.dump(Bfr_list, open('.Bfr_list.pickle', "wb"))
            pickle.dump(Bch_list, open('.Bch_list.pickle', "wb"))
            
            Bfr_list = pickle.load(open('.Bfr_list.pickle', 'rb'))
            Bch_list = pickle.load(open('.Bch_list.pickle', 'rb'))
            
            fig, ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.contourf(np.meshgrid(thickness_list,thickness_list2)[0], 
                        np.meshgrid(thickness_list,thickness_list2)[1], 
                        Bch_list)
            plt.show()
            
            fig, ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.contourf(np.meshgrid(thickness_list,thickness_list2)[0], 
                        np.meshgrid(thickness_list,thickness_list2)[1], 
                        Bfr_list)
            plt.show()
            
            fig, ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.contourf(np.meshgrid(thickness_list,thickness_list2)[0], 
                        np.meshgrid(thickness_list,thickness_list2)[1], 
                        Bch_list+Bfr_list)
            plt.show()
    
    
    
    
    tac = time.time()
    print("Done in {:.2f}s".format(tac-tic))
    
if __name__ == '__main__':
     main()