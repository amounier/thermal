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

from meteorology import get_historical_weather_data, get_safran_hourly_weather_data
from thermal_model import (refine_resolution, 
                           aggregate_resolution, 
                           run_thermal_model, 
                           plot_timeserie)
from behaviour import Behaviour
from administrative import Climat, France
from typologies import Typology
from future_meteorology import get_projected_weather_data


# models_period_dict = {0:{2:[2029,2049],
#                          4:[2064,2084],},
#                       1:{2:[2018,2038],
#                          4:[2056,2076],},
#                       2:{2:[2024,2044],
#                          4:[2066,2086],},
#                       3:{2:[2013,2033],
#                          4:[2056,2076],},
#                       4:{2:[2006,2024], # debut des projections en 2006
#                          4:[2046,2066],},}

models_period_dict = {0:{2:[2020,2040],
                         4:[2059,2079],},
                      1:{2:[2013,2033],
                         4:[2054,2074],},
                      2:{2:[2017,2037],
                         4:[2061,2081],},
                      3:{2:[2006,2025],
                         4:[2047,2067],},
                      4:{2:[2006,2021], # debut des projections en 2006
                         4:[2040,2060],},}


def compute_energy_needs_single_actions(component,typo_code,zcl,output_path,
                                        behaviour='conventionnel',period=[2000,2020],
                                        plot=True,nb_intervals=10,show=False,
                                        progressbar=False, model='explore2',
                                        nmod=3):
    
    city = zcl.center_prefecture
    typo = Typology(typo_code)
    
    # weather data
    if model == 'explore2':
        weather_data_checkfile = ".weather_data_{}_{}_{}_explore2_mod{}".format(city,period[0],period[1],nmod) + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_projected_weather_data(zcl.code, period,nmod=nmod)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            weather_data = pickle.load(open(weather_data_checkfile, 'rb'))

    else:
        weather_data_checkfile = ".weather_data_{}_{}_{}".format(city,period[0],period[1]) + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_historical_weather_data(city,period)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
    
    
    
    
    if behaviour == 'conventionnel':
        behaviour = Behaviour('conventionnel_th-bce_2020')
    
    type_dict = {'SFH':'Individual',
                 'TH':'Individual',
                 'MFH':'Collective',
                 'AB':'Collective'}
    type_long = type_dict.get(typo.type)
                                        
    dict_all_components = {'floor':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=nb_intervals)-0.05,
                                    'var_label':'Supplementary floor insulation thickness (m)',
                                    'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                    },
                           'walls':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=nb_intervals)-0.05,
                                    'var_label':'Supplementary walls insulation thickness (m)',
                                    'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                    },
                           'roof':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=nb_intervals)-0.05,
                                   'var_label':'Supplementary roof insulation thickness (m)',
                                   'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                    },
                           'albedo':{'var_space':['light','medium','dark','black'],
                                     'var_label':'External surface color',
                                     'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                     },
                           'ventilation':{'var_space':['natural','{} MV'.format(type_long),'{} DCV'.format(type_long),'{} HRV'.format(type_long)],
                                          'var_label':'Ventilation type',
                                          'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                          },
                           'shading':{'var_space':np.linspace(0, 2, nb_intervals),
                                      'var_label':'Solar shader length (m)',
                                      'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                          },
                           'windows':{'var_space':np.linspace(1, 5, nb_intervals),
                                      'var_label':'Windows U-value (W.m$^{-2}$.K$^{-1}$)',
                                      'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                          },
                           }
    
    
    dict_plot_top_value = {'SFH':350,
                           'TH':250,
                           'MFH':300,
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
                typo.ventilation = var_value
                typo.ventilation_efficiency = typo.get_ventilation_efficiency()
            if component == 'shading':
                typo.solar_shader_length = var_value
            if component == 'windows':
                typo.windows_U = var_value
                
            # typo.basement = False
                
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


def get_energy_needs_single_actions(component,typo_code,zcl,output_path,
                                    behaviour='conventionnel',period=[2000,2020],
                                    nb_intervals=10, model='explore2',nmod=3):
    
    city = zcl.center_prefecture
    typo = Typology(typo_code)
    
    # weather data
    if model == 'explore2':
        weather_data_checkfile = ".weather_data_{}_{}_{}_explore2_mod{}".format(city,period[0],period[1],nmod) + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_projected_weather_data(zcl.code, period,nmod=nmod)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            weather_data = pickle.load(open(weather_data_checkfile, 'rb'))

    else:
        weather_data_checkfile = ".weather_data_{}_{}_{}".format(city,period[0],period[1]) + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_historical_weather_data(city,period)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
    
    if behaviour == 'conventionnel':
        behaviour = Behaviour('conventionnel_th-bce_2020')
    
    type_dict = {'SFH':'Individual',
                 'TH':'Individual',
                 'MFH':'Collective',
                 'AB':'Collective'}
    type_long = type_dict.get(typo.type)
                                        
    dict_all_components = {'floor':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=nb_intervals)-0.05,
                                    'var_label':'Supplementary floor insulation thickness (m)',
                                    'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                    },
                           'walls':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=nb_intervals)-0.05,
                                    'var_label':'Supplementary walls insulation thickness (m)',
                                    'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                    },
                           'roof':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=nb_intervals)-0.05,
                                   'var_label':'Supplementary roof insulation thickness (m)',
                                   'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                    },
                           'albedo':{'var_space':['light','medium','dark','black'],
                                     'var_label':'External surface color',
                                     'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                     },
                           'ventilation':{'var_space':['natural','{} MV'.format(type_long),'{} DCV'.format(type_long),'{} HRV'.format(type_long)],
                                          'var_label':'Ventilation type',
                                          'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                          },
                           'shading':{'var_space':np.linspace(0, 2, nb_intervals),
                                      'var_label':'Solar shader length (m)',
                                      'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                          },
                           'windows':{'var_space':np.linspace(1, 5, nb_intervals),
                                      'var_label':'Windows U-value (W.m$^{-2}$.K$^{-1}$)',
                                      'var_saver':'action_{}_{}_{}_{}_{}-{}_mod{}'.format(component,typo_code,zcl.code,behaviour.name,period[0],period[1],nmod)
                                          },
                           }
    
    
    dict_plot_top_value = {'SFH':300,
                           'TH':250,
                           'MFH':300,
                           'AB':250}
    
    dict_components = dict_all_components.get(component)
    
    var_space = dict_components.get('var_space')
    
    Bch_list = [0]*len(var_space)
    Bfr_list = [0]*len(var_space)
    
    if '{}.pickle'.format(dict_components.get('var_saver')) not in os.listdir(output_path):
        compute_energy_needs_single_actions(component,typo_code,zcl,output_path,
                                            behaviour,period,
                                            plot=False,nb_intervals=nb_intervals,show=False,
                                            progressbar=False, model=model,
                                            nmod=nmod)
        
    Bch_list,Bfr_list = pickle.load(open(os.path.join(os.path.join(output_path),'{}.pickle'.format(dict_components.get('var_saver'))), 'rb'))
    
    Bch_list = np.asarray(Bch_list)/(1e3 * typo.surface)
    Bfr_list = np.asarray(Bfr_list)/(1e3 * typo.surface)
    return Bch_list, Bfr_list


def compute_energy_needs_typology(typo_code, typo_level,zcl,output_path,
                                  behaviour='conventionnel',period=[2000,2020],
                                  model='explore2',nmod=3):
    typo = Typology(typo_code, typo_level)
    city = zcl.center_prefecture
    
    # weather data
    if model == 'explore2':
        weather_data_checkfile = ".weather_data_{}_{}_{}_explore2_mod{}".format(city,period[0],period[1],nmod) + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_projected_weather_data(zcl.code, period,nmod=nmod)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            weather_data = pickle.load(open(weather_data_checkfile, 'rb'))

    else:
        weather_data_checkfile = ".weather_data_{}_{}_{}".format(city,period[0],period[1]) + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_historical_weather_data(city,period)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
    
    
    if behaviour == 'conventionnel':
        behaviour = Behaviour('conventionnel_th-bce_2020')
    
    var_saver = 'typology_{}_lvl-{}_{}_{}-{}_mod{}'.format(typo.code, typo.insulation_level, zcl.code, period[0],period[1],nmod)
    
    if '{}.pickle'.format(var_saver) not in os.listdir(output_path):
        
        simulation = run_thermal_model(typo, behaviour, weather_data, pmax_warning=False)
        simulation = aggregate_resolution(simulation, resolution='h')
        simulation = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
        
        pickle.dump(simulation, open(os.path.join(os.path.join(output_path),'{}.pickle'.format(var_saver)), "wb"))
    
    simulation = pickle.load(open(os.path.join(os.path.join(output_path),'{}.pickle'.format(var_saver)), 'rb'))
    
    return simulation


def draw_building_type_energy_needs(building_type, zcl, output_path, save=True,
                                    behaviour='conventionnel', period=[2000,2020],
                                    model="explore2",nmod=3):
    
    heating_needs = {}
    cooling_needs = {}
    
    for i in range(1,11):
        code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
        for level in ['initial','standard','advanced']:
            typology = Typology(code,level)
            simu = compute_energy_needs_typology(code,level,
                                                 zcl=zcl,
                                                 output_path=output_path,
                                                 behaviour=behaviour,
                                                 period=period,
                                                 model=model,
                                                 nmod=nmod)
            simu = simu/(1e3 * typology.surface)
            heating_needs[(code,level)] = simu.heating_needs.values
            cooling_needs[(code,level)] = simu.cooling_needs.values
            
    fig,ax = plt.subplots(figsize=(10,5),dpi=300)
    for i in range(1,11):
        code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
        
        j = i*7
        X = [j,j+2,j+4]
            
        for k,level in enumerate(['initial','standard','advanced']):
            if i==1 and k==1:
                label_heating ='Heating needs'
                label_cooling ='Cooling needs'
            else:
                label_heating = None
                label_cooling = None
            heating_color = 'tab:red'
            cooling_color = 'tab:blue'

            ax.bar([X[k]], heating_needs[(code,level)].mean(), 
                   yerr = heating_needs[(code,level)].std(),
                   width=1.6, label=label_heating, color=heating_color,alpha=0.5,
                   error_kw=dict(ecolor=heating_color,lw=1, capsize=2, capthick=1))
            
            ax.bar([X[k]], cooling_needs[(code,level)].mean(), 
                   bottom=heating_needs[(code,level)].mean(),
                   yerr = cooling_needs[(code,level)].std(),
                   width=1.6, label=label_cooling, color=cooling_color,alpha=0.5,
                   error_kw=dict(ecolor=cooling_color,lw=1, capsize=2, capthick=1))

    ax.set_ylim(bottom=0.)
    ax.set_ylabel('Energy needs (kWh.m$^{-2}$.yr$^{-1}$)')
    ax.legend()
    ax.set_title(zcl.code)
    ax.set_xticks([(i*7)+2 for i in range(1,11)],['{}.{:02d}'.format(building_type,i) for i in range(1,11)])
    
    if save:
        plt.savefig(os.path.join(output_path,'figs','{}.png'.format('typology_energy_needs_{}_{}_{}-{}'.format(building_type,zcl.code,period[0],period[1]))),bbox_inches='tight')
    plt.show()
    plt.close()
    return 



def draw_climate_impact_building_type_energy_needs(building_type, zcl, output_path, save=True,
                                                   behaviour='conventionnel', period_dict=models_period_dict,
                                                   model="explore2",nmod=3):
    
    # heating_needs = {}
    # cooling_needs = {}
    total_needs = {}
    
    for i in range(1,11):
        code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
        for level in ['initial','standard','advanced']:
            for climat in ['now',2,4]:
                if climat == 'now':
                    period = [2000,2020]
                else:
                    period = period_dict.get(nmod).get(climat)
                    
                typology = Typology(code,level)
                simu = compute_energy_needs_typology(code,level,
                                                     zcl=zcl,
                                                     output_path=output_path,
                                                     behaviour=behaviour,
                                                     period=period,
                                                     model=model,
                                                     nmod=nmod)
                simu = simu/(1e3 * typology.surface)
                total_needs[(code,level,climat)] = simu.cooling_needs.values + simu.heating_needs.values
            
    fig,ax = plt.subplots(figsize=(15,5),dpi=300)
    for i in range(1,11):
        code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
        
        j = i*7
        X = [j,j+2,j+4]
            
        for k,level in enumerate(['initial','standard','advanced']):
            # for climat in ['now',2,4]:
            if i==1 and k==1:
                label_now ='2000-2020'
                label_2 ='+2°C'
                label_4 ='+4°C'
            else:
                label_now = None
                label_2 = None
                label_4 = None
             
            color_now = 'k'
            color_2 = 'tab:blue'
            color_4 = 'tab:red'

            # ax.bar([X[k]], heating_needs[(code,level)].mean(), 
            #        yerr = heating_needs[(code,level)].std(),
            #        width=1.6, label=label_heating, color=heating_color,alpha=0.5,
            #        error_kw=dict(ecolor=heating_color,lw=1, capsize=2, capthick=1))
            
            # ax.bar([X[k]], cooling_needs[(code,level)].mean(), 
            #        bottom=heating_needs[(code,level)].mean(),
            #        yerr = cooling_needs[(code,level)].std(),
            #        width=1.6, label=label_cooling, color=cooling_color,alpha=0.5,
            #        error_kw=dict(ecolor=cooling_color,lw=1, capsize=2, capthick=1))
            
            ax.boxplot(total_needs[(code,level,'now')],positions=[X[k]-0.6],
                       widths=0.5,label=label_now,
                       boxprops=dict(color=color_now),
                       capprops=dict(color=color_now),
                       whiskerprops=dict(color=color_now),
                       flierprops=dict(markeredgecolor=color_now,markersize=2),
                       medianprops=dict(color=color_now),)
            
            ax.boxplot(total_needs[(code,level,2)],positions=[X[k]],
                       widths=0.5,label=label_2,
                       boxprops=dict(color=color_2),
                       capprops=dict(color=color_2),
                       whiskerprops=dict(color=color_2),
                       flierprops=dict(markeredgecolor=color_2,markersize=2),
                       medianprops=dict(color=color_2),)
            
            ax.boxplot(total_needs[(code,level,4)],positions=[X[k]+0.6],
                       widths=0.5,label=label_4,
                       boxprops=dict(color=color_4),
                       capprops=dict(color=color_4),
                       whiskerprops=dict(color=color_4),
                       flierprops=dict(markeredgecolor=color_4,markersize=2),
                       medianprops=dict(color=color_4),)
        
        

    ax.set_ylim(bottom=0.)
    ax.set_ylabel('Energy needs (kWh.m$^{-2}$.yr$^{-1}$)')
    ax.legend()
    ax.set_title(zcl.code)
    ax.set_xticks([(i*7)+2 for i in range(1,11)],['{}.{:02d}'.format(building_type,i) for i in range(1,11)])
    
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    for i in range(1,11):
        j = i*7
        X = [j-1.5,j+2,j+5.5]
        if i%2==0:
            ax.fill_between(X,[ylims[1]]*3,[ylims[0]]*3,color='lightgrey',alpha=0.37,zorder=-2)
    
    ax.set_xlim(xlims)
    if save:
        plt.savefig(os.path.join(output_path,'figs','{}.png'.format('typology_climate_impacts_energy_needs_{}_{}_{}-{}_nmod{}'.format(building_type,zcl.code,period[0],period[1],nmod))),bbox_inches='tight')
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
    if False:
        
        # Caractérisation du temps de calcul
        if False:
            # Localisation
            zcl = Climat('H1a')
            city = zcl.center_prefecture
            # city = 'Beauvais'
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
            
            
            weather_data = get_projected_weather_data(zcl.code, period)
            weather_data = refine_resolution(weather_data, resolution='600s')
            
            plot_timeserie(weather_data[['temperature_2m']],figsize=(15,5),ylabel='Energy needs (Wh)',
                           figs_folder=figs_folder,show=True)
            
            plot_timeserie(weather_data[['direct_sun_radiation_H']],figsize=(15,5),ylabel='Energy needs (Wh)',
                           figs_folder=figs_folder,show=True)
                
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
            
            plot_timeserie(simulation[['heating_needs','cooling_needs']],figsize=(15,5),ylabel='Energy needs (Wh)',
                           figs_folder=figs_folder,show=True)
            
            
        
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
            
            weather_source = 'ERA5' # ERA5
            
            # Checkpoint weather data
            if weather_source == 'ERA5':
                weather_data_checkfile = ".weather_data_{}_{}_{}_".format(city,period[0],period[1]) + today + ".pickle"
                if weather_data_checkfile not in os.listdir():
                    weather_data = get_historical_weather_data(city,period)
                    weather_data = refine_resolution(weather_data, resolution='600s')
                    pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
                else:
                    weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
            # elif weather_source == 'SAFRAN':
            #     weather_data = get_safran_hourly_weather_data(zcl_code,period)
            #     weather_data = refine_resolution(weather_data, resolution='600s')
                
            
            # Définition des habitudes
            conventionnel = Behaviour('conventionnel_th-bce_2020')
            # valeurs moyennes (enquete ifop et coda)
            # conventionnel.heating_rules = {i:[19.9]*24 for i in range(1,8)}
            # conventionnel.cooling_rules = {i:[22.6]*24 for i in range(1,8)}
            
            # typo_name = 'FR.N.SFH.03.Gen'
            typo_name = 'FR.N.MFH.03.Gen'
            # typo_name = 'FR.N.AB.03.Gen'
            typo = Typology(typo_name)
            
            # thickness_list = np.linspace(0.05, 0.3, 30)
            thickness_list = np.logspace(np.log10(0+0.05),np.log10(0.3),num=10)
            Bch_list = []
            Bfr_list = []
            
            for idx,thickness in tqdm.tqdm(enumerate(thickness_list),total=len(thickness_list)):
                typo.ceiling_supplementary_insulation_thickness = 0.05
                typo.floor_insulation_thickness = 0.05
                typo.windows_U = 1
                
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
        if False:
            # Localisation
            zcl = Climat('H1a')
            # zcl = Climat('H3')
            typo_code = 'FR.N.SFH.01.Gen'
            # typo_code = 'FR.N.SFH.07.Gen'
            
            # premier test
            if False:
                # compute_energy_needs_single_actions('roof',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True)
                
                # compute_energy_needs_single_actions('walls',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True)
                
                # compute_energy_needs_single_actions('floor',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True)
                
                # compute_energy_needs_single_actions('albedo',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True)
                
                # compute_energy_needs_single_actions('ventilation',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True)
                
                # compute_energy_needs_single_actions('shading',typo_code,zcl,
                #                      output_path=os.path.join(output, folder),
                #                      behaviour='conventionnel',
                #                      period=[2000,2020],
                #                      plot=True,show=True,
                #                      progressbar=True,model='era5')
                
                compute_energy_needs_single_actions('windows',typo_code,zcl,
                                     output_path=os.path.join(output, folder),
                                     behaviour='conventionnel',
                                     period=[2000,2020],
                                     plot=True,show=True,
                                     progressbar=True,model='era5')
            
            # parallelisation
            if False:
                zc_list = ['H1b','H2c','H3']
                typo_list = ['FR.N.SFH.01.Gen','FR.N.TH.01.Gen','FR.N.MFH.01.Gen','FR.N.AB.03.Gen']
                components = ['roof','walls','floor','albedo','ventilation','shading']
                
                nb_cpu = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(nb_cpu)
                
                run_list = []
                for zc in zc_list:
                    for typo in typo_list:
                        for comp in components:
                            run_list.append((comp,typo,Climat(zc),os.path.join(output, folder)))
                        
                pool.starmap(compute_energy_needs_single_actions, run_list)
                
            # optimisation par geste
            if False:
                zcl_list = ['H2c']
                
                run_list = []
                for zcl_code in zcl_list:
                    zcl = Climat(zcl_code)
                    # for building_type in ['SFH','TH','MFH','AB']:
                    for building_type in ['SFH']:
                        for i in range(1,11):
                            code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
                            run_list.append(('shading',
                                             code,
                                             zcl,
                                             os.path.join(output, folder),
                                             'conventionnel',
                                             [2010,2020]))
                
                nb_cpu = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(nb_cpu)
                pool.starmap(compute_energy_needs_single_actions, run_list)
        
        
        
        # Consommations énergétiques des typologies à différents niveaux
        if False:
            zcl_list = ['H1b','H3']
            zcl = Climat('H1a')
        
            run_list = []
            for zcl_code in zcl_list:
                zcl = Climat(zcl_code)
                for building_type in ['SFH','TH','MFH','AB']:
                # for building_type in ['SFH']:
                    for i in range(1,11):
                        code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
                        for level in ['initial','standard','advanced']:
                            run_list.append((code, level, zcl, os.path.join(output, folder)))
                        
            nb_cpu = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(nb_cpu)
            pool.starmap(compute_energy_needs_typology, run_list)
            
            for zcl_code in zcl_list:
                zcl = Climat(zcl_code)
                for building_type in ['SFH','TH','MFH','AB']:
                # for building_type in ['SFH']:
                    draw_building_type_energy_needs(building_type,zcl=zcl,output_path=os.path.join(output, folder))
                    

    #%% Changement de période climatique
    if False:
        
        # models_period_dict = {0:{2:[2029,2049],
        #                          4:[2064,2084],},
        #                       1:{2:[2018,2038],
        #                          4:[2056,2076],},
        #                       2:{2:[2024,2044],
        #                          4:[2066,2086],},
        #                       3:{2:[2013,2033],
        #                          4:[2056,2076],},
        #                       4:{2:[2006,2024], # debut des projections en 2006
        #                          4:[2046,2066],},}
        
        # premier test 
        if False:
            zcl = Climat('H1b')
            # zcl = Climat('H3')
            typo_code = 'FR.N.SFH.01.Gen'
            typo_code = 'FR.N.SFH.08.Gen'
            mod = 3
            # component = 'walls'
            # component = 'roof'
            component = 'windows'
            
            compute_energy_needs_single_actions(component,typo_code,zcl,
                                 output_path=os.path.join(output, folder),
                                 behaviour='conventionnel',
                                 period=[2000,2020],
                                 plot=True,show=True,
                                 progressbar=True,
                                 model='explore2',nmod=mod)
            
            compute_energy_needs_single_actions(component,typo_code,zcl,
                                 output_path=os.path.join(output, folder),
                                 behaviour='conventionnel',
                                 period=models_period_dict.get(mod).get(4),
                                 plot=True,show=True,
                                 progressbar=True,
                                 model='explore2',nmod=mod)
            
        # evolution des consommations totales pour les typologies
        if False:
            zcl_list = ['H1b','H3']
            zcl = Climat('H1b')
            mod = 3
        
            run_list = []
            for zcl_code in zcl_list:
                zcl = Climat(zcl_code)
                for building_type in ['SFH','TH','MFH','AB']:
                # for building_type in ['SFH']:
                    for i in range(1,11):
                        code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
                        for level in ['initial','standard','advanced']:
                            for climat in ['now',2,4]:
                                if climat == 'now':
                                    period = [2000,2020]
                                else:
                                    period = models_period_dict.get(mod).get(climat)
                                run_list.append((code, level, zcl, os.path.join(output, folder),
                                                 'conventionnel',period,'explore2',mod))
                                              
            nb_cpu = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(nb_cpu)
            pool.starmap(compute_energy_needs_typology, run_list)
            
            for zcl_code in zcl_list:
                zcl = Climat(zcl_code)
                for building_type in ['SFH','TH','MFH','AB']:
                # for building_type in ['SFH']:
                    draw_climate_impact_building_type_energy_needs(building_type,zcl=zcl,output_path=os.path.join(output, folder))
        
        
        # cadran des rénovations par gestes
        if False:
            
            # calcul des gains
            if True:
                # component = 'shading'
                mod = 3
                zcl_list = ['H1b','H3']
                
                run_list = []
                # for component in ['shading','walls','floor','roof','albedo','windows']:
                for component in ['windows']:
                    for zcl_code in zcl_list:
                        zcl = Climat(zcl_code)
                        # for building_type in ['SFH','TH','MFH','AB']:
                        for building_type in ['SFH']:
                            for i in range(1,11):
                                code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
                                
                                run_list.append((component,code,zcl,
                                                 os.path.join(output, folder),
                                                 'conventionnel',
                                                 [2000,2020],False))
                                
                                run_list.append((component,code,zcl,
                                                 os.path.join(output, folder),
                                                 'conventionnel',
                                                 models_period_dict.get(mod).get(2),False))
                                
                                run_list.append((component,code,zcl,
                                                 os.path.join(output, folder),
                                                 'conventionnel',
                                                 models_period_dict.get(mod).get(4),False))
                
                nb_cpu = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(nb_cpu)
                pool.starmap(compute_energy_needs_single_actions, run_list)
                
            # affichage du cadran
            if True:
                
                def find_nearest(array, value):
                    array = np.asarray(array)
                    idx = (np.abs(array - value)).argmin()
                    return idx, array[idx]


                dict_all_components = {'floor':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=10)-0.05,
                                                'var_label':'Supplementary floor insulation',
                                                'var_test':0.1,
                                                'var_ref':0,
                                                },
                                       'walls':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=10)-0.05,
                                                'var_label':'Supplementary walls insulation',
                                                'var_test':0.12,
                                                'var_ref':0,
                                                },
                                       'roof':{'var_space':np.logspace(np.log10(0+0.05),np.log10(0.4+0.05),num=10)-0.05,
                                               'var_label':'Supplementary roof insulation',
                                               'var_test':0.2,
                                               'var_ref':0,
                                               },
                                       'albedo':{'var_space':['light','medium','dark','black'],
                                                 'var_label':'External surface color',
                                                 'var_test':'light',
                                                 'var_ref':'dark',
                                                 },
                                       'ventilation':{'var_space':['natural','MV','DCV','HRV'],
                                                      'var_label':'Ventilation type',
                                                      'var_test':'HRV',
                                                      'var_ref':'natural',
                                                      },
                                       'shading':{'var_space':np.linspace(0, 2, 10),
                                                  'var_label':'Solar shader',
                                                  'var_test':1,
                                                  'var_ref':0,
                                                  },
                                       'windows':{'var_space':np.linspace(1, 5, 10),
                                                  'var_label':'Windows U-value',
                                                  'var_test':1.5,
                                                  'var_ref':4.6,
                                                  },
                                       }
                
                # component = 'windows'
                mod = 3
                zcl_list = ['H1b','H3']
                # zcl_list = ['H3']
                output_path = os.path.join(output, folder)
    

                # for component in ['shading','walls','floor','roof','albedo']:
                for component in ['windows']:
                    for zcl_code in zcl_list:
                        zcl = Climat(zcl_code)
                        # for building_type in ['SFH','TH','MFH','AB']:
                        for building_type in ['SFH']:
                            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                            max_Delta_x = 0
                            max_Delta_y = 0
                            for i in range(1,11):
                                code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
                                # print(code)
                                Bch_list,Bfr_list = get_energy_needs_single_actions(component,code,zcl,output_path, behaviour='conventionnel',period=[2000,2020], nmod=mod)
                                Bch_list2,Bfr_list2 = get_energy_needs_single_actions(component,code,zcl,output_path, behaviour='conventionnel',period=models_period_dict.get(mod).get(2), nmod=mod)
                                Bch_list4,Bfr_list4 = get_energy_needs_single_actions(component,code,zcl,output_path, behaviour='conventionnel',period=models_period_dict.get(mod).get(4), nmod=mod)
                                
                                Btot_list = Bch_list + Bfr_list
                                Bch_mean = Bch_list.mean(axis=1)
                                Bfr_mean = Bfr_list.mean(axis=1)
                                Btot_mean = Btot_list.mean(axis=1)
                                
                                Btot_list2 = Bch_list2 + Bfr_list2
                                Bch_mean2 = Bch_list2.mean(axis=1)
                                Bfr_mean2 = Bfr_list2.mean(axis=1)
                                Btot_mean2 = Btot_list2.mean(axis=1)
                                
                                Btot_list4 = Bch_list4 + Bfr_list4
                                Bch_mean4 = Bch_list4.mean(axis=1)
                                Bfr_mean4 = Bfr_list4.mean(axis=1)
                                Btot_mean4 = Btot_list4.mean(axis=1)
                                
                                var_space = dict_all_components.get(component).get('var_space')
                                var_ref = dict_all_components.get(component).get('var_ref')
                                var_test = dict_all_components.get(component).get('var_test')
                                
                                if var_test in var_space:
                                    nearest_test = var_test
                                    idx_test = var_space.index(var_test)
                                else:
                                    idx_test, nearest_test = find_nearest(var_space, var_test)
                                    
                                if var_ref in var_space:
                                    nearest_ref = var_ref
                                    idx_ref = list(var_space).index(var_ref)
                                else:
                                    idx_ref, nearest_ref = find_nearest(var_space, var_ref)
                                
                                
                                
                                idx_max = Btot_mean.argmax()
                                # idx_min = 3
                                # print(idx_min)
                                
                                Delta_Bch = Bch_mean[idx_ref]-Bch_mean[idx_test]
                                Delta_Bfr = Bfr_mean[idx_ref]-Bfr_mean[idx_test]
                                
                                Delta_Bch2 = Bch_mean2[idx_ref]-Bch_mean2[idx_test]
                                Delta_Bfr2 = Bfr_mean2[idx_ref]-Bfr_mean2[idx_test]
                                
                                Delta_Bch4 = Bch_mean4[idx_ref]-Bch_mean4[idx_test]
                                Delta_Bfr4 = Bfr_mean4[idx_ref]-Bfr_mean4[idx_test]
                                
                                max_Delta_y = np.max([max_Delta_y,np.abs(Delta_Bch),np.abs(Delta_Bch2),np.abs(Delta_Bch4)])
                                max_Delta_x = np.max([max_Delta_x,np.abs(Delta_Bfr),np.abs(Delta_Bfr2),np.abs(Delta_Bfr4)])
                                
                                cmap_dict = {'H3':plt.colormaps.get_cmap('Reds_r'),
                                             'H1b':plt.colormaps.get_cmap('Blues_r')}
                                cmap = cmap_dict.get(zcl.code)
                                
                                ax.plot([Delta_Bfr,Delta_Bfr2,Delta_Bfr4], 
                                        [Delta_Bch,Delta_Bch2,Delta_Bch4], 
                                        marker='o',color=cmap(i/11),
                                        label='{}.{:02d}'.format(building_type,i))
                                
                                ax.plot([Delta_Bfr], 
                                        [Delta_Bch], 
                                        marker='o',color=cmap(i/11),ls='',mfc='w')
                    
                            max_Delta_y *= 1.01
                            max_Delta_x *= 1.01
                            max_Delta = max(max_Delta_x, max_Delta_y)
                            ax.set_xlim([-max_Delta,max_Delta])
                            ax.set_ylim([-max_Delta,max_Delta])
                            
                            title = '{} - {}'.format(dict_all_components.get(component).get('var_label'), zcl.code) 
                            ax.set_title(title)
                            
                            ax.plot([0,0],[-max_Delta,max_Delta],ls=':',color='k',zorder=-1)
                            ax.plot([-max_Delta,max_Delta],[0,0],ls=':',color='k',zorder=-1)
                            ax.fill_between([-max_Delta,max_Delta],[max_Delta,-max_Delta],[-max_Delta,-max_Delta],
                                            color='lightgrey',alpha=0.5,zorder=-2)
                            
                            ax.set_xlabel('Gains in cooling needs (kWh.yr$^{-1}$.m$^{-2}$)')
                            ax.set_ylabel('Gains in heating needs (kWh.yr$^{-1}$.m$^{-2}$)')
                            ax.legend(ncol=2)
                            plt.savefig(os.path.join(figs_folder,'interactions_{}_{}_{}.png'.format(component,building_type,zcl.code)), bbox_inches='tight')
                            plt.show()
                        
        
        
    # %% Autres    
    if False:                           
    
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