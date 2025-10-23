#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:52:47 2025

@author: amounier
"""

import time 
import pandas as pd
import os 
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
import seaborn as sns
import multiprocessing
from scipy.ndimage import gaussian_filter

from typologies import Typology
from thermal_model import compute_U_values
from thermal_sensitivity import (open_electricity_consumption, 
                                 get_nationale_meteo,
                                 plot_thermal_sensitivity)
from future_meteorology import get_projected_weather_data
from thermal_model import (refine_resolution, 
                           aggregate_resolution, 
                           run_thermal_model)
from behaviour import Behaviour
from administrative import France, Climat


AUXILIARY_CONSUMPTION = 3
CONVERSION = 2.3
C_LIGHT = 0.9
P_LIGHT = 1.4 # W/m2
HOURS_LIGHT = 1500 # h
CONSUMPTION_LIGHT = (C_LIGHT * P_LIGHT * HOURS_LIGHT) / 1e3 # kWh/m2.a

def find_certificate(primary_consumption, heating_system, other_consumptions=None):
    """Returns energy performance certificate from A to G.

    Parameters
    ----------
    primary_consumption: float or pd.Series or pd.DataFrame
        Space heating energy consumption (kWh PE / m2.year)
    Returns
    -------
    float or pd.Series or pd.DataFrame
        Energy performance certificate.
    """
    CERTIFICATE_5USES_BOUNDARIES_ENERGY = {'A': [0, 70],
                                           'B': [70, 110],
                                           'C': [110, 180],
                                           'D': [180, 250],
                                           'E': [250, 330],
                                           'F': [330, 420],
                                           'G': [420, 9999],}
    
    CERTIFICATE_5USES_BOUNDARIES_EMISSION = {'A': [0, 6],
                                             'B': [6, 11],
                                             'C': [11, 30],
                                             'D': [30, 50],
                                             'E': [50, 70],
                                             'F': [70, 100],
                                             'G': [100, 1000],}
    
    CARBON_CONTENT = {'Electricity-Direct electric': 0.079,
                      'Electricity-Wood stove': 0.079,
                      'Electricity-Heat pump air': 0.079,
                      'Electricity-Heat pump': 0.079,
                      'Electricity-Heat pump water': 0.079,
                      'Natural gas-Performance boiler': 0.227,
                      'Natural gas-Standard boiler': 0.227,
                      'Natural gas-Collective boiler': 0.227,
                      'Oil fuel-Performance boiler': 0.324,
                      'Oil fuel-Standard boiler': 0.324,
                      'Oil fuel-Collective boiler': 0.324,
                      'Wood fuel-Performance boiler': 0.03,
                      'Wood fuel-Standard boiler': 0.03,
                      'Heating-District heating': 0.03}
    

    carbon_content = CARBON_CONTENT.get(heating_system)
    emission = primary_consumption * carbon_content
    primary_consumption += other_consumptions
    emission += other_consumptions * carbon_content

    if isinstance(primary_consumption, pd.Series):
        certificate_energy = pd.Series(dtype=str, index=primary_consumption.index)
        for key, item in CERTIFICATE_5USES_BOUNDARIES_ENERGY.items():
            cond = (primary_consumption > item[0]) & (primary_consumption <= item[1])
            certificate_energy[cond] = key

        certificate_emission = pd.Series(dtype=str, index=primary_consumption.index)
        for key, item in CERTIFICATE_5USES_BOUNDARIES_EMISSION.items():
            cond = (emission > item[0]) & (emission <= item[1])
            certificate_emission[cond] = key

        # maximum between energy and emission
        temp = pd.concat([certificate_energy, certificate_emission], axis=1, keys=['Energy', 'Emission'])
        certificate = temp.max(axis=1)

        return certificate
            

def reindex_mi(df, mi_index, levels=None, axis=0):
    """Return re-indexed DataFrame based on miindex using only few labels.

    Parameters
    -----------
    df: pd.DataFrame, pd.Series
        data to reindex
    mi_index: pd.MultiIndex, pd.Index
        master to index to reindex df
    levels: list, default df.index.names
        list of levels to use to reindex df
    axis: {0, 1}, default 0
        axis to reindex df

    Returns
    --------
    pd.DataFrame, pd.Series

    Example
    -------
        reindex_mi(surface_ds, segments, ['Occupancy status', 'Housing type']))
        reindex_mi(cost_invest_ds, segments, ['Heating energy final', 'Heating energy']))
    """

    if isinstance(df, (float, int)):
        return pd.Series(df, index=mi_index)

    if levels is None:
        if axis == 0:
            levels = df.index.names
        else:
            levels = df.columns.names

    if len(levels) > 1:
        tuple_index = (mi_index.get_level_values(level).tolist() for level in levels)
        new_miindex = pd.MultiIndex.from_tuples(list(zip(*tuple_index)))
        if axis == 0:
            df = df.reorder_levels(levels)
        else:
            df = df.reorder_levels(levels, axis=1)
    else:
        new_miindex = mi_index.get_level_values(levels[0])
    df_reindex = df.reindex(new_miindex, axis=axis)
    if axis == 0:
        df_reindex.index = mi_index
    elif axis == 1:
        df_reindex.columns = mi_index
    else:
        raise AttributeError('Axis can only be 0 or 1')

    return df_reindex
            
            
def conventional_heating_need(u_wall, u_floor, u_roof, u_windows, 
                              ratio_surface=pd.read_csv(os.path.join('data','Res-IRF','ratio_surface.csv')).set_index('Housing type'),
                              th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
                              air_rate=None, unobserved=None, climate=None, smooth=False, freq='year',
                              hourly_profile=None, temp_indoor=None, gain_utilization_factor=False,
                              zcl_thermal_parameters=None
                  ):
    """Seasonal method for space heating need.


    We apply a seasonal method according to EN ISO 13790 to estimate annual space heating demand by building type.
    The detailed calculation can be found in the TABULA project documentation (Loga, 2013).
    In a nutshell, the energy need for heating is the difference between the heat losses and the heat gain.
    The total heat losses result from heat transfer by transmission and ventilation during the heating season
    respectively proportional to the heat transfer coefficient $H_tr$ and $H_ve$.

    To not consider gain_utilization_factor create a difference of 5%. For consistency between results and

    Parameters
    ----------
    u_wall: pd.Series
        Index should include Housing type {'Single-family', 'Multi-family'}.
    u_floor: pd.Series
    u_roof: pd.Series
    u_windows: pd.Series
    ratio_surface: pd.Series
    th_bridging: {'Minimal', 'Low', 'Medium', 'High'}, default None
    vent_types: {'Ventilation naturelle', 'VMC SF auto et VMC double flux', 'VMC SF hydrogérable'}, default None
    infiltration: {'Minimal', 'Low', 'Medium', 'High'}, default None
    air_rate: pd.Series, default None
    unobserved: {'Minimal', 'High'}, default None
    climate: int, default None
        Climatic year to use to calculate heating need.
    smooth: bool, default False
        Use smooth daily data to calculate heating need.
    freq
    hourly_profile: optional, pd.Series
    temp_indoor: optional, default temp_indoor
    gain_utilization_factor: bool, default False
        If False, for simplification we use gain_utilization_factor = 1.

    Returns
    -------
    Conventional heating need (kWh/m2.a)
    """

    temp_ext = 7.1 # °C
    days_heating_season = 209
    solar_radiation = 306.4
    if temp_indoor is None:
        temp_indoor = 19


    surface_components = ratio_surface.copy()
    surface_components = {k:[v] for k,v in surface_components.items()}
    surface_components = pd.DataFrame(surface_components)

    df = pd.concat([u_wall, u_floor, u_roof, u_windows], axis=1, keys=['Wall', 'Floor', 'Roof', 'Windows'])
    surface_components.loc[0,'Floor'] *= 0.5
    surface_components = reindex_mi(surface_components, df.index)

    coefficient_transmission_transfer = (surface_components * df).sum(axis=1)
    coefficient_transmission_transfer += surface_components.sum(axis=1) * {'Minimal': 0,'Low': 0.05,'Medium': 0.1,'High': 0.15}[th_bridging]

    if air_rate is None:
        ventilation_dict = {'natural': 0.4,'Individual MV': 0.3,'Collective MV':0.3,'Individual DCV': 0.2,'Collective DCV':0.2,'Individual HRV':0.2,'Collective HRV':0.2}
        air_rate = ventilation_dict[vent_types] + {'minimal': 0.05,'low': 0.1,'medium': 0.2,'high': 0.5}[infiltration]

    coefficient_ventilation_transfer = 0.34 * air_rate * 2.5
    heat_transfer_coefficient = coefficient_ventilation_transfer + coefficient_transmission_transfer

    solar_coefficient = 0.6 * (1 - 0.3) * 0.9 * 0.62 * surface_components.loc[:, 'Windows']

    coefficient = 24 / 1000 * 0.9 * days_heating_season
    coefficient_climatic = coefficient * (temp_indoor - temp_ext)
    internal_heat_sources = 24 / 1000 * 4.17 * days_heating_season

    heat_transfer = heat_transfer_coefficient * coefficient_climatic
    solar_load = solar_coefficient * solar_radiation
    heat_gains = solar_load + internal_heat_sources

    if gain_utilization_factor is True:
        time_constant = 45 / (coefficient_transmission_transfer + coefficient_ventilation_transfer)
        a_h = 0.8 + time_constant / 30
        heat_balance_ratio = (internal_heat_sources + solar_load) / heat_transfer
        gain_utilization_factor = (1 - heat_balance_ratio ** a_h) / (1 - heat_balance_ratio ** (a_h + 1))
    else:
        gain_utilization_factor = 1

    heat_need = (heat_transfer - heat_gains * gain_utilization_factor) * 0.9

    return heat_need


def conventional_heating_final(u_wall, u_floor, u_roof, u_windows, ratio_surface, efficiency,
                               th_bridging='Medium', vent_types='Ventilation naturelle', infiltration='Medium',
                               air_rate=None, unobserved=None, climate=None, freq='year', smooth=False,
                               temp_indoor=None, gain_utilization_factor=False,
                               efficiency_hour=False, hourly_profile=None, temp_sink=None,
                               zcl_thermal_parameters=None):
    """Monthly stead-state space heating final energy delivered.


    Heat-pump formula come from Stafell et al., 2012.

    Parameters
    ----------
    u_wall: pd.Series
    u_floor: pd.Series
    u_roof: pd.Series
    u_windows: pd.Series
    ratio_surface: pd.Series
    efficiency: pd.Series
    th_bridging: {'Minimal', 'Low', 'Medium', 'High'}
    vent_types: {'Ventilation naturelle', 'VMC SF auto et VMC double flux', 'VMC SF hydrogérable'}
    infiltration: {'Minimal', 'Low', 'Medium', 'High'}
    air_rate: default None
    unobserved: {'Minimal', 'High'}, default None
    climate: int, default None
        Climatic year to use to calculate heating need.
    freq
    smooth: bool, default False
        Use smooth daily data to calculate heating need.
    temp_indoor
    gain_utilization_factor: bool, default False
        If False, for simplification we use gain_utilization_factor = 1.
    efficiency_hour

    Returns
    -------

    """
    heat_need = conventional_heating_need(u_wall, u_floor, u_roof, u_windows, ratio_surface,
                                          th_bridging=th_bridging, vent_types=vent_types,
                                          infiltration=infiltration, air_rate=air_rate, unobserved=unobserved,
                                          climate=climate, freq=freq, smooth=smooth,
                                          temp_indoor=temp_indoor, gain_utilization_factor=gain_utilization_factor,
                                          hourly_profile=hourly_profile,zcl_thermal_parameters=zcl_thermal_parameters)

    return heat_need / efficiency
        

def conventional_dhw_final(building_type, heating_system):
    """Calculate dhw final energy consumption.

    Parameters
    ----------
    index: pd.MultiIndex

    Returns
    -------

    """
    DHW_EFFICIENCY = {'Electricity-Direct electric': 0.95,
                      'Electricity-Wood stove': 0.95,
                      'Electricity-Heat pump air': 2.5, # previously at 0.95
                      'Electricity-Heat pump': 2.5,
                      'Electricity-Heat pump water': 2.5,
                      'Natural gas-Performance boiler': 0.6,
                      'Natural gas-Standard boiler': 0.6,
                      'Natural gas-Collective boiler': 0.6,
                      'Oil fuel-Performance boiler': 0.6,
                      'Oil fuel-Standard boiler': 0.6,
                      'Oil fuel-Collective boiler': 0.6,
                      'Wood fuel-Performance boiler': 0.6,
                      'Wood fuel-Standard boiler': 0.6,
                      'Heating-District heating': 0.6}
    
    DHW_NEED = {'SFH':15.3,'TH':15.3,'MFH':19.8,'AB':19.8}
    
    efficiency = DHW_EFFICIENCY.get(heating_system)
    dhw_need = DHW_NEED.get(building_type)
    return dhw_need / efficiency


def final2primary(heat_consumption, energy, conversion=2.3):
    if energy == 'Electricity':
        return heat_consumption * conversion
    else:
        return heat_consumption
            


def get_EPC(typology, heating_system='Electricity-Direct electric'):
    ratio_surface_dict = {'SFH':{'Wall':1.42,
                                 'Floor':0.75,
                                 'Roof':0.77,
                                 'Windows':0.17},
                          'MFH':{'Wall':0.78,
                                 'Floor':0.28,
                                 'Roof':0.29,
                                 'Windows':0.19}}
    ratio_surface_dict['TH'] = ratio_surface_dict['SFH']
    ratio_surface_dict['AB'] = ratio_surface_dict['MFH']
    
    dict_energy_efficiency = {'Electricity-Direct electric':0.95, 
                              'Electricity-Heat pump air':2,
                              'Electricity-Heat pump water':2.5,
                              'Heating-District heating':0.76,
                              'Natural gas-Performance boiler':0.76, 
                              'Oil fuel-Performance boiler':0.76,
                              'Wood fuel-Performance boiler':0.76,} 
    
    u_wall = pd.Series(data=[typology.tabula_Umur])
    u_floor = pd.Series(data=[typology.tabula_Upb])
    u_roof = pd.Series(data=[typology.tabula_Uph])
    u_windows = pd.Series(data=[typology.tabula_Uw])
    ratio_surface = ratio_surface_dict.get(typology.type)
    efficiency = dict_energy_efficiency.get(heating_system)
    vent_types = typology.ventilation
    infiltration = typology.air_infiltration
    
    heating_final = conventional_heating_final(u_wall, u_floor, u_roof, u_windows, ratio_surface, efficiency,
                                               th_bridging='Medium', vent_types=vent_types,
                                               infiltration=infiltration, air_rate=None, unobserved=None,
                                               zcl_thermal_parameters=None,)
    dhw_final = conventional_dhw_final(typology.type, heating_system)
    
    final_cons = heating_final + dhw_final
    
    primary_cons = final2primary(final_cons, heating_system.split('-')[0])
    other_consumptions = (CONSUMPTION_LIGHT + AUXILIARY_CONSUMPTION) * CONVERSION
    
    # print(primary_cons)
    certificate = find_certificate(primary_cons, heating_system, other_consumptions=other_consumptions)
    
    return certificate.iloc[0]



# =============================================================================
#                 
# =============================================================================
def get_complete_save_name(save_name, zcl, behaviour='conventionnel',period=[2020,2024],
                           model='explore2',nmod=0,natnocvent=True):
    
    if behaviour == 'conventionnel':
        behaviour = Behaviour('conventionnel_th-bce_2020')
        if natnocvent:
            behaviour.nocturnal_ventilation = True
            behaviour.update_name()
            
    if behaviour == 'conventionnel_noised':
        behaviour = Behaviour('conventionnel_th-bce_2020_gaussian_noise')
        if natnocvent:
            behaviour.nocturnal_ventilation = True
            behaviour.update_name()
            
    var_saver = save_name + '_bhv{}_zcl{}_nmod{}_start{}_end{}'.format(behaviour.full_name,zcl.code,nmod,period[0],period[1])
    return var_saver
    
    
def compute_energy_needs_segment(typo,zcl,output_path,save_name,
                                 behaviour='conventionnel',period=[2020,2024],
                                 model='explore2',nmod=0,natnocvent=True):
    
    city = zcl.center_prefecture
    
    # weather data
    if model == 'explore2':
        weather_data_checkfile = ".weather_data_{}_{}_{}_explore2_mod{}".format(city,period[0],period[1],nmod) + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_projected_weather_data(zcl.code, period,nmod=nmod)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            try:
                weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
            except pickle.UnpicklingError:
                weather_data = pickle.load(open(weather_data_checkfile, 'rb'))

    else:
        raise(NotImplementedError())
    
    
    if behaviour == 'conventionnel':
        behaviour = Behaviour('conventionnel_th-bce_2020')
        if natnocvent:
            behaviour.nocturnal_ventilation = True
            behaviour.update_name()
            
    if behaviour == 'conventionnel_noised':
        behaviour = Behaviour('conventionnel_th-bce_2020_gaussian_noise')
        if natnocvent:
            behaviour.nocturnal_ventilation = True
            behaviour.update_name()
    
    # var_saver = 'segment_{}_lvl-{}_{}_{}-{}_mod{}_{}'.format(typo.code, typo.insulation_level, zcl.code, period[0],period[1],nmod,behaviour.full_name)
    # var_saver = 'segment_start{}_end{}_typo{}__{}-{}_mod{}_{}'.format(period[0],period[1], typo.code, typo.insulation_level, zcl.code, ,nmod,behaviour.full_name)
    var_saver = save_name + '_bhv{}_zcl{}_nmod{}_start{}_end{}'.format(behaviour.full_name,zcl.code,nmod,period[0],period[1])
    
    # print(var_saver)
    if '{}.pickle'.format(var_saver) not in os.listdir(output_path):
        
        simulation = run_thermal_model(typo, behaviour, weather_data, pmax_warning=False)
        
        # TODO: peut-être sauvegarder des choses plus légères
        simulation = aggregate_resolution(simulation, resolution='h')
        
        pickle.dump(simulation, open(os.path.join(os.path.join(output_path),'{}.pickle'.format(var_saver)), "wb"))
    
    simulation = pickle.load(open(os.path.join(os.path.join(output_path),'{}.pickle'.format(var_saver)), 'rb'))
    # return 
    return simulation


def plot_weekly_consumption(data,figs_folder):
    data['weekday'] = data.index.dayofweek
    data['hour'] = data.index.hour
    
    weekdays = [x for xs in [[i]*24 for i in range(7)] for x in xs]
    hours = list(range(24))*7
    
    seasons = ['DJF','JJA']
    season_dict = {'JJA':[6,7,8],
                   'DJF':[12,1,2],
                   'MAM':[3,4,5],
                   'SON':[9,10,11]}
    
    dayofweek_dict = {0:'Monday',
                      1:'Tuesday',
                      2:'Wednesday',
                      3:'Thursday',
                      4:'Friday',
                      5:'Saturday',
                      6:'Sunday'}
    
    for season in seasons:
        weekly_cons = pd.DataFrame().from_dict({'weekday':weekdays,'hour':hours}).set_index(['weekday','hour'])
        
        mean_weekly = data[data.index.month.isin(season_dict.get(season))][['weekday','hour','total_needs']].groupby(by=['weekday','hour']).mean()
        std_weekly = data[data.index.month.isin(season_dict.get(season))][['weekday','hour','total_needs']].groupby(by=['weekday','hour']).std()
        
        mean_weekly = mean_weekly.rename(columns={'total_needs':'total_needs_mean'})
        std_weekly = std_weekly.rename(columns={'total_needs':'total_needs_std'})
        
        weekly_cons = weekly_cons.join(mean_weekly)
        weekly_cons = weekly_cons.join(std_weekly)
        weekly_cons['weekday_hour'] = [hour + 24*dow for dow,hour in weekly_cons.index]
        
        
        fig,ax = plt.subplots(figsize=(15,5),dpi=300)
        ax.plot(weekly_cons.weekday_hour, weekly_cons['total_needs_mean'],
                label=season)
        ax.fill_between(weekly_cons.weekday_hour, 
                        weekly_cons['total_needs_mean']+weekly_cons['total_needs_std'],
                        weekly_cons['total_needs_mean']-weekly_cons['total_needs_std'],
                        alpha=0.2)
        ylims = ax.get_ylim()
        # ylims = (0.,cons[[e for e in cons.columns if 'par_point_soutirage' in e]].max().max())
        # ylims = (0.,600)
        for e in range(1,7):
            ax.plot([e*24]*2,ylims,color='k',ls=':',zorder=-1)
        ax.set_ylim(ylims)
        xticks = list(range(0,weekly_cons['weekday_hour'].max()+6,6))
        ax.set_xlim([0,24*7])
        ax.set_xticks(xticks)
        ax.set_xticklabels([e%24 if e%24!=12 else '12\n{}'.format(dayofweek_dict.get(e//24)) for e in xticks])
        ax.set_ylabel('Mean hourly consumption by connection point (Wh)')
        plt.legend()
        # plt.savefig(os.path.join(figs_folder,'{}.png'.format('hourly_consumption_over_week_regions_{}_season_{}'.format('-'.join(map(str, regions)),season))), bbox_inches='tight')
        plt.show()
    return 


def plot_daily_consumption(data,output_path,normalize=True):
    data['hour'] = data.index.hour
    hours = list(range(24))
    
    seasons = ['DJF','JJA']
    season_dict = {'JJA':list(range(1,13)),
                   'DJF':list(range(1,13)),
                   'MAM':list(range(1,13)),
                   'SON':list(range(1,13))}
    colors = {'DJF':'tab:red','JJA':'tab:blue'}
    labels = {'DJF':'Heating','JJA':'Cooling'}
    
    daily_save = None
    
    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
    for season in seasons:
        col = {'DJF':'heating','JJA':'cooling'}.get(season)
        weekly_cons = pd.DataFrame().from_dict({'hour':hours})
        
        if normalize:
            data[col] = data[col]/data[col].mean()
            
        mean_weekly = data[data.index.month.isin(season_dict.get(season))][['hour',col]].groupby(by=['hour']).mean()
        std_weekly = data[data.index.month.isin(season_dict.get(season))][['hour',col]].groupby(by=['hour']).std()
        
        mean_weekly = mean_weekly.rename(columns={col:'total_needs_mean'})
        std_weekly = std_weekly.rename(columns={col:'total_needs_std'})
        
        weekly_cons = weekly_cons.join(mean_weekly)
        weekly_cons = weekly_cons.join(std_weekly)
        
        upper_bound = weekly_cons['total_needs_mean']+weekly_cons['total_needs_std']
        lower_bound = weekly_cons['total_needs_mean']-weekly_cons['total_needs_std']
        lower_bound = lower_bound.clip(lower=0.)
        
        if daily_save is None:
            daily_save = weekly_cons.rename(columns={'total_needs_mean':'{}_mean'.format(col),'total_needs_std':'{}_std'.format(col)}).set_index('hour')
        else:
            daily_save = daily_save.join(weekly_cons.rename(columns={'total_needs_mean':'{}_mean'.format(col),'total_needs_std':'{}_std'.format(col)}).set_index('hour'))
        
        ax.plot(weekly_cons.hour, weekly_cons['total_needs_mean'],
                label=labels.get(season),color=colors.get(season))
        ax.fill_between(weekly_cons.hour, upper_bound,lower_bound,
                        alpha=0.2,color=colors.get(season))
    ylims = ax.get_ylim()
    if normalize:
        ax.set_ylim([0,4])
    else:
        ax.set_ylim(ylims)
    ax.set_xlim([0,24])
    ax.set_ylabel('Mean hourly consumption by connection point (Wh)')
    plt.legend()
    # plt.savefig(os.path.join(figs_folder,'{}.png'.format('hourly_consumption_over_week_regions_{}_season_{}'.format('-'.join(map(str, regions)),season))), bbox_inches='tight')
    plt.show()
    
    daily_save.to_csv(os.path.join(output_path,'aggregate_elec_test.csv'))
    return 


def get_energy_prices():
    # en €/kWh
    ht_prices = pd.read_csv(os.path.join('data','Res-IRF','energy_prices_wt_ame2021.csv'))
    ht_prices = ht_prices.rename(columns={'Unnamed: 0':'year'}).set_index('year')
    
    taxes = pd.read_csv(os.path.join('data','Res-IRF','energy_taxes_ame2021.csv'))
    taxes = taxes.rename(columns={'Unnamed: 0':'year'}).set_index('year')
    
    htva_prices = ht_prices + taxes
    vat_dict = pd.read_csv(os.path.join('data','Res-IRF','energy_vat.csv')).set_index('energy')['vat'].to_dict()
    
    ttc_prices = htva_prices.copy()
    for energy in vat_dict.keys():
        ttc_prices[energy] = ttc_prices[energy]*(1+vat_dict.get(energy))
        
    # verification Res-IRF referecne rate
    if False:
        elec_rate = 0.0135
        start_year = 2019 # bizarre mais ok
        
        res_irf_years = np.asarray(list(range(start_year,2051)))
        res_irf_elec_price = np.asarray([ht_prices.Electricity.values[0]]*len(res_irf_years))
        for idx in range(1,len(res_irf_years)):
            res_irf_elec_price[idx] = res_irf_elec_price[idx-1]*(1+elec_rate)
            
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(ht_prices.index,ht_prices.Electricity,label='AME 2021')
        ax.plot(res_irf_years,res_irf_elec_price,label='Res-IRF Reference')
        ax.legend()
        ax.set_ylim(bottom=0.)
        plt.show()
        
    return ttc_prices


def get_energy_efficiency():
    # TODO: vérifier les coefficients pour les PAC
    # cf méthode 3CL-DPE
    dict_energy_efficiency = {'Electricity-Direct electric':0.95, 
                              'Electricity-Heat pump air':2,
                              'Electricity-Heat pump water':2.5,
                              'Electricity-Portable unit':1.,
                              'Heating-District heating':0.76,
                              'Natural gas-Performance boiler':0.76, 
                              'Oil fuel-Performance boiler':0.76,
                              'Wood fuel-Performance boiler':0.76,
                              'Electricity-No AC':0.} 
    
    return dict_energy_efficiency


def get_nb_resprinc_zcl():
    dict_ratio_res_princ_per_zcl = {'H1a':30/100, 
                                    'H1b':11/100, 
                                    'H1c':16/100, 
                                    'H2a':6/100, 
                                    'H2b':11/100, 
                                    'H2c':10/100, # correction pour rester à 100%
                                    'H2d':3/100, 
                                    'H3':13/100}
    return dict_ratio_res_princ_per_zcl


def compute_consumption(needs,typology,heating,cooling,year,zcl):
    return 


def aggregate_energy_consumption(stock,year,zcl,output_path,zeta=5):
    # prix de l'énergie tq Reference Res-IRF
    energy_prices = get_energy_prices()
    
    # efficacité des systèmes tq Res-IRF
    dict_energy_efficiency = get_energy_efficiency()
    
    consumption_heating = None
    consumption_cooling = None
    temperature = None
    
    for idx in tqdm.tqdm(range(len(stock)),desc=zcl):
        _,_,_,_,_,hs,cs,num,typo,save = stock.iloc[idx].values
        
        if cs == 'No AC':
            cs = 'Electricity-No AC'
            
        heating_energy_vector = hs.split('-')[0]
        cooling_energy_vector = cs.split('-')[0]
        
        period = [2018,2018]
        behaviour = 'conventionnel_noised'
        model='explore2'
        nmod=0
        complete_save = get_complete_save_name(save, Climat(zcl),behaviour=behaviour,period=period,model=model,nmod=nmod)

        temp = pickle.load(open(os.path.join(os.path.join(output_path),'{}.pickle'.format(complete_save)), 'rb'))
        temp = temp[['cooling_needs','heating_needs','temperature_2m']]
        temp['cooling_needs'] = temp['cooling_needs']/typo.households
        temp['heating_needs'] = temp['heating_needs']/typo.households
        temp = temp[temp.index.year==year]
        
        # passage de besoin à consommation conventionnelle
        temp['heating_consumption_standard'] = temp['heating_needs']*dict_energy_efficiency.get(hs)
        temp['cooling_consumption_standard'] = temp['cooling_needs']*dict_energy_efficiency.get(cs)
        
        # passage de conventionnel à réel 
        energy_price_heating_vector = energy_prices[heating_energy_vector].to_dict().get(year)
        energy_price_cooling_vector = energy_prices[cooling_energy_vector].to_dict().get(year)
        energy_bill = energy_price_heating_vector*temp['heating_consumption_standard'].sum()*1e-3 + energy_price_cooling_vector*temp['cooling_consumption_standard'].sum()*1e-3
        use_intensity = (energy_bill)**(-1/zeta)
        use_intensity = 1
        temp['heating_consumption'] = temp['heating_consumption_standard']*use_intensity
        temp['cooling_consumption'] = temp['cooling_consumption_standard']*use_intensity
        
        if consumption_heating is None:
            consumption_heating = temp[['heating_consumption']].rename(columns={'heating_consumption':idx})
        else:
            consumption_heating = pd.concat([consumption_heating,temp[['heating_consumption']].rename(columns={'heating_consumption':idx})],axis=1)
            
        if consumption_cooling is None:
            consumption_cooling = temp[['cooling_consumption']].rename(columns={'cooling_consumption':idx})
        else:
            consumption_cooling = pd.concat([consumption_cooling,temp[['cooling_consumption']].rename(columns={'cooling_consumption':idx})],axis=1)
        
        if temperature is None:
            temperature = temp[['temperature_2m']]
    return consumption_heating, consumption_cooling, stock, temperature

        
        


#%% ===========================================================================
# main script
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    
    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_resirf'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
    
    
    #%% Détermination des étiquettes énergétiques des typologies 
    if True:
        heating_system_list = ['Electricity-Direct electric', 
                               'Electricity-Heat pump air',
                               'Electricity-Heat pump water',
                               'Heating-District heating',
                               'Natural gas-Performance boiler', 
                               'Oil fuel-Performance boiler',
                               'Wood fuel-Performance boiler']
        
        letter2int= {chr(a):i+1 for i,a in enumerate(range(65, 91))}
        int2letter= {i+1:chr(a) for i,a in enumerate(range(65, 91))}
        
        dict_epc_typologies = {}
        for building_type in ['SFH','TH','MFH','AB']:
            for i in range(1,11):
                for level in ['initial']:##,'standard','advanced']:
                
                    code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
                    typo = Typology(code,level)
                    
                    dict_epc_typologies[typo.code] = []
                    
                    for hs in heating_system_list:
                        epc = letter2int.get(get_EPC(typo,hs))
                        dict_epc_typologies[typo.code].append(epc)
        
        for typ,epc in dict_epc_typologies.items():
            mean = np.asarray(epc).mean()
            print(typ, [int2letter.get(int(np.floor(mean))),int2letter.get(int(np.ceil(mean)))])
                
    #%% Premier essai sur le stock initial
    if False:
        stock = pd.read_csv(os.path.join('data','Res-IRF','buildingstock_sdes2018_update_hpdiff_ac_reduced.csv'))
        stock_init = stock.copy()
        mf_stock = stock[stock['Housing type']=='Multi-family']
        sf_stock = stock[stock['Housing type']=='Single-family']
        
        multi_family_typo_code = 'FR.N.AB.03.Gen'
        multi_family_typo_init = Typology(multi_family_typo_code)
        multi_family_typo = Typology(multi_family_typo_code)
        
        single_family_typo_code = 'FR.N.SFH.01.Gen'
        single_family_typo_init = Typology(single_family_typo_code)
        single_family_typo = Typology(single_family_typo_code)
        
        
        # graphes des évolution de la valeur U en fonction de l'épaisseur d'isolant
        if False:
            cmap = plt.get_cmap('viridis')
            colors = {'wall':cmap(0.4),
                      'roof':cmap(0.8),
                      'floor':cmap(0.1)}
            
            for title,typo_init,typo,sto in [('Multi-family',multi_family_typo_init,multi_family_typo,mf_stock),('Single-family',single_family_typo_init,single_family_typo,sf_stock)]:
                print(title)
                resirf_wall_values = [2.5,1.0,0.5,0.25,0.1]
                wall_insulation_supp_list = np.linspace(0,0.25,100)
                wall_uvalue_list = [np.nan]*len(wall_insulation_supp_list)
                for idx, l_insu_wall in enumerate(wall_insulation_supp_list):
                    typo.w0_insulation_thickness = typo_init.w0_insulation_thickness + l_insu_wall
                    typo.w1_insulation_thickness = typo_init.w1_insulation_thickness + l_insu_wall
                    typo.w2_insulation_thickness = typo_init.w2_insulation_thickness + l_insu_wall
                    typo.w3_insulation_thickness = typo_init.w3_insulation_thickness + l_insu_wall
                    
                    compute_U_values(typo)
                    wall_uvalue_list[idx] = typo.modelled_Umur
                
                wall_df = pd.DataFrame(index=resirf_wall_values)
                wall_insulation = pd.DataFrame({'uvalue':wall_uvalue_list,'supp_thickness':wall_insulation_supp_list}).set_index('uvalue')
                wall_df = wall_df.join(wall_insulation,how='outer')
                wall_df = wall_df.interpolate()
                
                print('Wall',wall_df.loc[resirf_wall_values].to_dict())
                
                # resirf_roof_values = sorted(sto.Roof.value_counts().index.to_list(),reverse=True)
                # roof_insulation_supp_list = np.linspace(0,0.25,100)
                # roof_uvalue_list = [np.nan]*len(roof_insulation_supp_list)
                # for idx, l_insu in enumerate(roof_insulation_supp_list):
                #     typo.ceiling_U = l_insu
                    
                #     compute_U_values(typo)
                #     roof_uvalue_list[idx] = typo.modelled_Uph
                
                # roof_df = pd.DataFrame(index=resirf_roof_values)
                # roof_insulation = pd.DataFrame({'uvalue':roof_uvalue_list,'supp_thickness':roof_insulation_supp_list}).set_index('uvalue')
                # roof_df = roof_df.join(roof_insulation,how='outer')
                # roof_df = roof_df.interpolate()
                
                resirf_floor_values = [2.0, 1.5, 0.5,0.3, 0.25]
                floor_insulation_supp_list = np.linspace(0.,0.25,50)
                floor_uvalue_list = [np.nan]*len(floor_insulation_supp_list)
                for idx, l_insu in enumerate(floor_insulation_supp_list):
                    typo.floor_insulation_thickness = typo_init.floor_insulation_thickness + l_insu
                    
                    compute_U_values(typo)
                    floor_uvalue_list[idx] = typo.modelled_Upb
                
                floor_df = pd.DataFrame(index=resirf_floor_values)
                floor_insulation = pd.DataFrame({'uvalue':floor_uvalue_list,'supp_thickness':floor_insulation_supp_list}).set_index('uvalue')
                floor_df = floor_df.join(floor_insulation,how='outer')
                floor_df = floor_df.interpolate()
                
                print('Floor',floor_df.loc[resirf_floor_values].to_dict())
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                ax.plot(wall_df.supp_thickness,wall_df.index,color=colors.get('wall'),label='Wall')
                ax.plot(floor_df.supp_thickness,floor_df.index,color=colors.get('floor'),label='Floor')
                # ax.plot(roof_df.supp_thickness,roof_df.index,color=colors.get('roof'),label='Roof')
                ax.legend()
                ax.set_ylim(bottom=0.)
                xlims = ax.get_xlim()
                
                for v in resirf_wall_values:
                    wall_val = wall_df.loc[v]
                    ax.plot(wall_val.values[0],v,marker='o',mec=colors.get('wall'),mfc='w')
                    ax.plot([wall_val.values[0]]*2,[0,v],ls=':',color=colors.get('wall'),)
                    
                # for v in resirf_roof_values:
                #     roof_val = roof_df.loc[v]
                #     ax.plot(roof_val.values[0],v,marker='o',mec=colors.get('roof'),mfc='w')
                #     ax.plot([roof_val.values[0]]*2,[0,v],ls=':',color=colors.get('roof'),)
                
                for v in resirf_floor_values:
                    floor_val = floor_df.loc[v]
                    ax.plot(floor_val.values[0],v,marker='o',mec=colors.get('floor'),mfc='w')
                    ax.plot([floor_val.values[0]]*2,[0,v],ls=':',color=colors.get('floor'),)
                
                ax.set_xlabel('Additional insulation thickness (m)')
                ax.set_ylabel('U-value (W.m$^{-2}$.K$^{-1}$)')
                ax.set_title(title)
                plt.savefig('{}_additionnal_uvalues.png'.format(title),bbox_inches='tight')
                plt.show()
        
        supp_insulation_layer_dict = {'Single-family':{'wall':{2.5: 0.0, 
                                                               1.0: 0.02904040404040404, 
                                                               0.5: 0.08459595959595961, 
                                                               0.25: 0.19570707070707072},
                                                       'floor':{2.0: 0.0, 
                                                                1.5: 0.0, 
                                                                0.5: 0.06377551020408162, 
                                                                0.25: 0.1760204081632653},
                                                       },
                                      'Multi-family':{'wall':{2.5: 0.003787878787878788, 
                                                              1.0: 0.036616161616161616, 
                                                              0.5: 0.09217171717171718, 
                                                              0.25: 0.20580808080808083},
                                                      'floor':{2.0: 0.0, 
                                                               1.5: 0.002551020408163265, 
                                                               0.5: 0.07397959183673469, 
                                                               0.25: 0.1913265306122449},
                                                      },
                                      }
        
        # period = [2020,2024]
        period = [2018,2018]
        # behaviour = 'conventionnel'
        behaviour = 'conventionnel_noised'
        
        # ajout des typo dans le stock
        typo_list = [None]*len(stock)
        typo_saver_list = [None]*len(stock)
        for idx in range(len(stock)):
            ht,wall,floor,roof,windows,hs,cs,_ = stock.iloc[idx].values
            
            # type de batiment
            if ht == 'Single-family':
                typo = Typology(single_family_typo_code)
            else:
                typo = Typology(multi_family_typo_code)
            
            # ajustement des valeurs U
            typo.ceiling_U = roof
            typo.windows_U = windows
            typo.w0_insulation_thickness = typo.w0_insulation_thickness + supp_insulation_layer_dict.get(ht).get('wall').get(wall)
            typo.w1_insulation_thickness = typo.w1_insulation_thickness + supp_insulation_layer_dict.get(ht).get('wall').get(wall)
            typo.w2_insulation_thickness = typo.w2_insulation_thickness + supp_insulation_layer_dict.get(ht).get('wall').get(wall)
            typo.w3_insulation_thickness = typo.w3_insulation_thickness + supp_insulation_layer_dict.get(ht).get('wall').get(wall)
            typo.floor_insulation_thickness = typo.floor_insulation_thickness + supp_insulation_layer_dict.get(ht).get('floor').get(floor)
            
            # puissance cooling à 0 si pas de clim (#TODO: vérfier pmax des clim fixes et portables)
            cooling = True
            if cs == 'No AC':
                typo.cooler_maximum_power = 0.
                cooling = False
            else:
                typo.cooler_maximum_power =  100*typo.surface
            typo.heater_maximum_power = 100*typo.surface
                
            typo_list[idx] = typo 
            typo_saver_list[idx] = 'segment_{}_heating{}_cooling{}_umur{}_floor{}_roof{}_windows{}'.format(typo.code,True,cooling,wall,floor,roof,windows).replace('.','')
            
        stock['typology'] = typo_list
        stock['save_name'] = typo_saver_list
        
        # print(stock.save_name.values[10])
        
        zcl_list = France().climats 
        
        run_list = []
        for zcl_code in zcl_list:
            zcl = Climat(zcl_code)
            for typo,seg_name in zip(stock.typology, stock.save_name):
                run_list.append((typo,
                                 zcl,
                                 os.path.join(output, folder),
                                 seg_name,
                                 behaviour,
                                 period))
        
        # print(run_list, len(run_list))
        
        # one thread run
        if False:
            for run in tqdm.tqdm(run_list):
                _ = compute_energy_needs_segment(*run)
                
        # parralel run
        if False:
            nb_cpu = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(nb_cpu)
            pool.starmap(compute_energy_needs_segment, run_list)
        
        test = compute_energy_needs_segment(*run_list[2*110])
        # test['heating_needs'] = test.heating_needs.rolling(3).mean()
        test['total_needs'] = test.heating_needs + test.cooling_needs
        

        # comparaison avec les besoins de chauffage annuel res_irf
        if False:
            stock_init = stock_init.set_index(['Housing type', 'Wall', 'Floor', 'Roof', 'Windows', 'Heating system','Cooling system'])
            
            idx = stock_init.index
            wall = pd.Series(idx.get_level_values('Wall'), index=idx)
            floor = pd.Series(idx.get_level_values('Floor'), index=idx)
            roof = pd.Series(idx.get_level_values('Roof'), index=idx)
            windows = pd.Series(idx.get_level_values('Windows'), index=idx)
            
            heat_needs = conventional_heating_need(wall, floor, roof, windows)
            stock['resirf_heating_needs_per_surface'] = heat_needs.values
    
            year_period = [2018,2018]
            year_range = 1 #year_period[1]-year_period[0]
            
            zcl = Climat('H1a')
            run_list = []
            for typo,seg_name in zip(stock.typology, stock.save_name):
                run_list.append((typo,
                                 zcl,
                                 os.path.join(output, folder),
                                 seg_name,
                                 behaviour,
                                 year_period))
                    
            heating_needs_list = [0]*len(stock)
            for idx in tqdm.tqdm(range(len(run_list))):
                annual_heating_needs = compute_energy_needs_segment(*run_list[idx]).heating_needs.sum()/year_range
                heating_needs_list[idx] = annual_heating_needs
            stock['heating_needs'] = heating_needs_list
            
            stock['heating_needs'] = stock['heating_needs']/np.asarray([e.households for e in stock['typology']])
            stock['heating_needs_per_surface'] = stock['heating_needs']/np.asarray([e.surface/e.households for e in stock['typology']])*1e-3
            
            if False:
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                sns.histplot(stock,x='heating_needs_per_surface',binrange=[0,400],binwidth=50)
                sns.histplot(stock,x='resirf_heating_needs_per_surface',binrange=[0,400],binwidth=50)
                plt.show()
                
            if False:
                stock['gap'] = (stock['resirf_heating_needs_per_surface']-stock['heating_needs_per_surface'])
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                sns.histplot(stock,x='gap',hue='Housing type')
                plt.show()
                    
            pass
            
            
        
        # affichage des times series
        if False:
            fig,ax = plt.subplots(figsize=(10,5),dpi=300)
            test[(test.index.year==2018)&(test.index.month==2)&(test.index.day.isin(list(range(1,14))))][['heating_needs','cooling_needs']].plot(ax=ax)
            plt.show()
            
            fig,ax = plt.subplots(figsize=(10,5),dpi=300)
            test[(test.index.year==2018)&(test.index.month==2)&(test.index.day.isin(list(range(1,14))))][['temperature_2m','internal_temperature']].plot(ax=ax)
            plt.show()
            
            fig,ax = plt.subplots(figsize=(10,5),dpi=300)
            test[(test.index.year==2018)&(test.index.month==8)&(test.index.day.isin(list(range(1,14))))][['temperature_2m','internal_temperature']].plot(ax=ax)
            plt.show()
            
            fig,ax = plt.subplots(figsize=(10,5),dpi=300)
            test[(test.index.year==2018)&(test.index.month==8)&(test.index.day.isin(list(range(1,14))))][['heating_needs','cooling_needs']].plot(ax=ax)
            plt.show()
        
        # test : affichage des graphes de thermosensibilité
        if False:
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.scatterplot(data=test,x='temperature_2m',y='total_needs',ax=ax)
            plt.show()
            
        
        # nouveau test d'aggrégation
        if True:
            # thermosensibilité par zcl 
            if False:
                zcl = 'H3'
                year = 2018
                
                consumption_heating, consumption_cooling, stock, temperature = aggregate_energy_consumption(stock,year,zcl,output_path=os.path.join(output, folder))
                elec_idx = stock[stock['Heating system'].str.startswith('Electricity')].index
                
                for i,idx in enumerate(elec_idx):
                    if i == 0:
                        modelled = consumption_heating[[idx]].rename(columns={idx:'heating'})*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl)
                        modelled['cooling'] = (consumption_cooling[[idx]]*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl))[idx]
                    else:
                        modelled['heating'] = modelled['heating'] + (consumption_heating[[idx]]*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl))[idx]
                        modelled['cooling'] = modelled['cooling'] + (consumption_cooling[[idx]]*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl))[idx]
                
                modelled['total_consumption'] = modelled['heating'] + modelled['cooling']
                modelled_per_housing = modelled/(stock.loc[elec_idx,'0'].sum()*get_nb_resprinc_zcl().get(zcl))
                modelled_per_housing['temperature'] = temperature.temperature_2m
                
                modelled_per_housing = modelled_per_housing.dropna(axis=0)
                modelled_per_housing_sorted = modelled_per_housing.copy().sort_values(by='temperature')
                
                C0_france = 0 #165.28 #Wh cf thermal_sensitivity.py
                
                x = modelled_per_housing_sorted.temperature.values
                y = modelled_per_housing_sorted['total_consumption'].values + C0_france
                
                plot_thermal_sensitivity(temperature=x,consumption=y,figs_folder=figs_folder,k_init=50,
                                         reg_code=zcl,reg_name=zcl,year='2018',set_ylim=None)
            
            # analyse nationale
            if False:
                year = 2018
                zcl_list = France().climats
                
                modelled = None
                for zcl in zcl_list:
                    consumption_heating, consumption_cooling, stock, temperature = aggregate_energy_consumption(stock,year,zcl,output_path=os.path.join(output, folder))
                    elec_idx = stock[stock['Heating system'].str.startswith('Electricity')].index
                    
                    for idx in elec_idx:
                        if modelled is None:
                            modelled = consumption_heating[[idx]].rename(columns={idx:'heating'})*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl)
                            modelled['cooling'] = (consumption_cooling[[idx]]*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl))[idx]
                        else:
                            modelled['heating'] = modelled['heating'] + (consumption_heating[[idx]]*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl))[idx]
                            modelled['cooling'] = modelled['cooling'] + (consumption_cooling[[idx]]*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl))[idx]
                
                modelled['total_consumption'] = modelled['heating'] + modelled['cooling']
                
                # thermosensibilité
                if True:
                    modelled_per_housing = modelled/stock.loc[elec_idx,'0'].sum()
                    modelled_per_housing['temperature'] = get_nationale_meteo([year]*2).temperature
                    
                    modelled_per_housing = modelled_per_housing.dropna(axis=0)
                    modelled_per_housing_sorted = modelled_per_housing.copy().sort_values(by='temperature')
                    
                    C0_france = 0 #165.28 #Wh cf thermal_sensitivity.py
                    
                    x = modelled_per_housing_sorted.temperature.values
                    y = modelled_per_housing_sorted['total_consumption'].values + C0_france
                    
                    plot_thermal_sensitivity(temperature=x,consumption=y,figs_folder=figs_folder,k_init=50,
                                             reg_code='FRA',reg_name='France',year='2018',set_ylim=None)
                    
                # graphe des consommations journalières
                if False:
                    plot_daily_consumption(modelled, output_path=os.path.join(output, folder),normalize=False)
                    plot_daily_consumption(modelled, output_path=os.path.join(output, folder),normalize=True)
                    
                # timeseries
                if True:
                    fig,ax = plt.subplots(figsize=(10,5),dpi=300)
                    modelled[['heating','cooling']].plot(ax=ax)
                    plt.show()
            
            # somme des consommations de chauffage par énergie (national) pour comparaison CEREN
            if True:
                year = 2018
                
                sdes_data_2018 = {'Electricity':48.9,
                                  'Natural gas':123.3+3.2,
                                  'Oil fuel':39.2,
                                  'Heating':11.4,
                                  'Wood fuel':73.9}
                sdes_data_2018 = {k:v*0.8 for k,v in sdes_data_2018.items()}
                
                zcl_list = France().climats
                energy_vector_list = list(set([e.split('-')[0] for e in get_energy_efficiency().keys()]))
                
                res_dict = {v:0 for v in energy_vector_list}
                
                for zcl in zcl_list:
                    modelled_dict = {v:None for v in energy_vector_list}
                    consumption_heating, consumption_cooling, stock, _ = aggregate_energy_consumption(stock,year,zcl,output_path=os.path.join(output, folder))
                    
                    for vector in energy_vector_list:
                        elec_idx = stock[stock['Heating system'].str.startswith(vector)].index
                        
                        modelled = modelled_dict.get(vector)
                        for idx in elec_idx:
                            if modelled is None:
                                modelled = consumption_heating[[idx]].rename(columns={idx:'heating'})*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl)
                                modelled['cooling'] = (consumption_cooling[[idx]]*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl))[idx]
                            else:
                                modelled['heating'] = modelled['heating'] + (consumption_heating[[idx]]*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl))[idx]
                                modelled['cooling'] = modelled['cooling'] + (consumption_cooling[[idx]]*stock['0'].loc[idx]*get_nb_resprinc_zcl().get(zcl))[idx]
                        modelled_dict[vector] = modelled
                        
                        res_dict[vector] += modelled.heating.sum()
                
                print('Consommations en TWh')
                for k,v in res_dict.items():
                    print('    {}: {:.1f} TWh'.format(k,v*1e-12))
                    print('ref {}: {:.1f} TWh'.format(k,sdes_data_2018.get(k)))
                print('sum : {:.1f} TWh (ref {:.1f} TWh)'.format(sum(list(res_dict.values()))*1e-12,sum(list(sdes_data_2018.values()))))
                    
                    
                    
                
                


                        
            
            
            
        # aggregation des consommations
        if False:
            # graphe des consos hebdo
            if False:
                agg_stock_consumption = aggregate_energy_consumption(stock, output_path=os.path.join(output, folder),behaviour=behaviour)
                
                print('Consommations annuelles...')
                print(agg_stock_consumption.sum()*1e-12,'TWh')
                
                # print('Thermosensibilité du stock global...')
                
                
                # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                # sns.scatterplot(data=agg_stock_consumption,x='france',y='total_needs',ax=ax)
                # plt.show()
                # plot_weekly_consumption(agg_stock_consumption_electricity, figs_folder=figs_folder)
                
            # graphe des consommations journalières (comparaison avec Staffell etc)
            if False:
                agg_stock_consumption_electricity = aggregate_energy_consumption(stock, output_path=os.path.join(output, folder),energy_filter='Electricity',behaviour=behaviour)
                nb_log_elec_heating = stock[stock['Heating system'].str.startswith('Electricity')]['0'].sum()
                
                print('Consommations annuelles électriques...')
                
                print(agg_stock_consumption_electricity.sum()*1e-12,'TWh')
                # plot_daily_consumption(agg_stock_consumption_electricity, output_path=os.path.join(output, folder),normalize=True)
                
                print('Thermosensibilité du stock global électrique...')
                # agg_stock_consumption_electricity['total_needs'] = agg_stock_consumption_electricity.heating_needs + agg_stock_consumption_electricity.cooling_needs
                
                # meteo_data = get_nationale_meteo([2020,2024])
                # agg_stock_consumption_electricity = agg_stock_consumption_electricity.join(meteo_data)
                
                agg_stock_per_housing = agg_stock_consumption_electricity.copy()
                agg_stock_per_housing['total_consumption'] = agg_stock_per_housing['total_consumption']/nb_log_elec_heating
                agg_stock_per_housing = agg_stock_per_housing[agg_stock_per_housing.index.year.isin(list(range(2022,2024+1)))]
                
                print(agg_stock_per_housing)
                
                agg_stock_per_housing = agg_stock_per_housing.dropna(axis=0)#[:20000]
                agg_stock_per_housing_sorted = agg_stock_per_housing.copy().sort_values(by='temperature_2m')
                
                C0_france = 165.28 #Wh cf thermal_sensitivity.py
                
                x = agg_stock_per_housing_sorted.temperature_2m.values
                y = agg_stock_per_housing_sorted['total_consumption'].values + C0_france
                
                plot_thermal_sensitivity(temperature=x,consumption=y,figs_folder=figs_folder,k_init=50,
                                         reg_code='france',reg_name='France',year='2022-2024',set_ylim=None)
                
    
        # test de recuperation des profils journaliers 
        if False:
            test_agg = pd.read_csv(os.path.join(output, folder,'aggregate_elec_test.csv')).set_index('hour')
            ninja = pd.read_excel("data/Ninja/41560_2023_1341_MOESM9_ESM_doubled.xlsx",sheet_name='Figure ED3').set_index('Hour')
            moreau = pd.read_csv('data/Res-IRF/hourly_profile_moreau_doubled.csv').set_index('hour')
            moreau['heating'] = moreau['value']/moreau['value'].mean()
            moreau = moreau[['heating']]
            
            cyclic = pd.concat([test_agg, test_agg, test_agg], ignore_index=False)
            
            test_agg = test_agg[['heating_needs_mean', 'cooling_needs_mean']]
            # test_agg['heating_needs_gaussian'] = gaussian_filter(test_agg.heating_needs_mean,sigma=5.3,mode='wrap')
            # test_agg['cooling_needs_gaussian'] = gaussian_filter(test_agg.cooling_needs_mean,sigma=5.3,mode='wrap')
            # test_agg['heating_needs_rolling'] = cyclic.heating_needs_mean.rolling(18,center=True).mean().values[24:24+24]
            # test_agg['cooling_needs_gaussian'] = gaussian_filter(test_agg.cooling_needs_mean,sigma=5.3,mode='wrap')
            
            
            
            ninja = ninja[['Cooling (mean)', 'Heating (mean)']]
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            test_agg.plot(ax=ax)
            ninja.plot(ax=ax)
            moreau.plot(ax=ax)
            ax.set_ylim([0,2])
            ax.set_xlim([0,23])
            plt.show()
    
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__=='__main__':
    main()