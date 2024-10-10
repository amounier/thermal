#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:00:06 2024

@author: amounier
"""

import time 
import os
from datetime import date
import pandas as pd
import tqdm
import numpy as np


from meteorology import get_historical_weather_data
from administrative import Climat


def open_zcl_daily_temperature():
    data = pd.read_csv(os.path.join('data','CORDEX','climat_region_daily_temperature.csv'))
    data['date'] = pd.to_datetime(data.date, format='%Y-%m-%d %H:%M:%S')
    data = data.set_index('date')
    return data


def delta_daily_temperature(zcl_codint,nmod,ref_period=[2010,2020]):
    data = open_zcl_daily_temperature()
    columns = [c for c in data.columns if str(zcl_codint) in c]
    columns = [c for c in columns if 'mod{}'.format(nmod) in c]
    data = data[columns]
    ref_years = list(range(ref_period[0],ref_period[1]+1))
    
    days = list(range(1,32))
    months = list(range(1,13))
    
    dict_ref_temperature = dict()
    for m in months:
        for d in days:
            ref_doy_temp = data[(data.index.day==d)&(data.index.month==m)&(data.index.year.isin(ref_years))]
            dict_ref_temperature[(m,d)] = ref_doy_temp.mean()
    
    data_delta_t = data.copy()
    for m in months:
        for d in days:
            filter_day = (data_delta_t.index.day==d)&(data_delta_t.index.month==m)
            data_delta_t[filter_day] = data_delta_t[filter_day] - dict_ref_temperature.get((m,d))
    return data_delta_t


def get_projected_weather_data(city,zcl_codint,nmod,rcp,future_period,principal_orientation,ref_year=2020):
    future_period_list = list(range(future_period[0],future_period[1]+1))
    dt = delta_daily_temperature(zcl_codint, nmod)
    col = 'proj_temperature_{}_mod{}_rcp{}'.format(zcl_codint,nmod,rcp)
    dt = dt[[col]]
    dt = dt[dt.index.year.isin(future_period_list)]
    
    # il faut que les deux periodes aient la même durée
    ref_period = [ref_year+1-len(future_period_list),ref_year]
    change_year_dict = {proj_year:ref_year for proj_year,ref_year in zip(future_period,ref_period)}
    
    new_index = [pd.to_datetime('{}-{}-{}'.format(change_year_dict.get(y),m,d)) for y,m,d in zip(dt.index.year,dt.index.month,dt.index.day)]
    dt.index = new_index

    weather_data = get_historical_weather_data(city,ref_period,principal_orientation)
    weather_data = weather_data.rename(columns={'temperature_2m':'ref_temperature_2m'})
    weather_data['temperature_2m'] = [np.nan]*len(weather_data)
    
    for y,m,d, delta_t in zip(dt.index.year.values, dt.index.month.values, dt.index.day.values, dt[col]):
        filter_day = (weather_data.index.year==y)&(weather_data.index.month==m)&(weather_data.index.day==d)
        
        projected_weather_data = pd.DataFrame(weather_data[filter_day]['ref_temperature_2m'] + delta_t).rename(columns={'ref_temperature_2m':'temperature_2m'})
        
        weather_data = weather_data.join(projected_weather_data, lsuffix='_l', rsuffix='_r')
        
        # coalesce cost column to get first non NA value
        weather_data['temperature_2m'] = weather_data['temperature_2m_l'].combine_first(weather_data['temperature_2m_r']).astype(float)
        
        # remove the cols
        weather_data = weather_data.drop(columns=['temperature_2m_r', 'temperature_2m_l'])
    
    return weather_data

#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_future_meteorology'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
        
    #%% Téléchargement des données CORDEX sur CDS (ne marche pas)
    if False:
        import cdsapi
        
        dataset = "projections-cordex-domains-single-levels"
        request = {
            "domain": "europe",
            "experiment": "rcp_8_5",
            "horizontal_resolution": "0_11_degree_x_0_11_degree",
            "temporal_resolution": "3_hours",
            "variable": [
                "2m_air_temperature",
                "surface_solar_radiation_downwards",
                "surface_thermal_radiation_downward"
            ],
            "gcm_model": "ipsl_cm5a_mr",
            "rcm_model": "knmi_racmo22e",
            "ensemble_member": "r1i1p1",
            "start_year": ["2078"],
            "end_year": ["2079"]
        }
        
        client = cdsapi.Client()
        client.retrieve(dataset, request).download()
        
    #%% Utilisation des projections de températures par zone climatique
    if True:
        zcl = Climat('H1a')
        nmod = 1
        data = delta_daily_temperature(zcl.codint,nmod)
        data.plot()
        
        data = get_projected_weather_data(city='Paris',
                                          zcl_codint=Climat('H1a').codint,
                                          nmod=0,
                                          rcp=85,
                                          future_period=[2090,2090],
                                          principal_orientation='S')
        print(data)
        data[['ref_temperature_2m','temperature_2m']].plot()
        # print(data)
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()

