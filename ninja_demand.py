#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:41:06 2025

@author: amounier
"""

import requests
import pandas as pd
import os
import time 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import numpy as np

from administrative import get_coordinates
from thermal_sensitivity import plot_thermal_sensitivity

token = 'ca2eeaded528fe7232c0d91c6c7daa8e5779bbc5'
api_base = 'https://www.renewables.ninja/api/'



def download_ninja_demand(city,year=2023,heating_threshold=18.,cooling_threshold=21.,
                          heating_power=0.02,cooling_power=0.02):
    lon,lat = get_coordinates(city)
    start_date, end_date = '{}-01-01'.format(year),'{}-12-31'.format(year)
    args = (lat,lon,start_date,end_date,heating_threshold,cooling_threshold,heating_power,cooling_power)
    
    output_path = os.path.join('data','Ninja')
    output_file = 'ninja_demand_lat{}_lon{}_start{}_end{}_hth{}_cth{}_hpw{}_cpw{}.csv'.format(*args)
    output_file_weather = 'ninja_weather_{}_{}_{}_{}.csv'.format(lat,lon,start_date,end_date)
    
    if output_file not in os.listdir(output_path):
        print("Downloading {} from renewables.ninja...".format(year))
        time.sleep(1)
        url = api_base + 'data/demand'
        s = requests.session()
        s.headers = {'Authorization': 'Token ' + token}
        
        args = {
            'lat': lat,
            'lon': lon,
            'date_from': start_date,
            'date_to': end_date,
            'heating_threshold': heating_threshold,
            'heating_power':heating_power,
            'cooling_threshold': cooling_threshold,
            'cooling_power':cooling_power,
            'use_diurnal_profile':True,
            'dataset': 'merra2',
            'format': 'csv'
        }
        
        r = s.get(url, params=args)
        
        with open(os.path.join(output_path,output_file), 'a') as f:
            f.write(r.text)
            
    if output_file_weather not in os.listdir(output_path):
        url = api_base + 'data/weather'
        s = requests.session()
        s.headers = {'Authorization': 'Token ' + token}
        
        args = {
            'lat': lat,
            'lon': lon,
            'date_from': start_date,
            'date_to': end_date,
            'var_t2m': True,
            'dataset': 'merra2',
            'format': 'csv'
        }
        
        r = s.get(url, params=args)
        
        with open(os.path.join(output_path,output_file_weather), 'a') as f:
            f.write(r.text)
    return
    
    
def get_ninja_demand(city,period=[2022,2024],heating_threshold=18.,cooling_threshold=21.,
                     heating_power=0.02,cooling_power=0.02):
    """
    Récupération des données produites par Renewables.ninja

    Parameters
    ----------
    city : TYPE
        DESCRIPTION.
    period : TYPE, optional
        DESCRIPTION. The default is [2022,2024].
    heating_threshold : TYPE, optional
        DESCRIPTION. The default is 18..
    cooling_threshold : TYPE, optional
        DESCRIPTION. The default is 21..
    heating_power : TYPE, optional
        DESCRIPTION. The default is 0.02.
    cooling_power : TYPE, optional
        DESCRIPTION. The default is 0.02.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    res = None
    for y in range(period[0],period[1]+1):
        lon,lat = get_coordinates(city)
        start_date, end_date = '{}-01-01'.format(y),'{}-12-31'.format(y)
        args = (lat,lon,start_date,end_date,heating_threshold,cooling_threshold,heating_power,cooling_power)
        
        output_path = os.path.join('data','Ninja')
        output_file = 'ninja_demand_lat{}_lon{}_start{}_end{}_hth{}_cth{}_hpw{}_cpw{}.csv'.format(*args)
        output_file_weather = 'ninja_weather_{}_{}_{}_{}.csv'.format(lat,lon,start_date,end_date)
        
        if output_file not in os.listdir(output_path) or output_file_weather not in os.listdir(output_path):
            download_ninja_demand(city,y,heating_threshold,cooling_threshold,heating_power,cooling_power)
        
        data = pd.read_csv(os.path.join(output_path,output_file),skiprows=3,).set_index('time')*1e3
        weather = pd.read_csv(os.path.join(output_path,output_file_weather),skiprows=3,).set_index('time')
    
        data = data.join(weather)
        data = data.reset_index()
        data['time'] = pd.to_datetime(data.time,format='%Y-%m-%d %H:%M')
        
        if res is None:
            res = data
        else:
            res = pd.concat([res,data])
    return res


#%% ===========================================================================
# script
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_ninja'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    #%% Essai de thermosensibilité
    if True:
        city = 'Marignane'
        # city = 'Paris'
        data = get_ninja_demand(city,
                                period=[2022,2024],
                                heating_threshold=17.,
                                cooling_threshold=22.,
                                heating_power=0.02,
                                cooling_power=0.02)
        
        data['total_demand'] = data['total_demand'] + 185.6
        # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        # sns.lineplot(data,x='time',y='total_demand',ax=ax)
        # ax.set_ylim(bottom=0.)
        # plt.show()
        
        # graphes 
        if False:
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.scatterplot(data,x='t2m',y='total_demand',ax=ax)
            ax.set_ylim(bottom=0.)
            plt.show()
        
            data_sorted = data.copy().sort_values(by='t2m')
            plot_thermal_sensitivity(data_sorted.t2m.values,data_sorted.total_demand.values,figs_folder,reg_code=93,reg_name='PACA',year='2022-2024')
    
        # profil journalier (std entre les régions climatiques)
        if True:
            moreau = pd.read_csv('data/Res-IRF/hourly_profile_moreau.csv')
            moreau['value'] = moreau['value']/moreau['value'].mean()
            
            conventionnel =  pd.read_csv('data/Ninja/heating_cooling_conventionnel.csv').rename(columns={'Unnamed: 0':'time'})
            conventionnel.time = pd.to_datetime(conventionnel.time,format='%Y-%m-%d %H:%M:%S')
            conventionnel['hour'] = conventionnel.time.dt.hour
            conventionnel['heating_temperature'] = conventionnel['heating_temperature']/conventionnel['heating_temperature'].mean()
            # conventionnel['cooling_temperature'] = -conventionnel['cooling_temperature'] + max(conventionnel['cooling_temperature'])
            conventionnel['cooling_temperature'] = conventionnel['cooling_temperature']/conventionnel['cooling_temperature'].mean()
            
            conv_hour_heating_avg = conventionnel[['hour','heating_temperature']].groupby('hour').mean()
            conv_hour_heating_avg = pd.concat([conv_hour_heating_avg,pd.DataFrame({'heating_temperature':[conv_hour_heating_avg.heating_temperature.values[0]]})],ignore_index=True)
            conv_hour_cooling_avg = conventionnel[['hour','cooling_temperature']].groupby('hour').mean()
            conv_hour_cooling_avg = pd.concat([conv_hour_cooling_avg,pd.DataFrame({'cooling_temperature':[conv_hour_cooling_avg.cooling_temperature.values[0]]})],ignore_index=True)
            
            data['hour'] = data.time.dt.hour
            
            data_heating = data.copy()
            data_heating = data_heating[(data_heating.time.dt.month.isin([12,1,2]))&(data_heating.time.dt.dayofweek.isin([1,2,3,4,5,6,7]))]
            data_heating['heating_demand'] = data_heating['heating_demand']/data_heating['heating_demand'].mean()
            hour_heating_avg = data_heating[['hour','heating_demand']].groupby('hour').mean()
            hour_heating_avg = pd.concat([hour_heating_avg,pd.DataFrame({'heating_demand':[hour_heating_avg.heating_demand.values[0]]})],ignore_index=True)
            hour_heating_std = data_heating[['hour','heating_demand']].groupby('hour').std()
            
            data_cooling = data.copy()
            data_cooling = data_cooling[data_cooling.time.dt.month.isin([6,7,8])]
            data_cooling['cooling_demand'] = data_cooling['cooling_demand']/data_cooling['cooling_demand'].mean()
            hour_cooling_avg = data_cooling[['hour','cooling_demand']].groupby('hour').mean()
            hour_cooling_avg = pd.concat([hour_cooling_avg,pd.DataFrame({'cooling_demand':[hour_cooling_avg.cooling_demand.values[0]]})],ignore_index=True)
            hour_cooling_std = data_cooling[['hour','cooling_demand']].groupby('hour').std()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(hour_heating_avg.index,hour_heating_avg.heating_demand,color='tab:red',label='Heating')
            # ax.plot(conv_hour_heating_avg.index,conv_hour_heating_avg.heating_temperature,color='tab:red',ls=':')
            # ax.plot(moreau.hour,moreau['value'],label='Moreau (2024)',color='tab:red',ls=':')
            # ax.fill_between(hour_heating_avg.index,
            #                 hour_heating_avg.heating_demand+hour_heating_std.heating_demand,
            #                 hour_heating_avg.heating_demand-hour_heating_std.heating_demand,color='tab:red',alpha=0.2)
            ax.plot(hour_cooling_avg.index,hour_cooling_avg.cooling_demand,color='tab:blue',label='Cooling')
            # ax.plot(conv_hour_cooling_avg.index,conv_hour_cooling_avg.cooling_temperature,color='tab:blue',ls=':')
            # ax.fill_between(hour_cooling_avg.index,
            #                 hour_cooling_avg.cooling_demand+hour_cooling_std.cooling_demand,
            #                 hour_cooling_avg.cooling_demand-hour_cooling_std.cooling_demand,color='tab:blue',alpha=0.2)
            # ax.plot([-1],[0],label='Staffell et al. (2023)',color='k')
            # ax.plot([-1],[0],label='Conventional behaviours',color='k',ls=':')
            ax.set_ylim(bottom=0.)
            ax.set_xlim([0,24])
            
            ylims = ax.get_ylim()
            for t in [0,12]:
                ax.fill_between([t,t+6],[ylims[1]]*2,[0]*2,color='tab:grey',alpha=0.1)
            
            ax.set_ylim(ylims)
            ax.set_ylabel('Average diurnal profiles (normalized)')
            ax.set_xlabel('Hours of the day')
            ax.legend()
            plt.show()
            
        # profil journalier de tmepérature 
        if True:
            data['hour'] = data.time.dt.hour
            
            data_jja = data.copy()
            data_jja = data_jja[(data_jja.time.dt.month.isin([6,7,8]))]
            hour_jja_avg = data_jja[['hour','t2m']].groupby('hour').mean()
            hour_jja_avg = pd.concat([hour_jja_avg,pd.DataFrame({'t2m':[hour_jja_avg.t2m.values[0]]})],ignore_index=True)
            
            data_djf = data.copy()
            data_djf = data_djf[(data_djf.time.dt.month.isin([12,1,2]))]
            hour_djf_avg = data_djf[['hour','t2m']].groupby('hour').mean()
            hour_djf_avg = pd.concat([hour_djf_avg,pd.DataFrame({'t2m':[hour_djf_avg.t2m.values[0]]})],ignore_index=True)
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(hour_jja_avg.index,hour_jja_avg.t2m,color='tab:red',label='JJA')
            ax.plot(hour_djf_avg.index,hour_djf_avg.t2m,color='tab:blue',label='DJF')
            ax.set_ylim(bottom=0.)
            ax.set_xlim([0,24])
            
            ylims = ax.get_ylim()
            for t in [0,12]:
                ax.fill_between([t,t+6],[ylims[1]]*2,[0]*2,color='tab:grey',alpha=0.1)
            
            ax.set_ylim(ylims)
            ax.set_ylabel('Average diurnal profiles (normalized)')
            ax.set_xlabel('Hours of the day')
            ax.legend()
            plt.show()
        
        # profil journalier de température de consigne
        if True:
            a_ch = hour_heating_avg.values.mean()/(19-hour_djf_avg.values.mean())
            hour_djf_avg['T_int_ch'] = hour_heating_avg.values/a_ch + hour_djf_avg.values
            
            a_fr = hour_cooling_avg.values.mean()/(hour_jja_avg.values.mean()-26)
            hour_jja_avg['T_int_fr'] = hour_cooling_avg.values/a_fr + hour_jja_avg.values
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(hour_jja_avg.index,hour_jja_avg.T_int_fr,color='tab:red',label='T int clim')
            ax.plot(hour_djf_avg.index,hour_djf_avg.T_int_ch,color='tab:blue',label='T int chauff')
            ax.set_ylim(bottom=0.)
            ax.set_xlim([0,24])
            
            ylims = ax.get_ylim()
            for t in [0,12]:
                ax.fill_between([t,t+6],[ylims[1]]*2,[0]*2,color='tab:grey',alpha=0.1)
            
            ax.set_ylim(ylims)
            ax.set_ylabel('Average diurnal profiles (normalized)')
            ax.set_xlabel('Hours of the day')
            ax.legend()
            plt.show()
            
            
                
            
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
    
if __name__ == '__main__':
    main()