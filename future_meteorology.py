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
import subprocess
import xarray as xr
import matplotlib.pyplot as plt
import pyproj
import cartopy.crs as ccrs
import matplotlib
import geopandas as gpd
from shapely.geometry import mapping
import cmocean
from pysolar.solar import get_altitude, get_altitude_fast, get_azimuth, get_azimuth_fast
import multiprocessing
import seaborn as sns
import pickle

from meteorology import (get_historical_weather_data, 
                         aggregate_resolution, 
                         get_direct_solar_irradiance_projection_ratio,
                         get_diffuse_solar_irradiance_projection_ratio,
                         get_meteo_data,
                         get_safran_weather_data,
                         get_safran_hourly_weather_data)
from administrative import Climat, get_coordinates, France, City, draw_climat_map, Climat_winter
from climate_zone_characterisation import map_xarray
from utils import blank_national_map, get_extent,plot_timeserie


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


def get_projected_weather_data_old(city,zcl_codint,nmod,rcp,future_period,ref_year=2020):
    future_period_list = list(range(future_period[0],future_period[1]+1))
    dt = delta_daily_temperature(zcl_codint, nmod)
    col = 'proj_temperature_{}_mod{}_rcp{}'.format(zcl_codint,nmod,rcp)
    dt = dt[[col]]
    dt = dt[dt.index.year.isin(future_period_list)]
    
    # il faut que les deux periodes aient la même durée
    # print(future_period_list)
    ref_period = [ref_year+1-len(future_period_list),ref_year]
    ref_period_list = list(range(ref_period[0],ref_period[1]+1))
    # print(ref_period_list)
    change_year_dict = {proj_year:ref_year for proj_year,ref_year in zip(future_period_list,ref_period_list)}
    # print(change_year_dict)
    new_index = [pd.to_datetime('{}-{}-{}'.format(change_year_dict.get(y),m,d),errors='coerce') for y,m,d in zip(dt.index.year,dt.index.month,dt.index.day)]
    dt.index = new_index

    weather_data = get_historical_weather_data(city,ref_period)
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



def compute_projected_weather_data(zcl_code,nmod):
    # Accéléré 72 fois grace aux bfill et ffill 
    zcl = Climat(zcl_code)
    city = City(zcl.center_prefecture)
    coordinates = city.coordinates
    
    path = os.path.join('data','Explore2','daily')
    
    daily_data = pd.read_csv(os.path.join(path,'tas_mod{}_{}.csv'.format(nmod,zcl_code))).rename(columns={'Unnamed: 0':'date'}).set_index('date')
    for cv in ['tasmin','tasmax','rsds']:
        daily_data = daily_data.join(pd.read_csv(os.path.join(path,'{}_mod{}_{}.csv'.format(cv,nmod,zcl_code))).rename(columns={'Unnamed: 0':'date'}).set_index('date'))
    daily_data.index = pd.to_datetime(daily_data.index)
    
    if '{}_mod{}.parquet'.format(zcl_code,nmod) not in os.listdir(os.path.join('data','Explore2','hourly')):
        
        hourly_data = pd.DataFrame(index=pd.date_range(daily_data.index[0],'{}-01-01'.format(daily_data.index[-1].year+1),freq='h'))
        hourly_data = hourly_data.drop(hourly_data.iloc[-1].name)
        
        dates = hourly_data.copy().index
        dates = dates.tz_localize(tz='CET',ambiguous='NaT',nonexistent='NaT')
        dates = dates.to_pydatetime()
        altitude = [get_altitude_fast(coordinates[1],coordinates[0],t) if t is not pd.NaT else np.nan for t in tqdm.tqdm(dates, desc='altitude')]
        azimuth = [float(get_azimuth_fast(coordinates[1],coordinates[0],t)) if t is not pd.NaT else np.nan for t in tqdm.tqdm(dates, desc='azimuth')]
        
        hourly_data['sun_azimuth'] = azimuth
        hourly_data['sun_altitude'] = altitude
        hourly_data['sunrise'] = hourly_data.sun_altitude.lt(0) & hourly_data.sun_altitude.shift(-1).ge(0)
        
        daily_data['hour_sunrise'] = hourly_data.index[hourly_data['sunrise']].hour.values[:len(daily_data)]
        
        # construction des données de temperature
        hourly_data['temp'] = [np.nan]*len(hourly_data)
        
        hour_min_rel_sunrise = 0
        hour_max = 15
        
        mask_max = hourly_data.index.hour == hour_max
        hourly_data.loc[mask_max,'temp'] = list(daily_data.tasmax.values) + [np.nan]*(len(hourly_data.loc[mask_max,'temp'])-len(daily_data))
        
        mask_min = hourly_data.sunrise.shift(hour_min_rel_sunrise).infer_objects(copy=False).fillna(False)
        hourly_data.loc[mask_min,'temp'] = list(daily_data.tasmin.values) + [np.nan]*(len(hourly_data.loc[mask_min,'temp'])-len(daily_data))
        
        # weather_data_modelled = weather_data_modelled[['temperature']]
        
        temperature_sin14R1 = [np.nan]*len(hourly_data)
        prev_T, prev_t, next_T, next_t = [np.nan]*4
        flag = False
        
        def get_previous_temperature(t, data):
            try:
                # d = data[data.index<t].dropna().iloc[-1]
                d = data.loc[t]
                t = d.date
                T = d.temp
                return t, T
            except IndexError:
                return np.nan, np.nan
            
        
        def get_next_temperature(t, data):
            try:
                # d = data[data.index>t].dropna().iloc[0]
                d = data.loc[t]
                t = d.date
                T = d.temp
                return t, T
            except IndexError:
                return np.nan, np.nan
        
        def compute_temperature_sin14R1(t,prev_T,prev_t,next_T,next_t):
            T = (next_T + prev_T)/2 - ((next_T-prev_T)/2 * np.cos(np.pi*(t-prev_t)/(next_t-prev_t)))
            return T
        
        # define bfill with temp and date
        hourly_bfill = hourly_data.copy()
        hourly_bfill['date'] = [pd.NaT]*len(hourly_bfill)
        hourly_bfill.loc[~hourly_bfill.temp.isnull(),'date'] = hourly_bfill.loc[~hourly_bfill.temp.isnull()].index
        hourly_bfill = hourly_bfill.bfill()
        
        # same for ffill
        hourly_ffill = hourly_data.copy()
        hourly_ffill['date'] = [pd.NaT]*len(hourly_ffill)
        hourly_ffill.loc[~hourly_ffill.temp.isnull(),'date'] = hourly_ffill.loc[~hourly_ffill.temp.isnull()].index
        hourly_ffill = hourly_ffill.ffill()
        
        for idx,t in tqdm.tqdm(enumerate(hourly_data.index),total=len(hourly_data), desc='temperature'):
            T = hourly_data.loc[t].temp
            if flag:
                prev_t, prev_T = get_previous_temperature(t, hourly_ffill)
                next_t, next_T = get_next_temperature(t, hourly_bfill)
                flag = False
            if not pd.isnull(T):
                flag = True
                temperature_sin14R1[idx] = T
            else:
                if pd.isnull(prev_t) or pd.isnull(next_t):
                    continue
                temperature_sin14R1[idx] = compute_temperature_sin14R1(t,prev_T,prev_t,next_T,next_t)
            
        hourly_data['temperature_2m'] = temperature_sin14R1
        
        hourly_data.to_parquet(os.path.join('data','Explore2','hourly','{}_mod{}.parquet'.format(zcl_code,nmod)))
        
        hourly_data['rsds'] = np.cos(np.deg2rad(90-hourly_data.sun_altitude))
        hourly_data['rsds'] = hourly_data['rsds'].clip(lower=0.)
        
        daily_data['rsds_model'] = aggregate_resolution(hourly_data[['rsds']],resolution='D', agg_method='mean')
        daily_data['rsds_factor'] = daily_data.rsds/daily_data.rsds_model # *1.09 # caution
        rsds_factor = np.asarray([[e]*24 for e in daily_data['rsds_factor']]).flatten()
        rsds_factor = list(rsds_factor) + [np.nan]*(len(hourly_data)-len(rsds_factor))
        
        hourly_data['rsds'] = hourly_data['rsds']*np.asarray(rsds_factor)
        
        direct_ratio = 0.749
        hourly_data['rsds_direct'] = hourly_data['rsds']*direct_ratio
        hourly_data['diffuse_radiation_instant'] = hourly_data['rsds']*(1-direct_ratio)
        
        normal_ratio = np.sin(np.deg2rad(np.maximum(hourly_data['sun_altitude'],0)))
        hourly_data['direct_normal_irradiance_instant'] = hourly_data['rsds_direct']/normal_ratio
        
        orientations = ['N','NE','E','SE','S','SW','W','NW','H']
        
        for ori in orientations:
            col_coef_dri = 'coefficient_direct_{}_irradiance'.format(ori)
            col_coef_dif = 'coefficient_diffuse_{}_irradiance'.format(ori)
            col_dri = 'direct_sun_radiation_{}'.format(ori)
            col_dif = 'diffuse_sun_radiation_{}'.format(ori)
            
            hourly_data[col_coef_dri] = get_direct_solar_irradiance_projection_ratio(ori, hourly_data.sun_azimuth, hourly_data.sun_altitude)
            hourly_data[col_coef_dif] = get_diffuse_solar_irradiance_projection_ratio(ori)
            hourly_data[col_dri] = hourly_data.direct_normal_irradiance_instant * hourly_data[col_coef_dri]
            hourly_data[col_dif] = hourly_data.diffuse_radiation_instant * hourly_data[col_coef_dif]
        
        hourly_data = hourly_data[['temperature_2m', 'diffuse_radiation_instant',
                                   'direct_normal_irradiance_instant', 'sun_altitude', 'sun_azimuth',
                                   'direct_sun_radiation_N', 'diffuse_sun_radiation_N',
                                   'direct_sun_radiation_NE', 'diffuse_sun_radiation_NE',
                                   'direct_sun_radiation_E', 'diffuse_sun_radiation_E',
                                   'direct_sun_radiation_SE', 'diffuse_sun_radiation_SE',
                                   'direct_sun_radiation_S', 'diffuse_sun_radiation_S',
                                   'direct_sun_radiation_SW', 'diffuse_sun_radiation_SW',
                                   'direct_sun_radiation_W', 'diffuse_sun_radiation_W',
                                   'direct_sun_radiation_NW', 'diffuse_sun_radiation_NW',
                                   'direct_sun_radiation_H', 'diffuse_sun_radiation_H']]
        
        
        hourly_data.to_parquet(os.path.join('data','Explore2','hourly','{}_mod{}.parquet'.format(zcl_code,nmod)))
    
    hourly_data = pd.read_parquet(os.path.join('data','Explore2','hourly','{}_mod{}.parquet'.format(zcl_code,nmod)))
    hourly_data = hourly_data.fillna(0.)
    return hourly_data


def get_projected_weather_data(zcl_code,period,nmod=3):
    """
    Récupération des données climatiques projetées

    Parameters
    ----------
    zcl_code : str
        code de la zone climatique zcl8.
    period : list
        list de deux éléments : année de départ, année de fin (inclues). Comprises entre 1951 et 2100
    nmod : int, optional
        Numéro du modèle climatique. The default is 3.

    Returns
    -------
    hourly : pandas DataFrame
        données climatiques.

    """
    hourly = compute_projected_weather_data(zcl_code, nmod=nmod)
    hourly = hourly[hourly.index.year.isin(list(range(period[0],period[1]+1)))]
    
    return hourly


                               
                               

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
    
    external_disk_connection = 'MPBE' in os.listdir('/media/amounier/')
        
    #%% Téléchargement des données CORDEX sur CDS (ne marche pas)
    if False:
        import cdsapi

        dataset = "projections-cordex-domains-single-levels"
        request = {
            "domain": "europe",
            "experiment": "historical",
            "horizontal_resolution": "0_11_degree_x_0_11_degree",
            "temporal_resolution": "daily_mean",
            "variable": [
                "2m_air_temperature",
                "maximum_2m_temperature_in_the_last_24_hours",
                "minimum_2m_temperature_in_the_last_24_hours",
                "surface_solar_radiation_downwards"
            ],
            "gcm_model": "cnrm_cerfacs_cm5",
            "rcm_model": "cnrm_aladin63",
            "ensemble_member": "r1i1p1",
            "start_year": ["1996"],
            "end_year": ["2000"]
        }
        
        client = cdsapi.Client()
        client.retrieve(dataset, request).download()

    # projections cliamtiques d'open-meteo CMIP6 HighRes
    if False:
        import openmeteo_requests

        import requests_cache
        # import pandas as pd
        from retry_requests import retry
        
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)
        
        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://climate-api.open-meteo.com/v1/climate"
        params = {
        	"latitude": 52.52,
        	"longitude": 13.41,
        	"start_date": "1950-01-01",
        	"end_date": "2050-12-31",
        	"models": ["CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S"],
        	"daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "shortwave_radiation_sum"]
        }
        responses = openmeteo.weather_api(url, params=params)
        
        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
        
        # Process daily data. The order of variables needs to be the same as requested.
        daily = response.Daily()
        daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
        daily_shortwave_radiation_sum = daily.Variables(3).ValuesAsNumpy()
        
        daily_data = {"date": pd.date_range(
        	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        	end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        	freq = pd.Timedelta(seconds = daily.Interval()),
        	inclusive = "left"
        )}
        
        daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
        
        daily_dataframe = pd.DataFrame(data = daily_data)
        print(daily_dataframe)


    
    
    #%% Utilisation des projections de températures par zone climatique
    if False:
        zcl = Climat('H1a')
        nmod = 1
        data = delta_daily_temperature(zcl.codint,nmod)
        data.plot()
        
        data = get_projected_weather_data_old(city='Paris',
                                          zcl_codint=Climat('H1a').codint,
                                          nmod=0,
                                          rcp=85,
                                          future_period=[2085,2090],
                                          principal_orientation='S')
        print(data)
        data[['ref_temperature_2m','temperature_2m']].plot()
        
        for t in ['ref_temperature_2m','temperature_2m']:
            print(data[t].mean())
        # print(data)
        
    
    #%% Gestion des données Explore2
    if False and external_disk_connection:
    
        data_folder = '/media/amounier/MPBE/heavy_data/Explore2'
        
        # telechargement sur le site de la DRIAS
        if False:
            # with open(os.path.join('data','Explore2','download_links.txt')) as f:
            #     dl_links = f.read().splitlines()
                
            with open(os.path.join('data','Explore2','download_links_adamont.txt')) as f:
                dl_links = f.read().splitlines()
            
            for url in dl_links:
                subprocess.run('wget -P {} {}'.format(data_folder,url),shell=True)
        
        
        # association des projections et modelisations historiques (temperature)
        if False:
            zcl = Climat('H3')
            city = zcl.center_prefecture
            coords = get_coordinates(city)
            
            tas_Explore2_hist = '/media/amounier/MPBE/heavy_data/Explore2/tasAdjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19510101-20051231.nc'
            tas_Explore2_proj = '/media/amounier/MPBE/heavy_data/Explore2/tasAdjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'
            
            tas_array_hist = xr.open_dataset(tas_Explore2_hist).tasAdjust - 273.15
            tas_array_proj = xr.open_dataset(tas_Explore2_proj).tasAdjust - 273.15
            
            tas_array_hist.rio.write_crs('epsg:27572', inplace=True)
            tas_array_proj.rio.write_crs('epsg:27572', inplace=True)
            
            geom = pd.Series(France().geometry).apply(mapping)
            tas_array_hist = tas_array_hist.rio.clip(geom, 'epsg:4326', drop=False)
            tas_array_proj = tas_array_proj.rio.clip(geom, 'epsg:4326', drop=False)
    
            transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27572", always_xy=True)
            coords_transformed = transformer.transform(*coords)
            
            # carte des temperature
            if True:
                cmap = matplotlib.colormaps.get_cmap('viridis')
                
                fig,ax = blank_national_map()
                
                tas = tas_array_hist.isel(time=42)
                img = tas.plot(ax=ax,transform=ccrs.epsg('27572'),add_colorbar=False,
                               cmap=cmap,vmin=tas.min(),vmax=tas.max())
                
                ax.plot(*coords_transformed, transform=ccrs.epsg('27572'), marker='o',color='red',mec='k')
                
                ax.set_title('')
                
                ax_cb = fig.add_axes([0,0,0.1,0.1])
                posn = ax.get_position()
                ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                fig.add_axes(ax_cb)
                cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
                cbar.set_label('Daily mean temperature (°C)')
                
                ax.set_extent(get_extent())
                
                plt.show()
            
            
            # détermination des périodes à +2 et +4 degrés 
            if True:
                mean_yt_deg2 = 11.37
                mean_yt_deg4 = 13.25
                
                hist = tas_array_hist.mean(('x','y'))
                # hist = tas_array_hist.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
                hist = hist.groupby('time.year').mean('time')
                
                proj = tas_array_proj.mean(('x','y'))
                # proj = tas_array_proj.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
                proj = proj.groupby('time.year').mean('time')
                
                data_hist = pd.DataFrame(index=hist.year, data=hist)
                data_proj = pd.DataFrame(index=proj.year, data=proj)
                data = pd.concat([data_hist, data_proj]).rename(columns={0:'temperature'})
                    
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                data.plot(ax=ax,color='k')
                ax.plot([data.index.min(),data.index.max()],[mean_yt_deg2]*2,color='tab:blue',label='+2°C')
                ax.plot([data.index.min(),data.index.max()],[mean_yt_deg4]*2,color='tab:red',label='+4°C')
                ax.legend()
                ax.set_ylabel('Yearly mean temperature over France (°C)')
                plt.show()
                
                period_list = []
                mean_20_years_list = []
                for y0 in data.index[10:-10]:
                    data_20 = data.loc[y0-10:y0+10]
                    mean_20 = data_20.mean()
                    period = y0
                    period_list.append(period)
                    mean_20_years_list.append(mean_20)
                    
                mean_20_years_list = np.asarray(mean_20_years_list).T[0]
                
                year_deg2 = period_list[next(x[0] for x in enumerate(mean_20_years_list) if x[1] > mean_yt_deg2)]
                year_deg4 = period_list[next(x[0] for x in enumerate(mean_20_years_list) if x[1] > mean_yt_deg4)]
                    
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                ax.plot(period_list,mean_20_years_list,color='k')
                ax.plot([period_list[0],year_deg2],[mean_yt_deg2]*2,color='tab:blue',label='+2°C ({}-{})'.format(year_deg2-10,year_deg2+10),)
                ax.plot([period_list[0],year_deg4],[mean_yt_deg4]*2,color='tab:red',label='+4°C ({}-{})'.format(year_deg4-10,year_deg4+10))
                ax.plot([year_deg2],[mean_yt_deg2],color='tab:blue',marker='o')
                ax.plot([year_deg4],[mean_yt_deg4],color='tab:red',marker='o')
                ax.legend()
                ax.set_ylabel('Yearly mean temperature over 20 years (°C)')
                ax.set_xlabel('Center year of mean period')
                plt.show()
    
        # comparaison avec les données ERA5
        if False:
            zcl = Climat('H3')
            city = zcl.center_prefecture
            coords = get_coordinates(city)
            period = [1990,2000]
            
            tas_Explore2_hist = '/media/amounier/MPBE/heavy_data/Explore2/tasAdjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19510101-20051231.nc'
            tas_array_hist = xr.open_dataset(tas_Explore2_hist).tasAdjust - 273.15
            
            tas_array_hist.rio.write_crs('epsg:27572', inplace=True)
            
            # geom = pd.Series(France().geometry).apply(mapping)
            # tas_array_hist = tas_array_hist.rio.clip(geom, 'epsg:4326', drop=False)
    
            transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27572", always_xy=True)
            coords_transformed = transformer.transform(*coords)
            
            explore2 = tas_array_hist.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
            explore2 = explore2.sel(time=slice("{}-01-01".format(period[0]), "{}-12-31".format(period[1])))
            
            era5 = get_historical_weather_data(city,period)
            era5 = aggregate_resolution(era5,resolution='D')
            
            if True:
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                explore2.plot(ax=ax)
                era5.temperature_2m.plot(ax=ax)
                plt.show()
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                ax.plot(era5.temperature_2m,explore2,marker='.',alpha=0.05,ls='')
                ax.plot([explore2.min(),explore2.max()],[explore2.min(),explore2.max()],color='k')
                plt.axis('equal')
                
                plt.show()
        
    # CDF temperature moyenne et solaire 
    if False:
        
        if external_disk_connection:
            data_folder = '/media/amounier/MPBE/heavy_data/Explore2'
        else:
            data_folder = os.path.join('data','Explore2')
        
        # association des projections et modelisations historiques (temperature)
        if True:
            zcl = Climat('H3')
            zcl = Climat('H1b')
            city = zcl.center_prefecture
            coords = get_coordinates(city)
            nmod = 0
            # calib_method = 'CDFt'
            calib_method = 'ADAMONT'
            variable = 'rsds' # tas
            variable = 'tas'
            season = True
    
            if calib_method == 'CDFt':
                models_dict = {0:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19510101-20051231.nc',
                                  # 'rcp45':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc',
                                  'rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                               
                               1:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_MOHC-HadREM3-GA7-05_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                                  'rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001230.nc'},
                               
                               2:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19500101-20051231.nc',
                                  # 'rcp45':'Adjust_France_ICHEC-EC-EARTH_rcp45_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc',
                                  'rcp85':'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                               
                               3:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                                  'rcp85':'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                               
                               4:{'historical':'Adjust_France_MOHC-HadGEM2-ES_historical_r1i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                                  'rcp85':'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-20991219.nc'},
                               }
            
            elif calib_method == 'ADAMONT':
                models_dict = {0:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19510101-20051231.nc',
                                  'rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                                  'historical_rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19510101-21001231.nc'},
                               
                               1:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                                  'rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001230.nc',
                                  'historical_rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-21001230.nc'},
                               
                               2:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19500101-20051231.nc',
                                  'rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                                  'historical_rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19500101-21001231.nc'},
                               
                               3:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                                  'rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                                  'historical_rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-21001231.nc'},
                               
                               4:{'historical':'Adjust_France_MOHC-HadGEM2-ES_historical_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                                  'rcp85':     'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-20991219.nc',
                                  'historical_rcp85':     'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20991219.nc'},
                               }
            
            rsds_Explore2_hist = os.path.join(data_folder,variable+models_dict.get(nmod).get('historical'))
            # rlds_Explore2_hist = os.path.join(data_folder,"rldsAdjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19510101-20051231.nc")
            # rsds_Explore2_proj = "/media/amounier/MPBE/heavy_data/Explore2/rsdsAdjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc"
            
            if variable == 'rsds':
                rsds_array_hist = xr.open_dataset(rsds_Explore2_hist).rsdsAdjust
                
            elif variable == 'tas':
                rsds_array_hist = xr.open_dataset(rsds_Explore2_hist).tasAdjust
                
            # rlds_array_hist = xr.open_dataset(rlds_Explore2_hist).rldsAdjust
            # rsds_array_proj = xr.open_dataset(rsds_Explore2_proj).rsdsAdjust
            
            rsds_array_hist.rio.write_crs('epsg:27572', inplace=True)
            # rlds_array_hist.rio.write_crs('epsg:27572', inplace=True)
            # rsds_array_proj.rio.write_crs('epsg:27572', inplace=True)
            
            
    
            transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27572", always_xy=True)
            coords_transformed = transformer.transform(*coords)
            
            if False:
                cmap = matplotlib.colormaps.get_cmap('viridis')
                geom = pd.Series(France().geometry).apply(mapping)
                rsds_array_hist = rsds_array_hist.rio.clip(geom, 'epsg:4326', drop=False)
                # rsds_array_proj = rsds_array_proj.rio.clip(geom, 'epsg:4326', drop=False)
                
                fig,ax = blank_national_map()
                
                tas = rsds_array_hist.isel(time=42)
                img = tas.plot(ax=ax,transform=ccrs.epsg('27572'),add_colorbar=False,
                               cmap=cmap,vmin=tas.min(),vmax=tas.max())
                
                ax.plot(*coords_transformed, transform=ccrs.epsg('27572'), marker='o',color='red',mec='k')
                
                ax.set_title('')
                
                ax_cb = fig.add_axes([0,0,0.1,0.1])
                posn = ax.get_position()
                ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                fig.add_axes(ax_cb)
                cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
                cbar.set_label('rsdsAdjust')
                
                ax.set_extent(get_extent())
                
                plt.show()
            
            
            if False:
                # hist = rsds_array_hist.mean(('x','y'))
                hist_rsds = rsds_array_hist.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
                # hist = hist.groupby('time.year').mean('time')
                # hist_rlds = rlds_array_hist.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
                
                # proj = rsds_array_proj.mean(('x','y'))
                # # proj = tas_array_proj.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
                # proj = proj.groupby('time.year').mean('time')
                
                # data_hist_rsds = pd.DataFrame(index=hist_rsds.time, data=hist_rsds).rename(columns={0:'rsds'})
                # data_hist_rlds = pd.DataFrame(index=hist_rlds.time, data=hist_rlds).rename(columns={0:'rlds'})
                # data_proj = pd.DataFrame(index=proj.year, data=proj)
                # data = data_hist_rsds.join(data_hist_rlds)
                
                # data['sum'] = data.rsds + data.rlds
                
                # fig,ax = plt.subplots(figsize=(15,5),dpi=300)
                # data_hist_rsds.plot(ax=ax)
                # # data_hist_rlds.plot(ax=ax)
                # # (data_hist_rsds.rsds+data_hist_rlds.rlds).plot(ax=ax)
                # # ax.legend()
                # ax.set_ylabel('rsdsAdjust')
                # ax.set_xlim([pd.to_datetime('2001-01-01'), pd.to_datetime('2001-12-31')])
                # plt.show()
                test = 1
                
            
            # comparaison des CDF entre ERA5 et explore2
            if True:
                # reanalysis = 'era5'
                # reanalysis = 'safran'
                
                data_hist_rsds = pd.DataFrame()
                for nmod in range(5):
                    
                    if variable == 'rsds':
                        rsds_Explore2_hist = os.path.join(data_folder,"rsds"+models_dict.get(nmod).get('historical'))
                        rsds_array_hist = xr.open_dataset(rsds_Explore2_hist).rsdsAdjust
                        
                    elif variable == 'tas':
                        rsds_Explore2_hist = os.path.join(data_folder,"tas"+models_dict.get(nmod).get('historical'))
                        rsds_array_hist = xr.open_dataset(rsds_Explore2_hist).tasAdjust
                        
                    # rsds_array_hist = xr.open_dataset(rsds_Explore2_hist).rsdsAdjust
                    rsds_array_hist.rio.write_crs('epsg:27572', inplace=True)
                    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27572", always_xy=True)
                    coords_transformed = transformer.transform(*coords)
                    hist_rsds = rsds_array_hist.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
                    
                    if variable == 'tas':
                        hist_rsds = hist_rsds - 273.15
                        
                    data_hist_rsds_mod = pd.DataFrame(index=hist_rsds.time, data=hist_rsds).rename(columns={0:'{}_{}'.format(variable,nmod)})
                    data_hist_rsds = data_hist_rsds.join(data_hist_rsds_mod,how='outer')
                
                data_hist_rsds = data_hist_rsds.dropna()
                
                period_calibration = [1976,2005]
                # period_calibration = [2000,2020]
                data_hist_rsds = data_hist_rsds[data_hist_rsds.index.year.isin(list(range(period_calibration[0],period_calibration[1]+1)))]
                
                # if reanalysis == 'era5':
                data_hist_era5 = get_meteo_data(city,[data_hist_rsds.index.year[0],data_hist_rsds.index.year[-1]],['shortwave_radiation_instant','temperature_2m'])
                data_hist_era5 = aggregate_resolution(data_hist_era5,resolution='D',agg_method='mean')
                
                # data_hist_mf = 
                    
                # elif reanalysis == 'safran':
                data_hist_safran = get_safran_weather_data(zcl.center_prefecture,[data_hist_rsds.index.year[0],data_hist_rsds.index.year[-1]],param=['SSI_Q','T_Q'])
                data_hist_safran = data_hist_safran.rename(columns={'SSI_Q':'shortwave_radiation_instant','T_Q':'temperature_2m'})
                data_hist_safran['shortwave_radiation_instant'] = data_hist_safran.shortwave_radiation_instant / 3600 / 1e-4 / 24 # from J.cm-2.day-1 to Wh.m-2.day-1
                    
                # # quantile001_era5 = np.quantile(data_hist,0.01)
                # quantile05_era5 = np.quantile(data_hist,0.5)
                # quantile099_era5 = np.quantile(data_hist,0.999)
                
                # # quantile001_explore2 = np.quantile(data_hist_rsds,0.01)
                # quantile05_explore2 = np.quantile(data_hist_rsds,0.5)
                # quantile099_explore2 = np.quantile(data_hist_rsds,0.999)
                
                # # val001 = 1 #quantile001_era5/quantile001_explore2
                # val005 = quantile05_era5/quantile05_explore2
                # val099 = quantile099_era5/quantile099_explore2
                
                # # print(val001,val099)
                # print(quantile05_explore2, val005, val099)
                
                # fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                # for idx,c in enumerate(data_hist_rsds.columns):
                #     if idx != 0:
                #         continue
                #     X = np.linspace(data_hist_rsds[c].min(),data_hist_rsds[c].max(),100)
                #     corr = [min(1,((X[i]-quantile05_explore2)/(quantile099_explore2-quantile05_explore2)*(val099-val005)+val005)) if X[i] > quantile05_explore2 else 1. for i in range(len(X))]
                #     ax.plot(X, corr,color='k')
                # ax.set_ylim(bottom=0.)
                # ax.set_ylabel('Solar correction function')
                # ax.set_xlabel('Daily mean Explore2 solar radiation (W.m$^{-2}$)')
                # plt.savefig(os.path.join(figs_folder,'solar_coorection_function_{}.png'.format(city)), bbox_inches='tight')
                # plt.show()
                
                # for c in data_hist_rsds.columns:
                #     corr = [min(1,((data_hist_rsds[c].values[i]-quantile05_explore2)/(quantile099_explore2-quantile05_explore2)*(val099-val005)+val005)) if data_hist_rsds[c].values[i] > quantile05_explore2 else 1. for i in range(len(data_hist_rsds))]
                #     data_hist_rsds[c] = data_hist_rsds[c] * np.asarray(corr)
                
                
                
                cmap = matplotlib.colormaps.get_cmap('viridis')
                
                if season == False:
                    for month in range(1,13):
                        data_hist_rsds_month = data_hist_rsds[data_hist_rsds.index.month==month]
                        data_hist_safran_month = data_hist_safran[data_hist_safran.index.month==month]
                        data_hist_era5_month = data_hist_era5[data_hist_era5.index.month==month]
                        
                        # print(month, data_hist_month.max())
                        
                        x_var = {'rsds':'shortwave_radiation_instant','tas':'temperature_2m'}.get(variable)
                        xlabel = {'rsds':'Daily mean shortwave solar radiation (W.m$^{-2}$)',
                                  'tas':'Daily mean temperature (°C)'}.get(variable)
                        
                        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                        sns.ecdfplot(data=data_hist_safran_month, x=x_var,ax=ax,color='k',label='SAFRAN',zorder=2)
                        # sns.ecdfplot(data=data_hist_era5_month, x="shortwave_radiation_instant",ax=ax,color='tab:red',label='ERA5',zorder=2)
                        for nmod in range(5):
                            sns.ecdfplot(data=data_hist_rsds_month, x="{}_{}".format(variable, nmod),ax=ax,
                                         color=cmap(nmod/5),label='Model {}'.format(nmod+1),ls=[':','--','-.',':','--'][nmod],zorder=1)
                        ax.set_title('{} - {} ({}-{})'.format(city, pd.to_datetime('2000-{:02d}-01'.format(month)).strftime('%B'),data_hist_rsds.index.year[0],data_hist_rsds.index.year[-1]))
                        ax.legend()
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel('Cumulative distribution function')
                        plt.savefig(os.path.join(figs_folder,'CDF_{}_{}_month{}_{}.png'.format(variable,city,month,calib_method.lower())), bbox_inches='tight')
                        plt.show()
                else:
                    for seas in ['DJF','MAM','JJA','SON']:
                        month_dict = {'DJF':[12,1,2],
                                      'MAM':[3,4,5],
                                      'JJA':[6,7,8],
                                      'SON':[9,10,11],}
                        
                        data_hist_rsds_month = data_hist_rsds[data_hist_rsds.index.month.isin(month_dict.get(seas))]
                        data_hist_safran_month = data_hist_safran[data_hist_safran.index.month.isin(month_dict.get(seas))]
                        data_hist_era5_month = data_hist_era5[data_hist_era5.index.month.isin(month_dict.get(seas))]
                        
                        # print(month, data_hist_month.max())
                        
                        x_var = {'rsds':'shortwave_radiation_instant','tas':'temperature_2m'}.get(variable)
                        xlabel = {'rsds':'Daily mean shortwave solar radiation (W.m$^{-2}$)',
                                  'tas':'Daily mean temperature (°C)'}.get(variable)
                        
                        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                        sns.ecdfplot(data=data_hist_safran_month, x=x_var,ax=ax,color='k',label='SAFRAN',zorder=2)
                        # sns.ecdfplot(data=data_hist_era5_month, x="shortwave_radiation_instant",ax=ax,color='tab:red',label='ERA5',zorder=2)
                        for nmod in range(5):
                            sns.ecdfplot(data=data_hist_rsds_month, x="{}_{}".format(variable, nmod),ax=ax,
                                         color=cmap(nmod/5),label='Model {}'.format(nmod+1),ls=[':','--','-.',':','--'][nmod],zorder=1)
                        ax.set_title('{} - {} ({}-{})'.format(city, seas,data_hist_rsds.index.year[0],data_hist_rsds.index.year[-1]))
                        ax.legend()
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel('Cumulative distribution function')
                        plt.savefig(os.path.join(figs_folder,'CDF_{}_{}_season{}_{}.png'.format(variable,city,seas,calib_method.lower())), bbox_inches='tight')
                        plt.show()
                
                # data_hist_rsds_month = data_hist_rsds.copy()
                # data_hist_month = data_hist.copy()
                
                # fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                # sns.ecdfplot(data=data_hist_month, x="shortwave_radiation_instant",ax=ax,color='k',label='ERA5')
                # for nmod in range(5):
                #     sns.ecdfplot(data=data_hist_rsds_month, x="rsds_{}".format(nmod),ax=ax,
                #                  color=cmap(nmod/5),label='Model {}'.format(nmod),ls=[':','--','-.',':','--'][nmod])
                # ax.set_title('{} ({}-{})'.format(city,data_hist_rsds.index.year[0],data_hist_rsds.index.year[-1]))
                # ax.legend()
                # ax.set_xlabel('Daily mean shortwave solar radiation (W.m$^{-2}$)')
                # ax.set_ylabel('Cumulative distribution function')
                # plt.savefig(os.path.join(figs_folder,'CDF_rsds_{}_year_adamont.png'.format(city)), bbox_inches='tight')
                # plt.show()
                    
                    
                    # fig,ax = plt.subplots(dpi=300,figsize=(5,5))
                    # data_hist.plot(ax=ax,color='k')
                    # data_hist_rsds.plot(ax=ax)
                    # ax.set_xlim([pd.to_datetime('2001-{:02d}-01'.format(month)), pd.to_datetime('2001-{:02d}-28'.format(month))])
                    # ax.set_xlabel('')
                    # plt.show()
                    
                
                
            if False:
                # hist = rsds_array_hist.mean(('x','y'))
                hist = rsds_array_hist.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
                hist = hist.groupby('time.year').mean('time')
                
                # proj = rsds_array_proj.mean(('x','y'))
                # proj = rsds_array_proj.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
                # proj = proj.groupby('time.year').mean('time')
                
                data_hist = pd.DataFrame(index=hist.year, data=hist)
                data_proj = pd.DataFrame(index=proj.year, data=proj)
                data = pd.concat([data_hist, data_proj]).rename(columns={0:'rsdsAdjust'})
                    
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                data.plot(ax=ax,color='k')
                ax.legend()
                ax.set_ylabel('rsdsAdjust')
                plt.show()
                
                
    #%% Formatage des données de modèles climatiques 
    if False:
    
        # calib_method = 'CDFt'
        calib_method = 'ADAMONT'

        if calib_method == 'CDFt':
            models_dict = {0:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19510101-20051231.nc',
                              # 'rcp45':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc',
                              'rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                           
                           1:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_MOHC-HadREM3-GA7-05_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                              'rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001230.nc'},
                           
                           2:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19500101-20051231.nc',
                              # 'rcp45':'Adjust_France_ICHEC-EC-EARTH_rcp45_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc',
                              'rcp85':'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                           
                           3:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                              'rcp85':'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                           
                           4:{'historical':'Adjust_France_MOHC-HadGEM2-ES_historical_r1i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                              'rcp85':'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-20991219.nc'},
                           }
        
        elif calib_method == 'ADAMONT':
            models_dict = {0:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19510101-20051231.nc',
                              'rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                              'historical_rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19510101-21001231.nc'},
                           
                           1:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                              'rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001230.nc',
                              'historical_rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-21001230.nc'},
                           
                           2:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19500101-20051231.nc',
                              'rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                              'historical_rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19500101-21001231.nc'},
                           
                           3:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                              'rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                              'historical_rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-21001231.nc'},
                           
                           4:{'historical':'Adjust_France_MOHC-HadGEM2-ES_historical_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                              'rcp85':     'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-20991219.nc',
                              'historical_rcp85':     'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20991219.nc'},
                           }
        
    
        
        if external_disk_connection:
            data_folder = '/media/amounier/MPBE/heavy_data/Explore2'
        else:
            data_folder = os.path.join('data','Explore2')
        
        climate_vars = ['tas','tasmax','tasmin','rsds']
        climate_vars = ['tas']
        # climate_vars = ['rsds']
        for climate_var in climate_vars:
            # print('climate var',climate_var)
        # climate_var = 'rsds' #'tas','tasmax','tasmin','rsds'
        
            for mod in range(0,5):
                # print('\tmodel',mod)
            # for mod in range(4,5):
                
                # Explore2_hist = os.path.join(data_folder,climate_var+models_dict.get(mod).get('historical'))
                # array_hist = xr.open_dataset(Explore2_hist)
                # array_hist = array_hist[list(array_hist.data_vars)[-1]]
                # if 'tas' in climate_var:
                #     array_hist = array_hist - 273.15
                # array_hist.rio.write_crs('epsg:27572', inplace=True)
                
                # Explore2_proj = os.path.join(data_folder,climate_var+models_dict.get(mod).get('rcp85'))
                # array_proj = xr.open_dataset(Explore2_proj)
                # array_proj = array_proj[list(array_proj.data_vars)[-1]]
                # if 'tas' in climate_var:
                #     array_proj = array_proj - 273.15
                # array_proj.rio.write_crs('epsg:27572', inplace=True)
                
                Explore2 = os.path.join(data_folder,climate_var+models_dict.get(mod).get('historical_rcp85'))
                array = xr.open_dataset(Explore2)
                array = array[list(array.data_vars)[-1]]
                array.rio.write_crs('epsg:27572', inplace=True)
                
                
                # concatenation des valeurs journalières
                if False:
                    climats = France().climats
                    for zcl in climats:
                        # TODO : changer les villes des zcl
                        # TODO : vérifier de que ça marche
                        zcl = Climat(zcl)
                        city = zcl.center_prefecture
                        coords = get_coordinates(city)
                        
                        save_name = '{}_mod{}_{}.csv'.format(climate_var,mod,zcl.code)
                        if save_name in os.listdir(os.path.join(output, folder)):
                            continue
                        
                        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27572", always_xy=True)
                        coords_transformed = transformer.transform(*coords)
                        
                        point_data = array.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
                        data = pd.DataFrame(index=point_data.time, data=point_data).rename(columns={0:climate_var})
                        
                        print('saving {}_mod{}_{}'.format(climate_var,mod,zcl.code))
                        data.to_csv(os.path.join(os.path.join(output, folder),save_name))
                    
        
                # affichage des données journalieres 
                if False:
                    zcl = Climat('H3')
                    city = zcl.center_prefecture # TODO : idem , à remplacer
                    coords = get_coordinates(city)
                    
                    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27572", always_xy=True)
                    coords_transformed = transformer.transform(*coords)
                    
                    hist_rsds = rsds_array_hist.sel(x=coords_transformed[0], y=coords_transformed[1], method='nearest')
        
                    data_hist_rsds = pd.DataFrame(index=hist_rsds.time, data=hist_rsds).rename(columns={0:climate_var})
                    
                    print(mod, data_hist_rsds[data_hist_rsds.index.year==2001].sum())
                        
                    fig,ax = plt.subplots(figsize=(15,5),dpi=300)
                    data_hist_rsds.plot(ax=ax)
                    ax.set_title(mod)
                    ax.set_ylabel('rsdsAdjust')
                    ax.set_xlim([pd.to_datetime('2001-01-01'), pd.to_datetime('2001-12-31')])
                    plt.show()
                
                # concatenation des valeurs annuelles
                if False and climate_var != 'tas':
    
                    data_array = array.mean(('x','y'))
                    data_array = data_array.groupby('time.year').mean('time')
                    data = pd.DataFrame(index=data_array.year, data=data_array).rename(columns={0:climate_var})
                    
                    # data.to_csv(os.path.join(os.path.join(output, folder),'{}_mod{}.csv'.format(climate_var,mod)))
                        
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    data.plot(ax=ax,color='k')
                    ax.legend()
                    # ax.set_ylim([8,17])
                    ax.set_title(mod)
                    # ax.set_ylabel('Yearly mean temperature over France (°C)')
                    plt.savefig(os.path.join(figs_folder,'{}_mod{}_evolution.png'.format(climate_var,mod)),bbox_inches='tight')
                    plt.show()
                    
                # concatenation des valeurs annuelles (temperature)
                if True and climate_var == 'tas':
                    # mean_yt_deg2 = 11.37
                    # mean_yt_deg4 = 13.25
                    
                    data_array = array.mean(('x','y'))
                    data_array = data_array.groupby('time.year').mean('time')
                    data = pd.DataFrame(index=data_array.year, data=data_array).rename(columns={0:climate_var})
                    
                    data.to_csv(os.path.join(os.path.join(output, folder),'{}_mod{}.csv'.format(climate_var,mod)))
                    
                    rolling_data = data.rolling(20,center=True).mean()
                    # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    # data.plot(ax=ax,color='k')
                    # ax.plot([data.index.min(),data.index.max()],[mean_yt_deg2]*2,color='tab:blue',label='+2°C')
                    # ax.plot([data.index.min(),data.index.max()],[mean_yt_deg4]*2,color='tab:red',label='+4°C')
                    # ax.legend()
                    # ax.set_ylim([8,17])
                    # ax.set_title('Climate model #{}'.format(mod))
                    # ax.set_ylabel('Yearly mean temperature over France (°C)')
                    # plt.savefig(os.path.join(figs_folder,'{}_mod{}_evolution.png'.format(climate_var,mod)),bbox_inches='tight')
                    # plt.show()
                    
                    # period_list = []
                    # mean_20_years_list = []
                    # for y0 in data.index[10:-10]:
                    #     data_20 = data.loc[y0-10:y0+10]
                    #     mean_20 = data_20.mean()
                    #     period = y0
                    #     period_list.append(period)
                    #     mean_20_years_list.append(mean_20)
                        
                    # mean_20_years_list = np.asarray(mean_20_years_list).T[0]
                    
                    # year_deg2 = period_list[next(x[0] for x in enumerate(mean_20_years_list) if x[1] > mean_yt_deg2)]
                    # year_deg4 = period_list[next(x[0] for x in enumerate(mean_20_years_list) if x[1] > mean_yt_deg4)]
                    
                    ref_temp_1976_2005 = data[data.index.isin(list(range(1976,2006)))].mean().values[0]
                    mean_yt_deg2 = ref_temp_1976_2005 + 1.4
                    mean_yt_deg27 = ref_temp_1976_2005 + 2.1
                    mean_yt_deg4 = ref_temp_1976_2005 + 3.4
                    
                    deg2_year = (data.rolling(20,center=True).mean()>=mean_yt_deg2).idxmax().values[0]
                    deg27_year = (data.rolling(20,center=True).mean()>=mean_yt_deg27).idxmax().values[0]
                    deg4_year = (data.rolling(20,center=True).mean()>=mean_yt_deg4).idxmax().values[0]
                    
                    print(models_dict.get(mod).get('historical_rcp85'))
                    print(2,deg2_year)
                    print(2.7,deg27_year)
                    print(4,deg4_year)
                    print()
                    
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    ax.plot(data.tas,color='k',alpha=0.2,label='Annual temperature')
                    ax.plot(rolling_data.tas,color='k',label='20-years rolling average')
                    # ax.plot(period_list,mean_20_years_list,color='k')
                    ax.fill_between([2000,2020],[17]*2,[8]*2,color='k',alpha=0.1)
                    ax.errorbar([deg2_year],[mean_yt_deg2],xerr=np.asarray([[10,9]]).T,color='gold',label='+2.0°C ({}-{})'.format(deg2_year-10,deg2_year+9),marker='o',mec='k',capsize=5,ls='')
                    ax.errorbar([deg27_year],[mean_yt_deg27],xerr=np.asarray([[10,9]]).T,color='tab:orange',label='+2.7°C ({}-{})'.format(deg27_year-10,deg27_year+9),marker='o',mec='k',capsize=5,ls='')
                    ax.errorbar([deg4_year],[mean_yt_deg4],xerr=np.asarray([[10,9]]).T,color='tab:red',label='+4.0°C ({}-{})'.format(deg4_year-10,deg4_year+9),marker='o',mec='k',capsize=5,ls='')
                    # ax.plot([deg27_year-10,deg27_year+9],[mean_yt_deg27]*2,color='tab:orange',label='+2.7°C ({}-{})'.format(deg27_year-10,deg27_year+9),)
                    # ax.plot([deg4_year-10,deg4_year+9],[mean_yt_deg4]*2,color='tab:red',label='+4°C ({}-{})'.format(deg4_year-10,deg4_year+9))
                    # ax.plot([year_deg2],[mean_yt_deg2],color='tab:blue',marker='o')
                    # ax.plot([year_deg4],[mean_yt_deg4],color='tab:red',marker='o')
                    ax.legend()
                    ax.set_ylim([8,17])
                    ax.set_xlim([1950,2100])
                    ax.set_title('Climate model #{} - RCP8.5'.format(mod))
                    ax.set_ylabel('Yearly mean temperature (°C)')
                    ax.set_xlabel('')
                    plt.savefig(os.path.join(figs_folder,'{}_mod{}_moving_mean_evolution.png'.format(climate_var,mod)),bbox_inches='tight')
                    plt.show()
                
                array.close()
                del array
    
    # Concaténation des données climatiques hist+rcp85
    if False:
        # calib_method = 'CDFt'
        calib_method = 'ADAMONT'

        if calib_method == 'CDFt':
            models_dict = {0:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19510101-20051231.nc',
                              # 'rcp45':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc',
                              'rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                           
                           1:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_MOHC-HadREM3-GA7-05_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                              'rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001230.nc'},
                           
                           2:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19500101-20051231.nc',
                              # 'rcp45':'Adjust_France_ICHEC-EC-EARTH_rcp45_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc',
                              'rcp85':'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                           
                           3:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                              'rcp85':'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                           
                           4:{'historical':'Adjust_France_MOHC-HadGEM2-ES_historical_r1i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                              'rcp85':'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-20991219.nc'},
                           }
        
        elif calib_method == 'ADAMONT':
            models_dict = {0:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19510101-20051231.nc',
                              'rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                              'historical_rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19510101-21001231.nc'},
                           
                           1:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                              'rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001230.nc',
                              'historical_rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-21001230.nc'},
                           
                           2:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19500101-20051231.nc',
                              'rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                              'historical_rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19500101-21001231.nc'},
                           
                           3:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                              'rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                              'historical_rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-21001231.nc'},
                           
                           4:{'historical':'Adjust_France_MOHC-HadGEM2-ES_historical_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                              'rcp85':     'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-20991219.nc',
                              'historical_rcp85':     'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20991219.nc'},
                           }
        
    
        
        if external_disk_connection:
            data_folder = '/media/amounier/MPBE/heavy_data/Explore2'
        else:
            data_folder = os.path.join('data','Explore2')
        
        
        climate_vars = ['tas','tasmax','tasmin','rsds']
        # climate_vars = ['tas']
        # climate_vars = ['tasmin']
        for climate_var in climate_vars:
            # print(climate_var)
            
            for idx,mod in enumerate(range(0,5)):
                
                # pas la mémoire pour tout faire d'un coup
                # if mod >= 2 :
                #     continue
            
                # if mod >= 4 :
                #     continue
                
                print(climate_var,mod)
                
                
                if climate_var+models_dict.get(mod).get('historical_rcp85') in os.listdir(data_folder):
                    continue
                
                Explore2_hist = os.path.join(data_folder,climate_var+models_dict.get(mod).get('historical'))
                array_hist = xr.open_dataset(Explore2_hist)
                array_hist = array_hist[list(array_hist.data_vars)[-1]]
                if 'tas' in climate_var:
                    array_hist = array_hist - 273.15
                array_hist.rio.write_crs('epsg:27572', inplace=True)
                
                Explore2_proj = os.path.join(data_folder,climate_var+models_dict.get(mod).get('rcp85'))
                array_proj = xr.open_dataset(Explore2_proj)
                array_proj = array_proj[list(array_proj.data_vars)[-1]]
                if 'tas' in climate_var:
                    array_proj = array_proj - 273.15
                array_proj.rio.write_crs('epsg:27572', inplace=True)
                
                geom = pd.Series(France().geometry).apply(mapping)
                array_hist = array_hist.rio.clip(geom, 'epsg:4326', drop=False)
                array_proj = array_proj.rio.clip(geom, 'epsg:4326', drop=False)
                
                array = xr.concat([array_hist,array_proj],dim='time')
                array.to_netcdf(os.path.join(data_folder,climate_var+models_dict.get(mod).get('historical_rcp85')),'w',format='NETCDF4')
                
                array_hist.close()
                array_proj.close()
                array.close()
                
                del array_hist
                del array_proj
                del array
                
    # Réalisation des cartes 
    if False:
        
        # calib_method = 'CDFt'
        calib_method = 'ADAMONT'

        if calib_method == 'CDFt':
            models_dict = {0:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19510101-20051231.nc',
                              # 'rcp45':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp45_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc',
                              'rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                           
                           1:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_MOHC-HadREM3-GA7-05_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                              'rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001230.nc'},
                           
                           2:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19500101-20051231.nc',
                              # 'rcp45':'Adjust_France_ICHEC-EC-EARTH_rcp45_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc',
                              'rcp85':'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                           
                           3:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                              'rcp85':'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-21001231.nc'},
                           
                           4:{'historical':'Adjust_France_MOHC-HadGEM2-ES_historical_r1i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_19520101-20051231.nc',
                              'rcp85':'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_LSCE-IPSL-CDFt-L-1V-0L-1976-2005_day_20060101-20991219.nc'},
                           }
        
        elif calib_method == 'ADAMONT':
            models_dict = {0:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19510101-20051231.nc',
                              'rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                              'historical_rcp85':'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CNRM-ALADIN63_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19510101-21001231.nc'},
                           
                           1:{'historical':'Adjust_France_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                              'rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001230.nc',
                              'historical_rcp85':     'Adjust_France_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v2_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-21001230.nc'},
                           
                           2:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19500101-20051231.nc',
                              'rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                              'historical_rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_KNMI-RACMO22E_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19500101-21001231.nc'},
                           
                           3:{'historical':'Adjust_France_ICHEC-EC-EARTH_historical_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                              'rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-21001231.nc',
                              'historical_rcp85':     'Adjust_France_ICHEC-EC-EARTH_rcp85_r12i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-21001231.nc'},
                           
                           4:{'historical':'Adjust_France_MOHC-HadGEM2-ES_historical_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20051231.nc',
                              'rcp85':     'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_20060101-20991219.nc',
                              'historical_rcp85':     'Adjust_France_MOHC-HadGEM2-ES_rcp85_r1i1p1_MOHC-HadREM3-GA7-05_v1_MF-ADAMONT-SAFRAN-1980-2011_day_19520101-20991219.nc'},
                           }
        
        # models_period_dict = {0:{2:[2020,2040],
        #                          4:[2059,2079],},
        #                       1:{2:[2013,2033],
        #                          4:[2054,2074],},
        #                       2:{2:[2017,2037],
        #                          4:[2061,2081],},
        #                       3:{2:[2006,2025],
        #                          4:[2047,2067],},
        #                       4:{2:[2006,2021], # debut des projections en 2006
        #                          4:[2040,2060],},}
        
        # nouvelles périodes 
        models_period_dict = {0:{2:  [2034,2053],
                                 2.7:[2046,2065],
                                 4:  [2072,2091],},
                              1:{2:  [2027,2046],
                                 2.7:[2042,2061],
                                 4:  [2063,2082],},
                              2:{2:  [2029,2048],
                                 2.7:[2037,2056],
                                 4:  [2069,2088],},
                              3:{2:  [2018,2037],
                                 2.7:[2033,2052],
                                 4:  [2060,2079],},
                              4:{2:  [2008,2027], 
                                 2.7:[2024,2043],
                                 4:  [2049,2068],},}
        
        # print(pd.DataFrame().from_dict(models_period_dict).T.to_latex())
        
        if external_disk_connection:
            data_folder = '/media/amounier/MPBE/heavy_data/Explore2'
        else:
            data_folder = os.path.join('data','Explore2')
            
        
        # carte par modèle climatique
        if False:
            climate_vars = ['tas','tasmax','tasmin','rsds']
            climate_vars = ['rsds']
            for climate_var in climate_vars:
                
                for idx,mod in enumerate([3,0,1,2,4]):
                # for idx,mod in enumerate(range(1,2)):
                    
                    # Explore2_hist = os.path.join(data_folder,climate_var+models_dict.get(mod).get('historical'))
                    # array_hist = xr.open_dataset(Explore2_hist)
                    # array_hist = array_hist[list(array_hist.data_vars)[-1]]
                    # if 'tas' in climate_var:
                    #     array_hist = array_hist - 273.15
                    # array_hist.rio.write_crs('epsg:27572', inplace=True)
                    
                    # Explore2_proj = os.path.join(data_folder,climate_var+models_dict.get(mod).get('rcp85'))
                    # array_proj = xr.open_dataset(Explore2_proj)
                    # array_proj = array_proj[list(array_proj.data_vars)[-1]]
                    # if 'tas' in climate_var:
                    #     array_proj = array_proj - 273.15
                    # array_proj.rio.write_crs('epsg:27572', inplace=True)
                    
                    Explore2 = os.path.join(data_folder,climate_var+models_dict.get(mod).get('historical_rcp85'))
                    array = xr.open_dataset(Explore2)
                    array = array[list(array.data_vars)[-1]]
                    array.rio.write_crs('epsg:27572', inplace=True)
                    
                    geom = pd.Series(France().geometry).apply(mapping)
                    array = array.rio.clip(geom, 'epsg:4326', drop=False)
                    
                    hist = array.sel(time=slice("2000-01-01", "2020-12-31")).mean('time')
                    
                    proj2_period = models_period_dict.get(mod).get(2)
                    proj27_period = models_period_dict.get(mod).get(2.7)
                    proj4_period = models_period_dict.get(mod).get(4)
                    
                    proj2 = array.sel(time=slice("{}-01-01".format(proj2_period[0]), "{}-12-31".format(proj2_period[1]))).mean('time')
                    proj27 = array.sel(time=slice("{}-01-01".format(proj27_period[0]), "{}-12-31".format(proj27_period[1]))).mean('time')
                    proj4 = array.sel(time=slice("{}-01-01".format(proj4_period[0]), "{}-12-31".format(proj4_period[1]))).mean('time')
                    
                    diff2 = proj2 - hist
                    diff27 = proj27 - hist
                    diff4 = proj4 - hist
                    
                    array.close()
                    del array
                    
                    # etat de base
                    if 'tas' in climate_var:
                        cmap = cmocean.cm.thermal
                    else:
                        cmap = cmocean.cm.solar
        
                    # carte de base
                    fig,ax = blank_national_map()
                    
                    if idx == 0:
                        vmin = hist.min()
                        vmax = hist.max()
                        
                        max_val = max(np.abs(diff4.max()), np.abs(diff4.min()))
                        
                    
                    img = hist.plot(ax=ax,transform=ccrs.epsg('27572'),add_colorbar=False,
                                   cmap=cmap,vmin=vmin,vmax=vmax)
                    
                    ax.set_title('Climate model #{} (2000-2020)'.format(mod))
                    
                    ax_cb = fig.add_axes([0,0,0.1,0.1])
                    posn = ax.get_position()
                    ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                    fig.add_axes(ax_cb)
                    cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
                    
                    cbar_label_dict = {'tas':'Mean daily temperature (°C)',
                                       'tasmax':'Mean of daily maximal temperature (°C)',
                                       'tasmin':'Mean of daily minimal temperature (°C)',
                                       'rsds':'Surface downwelling shortwave radiation (W.m$^{-2}$)'}
                    cbar.set_label(cbar_label_dict.get(climate_var))
                    
                    ax.set_extent(get_extent())
                    plt.savefig(os.path.join(figs_folder,'map_{}_mod{}_2000-2020.png'.format(climate_var,mod)),bbox_inches='tight')
                    plt.show()
                    plt.close()
                    
                    # difference sur les periode 2 et 4
                    cbar_label_diff_dict = {'tas':'Difference of daily mean temperature compared to 2000-2020 (°C)',
                                            'tasmax':'Difference of daily maximal temperature (°C)',
                                            'tasmin':'Difference of daily minimal temperature (°C)',
                                            'rsds':'Difference of daily RSDS (W.m$^{-2}$)'}
                    
                    cmap = cmocean.cm.balance
                    
                    # +2°C
                    fig,ax = blank_national_map()
                    
                    
                    img = diff2.plot(ax=ax,transform=ccrs.epsg('27572'),add_colorbar=False,
                                     cmap=cmap,vmin=-max_val,vmax=max_val)
                    
                    ax.set_title('Climate model #{} (+2°C)'.format(mod))
                    
                    ax_cb = fig.add_axes([0,0,0.1,0.1])
                    posn = ax.get_position()
                    ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                    fig.add_axes(ax_cb)
                    cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
                    cbar.set_label(cbar_label_diff_dict.get(climate_var))
                    
                    ax.set_extent(get_extent())
                    plt.savefig(os.path.join(figs_folder,'map_{}_mod{}_2deg.png'.format(climate_var,mod)),bbox_inches='tight')
                    plt.show()
                    plt.close()
                    
                    
                    # +2.7°C
                    fig,ax = blank_national_map()
                    
                    
                    img = diff27.plot(ax=ax,transform=ccrs.epsg('27572'),add_colorbar=False,
                                      cmap=cmap,vmin=-max_val,vmax=max_val)
                    
                    ax.set_title('Climate model #{} (+2.7°C)'.format(mod))
                    
                    ax_cb = fig.add_axes([0,0,0.1,0.1])
                    posn = ax.get_position()
                    ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                    fig.add_axes(ax_cb)
                    cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
                    cbar.set_label(cbar_label_diff_dict.get(climate_var))
                    
                    ax.set_extent(get_extent())
                    plt.savefig(os.path.join(figs_folder,'map_{}_mod{}_27deg.png'.format(climate_var,mod)),bbox_inches='tight')
                    plt.show()
                    plt.close()
                    
                    
                    # +4°C
                    fig,ax = blank_national_map()
                    
                    img = diff4.plot(ax=ax,transform=ccrs.epsg('27572'),add_colorbar=False,
                                     cmap=cmap,vmin=-max_val,vmax=max_val)
                    
                    ax.set_title('Climate model #{} (+4°C)'.format(mod))
                    
                    ax_cb = fig.add_axes([0,0,0.1,0.1])
                    posn = ax.get_position()
                    ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                    fig.add_axes(ax_cb)
                    cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
                    cbar.set_label(cbar_label_diff_dict.get(climate_var))
                    
                    ax.set_extent(get_extent())
                    plt.savefig(os.path.join(figs_folder,'map_{}_mod{}_4deg.png'.format(climate_var,mod)),bbox_inches='tight')
                    plt.show()
                    plt.close()
                    
                    del hist
                    del proj2
                    del proj27
                    del proj4
                    del diff2
                    del diff27
                    del diff4
            
        
        # enregistrement d'un netCDF moyen des 5 modèles
        if False:
            climate_vars = ['tas','tasmax','tasmin','rsds']
            climate_vars = ['rsds']
            
            for climate_var in climate_vars:
                print(climate_var)
                
                mean_array = None
                
                for idx,mod in enumerate(range(0,5)):
                    
                    Explore2 = os.path.join(data_folder,climate_var+models_dict.get(mod).get('historical_rcp85'))
                    array = xr.open_dataset(Explore2)
                    array = array[list(array.data_vars)[-1]]
                    array.rio.write_crs('epsg:27572', inplace=True)
                    
                    geom = pd.Series(France().geometry).apply(mapping)
                    array = array.rio.clip(geom, 'epsg:4326', drop=False)
                    
                    if mean_array is None:
                        mean_array = array
                    else:
                        mean_array = mean_array + array
                    
                    del array
                    
                mean_array = mean_array/5
                mean_array.to_netcdf(os.path.join(data_folder,'{}Adjust_France_meanModels_MF-ADAMONT-SAFRAN-1980-2011_day_1950-2100.nc'.format(climate_var)),'w',format='NETCDF4')
                
                del mean_array
                
        # carte moyenne des 5 modèles
        if True:
            climate_vars = ['tas','rsds']
            climate_vars = ['tas']
            
            for climate_var in climate_vars:
                
                Explore2 = os.path.join(data_folder,'{}Adjust_France_meanModels_MF-ADAMONT-SAFRAN-1980-2011_day_1950-2100.nc'.format(climate_var))
                array = xr.open_dataset(Explore2)
                array = array[list(array.data_vars)[-1]]
                array.rio.write_crs('epsg:27572', inplace=True)
                
                geom = pd.Series(France().geometry).apply(mapping)
                array = array.rio.clip(geom, 'epsg:4326', drop=False)
                
                
                hist = array.sel(time=slice("2000-01-01", "2020-12-31"))
                hist = hist.mean('time')
                
        
                # carte de base
                if False:
                    
                    # etat de base
                    if 'tas' in climate_var:
                        cmap = cmocean.cm.thermal
                    else:
                        cmap = cmocean.cm.solar
                        
                    fig,ax = blank_national_map()
                    # climats = [Climat_winter(e) for e in France().climats_winter]
                    # fig,ax = draw_climat_map({c:None for c in climats}, figs_folder=figs_folder,
                    #                          border_color='w',lw=1.,add_legend=False)
                    
                    vmin = hist.min()
                    vmax = hist.max()
                        
                    
                    img = hist.plot(ax=ax,transform=ccrs.epsg('27572'),add_colorbar=False,
                                   cmap=cmap,vmin=vmin,vmax=vmax)
                    
                    ax.set_title('Reference period (2000-2020)')
                    
                    ax_cb = fig.add_axes([0,0,0.1,0.1])
                    posn = ax.get_position()
                    ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                    fig.add_axes(ax_cb)
                    cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
                    
                    cbar_label_dict = {'tas':'Annual average of daily mean temperature (°C)',
                                       'tasmax':'Mean of daily maximal temperature (°C)',
                                       'tasmin':'Mean of daily minimal temperature (°C)',
                                       'rsds':'Surface downwelling shortwave radiation (W.m$^{-2}$)'}
                    cbar.set_label(cbar_label_dict.get(climate_var))
                    
                    ax.set_extent(get_extent())
                    plt.savefig(os.path.join(figs_folder,'map_{}_meanModels_2000-2020.png'.format(climate_var)),bbox_inches='tight')
                    plt.show()
                    plt.close()
                
                # carte des difference à +2, +4
                if False:
                    # ne marche pas, il faut trouver une période moyenne
                    proj2_period = models_period_dict.get(mod).get(2)
                    proj27_period = models_period_dict.get(mod).get(2.7)
                    proj4_period = models_period_dict.get(mod).get(4)
                    
                    proj2 = array.sel(time=slice("{}-01-01".format(proj2_period[0]), "{}-12-31".format(proj2_period[1]))).mean('time')
                    proj27 = array.sel(time=slice("{}-01-01".format(proj27_period[0]), "{}-12-31".format(proj27_period[1]))).mean('time')
                    proj4 = array.sel(time=slice("{}-01-01".format(proj4_period[0]), "{}-12-31".format(proj4_period[1]))).mean('time')
                    
                    diff2 = proj2 - hist
                    diff27 = proj27 - hist
                    diff4 = proj4 - hist
                    
                    cmap = cmocean.cm.balance
                    
                    fig,ax = blank_national_map()
                    
                    vmin = diff4.min()
                    vmax = diff4.max()
                    vmax = max(abs(vmin),vmax)
                    
                    img = diff4.plot(ax=ax,transform=ccrs.epsg('27572'),add_colorbar=False,
                                   cmap=cmap,vmin=-vmax,vmax=vmax)
                    
                    ax.set_title('+4°C period')
                    
                    ax_cb = fig.add_axes([0,0,0.1,0.1])
                    posn = ax.get_position()
                    ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
                    fig.add_axes(ax_cb)
                    cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
                    
                    cbar_label_diff_dict = {'tas':'Difference of daily temperature (°C)',
                                            'tasmax':'Difference of daily maximal temperature (°C)',
                                            'tasmin':'Difference of daily minimal temperature (°C)',
                                            'rsds':'RSDS difference (W.m$^{-2}$)'}
                    cbar.set_label(cbar_label_diff_dict.get(climate_var))
                    
                    ax.set_extent(get_extent())
                    plt.savefig(os.path.join(figs_folder,'map_{}_4deg.png'.format(climate_var)),bbox_inches='tight')
                    plt.show()
                    plt.close()
                
                
                del array
    

            
    #%% Détermination des périodes de +2, +4 degrés d'après les données de l'atlas interactif
    # ancienne méthode : problème
    if False:
        
        deg2_data = xr.open_dataset(os.path.join('data','CORDEX','CORDEX Europe - Mean temperature (T) deg C - Warming 2°C RCP8.5 - Annual (47 models)','map.nc'))
        deg2_data = deg2_data.tas
        deg2_data.rio.write_crs('epsg:4326', inplace=True)
        
        deg4_data = xr.open_dataset(os.path.join('data','CORDEX','CORDEX Europe - Mean temperature (T) deg C - Warming 4°C RCP8.5 - Annual (39 models)','map.nc'))
        deg4_data = deg4_data.tas
        deg4_data.rio.write_crs('epsg:4326', inplace=True)
        
        geom = pd.Series(France().geometry).apply(mapping)
        deg2_data_cropped = deg2_data.rio.clip(geom, 'epsg:4326', drop=False)
        deg4_data_cropped = deg4_data.rio.clip(geom, 'epsg:4326', drop=False)
        
        mean_yt_deg2 = deg2_data_cropped.mean()
        mean_yt_deg4 = deg4_data_cropped.mean()
        
        print('Monde 2°C',float(deg2_data.mean()))
        print('France 2°C',float(mean_yt_deg2))
        
        print('Monde 4°C',float(deg4_data.mean()))
        print('France 4°C',float(mean_yt_deg4))
        
        
        if False:
            cmap = matplotlib.colormaps.get_cmap('viridis')
            cmap = cmocean.cm.thermal
            
            fig,ax = blank_national_map()
            img = deg2_data_cropped.plot(ax=ax,transform=ccrs.PlateCarree(),add_colorbar=False,
                                         cmap=cmap,vmin=-3,vmax=16)
            
            ax.set_title('Yearly mean temperature over France : {:.2f}°C'.format(mean_yt_deg2))
            
            ax_cb = fig.add_axes([0,0,0.1,0.1])
            posn = ax.get_position()
            ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
            fig.add_axes(ax_cb)
            cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
            cbar.set_label('Yearly mean temperature (°C) - Global warming +2°C')
            
            ax.set_extent(get_extent())
            plt.savefig(os.path.join(figs_folder,'mean_yearly_temperature_france_gw2deg.png'), bbox_inches='tight')
            plt.show()
            
            
            fig,ax = blank_national_map()
            img = deg4_data_cropped.plot(ax=ax,transform=ccrs.PlateCarree(),add_colorbar=False,
                                         cmap=cmap,vmin=-3,vmax=16)
            
            
            ax.set_title('Yearly mean temperature over France : {:.2f}°C'.format(mean_yt_deg4))
            
            ax_cb = fig.add_axes([0,0,0.1,0.1])
            posn = ax.get_position()
            ax_cb.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
            fig.add_axes(ax_cb)
            cbar = plt.colorbar(img,cax=ax_cb,extendfrac=0.02)
            cbar.set_label('Yearly mean temperature (°C) - Global warming +4°C')
            
            ax.set_extent(get_extent())
            plt.savefig(os.path.join(figs_folder,'mean_yearly_temperature_france_gw4deg.png'), bbox_inches='tight')
            plt.show()
            
            
            
    # %% Test des projections climatiques propres
    if False:
        zcl_list = France().climats
        mod_list = list(range(5))
        
        run_list = []
        for zcl_code in zcl_list:
            for mod in mod_list:
                run_list.append((zcl_code, mod))
        
        # ne marche pas je crois (ram ?)
        # nb_cpu = multiprocessing.cpu_count()
        # pool = multiprocessing.Pool(nb_cpu)
        # pool.starmap(get_projected_weather_data, run_list)
        
        for run in run_list:
            print(run)
            compute_projected_weather_data(*run)
        
        # explore2 = get_projected_weather_data(zcl_code=zcl_code, period=[2090,2090],nmod=mod)
        # print(explore2)
        
    if False:
        test = get_projected_weather_data('H3',[2075,2075])
        agg = aggregate_resolution(test[['direct_sun_radiation_H']],resolution='D',agg_method='mean')
        print(agg.direct_sun_radiation_H.plot())
    
    # caractérisation des évolutions de températures
    if True:
        zcl = Climat('H1a')
        var = 'temperature_2m'
        # var = 'direct_sun_radiation_H'
        period=[2000,2100]
        agg_period = 'YS'
        
        # data = get_historical_weather_data(zcl.center_prefecture, period)
        data = get_safran_hourly_weather_data(zcl.code,period)
        data = aggregate_resolution(data[[var]],resolution=agg_period,agg_method='mean')
        data = data.rename(columns={var:'SAFRAN'})
        
        for nmod in range(5):
            explore2 = get_projected_weather_data('H1a', period,nmod=nmod)
            explore2 = aggregate_resolution(explore2[[var]],resolution=agg_period,agg_method='mean')
            explore2 = explore2.rename(columns={var:'Explore2 - mod n°{}'.format(nmod+1)})
            data = data.join(explore2,how='outer')
        
        cmap = matplotlib.colormaps.get_cmap('viridis')
        fig,ax = plot_timeserie(data, show=False,figsize=(8,5),
                       colors=['k']+[cmap(i/5) for i in range(5)],
                       linestyles=['-']+[':','--','-.',':','--'], 
                       ylabel=var)
        plt.show()
        
        # fort écart sur les flux solaire: à comparer avec les données d'observation MF
        
        
    # graphe des évolutions de température hivernales et estivales par période et zone climatique
    if False:
        models_period_dict = {0:{2:  [2034,2053],
                                 2.7:[2046,2065],
                                 4:  [2072,2091],},
                              1:{2:  [2027,2046],
                                 2.7:[2042,2061],
                                 4:  [2063,2082],},
                              2:{2:  [2029,2048],
                                 2.7:[2037,2056],
                                 4:  [2069,2088],},
                              3:{2:  [2018,2037],
                                 2.7:[2033,2052],
                                 4:  [2060,2079],},
                              4:{2:  [2008,2027], 
                                 2.7:[2024,2043],
                                 4:  [2049,2068],},}
        
        for mod in range(5):
            models_period_dict[mod].update({0:[2000,2020]})
        
        zcl_list = France().climats
        temperature_dict = dict()
        
        if False:
            for mod in range(5):
                for zcl in zcl_list:
                    weather = compute_projected_weather_data(zcl, mod)
                    for period in [0,2,2.7,4]:
                        p = models_period_dict.get(mod).get(period)
                        weather_period = weather[weather.index.year.isin(list(range(p[0],p[1]+1)))]
                        weather_period = aggregate_resolution(weather_period[['temperature_2m']],resolution='D',agg_method='mean')
                        
                        temperature_dict[(mod,zcl,period,'ALL')] = weather_period.temperature_2m.to_list()
                        temperature_dict[(mod,zcl,period,'DJF')] = weather_period[weather_period.index.month.isin([12,1,2])].temperature_2m.to_list()
                        temperature_dict[(mod,zcl,period,'JJA')] = weather_period[weather_period.index.month.isin([6,7,8])].temperature_2m.to_list()
                        
            pickle.dump(temperature_dict, open('.temperature_dict.pickle', "wb"))
        else:
            temperature_dict = pickle.load(open('.temperature_dict.pickle', 'rb'))
        
        # affichage des graphes par modèles 
        if False:
            # mod = 1
            models_dict = {0:'CNRM-CM5_ALADIN63',
                           1:'CNRM-CM5_HadREM3-GA7',
                           2:'EC-EARTH_HadREM3-GA7',
                           3:'EC-EARTH_RACMO22E',
                           4:'HadGEM2-ES_HadREM3-GA7'}
            
            for mod in range(5):
                fig,ax = plt.subplots(figsize=(10,5),dpi=300)
                for idx,zcl in enumerate(zcl_list):
                    
                    j = idx*7
                    X = [j,j+2,j+4]
                        
                    for k,period in enumerate([0,2,4]):
                        if idx==1 and k==1:
                            label_jja ='JJA'
                            label_djf ='DJF'
                        else:
                            label_jja = None
                            label_djf = None
                            
                        jja_color = 'tab:red'
                        djf_color = 'tab:blue'
                        
                        bplot_jja = ax.boxplot(temperature_dict[(mod,zcl,period,'JJA')],positions=[X[k]-0.2],
                                               widths=0.5,label=label_jja,
                                               patch_artist=True,
                                               boxprops=dict(color=jja_color),
                                               capprops=dict(color=jja_color),
                                               whiskerprops=dict(color=jja_color),
                                               flierprops=dict(markeredgecolor=jja_color,markersize=2),
                                               medianprops=dict(color=jja_color),)
                        
                        for box in bplot_jja['boxes']:
                            box.set(facecolor='w',lw=1.5)
                        
                        bplot_djf = ax.boxplot(temperature_dict[(mod,zcl,period,'DJF')],positions=[X[k]+0.2],
                                               widths=0.5,label=label_djf,
                                               patch_artist=True,
                                               boxprops=dict(color=djf_color),
                                               capprops=dict(color=djf_color),
                                               whiskerprops=dict(color=djf_color),
                                               flierprops=dict(markeredgecolor=djf_color,markersize=2),
                                               medianprops=dict(color=djf_color),)
                        
                        for box in bplot_djf['boxes']:
                            box.set(facecolor='w',lw=1.5)
                    
                    list_temp_jja = [np.mean(temperature_dict[(mod,zcl,p,'JJA')]) for p in [0,2,4]]
                    list_temp_djf = [np.mean(temperature_dict[(mod,zcl,p,'DJF')]) for p in [0,2,4]]
                    
                    ax.plot(X,list_temp_jja,color=jja_color,zorder=-1,alpha=0.7)
                    ax.plot(X,list_temp_djf,color=djf_color,zorder=-1,alpha=0.7)
        
                # ax.set_ylim(bottom=0.)
                ax.set_ylabel('Mean temperature (°C)')
                ax.legend()
                ax.set_title(models_dict.get(mod).replace('_',' - '))
                ax.set_xticks([(i*7)+2 for i in range(len(zcl_list))],zcl_list)
                
                ylims = ax.get_ylim()
                ylims = [-15,40]
                xlims = ax.get_xlim()
                for i in range(len(zcl_list)):
                    j = i*7
                    X = [j-1.5,j+2,j+5.5]
                    if i%2==0:
                        ax.fill_between(X,[ylims[1]]*3,[ylims[0]]*3,color='lightgrey',alpha=0.37,zorder=-2)
                
                ax.set_xlim([xlims[0]-0.5,xlims[-1]+0.5])
                ax.set_ylim(ylims)
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('temperature_evolution_zcl_nmod{}'.format(mod))),bbox_inches='tight')
                
                plt.show()
                
        # affichage des moyennes des modèles
        if True:
            fig,ax = plt.subplots(figsize=(10,5),dpi=300)
            for idx,zcl in enumerate(zcl_list):
                
                j = idx*7
                X = [j,j+2,j+4]
                    
                for k,period in enumerate([0,2,4]):
                    if idx==1 and k==1:
                        label_jja ='JJA'
                        label_djf ='DJF'
                    else:
                        label_jja = None
                        label_djf = None
                        
                    jja_color = 'tab:red'
                    djf_color = 'tab:blue'
                    
                    list_jja = [np.mean(temperature_dict[(m,zcl,period,'JJA')]) for m in range(5)]
                    
                    ax.errorbar([X[k]-0.2],np.mean(list_jja),
                                yerr=np.asarray([[np.mean(list_jja)-np.min(list_jja),np.max(list_jja)-np.mean(list_jja)]]).T,
                                marker='o',mec=jja_color,mfc='w',ls='',ecolor=jja_color,capsize=2,
                                label=label_jja,)
                    
                    list_djf = [np.mean(temperature_dict[(m,zcl,period,'DJF')]) for m in range(5)]
                    
                    ax.errorbar([X[k]+0.2],np.mean(list_djf),
                                yerr=np.asarray([[np.mean(list_djf)-np.min(list_djf),np.max(list_djf)-np.mean(list_djf)]]).T,
                                marker='o',mec=djf_color,mfc='w',ls='',ecolor=djf_color,capsize=2,
                                label=label_djf,)
                
                list_temp_jja = [np.mean([np.mean(temperature_dict[(m,zcl,p,'JJA')]) for m in range(5)]) for p in [0,2,4]]
                list_temp_djf = [np.mean([np.mean(temperature_dict[(m,zcl,p,'DJF')]) for m in range(5)]) for p in [0,2,4]]
                
                ax.plot(X,list_temp_jja,color=jja_color,zorder=-1,alpha=0.7)
                ax.plot(X,list_temp_djf,color=djf_color,zorder=-1,alpha=0.7)
    
            # ax.set_ylim(bottom=0.)
            ax.set_ylabel('Mean temperature (°C)')
            ax.legend()
            ax.set_title('')
            ax.set_xticks([(i*7)+2 for i in range(len(zcl_list))],zcl_list)
            
            ylims = ax.get_ylim()
            # ylims = [-15,40]
            xlims = ax.get_xlim()
            for i in range(len(zcl_list)):
                j = i*7
                X = [j-1.5,j+2,j+5.5]
                if i%2==0:
                    ax.fill_between(X,[ylims[1]]*3,[ylims[0]]*3,color='lightgrey',alpha=0.37,zorder=-2)
            
            ax.set_xlim([-1.5,54.5])
            ax.set_ylim(ylims)
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('temperature_evolution_zcl')),bbox_inches='tight')
            
            plt.show()
            
    # idem pour les flux solaires
    if False:
        models_period_dict = {0:{2:  [2034,2053],
                                 2.7:[2046,2065],
                                 4:  [2072,2091],},
                              1:{2:  [2027,2046],
                                 2.7:[2042,2061],
                                 4:  [2063,2082],},
                              2:{2:  [2029,2048],
                                 2.7:[2037,2056],
                                 4:  [2069,2088],},
                              3:{2:  [2018,2037],
                                 2.7:[2033,2052],
                                 4:  [2060,2079],},
                              4:{2:  [2008,2027], 
                                 2.7:[2024,2043],
                                 4:  [2049,2068],},}
        
        for mod in range(5):
            models_period_dict[mod].update({0:[2000,2020]})
        
        zcl_list = France().climats
        normal_direct_radiation_dict = dict()
        
        if False:
            for mod in range(5):
                for zcl in zcl_list:
                    weather = compute_projected_weather_data(zcl, mod)
                    for period in [0,2,2.7,4]:
                        p = models_period_dict.get(mod).get(period)
                        weather_period = weather[weather.index.year.isin(list(range(p[0],p[1]+1)))]
                        weather_period = aggregate_resolution(weather_period[['direct_normal_irradiance_instant']],resolution='D',agg_method='mean')
                        
                        normal_direct_radiation_dict[(mod,zcl,period,'ALL')] = weather_period.direct_normal_irradiance_instant.to_list()
                        normal_direct_radiation_dict[(mod,zcl,period,'DJF')] = weather_period[weather_period.index.month.isin([12,1,2])].direct_normal_irradiance_instant.to_list()
                        normal_direct_radiation_dict[(mod,zcl,period,'JJA')] = weather_period[weather_period.index.month.isin([6,7,8])].direct_normal_irradiance_instant.to_list()
                        
            pickle.dump(normal_direct_radiation_dict, open('.normal_direct_radiation_dict.pickle', "wb"))
        else:
            normal_direct_radiation_dict = pickle.load(open('.normal_direct_radiation_dict.pickle', 'rb'))
        
        # affichage des graphes par modèles 
        if True:
            # mod = 1
            models_dict = {0:'CNRM-CM5_ALADIN63',
                           1:'CNRM-CM5_HadREM3-GA7',
                           2:'EC-EARTH_HadREM3-GA7',
                           3:'EC-EARTH_RACMO22E',
                           4:'HadGEM2-ES_HadREM3-GA7'}
            
            for mod in range(5):
                fig,ax = plt.subplots(figsize=(10,5),dpi=300)
                for idx,zcl in enumerate(zcl_list):
                    
                    j = idx*7
                    X = [j,j+2,j+4]
                        
                    for k,period in enumerate([0,2,4]):
                        if idx==1 and k==1:
                            label_jja ='JJA'
                            label_djf ='DJF'
                        else:
                            label_jja = None
                            label_djf = None
                            
                        jja_color = 'tab:red'
                        djf_color = 'tab:blue'
                        
                        bplot_jja = ax.boxplot(normal_direct_radiation_dict[(mod,zcl,period,'JJA')],positions=[X[k]-0.2],
                                               widths=0.5,label=label_jja,
                                               patch_artist=True,
                                               boxprops=dict(color=jja_color),
                                               capprops=dict(color=jja_color),
                                               whiskerprops=dict(color=jja_color),
                                               flierprops=dict(markeredgecolor=jja_color,markersize=2),
                                               medianprops=dict(color=jja_color),)
                        
                        for box in bplot_jja['boxes']:
                            box.set(facecolor='w',lw=1.5)
                        
                        bplot_djf = ax.boxplot(normal_direct_radiation_dict[(mod,zcl,period,'DJF')],positions=[X[k]+0.2],
                                               widths=0.5,label=label_djf,
                                               patch_artist=True,
                                               boxprops=dict(color=djf_color),
                                               capprops=dict(color=djf_color),
                                               whiskerprops=dict(color=djf_color),
                                               flierprops=dict(markeredgecolor=djf_color,markersize=2),
                                               medianprops=dict(color=djf_color),)
                        
                        for box in bplot_djf['boxes']:
                            box.set(facecolor='w',lw=1.5)
                    
                    list_temp_jja = [np.mean(normal_direct_radiation_dict[(mod,zcl,p,'JJA')]) for p in [0,2,4]]
                    list_temp_djf = [np.mean(normal_direct_radiation_dict[(mod,zcl,p,'DJF')]) for p in [0,2,4]]
                    
                    ax.plot(X,list_temp_jja,color=jja_color,zorder=-1,alpha=0.7)
                    ax.plot(X,list_temp_djf,color=djf_color,zorder=-1,alpha=0.7)
        
                # ax.set_ylim(bottom=0.)
                ax.set_ylabel('Mean direct normal radiation (W.m$^{-2}$)')
                ax.legend()
                ax.set_title(models_dict.get(mod).replace('_',' - '))
                ax.set_xticks([(i*7)+2 for i in range(len(zcl_list))],zcl_list)
                
                ylims = ax.get_ylim()
                ylims = [0,700]
                xlims = ax.get_xlim()
                for i in range(len(zcl_list)):
                    j = i*7
                    X = [j-1.5,j+2,j+5.5]
                    if i%2==0:
                        ax.fill_between(X,[ylims[1]]*3,[ylims[0]]*3,color='lightgrey',alpha=0.37,zorder=-2)
                
                ax.set_xlim([xlims[0]-0.5,xlims[-1]+0.5])
                ax.set_ylim(ylims)
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('radiation_evolution_zcl_nmod{}'.format(mod))),bbox_inches='tight')
                
                plt.show()
                
        # affichage des moyennes des modèles
        if False:
            fig,ax = plt.subplots(figsize=(10,5),dpi=300)
            for idx,zcl in enumerate(zcl_list):
                
                j = idx*7
                X = [j,j+2,j+4]
                    
                for k,period in enumerate([0,2,4]):
                    if idx==1 and k==1:
                        label_jja ='JJA'
                        label_djf ='DJF'
                    else:
                        label_jja = None
                        label_djf = None
                        
                    jja_color = 'tab:red'
                    djf_color = 'tab:blue'
                    
                    list_jja = [np.mean(normal_direct_radiation_dict[(m,zcl,period,'JJA')]) for m in range(5)]
                    
                    ax.errorbar([X[k]-0.2],np.mean(list_jja),
                                yerr=np.asarray([[np.mean(list_jja)-np.min(list_jja),np.max(list_jja)-np.mean(list_jja)]]).T,
                                marker='o',mec=jja_color,mfc='w',ls='',ecolor=jja_color,capsize=2,
                                label=label_jja,)
                    
                    list_djf = [np.mean(normal_direct_radiation_dict[(m,zcl,period,'DJF')]) for m in range(5)]
                    
                    ax.errorbar([X[k]+0.2],np.mean(list_djf),
                                yerr=np.asarray([[np.mean(list_djf)-np.min(list_djf),np.max(list_djf)-np.mean(list_djf)]]).T,
                                marker='o',mec=djf_color,mfc='w',ls='',ecolor=djf_color,capsize=2,
                                label=label_djf,)
                
                list_temp_jja = [np.mean([np.mean(normal_direct_radiation_dict[(m,zcl,p,'JJA')]) for m in range(5)]) for p in [0,2,4]]
                list_temp_djf = [np.mean([np.mean(normal_direct_radiation_dict[(m,zcl,p,'DJF')]) for m in range(5)]) for p in [0,2,4]]
                
                ax.plot(X,list_temp_jja,color=jja_color,zorder=-1,alpha=0.7)
                ax.plot(X,list_temp_djf,color=djf_color,zorder=-1,alpha=0.7)
    
            # ax.set_ylim(bottom=0.)
            ax.set_ylabel('Mean direct normal radiation (W.m$^{-2}$)')
            ax.legend()
            ax.set_title('')
            ax.set_xticks([(i*7)+2 for i in range(len(zcl_list))],zcl_list)
            
            ylims = ax.get_ylim()
            # ylims = [-15,40]
            xlims = ax.get_xlim()
            for i in range(len(zcl_list)):
                j = i*7
                X = [j-1.5,j+2,j+5.5]
                if i%2==0:
                    ax.fill_between(X,[ylims[1]]*3,[ylims[0]]*3,color='lightgrey',alpha=0.37,zorder=-2)
            
            ax.set_xlim([-1.5,54.5])
            ax.set_ylim(ylims)
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('radiation_evolution_zcl')),bbox_inches='tight')
            
            plt.show()
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()

