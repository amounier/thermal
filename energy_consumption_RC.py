#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:08:40 2024

@author: amounier
"""


import time 
import os 
import numpy as np
import tqdm
from datetime import date
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.linalg import expm
from numpy.linalg import inv

# Défintion de la date du jour
today = pd.Timestamp(date.today()).strftime('%Y%m%d')

# Défintion des dossiers de sortie 
output = 'output'
folder = '{}_RC_models'.format(today)
figs_folder = os.path.join(output, folder, 'figs')



def get_coordinates(city):
    """
    Récupération des coordonnées d'une ville via l'API OSM. 

    Parameters
    ----------
    city : str
        DESCRIPTION.

    Returns
    -------
    longitude : float
        DESCRIPTION.
    latitude : float
        DESCRIPTION.

    """
    coordinates_dict = {'Paris':(2.352222, 48.856614),
                        'Marseille':(5.369780, 43.296482)
                       }
    
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:129.0) Gecko/20100101 Firefox/129.0',
               'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8'}
    try:
        url = 'https://nominatim.openstreetmap.org/search?q={},France&format=json'.format(city)
        response = requests.get(url, headers=headers)
        json_data = response.json()
        longitude = float(json_data[0].get('lon'))
        latitude = float(json_data[0].get('lat'))
    except (requests.exceptions.ConnectionError, requests.exceptions.JSONDecodeError):
        try:
            longitude, latitude = coordinates_dict[city]
        except KeyError:
            raise KeyError('No internet connexion, only availables cities are : {}'.format(', '.join(list(coordinates_dict.keys()))))
    return longitude, latitude

    
def get_open_meteo_url(longitude, latitude, year, hourly_variables):
    """
    Récupération de l'url de l'API Open-Météo

    Parameters
    ----------
    longitude : float
        DESCRIPTION.
    latitude : float
        DESCRIPTION.
    year : int
        DESCRIPTION.
    hourly_variables : list of str or str
        DESCRIPTION.

    Returns
    -------
    url : str
        DESCRIPTION.

    """
    if isinstance(hourly_variables, list):
        hourly_variables = ','.join(hourly_variables)
    url = 'https://archive-api.open-meteo.com/v1/archive?latitude={}&longitude={}&start_date={}-01-01&end_date={}-12-31&hourly={}&timezone=Europe%2FBerlin'.format(latitude,longitude,year,year,hourly_variables)
    return url


def open_meteo_historical_data(longitude, latitude, year, hourly_variables=['temperature_2m','direct_radiation_instant'], force=False):
    """
    Ouverture des fichiers meteo

    Parameters
    ----------
    longitude : float
        DESCRIPTION.
    latitude : float
        DESCRIPTION.
    year : int
        DESCRIPTION.
    hourly_variables : str or list of str, optional
        DESCRIPTION. The default is ['temperature_2m','direct_radiation_instant'].
    force : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    data : pandas DataFrame
        DESCRIPTION.

    """
    if isinstance(hourly_variables, list):
        hourly_variables_str = ','.join(hourly_variables)
    else:
        hourly_variables_str = hourly_variables
        
    save_path = os.path.join('data','Open-Meteo')
    save_name = '{}_{}_{}_{}.csv'.format(hourly_variables_str, year, longitude, latitude)
    save_name_units = '{}_{}_{}_{}_units.txt'.format(hourly_variables_str, year, longitude, latitude)

    if save_name not in os.listdir(save_path) or force:
        url = get_open_meteo_url(longitude, latitude, year, hourly_variables)
        response = requests.get(url)
        json_data = response.json()

        units = json_data.get('hourly_units')
        with open(os.path.join(save_path,save_name_units), 'w') as f:
            for col, unit in units.items():
                f.write('{} : {} \n'.format(col,unit))
        
        data = pd.DataFrame().from_dict(json_data.get('hourly'))
        data.to_csv(os.path.join(save_path,save_name), index=False)
        
    data = pd.read_csv(os.path.join(save_path,save_name))
    data = data.set_index('time')
    data.index = pd.to_datetime(data.index)
    return data


def get_meteo_units(longitude, latitude, year, hourly_variables=['temperature_2m','direct_radiation_instant']):
    """
    Récupération du dictionnaire des unités des variables météorologiques

    Parameters
    ----------
    longitude : float
        DESCRIPTION.
    latitude : float
        DESCRIPTION.
    year : int
        DESCRIPTION.
    hourly_variables : str or list of str, optional
        DESCRIPTION. The default is ['temperature_2m','direct_radiation_instant'].

    Returns
    -------
    d : dict
        DESCRIPTION.

    """
    if isinstance(hourly_variables, list):
        hourly_variables_str = ','.join(hourly_variables)
    else:
        hourly_variables_str = hourly_variables
        
    unit_path = os.path.join('data','Open-Meteo','{}_{}_{}_{}_units.txt'.format(hourly_variables_str, year, longitude, latitude))
    
    with open(unit_path) as f:
        d = {k: v for line in f for (k, v) in [line.strip().split(' : ')]}
    return d


def plot_timeserie(data,figsize=(5,5),dpi=300,labels=None,save_fig=None,show=True,xlim=None,figax=None,**kwargs):
    """
    Fonction d'affichage de séries temporelles

    """
    if figax is None:
        fig,ax = plt.subplots(dpi=dpi,figsize=figsize)
    else:
        fig,ax = figax
    
    data_plot = data.copy()
    if xlim is not None:
        ax.set_xlim(xlim)
        data_plot = data_plot[(data_plot.index >= xlim[0])&(data_plot.index <= xlim[1])]
    
    if labels is None:
        labels = data_plot.columns
        
    for i,c in enumerate(data_plot.columns):
        ax.plot(data_plot[c],label=labels[i],**kwargs)
    ax.legend()
        
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    #formatter = mdates.DateFormatter('%b')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    if save_fig is not None:
        plt.savefig(os.path.join(figs_folder,'{}.png'.format(save_fig)),bbox_inches='tight')
    if show:
        plt.show()
        plt.close()
        return 
    return fig,ax


def dot3(A,B,C):
    return np.dot(A,np.dot(B,C))


def get_P_heater(Ti, Ti_min, Pmax, method='all_or_nothing'):
    if method == 'all_or_nothing':
        if Ti < Ti_min:
            P_heater = Pmax
        else:
            P_heater = 0
            
    elif method == 'linear_tolerance':
        tolerance = 1 #°C
        if Ti < Ti_min - tolerance:
            P_heater = Pmax
        elif Ti >= Ti_min + tolerance:
            P_heater = 0
        else:
            P_heater = (-(Ti-(Ti_min-tolerance))/(2*tolerance)+1)*Pmax
    return P_heater


def get_P_cooler(Ti, Ti_max, Pmax, method='all_or_nothing'):
    if method == 'all_or_nothing':
        if Ti > Ti_max:
            P_cooler = Pmax
        else:
            P_cooler = 0
            
    elif method == 'linear_tolerance':
         tolerance = 1 #°C
         if Ti > Ti_max + tolerance:
             P_cooler = Pmax
         elif Ti <= Ti_max - tolerance:
             P_cooler = 0
         else:
             P_cooler = ((Ti-(Ti_max-tolerance))/(2*tolerance))*Pmax
    return -P_cooler
    

def run_R2C2_model_simulation(data, R1, R2, C1, C2, Ti_min, Ti_max, P_heater_max, P_cooler_max, P_internal, solar_gain, heater_method, cooler_method):
    time_ = np.asarray(data.index)
    delta_t = (time_[1]-time_[0]) / np.timedelta64(1, 's')
    
    # Matrices A et B de la modélsiation R2C2 (cf p186 carnet)
    A_R2C2 = np.asarray([[-1/(R1*C1)-1/(R2*C1),  1/(R2*C1)],
                         [ 1/(R2*C2)          , -1/(R2*C2)]])
    B_R2C2 = np.asarray([[1/(R1*C1) ,  0],
                         [0          , 1/C2]])
    
    # Matrices discretisées 
    F = expm(A_R2C2 * delta_t)
    G = dot3(inv(A_R2C2), F-np.eye(2), B_R2C2)
    
    # État initial
    X = np.zeros((len(time_), 2))
    P_flux = np.zeros((len(time_), 5))
    
    Ti0 = Ti_min
    Te0 = (data.temperature_2m.values[0] + Ti0)/2
    X[0] = [Te0,Ti0]
    
    # Simulation
    for i in tqdm.tqdm(range(1,len(time_)), total=len(time_)-1):
        Ta = data.temperature_2m.values[i]
        DSI = data.direct_radiation_instant.values[i]
        
        Ti = X[i-1,1]
        Te = X[i-1,0]
        P_heater = get_P_heater(Ti, Ti_min=Ti_min, Pmax=P_heater_max,method=heater_method)
        P_cooler = get_P_cooler(Ti, Ti_max=Ti_max, Pmax=P_cooler_max,method=cooler_method)
        P_emitters = P_heater + P_cooler
        P_solar  = DSI*solar_gain
        P_th = 1/(R1+R2)*(Ta-Ti)
        P_total = P_emitters + P_internal + P_solar
        
        P_flux[i] = np.asarray([P_heater, P_cooler, P_solar, P_internal, P_th])
        X[i] = np.dot(F,X[i-1]) + np.dot(G, np.asarray([Ta, P_total]))
        
    return X, P_flux


def run_R1C0_model_simulation(data, R1, R2, C1, C2, Ti_min, Ti_max, P_heater_max, P_cooler_max, P_internal, solar_gain, heater_method, cooler_method):
    time_ = np.asarray(data.index)
    
    # État initial
    X = np.zeros((len(time_), 2))
    P_flux = np.zeros((len(time_), 5))
    
    Ti0 = Ti_min
    Te0 = (data.temperature_2m.values[0] + Ti0)/2
    X[0] = [Te0,Ti0]
    
    # Simulation
    for i in tqdm.tqdm(range(1,len(time_)), total=len(time_)-1):
        Ta = data.temperature_2m.values[i]
        DSI = data.direct_radiation_instant.values[i]
        
        Ti = X[i-1,1]
        Te = X[i-1,0]
        
        P_solar  = DSI*solar_gain
        if Ti<Ti_min:
            P_heater = max(0,1/(R1+R2)*(Ti_min-Ta) - P_internal - P_solar)
        else:
            P_heater = 0
        # print(P_heater)
        if Ti>Ti_max:
            P_cooler = min(0,1/(R1+R2)*(Ti_max-Ta) - P_internal - P_solar)
        else:
            P_cooler = 0
        P_emitters = P_heater + P_cooler
        P_solar  = DSI*solar_gain
        P_th = 1/(R1+R2)*(Ta-Ti)
        P_total = P_emitters + P_internal + P_solar
        
        P_flux[i] = np.asarray([P_heater, P_cooler, P_solar, P_internal, P_th])
        X[i][0] = (R2*Ta + R1*Ti)/(R1+R2)
        X[i][1] = R2*P_total+X[i][0]
        # X[i][0] = (R2*Ta + R1*Ti)/(R1+R2)
        
    return X, P_flux


# =============================================================================
# script principal
# =============================================================================

def main():
    tic = time.time()
    
    # Unité temporelle de base : horaire 
    
    # Définition des paramètres spatio-temporels
    year = 2019
    city = 'Marseille'
    
    # Définition des paramètres géométriques
    # Pièce carrée de surface S, de H m de haut et des murs d'épaisseur e m
    floor_surface     = 60 # m2
    height            = 3 # m
    wall_thickness    = 0.3 # m
    volume            = floor_surface * height
    wall_length       = np.sqrt(floor_surface)
    perimeter         = wall_length * 4
    heat_loss_surface = perimeter * height
    
    # Définition des paramètres matériaux (https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/2-fascicule_materiaux.pdf)
    lambda_wall = 2.8 # W/(m.K)
    rho_wall    = 2600 # kg/m3
    Cp_wall     = 1000 # J/(kg.K)
    volume_wall = heat_loss_surface * wall_thickness
    mass_wall   = rho_wall * volume_wall
    
    # lambda_air = 0.025 # W/(m.K)
    # rho_air    = 1.2 # kg/m3
    # Cp_air     = 1000 # J/(kg.K)
    # mass_air   = rho_air * volume
    
    R_wall = wall_thickness/(lambda_wall*heat_loss_surface)
    C_wall = Cp_wall*mass_wall
    # C_air = Cp_air*mass_air
    
    # Définition des consignes de température
    Ti_min = 20 # °C
    Ti_max = 25 # °C
    
    # Définition des puissances des émetteurs (https://calculis.net/puissance#radiateur, https://particuliers.engie.fr/depannages-services/conseils-equipements-chauffage/conseils-installation-climatisation/calcul-puissance-clim.html)
    q_max_heater = 14000 # W
    q_max_cooler = 10000 # W
    
    # Défintion des apports internes et solaires 
    q_internal = 400 # W
    # q_internal *= 0
    surface_windows = 4 # m2
    q_solar_gain =  surface_windows * 0.4 # m2
    # q_solar_gain *= 0
    
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
    
    # Récupération des données météo
    coordinates = get_coordinates(city)
    data = open_meteo_historical_data(longitude=coordinates[0], latitude=coordinates[1], year=year)
    meteo_units = get_meteo_units(longitude=coordinates[0], latitude=coordinates[1], year=year)
    
    data_high_res = pd.DataFrame(index=pd.date_range(data.index[0],data.index[-1],freq='30s'))
    data_high_res = data_high_res.join(data,how='left')
    data_high_res = data_high_res.interpolate()
    
    R1_test, R2_test, C1_test, C2_test = 2.13e-2, 2.37e-3, 1.56e7, 1.93e6
    
    # Résolution du modèle R2C2
    X_R2C2, P_th = run_R2C2_model_simulation(data=data_high_res, 
                                             R1=R_wall/2, 
                                             R2=R_wall/2, 
                                             C1=C_wall/2, 
                                             C2=C_wall/2, 
                                             Ti_min=Ti_min, 
                                             Ti_max=Ti_max, 
                                             P_heater_max=q_max_heater, 
                                             P_cooler_max=q_max_cooler,
                                             P_internal=q_internal,
                                             solar_gain=q_solar_gain,
                                             heater_method='all_or_nothing',
                                             cooler_method='all_or_nothing')
    
    # X_R2C2, P_th = run_R2C2_model_simulation(data=data_high_res, 
    #                                          R1=R1_test, 
    #                                          R2=R2_test, 
    #                                          C1=C1_test, 
    #                                          C2=C2_test, 
    #                                          Ti_min=Ti_min, 
    #                                          Ti_max=Ti_max, 
    #                                          P_heater_max=q_max_heater, 
    #                                          P_cooler_max=q_max_cooler,
    #                                          P_internal=q_internal,
    #                                          solar_gain=q_solar_gain)
    
    data_high_res['internal_wall_temperature'] = X_R2C2.T[0]
    data_high_res['indoor_temperature'] = X_R2C2.T[1]
    data_high_res['q_heater'] = P_th.T[0]
    data_high_res['q_cooler'] = P_th.T[1]
    data_high_res['q_solar'] = P_th.T[2]
    data_high_res['q_internal'] = P_th.T[3]
    data_high_res['q_thermal'] = P_th.T[4]
    
    hourly_data_high_res = data_high_res.groupby(pd.Grouper(freq='h')).mean()
    for c in ['temperature_2m', 'internal_wall_temperature', 'indoor_temperature','q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']:
        data[c] = hourly_data_high_res[c]
        
    
    X_R2C0, P_th_lin = run_R1C0_model_simulation(data=data_high_res, 
                                                 R1=R_wall/2, 
                                                 R2=R_wall/2, 
                                                 C1=C_wall/2, 
                                                 C2=C_wall/2, 
                                                 Ti_min=Ti_min, 
                                                 Ti_max=Ti_max, 
                                                 P_heater_max=q_max_heater*10, 
                                                 P_cooler_max=q_max_cooler*10,
                                                 P_internal=q_internal,
                                                 solar_gain=q_solar_gain,
                                                 heater_method='all_or_nothing',
                                                 cooler_method='all_or_nothing')
    
    data_high_res['internal_wall_temperature_linear'] = X_R2C0.T[0]
    data_high_res['indoor_temperature_linear'] = X_R2C0.T[1]
    data_high_res['q_heater_linear'] = P_th_lin.T[0]
    data_high_res['q_cooler_linear'] = P_th_lin.T[1]
    data_high_res['q_solar_linear'] = P_th_lin.T[2]
    data_high_res['q_internal_linear'] = P_th_lin.T[3]
    data_high_res['q_thermal_linear'] = P_th_lin.T[4]
    
    hourly_data_high_res = data_high_res.groupby(pd.Grouper(freq='h')).mean()
    for c in [e+'_linear' for e in ['internal_wall_temperature', 'indoor_temperature','q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']]:
        data[c] = hourly_data_high_res[c]
    
    
    
    
    
    if False:
        cols = ['temperature_2m', 'internal_wall_temperature', 'indoor_temperature']
        plot_timeserie(data[cols], labels=['{} (°C)'.format(c) for c in cols], figsize=(15,5),
                       xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
  
                  
    if False:
        cols = ['q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']
        plot_timeserie(data[cols], labels=['{} (W)'.format(c) for c in cols], figsize=(15,5),
                       xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
        
    if False:
        cols = ['temperature_2m', 'internal_wall_temperature_linear', 'indoor_temperature_linear']
        plot_timeserie(data[cols], labels=['{} (°C)'.format(c) for c in cols], figsize=(15,5),
                       xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
  
                  
    if False:
        cols = [e+'_linear' for e in ['q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']]
        plot_timeserie(data[cols], labels=['{} (W)'.format(c) for c in cols], figsize=(15,5),
                       xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
    
    if False:
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(data[data.q_heater>0]['temperature_2m'], data[data.q_heater>0]['q_heater'], ls='',marker='.',color='k',alpha=0.4)
        ax.plot(data[data.q_cooler<0]['temperature_2m'], -data[data.q_cooler<0]['q_cooler'], ls='',marker='.',color='k',alpha=0.4)
        plt.show()
        
    monthly_data = data[['q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']].groupby(pd.Grouper(freq='MS')).sum()
    
    if True:
        print('Consommation de chauffage :')
        for d in monthly_data.index:
            print('{} : {:.0f} kW'.format(d.strftime('%b'), monthly_data.loc[d].q_heater/1000))
        print('Total : {:.0f} kW'.format(monthly_data.q_heater.sum()/1000))
        print()
        print('Consommation de refroidissement :')
        for d in monthly_data.index:
            print('{} : {:.0f} kW'.format(d.strftime('%b'), -monthly_data.loc[d].q_cooler/1000))
        print('Total : {:.0f} kW'.format(-monthly_data.q_cooler.sum()/1000))
        
        fig,ax = plot_timeserie(monthly_data[['q_heater']], show=False)
        fig,ax = plot_timeserie(-1*monthly_data[['q_cooler']], show=False, figax=(fig,ax))
        
        
    monthly_data_linear = data[[e+'_linear' for e in ['q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']]].groupby(pd.Grouper(freq='MS')).sum()
    
    if True:
        print('Consommation de chauffage :')
        for d in monthly_data_linear.index:
            print('{} : {:.0f} kW'.format(d.strftime('%b'), monthly_data_linear.loc[d].q_heater_linear/1000))
        print('Total : {:.0f} kW'.format(monthly_data_linear.q_heater_linear.sum()/1000))
        print()
        print('Consommation de refroidissement :')
        for d in monthly_data_linear.index:
            print('{} : {:.0f} kW'.format(d.strftime('%b'), -monthly_data_linear.loc[d].q_cooler_linear/1000))
        print('Total : {:.0f} kW'.format(-monthly_data_linear.q_cooler_linear.sum()/1000))
        
        fig,ax = plot_timeserie(monthly_data_linear[['q_heater_linear']], show=False, figax=(fig,ax))
        plot_timeserie(-1*monthly_data_linear[['q_cooler_linear']], figax=(fig,ax))
    
    
    
    
# =============================================================================
# Affichage 
# =============================================================================
    # Affichage des données météo
    if False:
        for c in ['temperature_2m','direct_radiation_instant']:
            plot_timeserie(data[[c]], labels=['{} ({})'.format(c,meteo_units.get(c))], figsize=(15,5),
                           xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
    
    
    
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()

