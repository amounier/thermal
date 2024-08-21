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
import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit
from scipy.linalg import expm
from numpy.linalg import inv

from utils import plot_timeserie
from meteorology import get_coordinates, open_meteo_historical_data



def dot3(A,B,C):
    return np.dot(A,np.dot(B,C))


def get_P_heater(Ti, Ti_min, Pmax, method='all_or_nothing'):
    """
    Renvoie la puissance des équipemetns de chauffage

    Parameters
    ----------
    Ti : float
        DESCRIPTION.
    Ti_min : float
        DESCRIPTION.
    Pmax : float
        DESCRIPTION.
    method : str, optional
        DESCRIPTION. The default is 'all_or_nothing'.

    Returns
    -------
    P_heater : float
        DESCRIPTION.

    """
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
    """
    Renvoie la puissance des équipements de refroidissement

    Parameters
    ----------
    Ti : float
        DESCRIPTION.
    Ti_max : float
        DESCRIPTION.
    Pmax : float
        DESCRIPTION.
    method : str, optional
        DESCRIPTION. The default is 'all_or_nothing'.

    Returns
    -------
    P_cooler : float
        DESCRIPTION.

    """
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
    
    # flux de refroidissement négatif
    P_cooler = - P_cooler
    return P_cooler
    

def run_R2C2_model_simulation(data, R1, R2, C1, C2, Ti_min, Ti_max, P_heater_max, P_cooler_max, P_internal, solar_gain, heater_method, cooler_method):
    """
    Simulation d'évolution de la température intéreiure et des consommations associés
    pour un système thermique R2C2

    Parameters
    ----------
    data : pandas DataFrame
        DESCRIPTION.
    R1 : float
        DESCRIPTION.
    R2 : float
        DESCRIPTION.
    C1 : float
        DESCRIPTION.
    C2 : float
        DESCRIPTION.
    Ti_min : float
        DESCRIPTION.
    Ti_max : float
        DESCRIPTION.
    P_heater_max : float
        DESCRIPTION.
    P_cooler_max : float
        DESCRIPTION.
    P_internal : float
        DESCRIPTION.
    solar_gain : float
        DESCRIPTION.
    heater_method : str
        DESCRIPTION.
    cooler_method : str
        DESCRIPTION.

    Returns
    -------
    X : numpy Array
        DESCRIPTION.
    P_flux : numpy Array
        DESCRIPTION.

    """
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
        # Te = X[i-1,0]
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
    """
    Simulation d'évolution de la température intéreiure et des consommations associés
    pour un système thermique R1

    Parameters
    ----------
    data : pandas DataFrame
        DESCRIPTION.
    R1 : float
        DESCRIPTION.
    R2 : float
        DESCRIPTION.
    C1 : float
        DESCRIPTION.
    C2 : float
        DESCRIPTION.
    Ti_min : float
        DESCRIPTION.
    Ti_max : float
        DESCRIPTION.
    P_heater_max : float
        DESCRIPTION.
    P_cooler_max : float
        DESCRIPTION.
    P_internal : float
        DESCRIPTION.
    solar_gain : float
        DESCRIPTION.
    heater_method : str
        DESCRIPTION.
    cooler_method : str
        DESCRIPTION.

    Returns
    -------
    X : numpy Array
        DESCRIPTION.
    P_flux : numpy Array
        DESCRIPTION.

    """
    # TODO : supprimer les variables d'entrée non utilisées
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
        # Te = X[i-1,0]
        
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
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_RC_models'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    #--------------------------------------------------------------------------
    
    # Unité temporelle de base : horaire 
    
    # Définition des paramètres spatio-temporels
    year = 2023
    city = 'Marseille'
    
    # Définition des paramètres géométriques
    # Pièce carrée de surface S, de H m de haut et des murs d'épaisseur e m
    floor_surface     = 60 # m2
    height            = 3 # m
    wall_thickness    = 0.3 # m
    # volume            = floor_surface * height
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
    
    # Récupération des données météo
    coordinates = get_coordinates(city)
    data = open_meteo_historical_data(longitude=coordinates[0], latitude=coordinates[1], year=year)
    # meteo_units = get_meteo_units(longitude=coordinates[0], latitude=coordinates[1], year=year)
    
    data_high_res = pd.DataFrame(index=pd.date_range(data.index[0],data.index[-1],freq='30s'))
    data_high_res = data_high_res.join(data,how='left')
    data_high_res = data_high_res.interpolate()
    
    
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
                                             heater_method='linear_tolerance',
                                             cooler_method='all_or_nothing')
    
    # R1_test, R2_test, C1_test, C2_test = 2.13e-2, 2.37e-3, 1.56e7, 1.93e6
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
        plot_timeserie(data[cols], labels=['{} (°C)'.format(c) for c in cols], figsize=(15,5), figs_folder = figs_folder,
                       xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
  
                  
    if False:
        cols = ['q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']
        plot_timeserie(data[cols], labels=['{} (W)'.format(c) for c in cols], figsize=(15,5), figs_folder = figs_folder,
                       xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
        
    if False:
        cols = ['temperature_2m', 'internal_wall_temperature_linear', 'indoor_temperature_linear']
        plot_timeserie(data[cols], labels=['{} (°C)'.format(c) for c in cols], figsize=(15,5), figs_folder = figs_folder,
                       xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
  
                  
    if False:
        cols = [e+'_linear' for e in ['q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']]
        plot_timeserie(data[cols], labels=['{} (W)'.format(c) for c in cols], figsize=(15,5), figs_folder = figs_folder,
                       xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
    
    if True:
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(data[data.q_heater>0]['temperature_2m'], data[data.q_heater>0]['q_heater'], ls='',marker='.',color='k',alpha=0.4)
        ax.plot(data[data.q_cooler<0]['temperature_2m'], -data[data.q_cooler<0]['q_cooler'], ls='',marker='.',color='k',alpha=0.4)
        plt.show()
        
    monthly_data = data[['q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']].groupby(pd.Grouper(freq='MS')).sum()
    
    if False:
        print('Consommation de chauffage :')
        for d in monthly_data.index:
            print('{} : {:.0f} kW'.format(d.strftime('%b'), monthly_data.loc[d].q_heater/1000))
        print('Total : {:.0f} kW'.format(monthly_data.q_heater.sum()/1000))
        print()
        print('Consommation de refroidissement :')
        for d in monthly_data.index:
            print('{} : {:.0f} kW'.format(d.strftime('%b'), -monthly_data.loc[d].q_cooler/1000))
        print('Total : {:.0f} kW'.format(-monthly_data.q_cooler.sum()/1000))
        
        fig,ax = plot_timeserie(monthly_data[['q_heater']], show=False, figs_folder = figs_folder,)
        fig,ax = plot_timeserie(-1*monthly_data[['q_cooler']], show=False, figax=(fig,ax), figs_folder = figs_folder,)
        
        
    monthly_data_linear = data[[e+'_linear' for e in ['q_heater', 'q_cooler', 'q_solar', 'q_internal', 'q_thermal']]].groupby(pd.Grouper(freq='MS')).sum()
    
    if False:
        print('Consommation de chauffage :')
        for d in monthly_data_linear.index:
            print('{} : {:.0f} kW'.format(d.strftime('%b'), monthly_data_linear.loc[d].q_heater_linear/1000))
        print('Total : {:.0f} kW'.format(monthly_data_linear.q_heater_linear.sum()/1000))
        print()
        print('Consommation de refroidissement :')
        for d in monthly_data_linear.index:
            print('{} : {:.0f} kW'.format(d.strftime('%b'), -monthly_data_linear.loc[d].q_cooler_linear/1000))
        print('Total : {:.0f} kW'.format(-monthly_data_linear.q_cooler_linear.sum()/1000))
        
        fig,ax = plot_timeserie(monthly_data_linear[['q_heater_linear']], show=False, figax=(fig,ax), figs_folder = figs_folder,)
        plot_timeserie(-1*monthly_data_linear[['q_cooler_linear']], figax=(fig,ax), figs_folder = figs_folder,)
    
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()

