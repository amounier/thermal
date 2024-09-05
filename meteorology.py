#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:42:33 2024

@author: amounier
"""

import time 
import os
import pandas as pd
from datetime import date
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import tqdm

from utils import plot_timeserie


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
    coordinates_dict = {'Paris':(2.320041, 48.85889),
                        'Marseille':(5.369953, 43.296174),
                        'Brest':(-4.486009, 48.390528)
                       }
    
    if city in coordinates_dict.keys():
        longitude, latitude = coordinates_dict[city]
        return longitude, latitude
    else:
        try:
            # initialisation de l'instance Nominatim (API OSM), changer l'agent si besoin
            geolocator = Nominatim(user_agent="amounier")
            location = geolocator.geocode(city)
            longitude, latitude = round(location.longitude,ndigits=6), round(location.latitude, ndigits=6)
        except GeocoderUnavailable:
            raise KeyError('No internet connexion, offline availables cities are : {}'.format(', '.join(list(coordinates_dict.keys()))))
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
    tod = pd.Timestamp(date.today())
    
    # Si l'année demandée n'est pas terminée, il faut modifier les périodes requêtées
    end_month, end_day = 12, 31
    if year == tod.year:
        end_day = tod.strftime('%d')
        end_month = tod.strftime('%m')
        
    url = 'https://archive-api.open-meteo.com/v1/archive?latitude={}&longitude={}&start_date={}-01-01&end_date={}-{}-{}&hourly={}&timezone=Europe%2FBerlin'.format(latitude,longitude,year,year,end_month,end_day,hourly_variables)
    # print(url)
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
    # TODO : peut-etre mettre un nom de ville en entrée et en faire des nom de sauvegarde pluis lisible
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
        # print(year, response)
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


def get_meteo_data(city, period=[2020,2024]):
    longitude, latitude = get_coordinates(city)
    data = None
    for y in range(period[0],period[1]+1):
        yearly_data = open_meteo_historical_data(longitude, latitude, y)
        if data is None:
            data = yearly_data
        else:
            data = pd.concat([data, yearly_data])
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



def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_meteorology'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    # -------------------------------------------------------------------------
    
    # Affichage des données météo
    if False:
        city = 'Paris'
        year = 2022
        
        coordinates = get_coordinates(city)
        data = open_meteo_historical_data(longitude=coordinates[0], latitude=coordinates[1], year=year)
        meteo_units = get_meteo_units(longitude=coordinates[0], latitude=coordinates[1], year=year)
        
        for c in ['temperature_2m','direct_radiation_instant']:
            plot_timeserie(data[[c]], labels=['{} ({})'.format(c,meteo_units.get(c))], figsize=(15,5),figs_folder = figs_folder,
                           xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
    
    # étude de la température du sol en fonction de la température de surface 
    if True:
        city = 'Paris'
        year = 2022
        variables = ['temperature_2m','soil_temperature_0_to_7cm','soil_temperature_7_to_28cm','soil_temperature_28_to_100cm','soil_temperature_100_to_255cm']
        
        coordinates = get_coordinates(city)
        data = open_meteo_historical_data(longitude=coordinates[0], latitude=coordinates[1], year=year, hourly_variables=variables)
        meteo_units = get_meteo_units(longitude=coordinates[0], latitude=coordinates[1], year=year, hourly_variables=variables)
        
        # Modélisation des températures souterraines en fonction des températures de surface 
        if True:
            
            def ground_temperature(theta_g, t, x, D,h, data):
                time_ = np.asarray(data.index)
                delta_t = (time_[1]-time_[0]) / np.timedelta64(1, 's')
                D = D*delta_t
                t = min(t,len(data)-1)
                theta_e = data.temperature_2m.iloc[int(t)]
                # RC = x/h + x**2/D
                RC = x**2/D
                dtheta_gdt = (1/RC)*(theta_e-theta_g)
                return dtheta_gdt
            
            
            def get_init_ground_temperature(x,data,xi=1.5,envelope=False):
                if envelope:
                    theta_gs = data.temperature_2m.quantile(0.001)
                else:
                    theta_gs = data.temperature_2m.values[0]
                theta_ginf = data.temperature_2m.median()
                res = theta_gs + (theta_ginf-theta_gs) * (1-np.exp(-x/xi))
                return res
            
            
            def model_ground_temperature(depth_list, data, lambda_th=1.5,cp=1000,rho=2500,h=0.05, res='1h'):
                data_res = data.copy()
                
                # si besoin d'augmentaer la résolution temporelle pour éviter l'instabilité d'intégration
                data_high_res = pd.DataFrame(index=pd.date_range(data_res.index[0],data_res.index[-1],freq=res))
                data_high_res = data_high_res.join(data_res,how='left')
                data_high_res = data_high_res.interpolate()
            
                # paramètres thermiques du sol
                # lambda_th = 1.5 #W/(m.K) https://energieplus-lesite.be/donnees/enveloppe44/caracteristiques-thermiques-des-sols/
                # cp = 1000 #K/(kg.K)
                # rho = 2500 #kg/m3
                # h = 0.05 #W/(m2.K) https://fr.wikipedia.org/wiki/Coefficient_de_convection_thermique
                D = lambda_th/(rho*cp) #m2/s
            
                t = np.arange(0,len(data_high_res))
                # theta_g0 = data_res.temperature_2m.mean()-4 # TODO, à modifier en fonction de la profondeur
                
                
                depth_col_list = []
                for depth in tqdm.tqdm(depth_list):
                    # initialisation en fonction de la profondeur
                    theta_g0 = get_init_ground_temperature(depth, data_res)
                    
                    modelled_col = 'modelled_soil_temperature_{:.0f}cm'.format(depth*100)
                    depth_col_list.append(modelled_col)
                    data_high_res[modelled_col] = odeint(ground_temperature, theta_g0, t, args=(depth,D,h, data_high_res)).T[0]
                    
                hourly_data_high_res = data_high_res.groupby(pd.Grouper(freq='h')).mean()
                for col in depth_col_list:
                    data_res[col] = hourly_data_high_res[col]
                
                return data_res
                
            
            # list des profondeur modélisées
            # depth_list = [x/100 for x in [7, 28, 100, 255, 500]]
            # depth_list = [x/100 for x in np.linspace(5,1000,30)]
            
            # data = model_ground_temperature(depth_list, data)
            
            # Affichage des séries temporelles
            if True:
                depth_list = [x/100 for x in [64,178]]
                data = model_ground_temperature(depth_list, data)
                
                plot_timeserie(data[['soil_temperature_28_to_100cm','modelled_soil_temperature_64cm',
                                     'soil_temperature_100_to_255cm','modelled_soil_temperature_178cm']], 
                               figsize=(15,5),figs_folder = figs_folder,
                               xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
            
            # Affichage de la température en fonction de la profondeur à partir des données Open-Météo
            if False:
                fig, ax = plt.subplots(figsize=(5,5),dpi=300)
                dict_col = {4/100:'soil_temperature_0_to_7cm',
                            18/100:'soil_temperature_7_to_28cm', 
                            64/100:'soil_temperature_28_to_100cm',
                            178/100:'soil_temperature_100_to_255cm'}
                for depth in dict_col.keys():
                    modelled_col = dict_col.get(depth)
                    ax.plot(data[modelled_col].values, [depth]*len(data),ls='',marker='.',color='tab:blue',alpha=0.01)
                ax.set_ylim(ax.get_ylim()[::-1])
                
                # calibration de xi pour l'estimation de la température initiale
                if False:
                    Y = np.linspace(0,3)
                    X = [get_init_ground_temperature(y, data,xi=1.5,envelope=True) for y in Y]
                    ax.plot(X,Y,color='k')
                    
                ax.plot([data.temperature_2m.median(),data.temperature_2m.median()],[min(depth_list), max(depth_list)],color='k')
                plt.show()
                
            # Affichage de la température en fonction de la profondeur par modélisation
            if False:
                fig, ax = plt.subplots(figsize=(5,5),dpi=300)
                for depth in depth_list:
                    modelled_col = 'modelled_soil_temperature_{:.0f}cm'.format(depth*100)
                    ax.plot(data[modelled_col].values, [depth]*len(data),ls='',marker='.',color='tab:blue',alpha=0.01)
                ax.set_ylim(ax.get_ylim()[::-1])
                ax.plot([data.temperature_2m.median(),data.temperature_2m.median()],[min(depth_list), max(depth_list)],color='k')
                ax.set_ylabel('Profondeur (m)')
                ax.set_ylabel('Température du sol (°C)')
                plt.show()
            
            
            # Même affichage avec des moyennes par saison (ou par mois ?)
            if False:
                fig, ax = plt.subplots(figsize=(5,5),dpi=300)
                seasons_dict = {'DJF':([12,1,2],'tab:blue'),
                                'MAM':([3,4,5],'tab:green'),
                                'JJA':([6,7,8],'tab:red'),
                                'SON':([9,10,11],'tab:orange')}
                for season in seasons_dict.keys():
                    season_months, season_color = seasons_dict.get(season)
                    season_mean = data[data.index.month.isin(season_months)]
                    season_mean = pd.DataFrame(season_mean.mean()).T
                    
                    season_plot = []
                    for depth in depth_list:
                        modelled_col = 'modelled_soil_temperature_{:.0f}cm'.format(depth*100)
                        season_plot.append(season_mean[modelled_col].values[0])
                        
                    ax.plot(season_plot, depth_list,color=season_color,label=season)
                    
                ax.set_ylim(ax.get_ylim()[::-1])
                ax.legend()
                ax.plot([data.temperature_2m.median(),data.temperature_2m.median()],[min(depth_list), max(depth_list)],color='k')
                ax.set_ylabel('Profondeur (m)')
                ax.set_xlabel('Température du sol (°C)')
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('modelling_of_ground_temperature')),bbox_inches='tight')
                plt.show()
            
            
        
        # Affichage des données annuelles
        if False:
            for i,c in enumerate(variables):
                if c == variables[0]:
                    fig, ax = plot_timeserie(data[[c]], labels=['{} ({})'.format(c,meteo_units.get(c))], figsize=(15,5),figs_folder = figs_folder, show=False,
                                             xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
                elif c == variables[-1]:
                    plot_timeserie(data[[c]], labels=['{} ({})'.format(c,meteo_units.get(c))], figsize=(15,5),figs_folder = figs_folder, show=True, figax=(fig,ax),
                                   xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],save_fig='soil_temperature_{}_{}'.format(city,year))
                else:
                    fig, ax = plot_timeserie(data[[c]], labels=['{} ({})'.format(c,meteo_units.get(c))], figsize=(15,5),figs_folder = figs_folder, show=False,figax=(fig,ax),
                                             xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
                
        
        # Affichage des versus plot (diagramme de phase)
        if False:
            for i,c in enumerate(variables[1:]):
                fig,ax = plt.subplots(figsize=(5,5), dpi=300)
                ax.plot(data['temperature_2m'],data[c], marker='o',ls='',alpha=0.2)
                min_t, max_t = [data.temperature_2m.min(), data.temperature_2m.max()]
                ax.set_xlim([min_t, max_t])
                ax.set_ylim([min_t, max_t])
                ax.plot([min_t, max_t], [min_t, max_t], color='k')
                ax.set_ylabel('{} ({})'.format(c,meteo_units.get(c)))
                ax.set_xlabel('{} ({})'.format(c,meteo_units.get('temperature_2m')))
                ax.set_title('{} ({})'.format(city,year))
                plt.show()
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()