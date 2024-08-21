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
    if True:
        city = 'Paris'
        year = 2022
        
        coordinates = get_coordinates(city)
        data = open_meteo_historical_data(longitude=coordinates[0], latitude=coordinates[1], year=year)
        meteo_units = get_meteo_units(longitude=coordinates[0], latitude=coordinates[1], year=year)
        
        for c in ['temperature_2m','direct_radiation_instant']:
            plot_timeserie(data[[c]], labels=['{} ({})'.format(c,meteo_units.get(c))], figsize=(15,5),figs_folder = figs_folder,
                           xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
    
    
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()