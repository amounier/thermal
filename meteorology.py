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
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from pysolar.solar import get_altitude, get_altitude_fast, get_azimuth, get_azimuth_fast
import tqdm

# Pour ne pas utiliser numpy dans la gestion des dates de Pysolar (moins efficace est plus à jour)
import pysolar
pysolar.use_math()

from utils import plot_timeserie
from administrative import get_coordinates

import warnings




    
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
    # TODO : peut-etre mettre un nom de ville en entrée et en faire des nom de sauvegarde plus lisible
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


def get_meteo_data(city, period=[2020,2024],variables=['temperature_2m','direct_radiation_instant']):
    longitude, latitude = get_coordinates(city)
    data = None
    for y in range(period[0],period[1]+1):
        yearly_data = open_meteo_historical_data(longitude, latitude, y, hourly_variables=variables)
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


def get_direct_solar_irradiance_projection_ratio(orientation,sun_azimuth,sun_altitude):
    """
    Coefficient de puissance solaire directe pour une paroi donnée.

    Parameters
    ----------
    orientation : str or int or float
        Orientation  de la paroi (str conseillé).
    sun_azimuth : int or float
        Angle azimutal du soleil en degré.
    sun_altitude : int or float
        Angle d'altitude du soleil en degré.

    Raises
    ------
    ValueError
        Si l'orientation donnée est fausse.

    Returns
    -------
    ratio : float
        Coefficient compris entre 0 et 1.

    """
    
    # gestion rapide du cas d'une paroi horizontale
    if orientation == 'H':
        ratio = np.sin(np.deg2rad(np.maximum(sun_altitude,0)))
    
    # cas des parois verticales
    else:
        # passage d'une orientation str en angle en degrés
        valid_orientations = ['N','NE','E','SE','S','SW','W','NW']
        dict_angle_orientation = {i*45:o for i,o in enumerate(valid_orientations)}
        dict_orientation_angle = {v:k for k,v in dict_angle_orientation.items()}
        if isinstance(orientation,str):
            # Vérification de l'orientation 
            if orientation not in valid_orientations:
                raise ValueError("'{}' is not a valid orientation (must be in ['{}']).".format(orientation,"', '".join(valid_orientations)))
                
            orientation_angle = dict_orientation_angle.get(orientation)
        
        # l'orientation peut directement être entrée en degrés (déconseillé)
        elif isinstance(orientation,int) or isinstance(orientation,float):
            orientation_angle = orientation%360
          
        ratio = np.float64(sun_altitude>0) * np.cos(np.deg2rad(sun_altitude)) * np.maximum(np.cos(np.deg2rad(list(sun_azimuth-orientation_angle))),0)
    return ratio


def get_diffuse_solar_irradiance_projection_ratio(orientation):
    """
    Coefficient de puissance solaire diffuse pour une paroi donnée.

    Parameters
    ----------
    orientation : int or str
        Orientation de la paroi.

    Returns
    -------
    ratio : float
        Coefficient de puissance.

    """
    if orientation == 'H':
        ratio = 1.
    else:
        ratio = 0.5
    return ratio


def get_init_ground_temperature(x, data, xi_1 = 0.8, xi_2=0.4, w0=1.37, second_order=True):
    theta_ginit = data[(data.index.month==1)&(data.index.day==1)].temperature_2m.min()
    theta_ginf = data.temperature_2m.median()
    
    if second_order:
        res = theta_ginit + (theta_ginf-theta_ginit)*(1 - np.exp(-xi_2*w0*x) * np.cos(w0*np.sqrt(1-xi_2**2)*x))
                                        
    else:
        # data_january = data[data.index.month==1]
        # TODO : tendance à sous estimer les valeurs : à modifier
        # if envelope:
        #     theta_gs = data_january.temperature_2m.mean()
        # else:
        # theta_gs = data.temperature_2m.values[0]
        # theta_ginf = data.temperature_2m.median()
        res = theta_ginit + (theta_ginf-theta_ginit) * (1-np.exp(-x/xi_1))
    return res 


def get_list_orientations(principal_orientation):
    # définition des dictionnaires d'angles
    valid_orientations = ['N','NE','E','SE','S','SW','W','NW']
    dict_angle_orientation = {i*45:o for i,o in enumerate(valid_orientations)}
    dict_orientation_angle = {v:k for k,v in dict_angle_orientation.items()}
    
    # liste des orientations
    orientation_0 = principal_orientation
    orientation_1 = dict_angle_orientation.get((dict_orientation_angle.get(orientation_0)+90)%360)
    orientation_2 = dict_angle_orientation.get((dict_orientation_angle.get(orientation_1)+90)%360)
    orientation_3 = dict_angle_orientation.get((dict_orientation_angle.get(orientation_2)+90)%360)
    orientations_list = [orientation_0, orientation_1, orientation_2, orientation_3] + ['H']
    return orientations_list



def get_historical_weather_data(city, period, principal_orientation, display_units=False):
    # initialisation des données météo
    variables = ['temperature_2m','diffuse_radiation_instant','direct_normal_irradiance_instant']
    coordinates = get_coordinates(city)
    data = get_meteo_data(city,period,variables=variables)
    
    # formatage des dates sur le bon fuseau horaire
    dates = data.copy().index
    dates = dates.tz_localize(tz='CET',ambiguous='NaT',nonexistent='NaT')
    dates = dates.to_pydatetime()
    
    # ajout de la hauteur du soleil (en degrés)
    altitude = [get_altitude_fast(coordinates[1],coordinates[0],t) if t is not pd.NaT else np.nan for t in dates]
    data['sun_altitude'] = altitude
    
    # ajout de l'azimuth du soleil (en degrés)
    azimuth = [float(get_azimuth_fast(coordinates[1],coordinates[0],t)) if t is not pd.NaT else np.nan for t in dates]
    data['sun_azimuth'] = azimuth
    
    # défintion de la liste des orientations du bâtiment
    orientations = get_list_orientations(principal_orientation)
    
    for ori in orientations:
        col_coef_dri = 'coefficient_direct_{}_irradiance'.format(ori)
        col_coef_dif = 'coefficient_diffuse_{}_irradiance'.format(ori)
        col_dri = 'direct_sun_radiation_{}'.format(ori)
        col_dif = 'diffuse_sun_radiation_{}'.format(ori)
        # global_ri = 'sun_radiation_{}'.format(ori)
        data[col_coef_dri] = get_direct_solar_irradiance_projection_ratio(ori, data.sun_azimuth, data.sun_altitude)
        data[col_coef_dif] = get_diffuse_solar_irradiance_projection_ratio(ori)
        data[col_dri] = data.direct_normal_irradiance_instant * data[col_coef_dri]
        data[col_dif] = data.diffuse_radiation_instant * data[col_coef_dif]
        # je garde les apports diffus et directs séparés pour pouvoir traiter les masquages différemment
        # data[global_ri] = data[col_dri] + data[col_dif]
        
        # suppression des colonnes intermédiaires de calcul
        data = data.drop(columns=[col_coef_dri,col_coef_dif])

    if display_units:
        meteo_units = get_meteo_units(longitude=coordinates[0], latitude=coordinates[1], year=period[0], hourly_variables=variables)
        meteo_units['sun_altitude'] = '°'
        meteo_units['sun_azimuth'] = '°'
        meteo_units['direct_sun_radiation'] = 'W/m²'
        meteo_units['diffuse_sun_radiation'] = 'W/m²'
        print(meteo_units)
        
    return data
    

#%% ===========================================================================
# Script principal
# =============================================================================

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
    
    #%% Affichage des données météo
    if False:
        city = 'Paris'
        year = 2022
        
        coordinates = get_coordinates(city)
        data = open_meteo_historical_data(longitude=coordinates[0], latitude=coordinates[1], year=year)
        meteo_units = get_meteo_units(longitude=coordinates[0], latitude=coordinates[1], year=year)
        
        for c in ['temperature_2m','direct_radiation_instant']:
            plot_timeserie(data[[c]], labels=['{} ({})'.format(c,meteo_units.get(c))], figsize=(15,5),figs_folder = figs_folder,
                           xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
    
    
    #%% Étude de la température du sol en fonction de la température de surface 
    if False:
        city = 'Marseille'
        year = 2021
        # 2021
        variables = ['temperature_2m','soil_temperature_0_to_7cm','soil_temperature_7_to_28cm','soil_temperature_28_to_100cm','soil_temperature_100_to_255cm']
        
        coordinates = get_coordinates(city)
        data = open_meteo_historical_data(longitude=coordinates[0], latitude=coordinates[1], year=year, hourly_variables=variables)
        meteo_units = get_meteo_units(longitude=coordinates[0], latitude=coordinates[1], year=year, hourly_variables=variables)
        
        # Modélisation des températures souterraines en fonction des températures de surface 
        if True:
            
            def ground_temperature(theta_g, t, x, D,h, data,x_stationary=10):
                time_ = np.asarray(data.index)
                delta_t = (time_[1]-time_[0]) / np.timedelta64(1, 's')
                D = D*delta_t
                t = min(t,len(data)-1)
                theta_e = data.temperature_2m.iloc[int(t)]
                te_mean = data.temperature_2m.mean()
                # RC = x/h + x**2/D
                RC = x**2/D
                dtheta_gdt = (1/RC)*(theta_e-theta_g) #+ 1/(x*x_stationary/D-x**2/D) * (te_mean-theta_g)
                return dtheta_gdt
            
            
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
                
                depth_col_list = []
                for depth in tqdm.tqdm(depth_list):
                    # initialisation en fonction de la profondeur
                    theta_g0 = get_init_ground_temperature(depth, data_res, second_order=True)
                    
                    modelled_col = 'modelled_soil_temperature_{:.0f}cm'.format(depth*100)
                    depth_col_list.append(modelled_col)
                    data_high_res[modelled_col] = odeint(ground_temperature, theta_g0, t, args=(depth,D,h, data_high_res)).T[0]
                    
                hourly_data_high_res = data_high_res.groupby(pd.Grouper(freq='h')).mean()
                for col in depth_col_list:
                    data_res[col] = hourly_data_high_res[col]
                
                return data_res
                
            
            # Affichage des séries temporelles
            if True:
                depth_list = [((3/5)*mh+(2/5)*ml)/100 for ml,mh in [(28,100),(100,255)]]
                # depth_list = [((1/2)*mh+(1/2)*ml)/100 for ml,mh in [(28,100),(100,255)]]
                ground_data = model_ground_temperature(depth_list, data)
                
                fig,ax = plot_timeserie(ground_data[['soil_temperature_28_to_100cm',
                                                     'modelled_soil_temperature_{:.0f}cm'.format(depth_list[0]*100),
                                                     'soil_temperature_100_to_255cm',
                                                     'modelled_soil_temperature_{:.0f}cm'.format(depth_list[1]*100),]],
                                        colors = ['tab:blue','tab:blue','tab:red','tab:red'],
                                        labels=['']*4,
                                        linestyles=['-',':','-',':'],show=False,
                                        figsize=(15,5),figs_folder = figs_folder,
                                        xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))])
                ax.set_ylabel('Soil temperature (°C)')
                ax.plot(ground_data.iloc[0]['soil_temperature_28_to_100cm'],ls='-',color='tab:blue',label='$d \\in [28,100]~cm$')
                ax.plot(ground_data.iloc[0]['soil_temperature_100_to_255cm'],ls='-',color='tab:red',label='$d \\in [100,255]~cm$')
                
                # ax2 = ax.twinx()
                # ax2.tick_params(right=False)
                # ax2.set(yticklabels=[])
                
                ax.plot(ground_data.iloc[0]['soil_temperature_28_to_100cm'],ls='-',color='k',label='$Data$')
                ax.plot(ground_data.iloc[0]['soil_temperature_100_to_255cm'],ls=':',color='k',label='$Model$')
                # ax2.legend(loc='upper right')
                ax.legend(loc='upper left')
                
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('timeserie_ground_temperature_{}_{}'.format(city,year))),bbox_inches='tight')
                plt.show()
                
            # Affichage de la température en fonction de la profondeur à partir des données Open-Météo
            if False:
                fig, ax = plt.subplots(figsize=(5,5),dpi=300)
                dict_col = {4/100:'soil_temperature_0_to_7cm',
                            18/100:'soil_temperature_7_to_28cm', 
                            64/100:'soil_temperature_28_to_100cm',
                            178/100:'soil_temperature_100_to_255cm'}
                for depth in dict_col.keys():
                    modelled_col = dict_col.get(depth)
                    # ax.plot(data[modelled_col].values, [depth]*len(data),ls='',marker='.',color='tab:blue',alpha=0.01)
                    ax.plot(data[(data.index.month==1)&(data.index.day==1)&(data.index.hour==0)][modelled_col].values, [depth]*len(data[(data.index.month==1)&(data.index.day==1)&(data.index.hour==0)]),ls='',marker='.',color='tab:red',alpha=0.5)
                
                
                # calibration de xi pour l'estimation de la température initiale
                if True:
                    Y = np.linspace(0,3)
                    X = [get_init_ground_temperature(y, data) for y in Y]
                    ax.plot(X,Y,color='k')
                
                ax.set_ylim(ax.get_ylim()[::-1])
                depth_list = [((1/2)*mh+(1/2)*ml)/100 for ml,mh in [(28,100),(100,255)]]
                ax.plot([data.temperature_2m.median(),data.temperature_2m.median()],[min(depth_list), max(depth_list)],color='k')
                plt.show()
        
        
            # Affichage (moche) de la température en fonction de la profondeur par modélisation
            if False:
                # liste des profondeur modélisées
                depth_list = [x/100 for x in np.linspace(5,1000,20)]
                ground_data = model_ground_temperature(depth_list, data)
                
                fig, ax = plt.subplots(figsize=(5,5),dpi=300)
                for depth in depth_list:
                    modelled_col = 'modelled_soil_temperature_{:.0f}cm'.format(depth*100)
                    ax.plot(ground_data[modelled_col].values, [depth]*len(ground_data),ls='',marker='.',color='tab:blue',alpha=0.01)
                ax.set_ylim(ax.get_ylim()[::-1])
                ax.plot([ground_data.temperature_2m.median(),ground_data.temperature_2m.median()],[min(depth_list), max(depth_list)],color='k')
                ax.set_ylabel('Profondeur (m)')
                ax.set_ylabel('Température du sol (°C)')
                plt.show()
            
            
            # Affichage des températures sous terraines avec des moyennes par saison (ou par mois ?)
            if False:
                season = False
                month = True
                
                # liste des profondeur modélisées
                depth_list = [x/100 for x in np.linspace(5,1000,30)]
                ground_data = model_ground_temperature(depth_list, data)
                
                if season:
                    fig, ax = plt.subplots(figsize=(5,5),dpi=300)
                    seasons_dict = {'DJF':([12,1,2],'tab:blue'),
                                    'MAM':([3,4,5],'tab:green'),
                                    'JJA':([6,7,8],'tab:red'),
                                    'SON':([9,10,11],'tab:orange')}
                    for season in seasons_dict.keys():
                        season_months, season_color = seasons_dict.get(season)
                        season_mean = ground_data[ground_data.index.month.isin(season_months)]
                        season_mean = pd.DataFrame(season_mean.mean()).T
                        
                        season_plot = []
                        for depth in depth_list:
                            modelled_col = 'modelled_soil_temperature_{:.0f}cm'.format(depth*100)
                            season_plot.append(season_mean[modelled_col].values[0])
                            
                        ax.plot(season_plot, depth_list,color=season_color,label=season)
                        
                    ax.set_ylim(ax.get_ylim()[::-1])
                    ax.legend()
                    ax.plot([ground_data.temperature_2m.median(),ground_data.temperature_2m.median()],[min(depth_list), max(depth_list)],color='k')
                    ax.set_ylabel('Profondeur (m)')
                    ax.set_xlabel('Température du sol (°C)')
                    plt.savefig(os.path.join(figs_folder,'{}.png'.format('modelling_of_ground_temperature_seasons')),bbox_inches='tight')
                    plt.show()
                    
                if month:
                    fig, ax = plt.subplots(figsize=(5,5),dpi=300)
                    cmap = plt.colormaps.get_cmap('viridis')
                    line_styles = ['-',':','--','-.']
                    
                    month_dict = {pd.to_datetime('2000-{:02d}-01'.format(m)).strftime('%B'):([m],cmap(0.5-np.cos(2*np.pi*(m-1)/12)/2),line_styles[m%4]) for m in range(1,13)}

                    for month in month_dict.keys():
                        month_months, month_color, ls_month = month_dict.get(month)
                        month_mean = ground_data[ground_data.index.month.isin(month_months)]
                        month_mean = pd.DataFrame(month_mean.mean()).T
                        
                        month_plot = []
                        for depth in depth_list:
                            modelled_col = 'modelled_soil_temperature_{:.0f}cm'.format(depth*100)
                            month_plot.append(month_mean[modelled_col].values[0])
                            
                        ax.plot(month_plot, depth_list,color=month_color,ls=ls_month,label=month)
                        
                    ax.set_ylim(ax.get_ylim()[::-1])
                    ax.legend()
                    ax.plot([ground_data.temperature_2m.median(),ground_data.temperature_2m.median()],[min(depth_list), max(depth_list)],color='k')
                    ax.set_ylabel('Profondeur (m)')
                    ax.set_xlabel('Température du sol (°C)')
                    plt.savefig(os.path.join(figs_folder,'{}.png'.format('modelling_of_ground_temperature_months_{}_{}'.format(city,year))),bbox_inches='tight')
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
                
                
                
    #%% Caractérisation et étude du flux solaire par orientation
    if False:
        variables = ['direct_radiation_instant','diffuse_radiation_instant','direct_normal_irradiance_instant']
        city = 'Marseille'
        year = 2022
        
        coordinates = get_coordinates(city)
        data = open_meteo_historical_data(longitude=coordinates[0], latitude=coordinates[1], year=year, hourly_variables=variables)
        meteo_units = get_meteo_units(longitude=coordinates[0], latitude=coordinates[1], year=year, hourly_variables=variables)
        
        # Affichage des données annuelles
        if False:
            for i,c in enumerate(variables):
                plot_timeserie(data[[c]], labels=['{} ({})'.format(c,meteo_units.get(c))], figsize=(15,5),figs_folder = figs_folder, show=True,
                               xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],save_fig='{}_{}_{}'.format(c,city,year))
        
        # Étude de l'azimuth et de l'élévation
        if False:
            warnings.simplefilter("ignore")
    
            dates = data.copy().index
            dates = dates.tz_localize(tz='CET',ambiguous='NaT',nonexistent='NaT')
            altitude = [get_altitude_fast(coordinates[1],coordinates[0],t) if t is not pd.NaT else np.nan for t in dates]
            azimuth = [get_azimuth_fast(coordinates[1],coordinates[0],t) if t is not pd.NaT else np.nan for t in dates]
            
            # ajout de la hauteur du soleil (en degrés)
            data['sun_altitude'] = altitude
            
            # ajout de l'azimuth du soleil (en degrés)
            data['sun_azimuth'] = azimuth
            
            # vérification du lien entre l'irradiance normale et directe
            if False:
                data.sun_altitude = [max(0,e) for e in data.sun_altitude]
                data['verif_sun_direct_irradiance'] = np.sin(np.deg2rad(data.sun_altitude))*data.direct_normal_irradiance_instant
                plot_timeserie(data[['verif_sun_direct_irradiance','direct_radiation_instant']], labels=['{} ({})'.format('direct_normal_irradiance_instant',meteo_units.get('direct_normal_irradiance_instant'))]*2, figsize=(15,5),figs_folder = figs_folder, show=True,alpha=0.5,
                               xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],save_fig='verif_solar_{}_{}'.format(city,year))
            
            # graphe polaire du parcours du soleil dans l'année
            if False:
                fig,ax = plt.subplots(dpi=300,figsize=(5,5),subplot_kw={'projection': 'polar'})

                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_rlabel_position(0)
                ax.set_rlim(bottom=90, top=0)
                ax.set_rticks(list(range(0,91,20)))
                ax.set_yticklabels(['{}°'.format(e) if e!=0 else '' for e in ax.get_yticks()])
                ax.set_xticks(np.pi/180. * np.linspace(0,  360, 12, endpoint=False))
                ax.grid(True)
                ax.set_xticklabels([{0:'N',90:'E',180:'S',270:'W'}.get(e,'{:.0f}°'.format(e)) for e in np.linspace(0,  360, 12, endpoint=False)])
                
                data_solstice_summer = data[(data.index.day==21)&(data.index.month==6)]
                data_solstice_winter = data[(data.index.day==21)&(data.index.month==12)]
                
                ax.plot(2*np.pi/360*data_solstice_summer.sun_azimuth,data_solstice_summer.sun_altitude,color='tab:blue',zorder=3)
                ax.plot(2*np.pi/360*data_solstice_winter.sun_azimuth,data_solstice_winter.sun_altitude,color='tab:blue',zorder=3)
                
                ax.fill_between(list(2*np.pi/360*data_solstice_summer.sun_azimuth.values),0,list(data_solstice_summer.sun_altitude.values), alpha=0.2,color='tab:blue')
                ax.fill_between(list(2*np.pi/360*data_solstice_winter.sun_azimuth.values),0,list(data_solstice_winter.sun_altitude.values), color='w')
                
                # dessin de l'analemme à 12h UTC (13h en hiver en heure locale)
                data_analemme = data.copy()
                data_analemme = data_analemme.tz_localize(tz='CET',ambiguous='NaT',nonexistent='NaT').tz_convert('UTC')
                data_analemme = data_analemme[(data_analemme.index.hour==12)]
                ax.plot(2*np.pi/360*data_analemme.sun_azimuth,data_analemme.sun_altitude,color='tab:blue',lw=1,alpha=0.5,zorder=2)
                
                ax.set_title('{}'.format(city))
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('sun_path_{}_{}'.format(city,year))),bbox_inches='tight')
                plt.show()
                
            # graphe polaire du parcours du soleil dans l'année (comparaison entre villes)
            if False:
                variables = ['direct_radiation_instant','diffuse_radiation_instant','direct_normal_irradiance_instant']
                year = 2022
                
                city_north = 'Brest'
                city_south = 'Nice'
                
                coordinates_Marseille = get_coordinates(city_south)
                coordinates_Lille = get_coordinates(city_north)
                
                data_Lille = open_meteo_historical_data(longitude=coordinates_Lille[0], latitude=coordinates_Lille[1], year=year, hourly_variables=variables)
                data_Marseille = open_meteo_historical_data(longitude=coordinates_Marseille[0], latitude=coordinates_Marseille[1], year=year, hourly_variables=variables)
                
                dates_Lille = data_Lille.copy().index
                dates_Lille = dates_Lille.tz_localize(tz='CET',ambiguous='NaT',nonexistent='NaT')
                altitude_Lille = [get_altitude_fast(coordinates_Lille[1],coordinates_Lille[0],t) if t is not pd.NaT else np.nan for t in dates_Lille]
                azimuth_Lille = [get_azimuth_fast(coordinates_Lille[1],coordinates_Lille[0],t) if t is not pd.NaT else np.nan for t in dates_Lille]
                data_Lille['sun_altitude'] = altitude_Lille
                data_Lille['sun_azimuth'] = azimuth_Lille
                
                dates_Marseille = data_Marseille.copy().index
                dates_Marseille = dates_Marseille.tz_localize(tz='CET',ambiguous='NaT',nonexistent='NaT')
                altitude_Marseille = [get_altitude_fast(coordinates_Marseille[1],coordinates_Marseille[0],t) if t is not pd.NaT else np.nan for t in dates_Marseille]
                azimuth_Marseille = [get_azimuth_fast(coordinates_Marseille[1],coordinates_Marseille[0],t) if t is not pd.NaT else np.nan for t in dates_Marseille]
                data_Marseille['sun_altitude'] = altitude_Marseille
                data_Marseille['sun_azimuth'] = azimuth_Marseille
                
                
                fig,ax = plt.subplots(dpi=300,figsize=(5,5),subplot_kw={'projection': 'polar'})

                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_rlabel_position(0)
                ax.set_rlim(bottom=90, top=0)
                ax.set_rticks(list(range(0,91,20)))
                ax.set_yticklabels(['{}°'.format(e) if e!=0 else '' for e in ax.get_yticks()])
                ax.set_xticks(np.pi/180. * np.linspace(0,  360, 12, endpoint=False))
                ax.grid(True)
                ax.set_xticklabels([{0:'N',90:'E',180:'S',270:'W'}.get(e,'{:.0f}°'.format(e)) for e in np.linspace(0,  360, 12, endpoint=False)])
                
                data_solstice_summer_Lille = data_Lille[(data_Lille.index.day==21)&(data_Lille.index.month==6)]
                data_solstice_winter_Lille = data_Lille[(data_Lille.index.day==21)&(data_Lille.index.month==12)]
                
                data_solstice_summer_Marseille = data_Marseille[(data_Marseille.index.day==21)&(data_Marseille.index.month==6)]
                data_solstice_winter_Marseille = data_Marseille[(data_Marseille.index.day==21)&(data_Marseille.index.month==12)]
                
                ax.plot(2*np.pi/360*data_solstice_summer_Lille.sun_azimuth,data_solstice_summer_Lille.sun_altitude,color='tab:blue',zorder=3)
                ax.plot(2*np.pi/360*data_solstice_winter_Lille.sun_azimuth,data_solstice_winter_Lille.sun_altitude,color='tab:blue',zorder=3)
                ax.fill_between(list(2*np.pi/360*data_solstice_summer_Lille.sun_azimuth.values),0,list(data_solstice_summer_Lille.sun_altitude.values), alpha=0.2,color='tab:blue',label=city_north)
                ax.fill_between(list(2*np.pi/360*data_solstice_winter_Lille.sun_azimuth.values),0,list(data_solstice_winter_Lille.sun_altitude.values), color='w')
                
                ax.plot(2*np.pi/360*data_solstice_summer_Marseille.sun_azimuth,data_solstice_summer_Marseille.sun_altitude,color='tab:red',zorder=3)
                ax.plot(2*np.pi/360*data_solstice_winter_Marseille.sun_azimuth,data_solstice_winter_Marseille.sun_altitude,color='tab:red',zorder=3)
                ax.fill_between(list(2*np.pi/360*data_solstice_summer_Marseille.sun_azimuth.values),0,list(data_solstice_summer_Marseille.sun_altitude.values), alpha=0.2,color='tab:red',label=city_south)
                ax.fill_between(list(2*np.pi/360*data_solstice_winter_Marseille.sun_azimuth.values),0,list(data_solstice_winter_Marseille.sun_altitude.values), color='w')
                
                # dessin de l'analemme à 12h UTC (13h en hiver en heure locale)
                data_analemme_Lille = data_Lille.copy()
                data_analemme_Lille = data_analemme_Lille.tz_localize(tz='CET',ambiguous='NaT',nonexistent='NaT').tz_convert('UTC')
                data_analemme_Lille = data_analemme_Lille[(data_analemme_Lille.index.hour==12)]
                ax.plot(2*np.pi/360*data_analemme_Lille.sun_azimuth,data_analemme_Lille.sun_altitude,color='tab:blue',lw=1,alpha=0.5,zorder=2)
                
                data_analemme_Marseille = data_Marseille.copy()
                data_analemme_Marseille = data_analemme_Marseille.tz_localize(tz='CET',ambiguous='NaT',nonexistent='NaT').tz_convert('UTC')
                data_analemme_Marseille = data_analemme_Marseille[(data_analemme_Marseille.index.hour==12)]
                ax.plot(2*np.pi/360*data_analemme_Marseille.sun_azimuth,data_analemme_Marseille.sun_altitude,color='tab:red',lw=1,alpha=0.5,zorder=2)
                
                ax.legend()
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('sun_path_{}_{}_{}'.format(city_north,city_south,year))),bbox_inches='tight')
                plt.show()
        
            # Récupération des flux solaires par orientation (projections sur paroi)
            if False:
                orientations = ['N','E','S','W','H']
                list_orientation_coef_dri_cols = []
                list_orientation_coef_dif_cols = []
                list_orientation_dri = []
                list_orientation_dif = []
                list_orientation_glob = []
                for ori in orientations:
                    col_coef_dri = 'coefficient_direct_{}_irradiance'.format(ori)
                    col_coef_dif = 'coefficient_diffuse_{}_irradiance'.format(ori)
                    col_dri = 'direct_radiation_{}'.format(ori)
                    col_dif = 'diffuse_radiation_{}'.format(ori)
                    global_ri = 'global_radiation_{}'.format(ori)
                    data[col_coef_dri] = get_direct_solar_irradiance_projection_ratio(ori, data.sun_azimuth, data.sun_altitude)
                    data[col_coef_dif] = get_diffuse_solar_irradiance_projection_ratio(ori)
                    data[col_dri] = data.direct_normal_irradiance_instant * data[col_coef_dri]
                    data[col_dif] = data.diffuse_radiation_instant * data[col_coef_dif]
                    data[global_ri] = data[col_dri] + data[col_dif]
                    
                    list_orientation_coef_dri_cols.append(col_coef_dri)
                    list_orientation_coef_dif_cols.append(col_coef_dif)
                    list_orientation_dri.append(col_dri)
                    list_orientation_dif.append(col_dif)
                    list_orientation_glob.append(global_ri)
                    
                    
                plot_timeserie(data[list_orientation_coef_dri_cols], labels=orientations, figsize=(5,5),figs_folder = figs_folder, show=True,ylabel='Direct solar irradiation ratio (no unit)',legend_loc='upper left',ylim_top=1.,
                               xlim=[pd.to_datetime('{}-12-21'.format(year)), pd.to_datetime('{}-12-22'.format(year))],save_fig='direct_solar_coefficient_orientations_{}_{}_winter'.format(city,year))
                plot_timeserie(data[list_orientation_coef_dri_cols], labels=orientations, figsize=(5,5),figs_folder = figs_folder, show=True,ylabel='Direct solar irradiation ratio (no unit)',legend_loc='upper left',ylim_top=1.,
                               xlim=[pd.to_datetime('{}-06-21'.format(year)), pd.to_datetime('{}-06-22'.format(year))],save_fig='direct_solar_coefficient_orientations_{}_{}_summer'.format(city,year))
                plot_timeserie(data[list_orientation_coef_dri_cols], labels=orientations, figsize=(5,5),figs_folder = figs_folder, show=True,ylabel='Direct solar irradiation ratio (no unit)',legend_loc='upper left',ylim_top=1.,
                               xlim=[pd.to_datetime('{}-03-21'.format(year)), pd.to_datetime('{}-03-22'.format(year))],save_fig='direct_solar_coefficient_orientations_{}_{}_equinoxe'.format(city,year))
                
                plot_timeserie(data[list_orientation_dri], labels=orientations, figsize=(15,5),figs_folder = figs_folder, show=True,ylabel='Direct solar irradiation (W/m2)',
                               xlim=[pd.to_datetime('{}-12-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],save_fig='direct_solar_orientations_{}_{}_winter'.format(city,year))
                plot_timeserie(data[list_orientation_dri], labels=orientations, figsize=(15,5),figs_folder = figs_folder, show=True,ylabel='Direct solar irradiation (W/m2)',
                               xlim=[pd.to_datetime('{}-06-01'.format(year)), pd.to_datetime('{}-06-30'.format(year))],save_fig='direct_solar_orientations_{}_{}_summer'.format(city,year))
                plot_timeserie(data[list_orientation_dri], labels=orientations, figsize=(15,5),figs_folder = figs_folder, show=True,ylabel='Direct solar irradiation (W/m2)',
                               xlim=[pd.to_datetime('{}-03-01'.format(year)), pd.to_datetime('{}-03-31'.format(year))],save_fig='direct_solar_orientations_{}_{}_equinoxe'.format(city,year))
                
                plot_timeserie(data[list_orientation_glob], labels=orientations, figsize=(15,5),figs_folder = figs_folder, show=True,ylabel='Direct solar irradiation (W/m2)',
                               xlim=[pd.to_datetime('{}-12-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],save_fig='global_solar_orientations_{}_{}_winter'.format(city,year))
                plot_timeserie(data[list_orientation_glob], labels=orientations, figsize=(15,5),figs_folder = figs_folder, show=True,ylabel='Direct solar irradiation (W/m2)',
                               xlim=[pd.to_datetime('{}-06-01'.format(year)), pd.to_datetime('{}-06-30'.format(year))],save_fig='global_solar_orientations_{}_{}_summer'.format(city,year))
                plot_timeserie(data[list_orientation_glob], labels=orientations, figsize=(15,5),figs_folder = figs_folder, show=True,ylabel='Direct solar irradiation (W/m2)',
                               xlim=[pd.to_datetime('{}-03-01'.format(year)), pd.to_datetime('{}-03-31'.format(year))],save_fig='global_solar_orientations_{}_{}_equinoxe'.format(city,year))
    
    #%% Test de préparation des données météo pour le calcul thermique
    if False:
        city = 'Marseille'
        period = [2010,2020]
        principal_orientation = 'S'
        
        weather_data = get_historical_weather_data(city,period,principal_orientation,display_units=False)
        
        plot_timeserie(weather_data[['temperature_2m']], figsize=(15,5),figs_folder = figs_folder, show=True,ylabel='External temperature (°C)',labels=['{}'.format(city)],
                       save_fig='external_temperature_{}_{}-{}'.format(city,period[0],period[1]))
        
        plot_timeserie(weather_data[['direct_sun_radiation_{}'.format(principal_orientation),'diffuse_sun_radiation_{}'.format(principal_orientation)]], 
                       figsize=(15,5),figs_folder = figs_folder, show=True,ylabel='Sun radiation - {} façade '.format(principal_orientation) + '(W.m$^{-2}$)',
                       labels=['Direct radiation - {}'.format(city),'Diffuse radiation - {}'.format(city)],
                       colors=['tab:red','tab:blue'], save_fig='solar_radiation_{}_{}-{}'.format(city,period[0],period[1]))
        
    #%% Étude sur l'influence du vent et son rôle dans la thermique du bâtiment
    if False:
        # TODO à faire
        pass
            
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()