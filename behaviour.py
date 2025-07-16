#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:43:59 2024

@author: amounier
"""

import time
import os
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

class Behaviour():
    def __init__(self,name):
        self.name = name
        self.cst_internal_gains = None
        self.nocturnal_ventilation = False 
        self.closing_shutters = False # TODO: à intégrer
        self.full_name = self.name
        
        if self.name == 'conventionnel_th-bce_2020':
            self.heating_rules = {1:[19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 16, 16, 16, 16, 16, 16, 16, 16, 19, 19, 19, 19, 19, 19],
                                  2:[19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 16, 16, 16, 16, 16, 16, 16, 16, 19, 19, 19, 19, 19, 19],
                                  3:[19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 16, 16, 16, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19],
                                  4:[19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 16, 16, 16, 16, 16, 16, 16, 16, 19, 19, 19, 19, 19, 19],
                                  5:[19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 16, 16, 16, 16, 16, 16, 16, 16, 19, 19, 19, 19, 19, 19],
                                  6:[19]*24,
                                  7:[19]*24,}
            
            self.cooling_rules = {1:[26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 30, 30, 30, 30, 30, 30, 30, 30, 26, 26, 26, 26, 26, 26],
                                  2:[26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 30, 30, 30, 30, 30, 30, 30, 30, 26, 26, 26, 26, 26, 26],
                                  3:[26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 30, 30, 30, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26],
                                  4:[26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 30, 30, 30, 30, 30, 30, 30, 30, 26, 26, 26, 26, 26, 26],
                                  5:[26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 30, 30, 30, 30, 30, 30, 30, 30, 26, 26, 26, 26, 26, 26],
                                  6:[26]*24,
                                  7:[26]*24,}
            
            self.presence_rules = {1:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                   2:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                   3:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   4:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                   5:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                                   6:[1]*24,
                                   7:[1]*24,}
            
            self.sleeping_rules = {1:[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                   2:[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                   3:[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                   4:[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                   5:[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                   6:[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                   7:[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],}
            
        elif self.name == 'conventionnel_th-bce_2020_cst':
            self.heating_rules = {i:[19]*24 for i in range(1,8)}
            self.cooling_rules = {i:[26]*24 for i in range(1,8)}
            self.presence_rules = {i:[1]*24 for i in range(1,8)}
            self.sleeping_rules = {i:[1]*24 for i in range(1,8)}
            
        else:
            self.heating_rules = None
            self.cooling_rules = None
            self.presence_rules = None
            
        
    def __str__(self):
        return self.full_name
    
    def update_name(self):
        if self.nocturnal_ventilation:
            self.full_name = self.name + '_natnocvent'
            
    
    def get_set_point_temperature(self, weather_data):
        
        heating_temperature = [self.heating_rules[d.dayofweek+1][d.hour] for d in weather_data.index]
        cooling_temperature = [self.cooling_rules[d.dayofweek+1][d.hour] for d in weather_data.index]
        
        return heating_temperature, cooling_temperature
    
    
    def get_number_equivalent_adults(self, surface, building_type, figs_folder=None):
        def get_nb_max(surface, building_type):
            # print(surface, building_type)
            if building_type in ['TH','SFH']:
                if surface < 30:
                    nb_max = 1
                elif surface < 70:
                    nb_max = 1.75 - 0.01875 * (70 - surface)
                else:
                    nb_max = 0.025 * surface
                return nb_max
            
            if building_type in ['AB','MFH']:
                if surface < 10:
                    nb_max = 1
                elif surface < 50:
                    nb_max = 1.75 - 0.01875 * (50 - surface)
                else:
                    nb_max = 0.035 * surface
                return nb_max
            return 
        
        if self.name in ['conventionnel_th-bce_2020' ,'conventionnel_3cl-dpe_2021', 'conventionnel_th-bce_2020_cst']:
            nb_max = get_nb_max(surface, building_type)
            if nb_max < 1.75:
                nb_ea = nb_max
            else:
                nb_ea = 1.75 + 0.3 * (nb_max - 1.75)
        else:
            nb_ea = None
            
        if figs_folder is not None:
            surface_list = np.linspace(5,150,100)
            n_max_list_sf = [get_nb_max(s,'SFH') for s in surface_list]
            n_max_list_mf = [get_nb_max(s,'MFH') for s in surface_list]
            n_ea_list_sf = [self.get_number_equivalent_adults(s,'SFH',None) for s in surface_list]
            n_ea_list_mf = [self.get_number_equivalent_adults(s,'MFH',None) for s in surface_list]
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot(surface_list, n_max_list, color='tab:red',label='N$_{max}$')
            # ax.plot(surface_list, n_ea_list, color='tab:blue',label='N$_{adeq}$')
            ax.plot(surface_list, n_ea_list_sf, color='tab:blue',label='Single-family')
            ax.plot(surface_list, n_ea_list_mf, color='tab:red',label='Multi-family')
            ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_ylabel('Number of equivalents adults')
            ax.set_xlabel('Surface (m$^2$)')
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('{}_nb_adulte_eq'.format(self.name))),bbox_inches='tight')
            plt.show()
        return nb_ea
        
    
    def get_internal_gains(self, surface, building_type, weather_data):
        if self.name in ['conventionnel_th-bce_2020' ,'conventionnel_3cl-dpe_2021', 'conventionnel_th-bce_2020_cst']:
            internal_gains = []
            
            for d in weather_data.index:
                day_of_week = d.dayofweek + 1
                hour = d.hour
                
                sleeping_time = bool(self.sleeping_rules[day_of_week][hour])
                presence_time = bool(self.presence_rules[day_of_week][hour])
                
                if presence_time and not sleeping_time:
                    equipment_gains = 5.7 * surface # W
                else:
                    equipment_gains = 1.1 * surface # W
                
                if presence_time and not sleeping_time:
                    light_gains = 1.4 * surface # W
                else:
                    light_gains = 0 # W
                    
                if presence_time and not sleeping_time:
                    nb_equivalent_adults = self.get_number_equivalent_adults(surface,building_type)
                    people_gains = 90 * nb_equivalent_adults # W
                elif presence_time and sleeping_time:
                    nb_equivalent_adults = self.get_number_equivalent_adults(surface,building_type)
                    people_gains = 63 * nb_equivalent_adults # W
                else:
                    people_gains = 0 # W
                
                total_gains = equipment_gains + light_gains + people_gains
                internal_gains.append(total_gains)
        else:
            internal_gains = [None]*len(weather_data)
            
        if self.cst_internal_gains is not None:
            gains = self.cst_internal_gains*surface
            internal_gains = [gains]*len(weather_data)
                
        return internal_gains
        
        
    def plot_rules(self,figs_folder=None):
        year=date.today().year
        
        start_date = pd.to_datetime('{}-01-01'.format(year)) # de préférence commençant un lundi
        while start_date.dayofweek > 0:
            start_date += np.timedelta64(1, 'D')
            
        end_date = start_date + 7 * np.timedelta64(1, 'D')
        
        data = pd.DataFrame(index=pd.date_range(start=start_date,end=end_date,freq='h'))
        heating_temperature = [self.heating_rules[d.dayofweek+1][d.hour] for d in data.index]
        cooling_temperature = [self.cooling_rules[d.dayofweek+1][d.hour] for d in data.index]
        data['heating_temperature'] = heating_temperature
        data['cooling_temperature'] = cooling_temperature
        
        fig, ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(data.index,data.heating_temperature,label='Heating',color='tab:red')
        ax.plot(data.index,data.cooling_temperature,label='Cooling',color='tab:blue')
        ax.legend()
        ax.set_ylim(bottom=10.,top=31)
        ax.set_xlim([start_date,end_date])
        ax.set_ylabel('Setpoint temperature (°C)')
        
        # data.to_csv('heating_cooling_conventionnel.csv')
        
        locator = mdates.AutoDateLocator()
        # formatter = mdates.ConciseDateFormatter(locator)
        formatter = mdates.DateFormatter('%a')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        if figs_folder is not None:
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('{}_heating_cooling_rules'.format(self.name))),bbox_inches='tight')
            
        plt.show()
    
    
#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_behaviour'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
    
    #%% Test de la classe Behaviour
    if True:
        behaviour = 'conventionnel_th-bce_2020'
        conventionnel = Behaviour(behaviour)
        
        
        # Affichage des règles de consommation conventionnelle
        if True:
            # conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
            # conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
            
            conventionnel.plot_rules(figs_folder)
            # conventionnel.get_number_equivalent_adults(0,figs_folder)
        
        from meteorology import get_coordinates, open_meteo_historical_data
        
        city = 'Marseille'
        year = 2021
        variables = ['temperature_2m']
        coordinates = get_coordinates(city)
        weather_data = open_meteo_historical_data(longitude=coordinates[0], latitude=coordinates[1], year=year, hourly_variables=variables)
        
        # Graphe pour des apports internes constants
        # conventionnel.cst_internal_gains = 4.17 # W/m2
        
        heating_setpoint, cooling_setpoint = conventionnel.get_set_point_temperature(weather_data)
        internal_gains = conventionnel.get_internal_gains(80,'SFH',weather_data)
        weather_data['internal_gains'] = internal_gains
        
        # Graphe des apports internes
        if True:
            start_date = pd.to_datetime('{}-01-01'.format(year)) # de préférence commençant un lundi
            while start_date.dayofweek > 0:
                start_date += np.timedelta64(1, 'D')
                
            end_date = start_date + 7 * np.timedelta64(1, 'D')
            
            data = pd.DataFrame(index=pd.date_range(start=start_date,end=end_date,freq='h'))
            data = data.join(weather_data[['internal_gains']])
            
            
            fig, ax = plt.subplots(dpi=300,figsize=(5,5))
            ax.plot(data.index,data.internal_gains,label='Internal gains',color='k')
            ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_ylabel('Internal gains for a 80m$^2$ single-family house (W)')
            ax.set_xlim([start_date,end_date])
            
            locator = mdates.AutoDateLocator()
            # formatter = mdates.ConciseDateFormatter(locator)
            formatter = mdates.DateFormatter('%a')
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            
            if figs_folder is not None:
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('{}_internal_gains_rules'.format(behaviour))),bbox_inches='tight')
        
        # graphe du nombre d'adultes équivalent 
        if True:
            conventionnel.get_number_equivalent_adults(20,'SFH',figs_folder=figs_folder)
        
        
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()