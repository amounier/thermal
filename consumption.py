#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 15:04:48 2025

@author: amounier
"""

import time 
import os 
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pickle
import multiprocessing
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.optimize import curve_fit
from matplotlib.cbook import get_sample_data
import tqdm
import matplotlib as mpl

from administrative import France, Climat
from typologies import Typology
from thermal_model import compute_U_values
from behaviour import Behaviour
from future_meteorology import get_projected_weather_data
from thermal_model import refine_resolution, aggregate_resolution, run_thermal_model
from utils import get_scenarios_color


# depuis resirf_stock.py, cf figures data/Res-IRF
INSULATION_LAYER = {'Single-family':{'wall':{2.5: 0.0, 1.0: 0.02904040404040404, 0.5: 0.08459595959595961, 0.25: 0.19570707070707072,0.1:0.5},
                                     'floor':{2.0: 0.0, 1.5: 0.0, 0.5: 0.06377551020408162, 0.25: 0.1760204081632653,0.3:0.14},},
                    'Multi-family':{'wall':{2.5: 0.003787878787878788, 1.0: 0.036616161616161616, 0.5: 0.09217171717171718, 0.25: 0.20580808080808083,0.1:0.5},
                                    'floor':{2.0: 0.0, 1.5: 0.002551020408163265, 0.5: 0.07397959183673469, 0.25: 0.1913265306122449,0.3:0.15},},}

REFERENCE_TYPOLOGY_CODE = {'Single-family':'FR.N.SFH.01.Gen',
                           'Multi-family':'FR.N.AB.03.Gen'}

CLIMATE_MODELS_NUMBERS = {'CNRM-CM5_ALADIN63':0,
                          'CNRM-CM5_HadREM3-GA7':1,
                          'EC-EARTH_RACMO22E':2,
                          'EC-EARTH_HadREM3-GA7':3,
                          'HadGEM2-ES_HadREM3-GA7':4}

ZCL_LIST = France().climats

ZCL_POPULATION_DISTRIBUTION = {'H1a':0.3, 
                               'H1b':0.11, 
                               'H1c':0.16, 
                               'H2a':0.06, 
                               'H2b':0.11, 
                               'H2c':0.1, 
                               'H2d':0.03, 
                               'H3':0.13}

NORTH_ZCL = ['H1a', 'H1b', 'H2a']
SOUTH_ZCL = sorted(list(set(ZCL_LIST)-set(NORTH_ZCL)))

ENERGY_EFFICIENCY_HEATER = {'Electricity-Direct electric':0.95, 
                            'Electricity-Heat pump air':2,
                            'Electricity-Heat pump water':2.5,
                            'Heating-District heating':0.76,
                            'Natural gas-Performance boiler':0.76, 
                            'Oil fuel-Performance boiler':0.76,
                            'Wood fuel-Performance boiler':0.76,} 

ENERGY_EFFICIENCY_COOLER = {'Electricity-Heat pump air':5.65/0.9,
                            'Electricity-Portable unit':3.1/0.9,
                            'No AC':1.} 


def get_carbon_intensity(update=False,plot=False):
    data = pd.read_csv('data/Res-IRF/carbon_emission_tend.csv').rename(columns={'Unnamed: 0':'year'}).set_index('year')
    
    if plot:
        data_to_plot = data.copy()
        data_to_plot = data_to_plot.rename(columns={'Heating':'District heating'})
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ls_list = ['-','--','-.']
        for idx,c in enumerate(data_to_plot.columns):
            ax.plot(data_to_plot[c],label=c,color=plt.get_cmap('viridis')(idx/len(data_to_plot.columns)),ls=ls_list[idx%3])
        # data_to_plot.plot(ax=ax,cmap=plt.get_cmap('viridis'),ls=['-','-.'])
        ax.set_ylabel('Carbon intensity (kgCO$_2$eq.kWh$^{-1}$)')
        ax.set_xlim([2018,2050])
        ax.set_xlabel('')
        ax.legend()
        ax.set_ylim(bottom=0.)
        plt.show()
        
    if update:
        # https://assets.rte-france.com/analyse-et-donnees/2025-09/BE2024%20-%20Rapport%20Complet.pdf p91
        data['Electricity'] = data['Electricity'].clip(upper=0.045)
    
    data['Electricity2'] = data['Electricity']
    data['Electricity3'] = data['Electricity']
    data = data.rename(columns={'Electricity':'Electricity-Direct electric',
                                'Electricity2':'Electricity-Heat pump air', 
                                'Electricity3':'Electricity-Heat pump water',
                                'Natural gas':'Natural gas-Performance boiler', 
                                'Oil fuel':'Oil fuel-Performance boiler', 
                                'Wood fuel':'Wood fuel-Performance boiler', 
                                'Heating':'Heating-District heating',})
    
    return data 
    
ENERGY_CARBON_INTENSITY = get_carbon_intensity()
ENERGY_CARBON_INTENSITY_UPDATE = get_carbon_intensity(update=True)


class Stock:
    def __init__(self,ac_scenario='REF',pz_scenario='REF',zcl='H1a',
                 climate_model='EC-EARTH_HadREM3-GA7',period=[2018,2050],folder=None):
        """
        Stock résidentiel en sortie de Res-IRF'

        Parameters
        ----------
        ac_scenario : str, optional
            scenario AC development: ACM, REF, ACP. The default is 'REF'.
        pz_scenario : str, optional
            scenario priority climate zone: NOF,REF,SOF. The default is 'REF'.
        zcl : str, optional
            climate zone (zcl8). The default is 'H1a'.
        climate_model : str, optional
            GCM/RCM combination. The default is 'EC-EARTH_HadREM3-GA7'.
        """
        self.ac_scenario = ac_scenario
        self.pz_scenario = pz_scenario
        self.zcl = zcl
        self.climate_model = climate_model
        self.climate_model_number = CLIMATE_MODELS_NUMBERS.get(self.climate_model)
        self.start_year = period[0]
        self.end_year = period[1]
        self.zcl_city = Climat(self.zcl).center_prefecture
        
        self.path = folder
        self.figs_path = os.path.join(self.path, 'figs')
        
        self.ac_pz_scenario = '{}_{}'.format(self.ac_scenario,self.pz_scenario)
        self.scenario_name = '{}__{}__{}__{}'.format(self.ac_scenario,self.pz_scenario,self.zcl,self.climate_model)
        self.years = list(range(self.start_year,self.end_year+1))
        
        self.zcl_color = plt.get_cmap('viridis')(ZCL_LIST.index(self.zcl)/len(ZCL_LIST))
        self.scenario_color = get_scenarios_color().get(self.ac_pz_scenario)
        
        self.stock_folder = os.path.join('data','Res-IRF','SCENARIOS',self.scenario_name)
        self.full_stock_path = os.path.join(self.stock_folder,'full_stock.csv')
        self.full_stock = pd.read_csv(self.full_stock_path).rename(columns={str(y):y for y in self.years})
        self.rc_path = os.path.join(self.stock_folder,'RC_needs')
        if 'RC_needs' not in os.listdir(self.stock_folder):
            os.mkdir(self.rc_path)
        self.full_columns = ['Existing','Occupancy status','Income owner','Income tenant','Housing type','Cooling system','Wall','Floor','Roof','Windows','Heating system']
        self.technical_columns = ['Housing type', 'Cooling system', 'Wall', 'Floor', 'Roof', 'Windows', 'Heating system']
        self.technical_stock = self.full_stock.groupby(self.technical_columns)[self.years].sum()
        
        # attributs calculés
        self.reduced_technical_stock = None
        self.reduced_technical_stock_heater_agg = None
        self.typologies = None
        self.energy_needs_heating = None
        self.energy_needs_cooling = None
        self.stock_energy_needs_heating = None
        self.stock_energy_needs_cooling = None
        self.energy_consumption_heating_conv = None
        self.energy_consumption_cooling_conv = None
        self.stock_energy_consumption_heating_conv = None
        self.stock_energy_consumption_cooling_conv = None
        self.heating_use_intensity = None
        self.cooling_use_intensity = None
        self.energy_consumption_heating = None
        self.energy_consumption_cooling = None
        self.stock_energy_consumption_heating = None
        self.stock_energy_consumption_cooling = None
        
        self.energy_needs_hourly = None
        
    def __str__(self):
        return self.scenario_name
    
    
    def compute_reduced_technical_stock(self, threshold=0.8, plot=False):
        print('{} - Stock reduction...'.format(self.scenario_name))
        # on garde les plus grands segments, représentant 80% du parc
        end_stock = self.technical_stock.sum(axis=1)
        # end_stock = self.technical_stock[list(range(2030,2051))].sum(axis=1)
        segments = len(end_stock[end_stock.sort_values(ascending=False).cumsum()<(end_stock.sum()*threshold)])
        largest = end_stock.nlargest(segments)
        reduced_stock = self.technical_stock.loc[largest.index]
        
        # recalage homogène du nombre total de logements
        gap = self.technical_stock.sum()/reduced_stock.sum()
        reduced_stock = reduced_stock * gap
        
        if plot:
            cumsum = end_stock.sort_values(ascending=False).cumsum().values
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(cumsum/cumsum[-1],color=self.zcl_color)
            ax.set_xscale('log')
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
            ax.plot([segments]*2,ylims,color='k',alpha=0.5,zorder=-1)
            ax.plot(xlims,[threshold]*2,color='k',alpha=0.5,zorder=-1)
            ax.text(segments+50,0.05,'{}'.format(segments),va='center',ha='left')
            ax.set_ylim(ylims)
            ax.set_xlim(xlims)
            ax.set_ylabel('Fraction of total households (ratio)')
            ax.set_xlabel('Cumulative count (out of {})'.format(len(self.technical_stock)))
            ax.set_ylim(bottom=0.,top=1.)
            ax.set_title(self.zcl)
            plt.savefig(os.path.join(self.figs_path,'reduced_stock__{}.png'.format(self.scenario_name)), bbox_inches='tight')
            plt.show()
            
            # représentativit du parc par année
            if True:
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                plot_years = [self.start_year,2020,2030,2040,self.end_year]
                for y in plot_years:
                    X = np.linspace(0,len(self.technical_stock),50)
                    Y = [0.]*len(X)
                    for idx,n_seg in enumerate(X):
                        largest = self.technical_stock[y].nlargest(int(n_seg))
                        Y[idx] = largest.sum()/self.technical_stock[y].sum()
                    
                    ax.plot(X,Y,label=y)
                ax.set_xscale('log')
                ylims = ax.get_ylim()
                xlims = ax.get_xlim()
                ax.plot([segments]*2,ylims,color='k',alpha=0.5,zorder=-1)
                
                ax.set_ylim(ylims)
                ax.set_xlim(xlims)
                # ax.set_ylim(bottom=0.,top=1.)
                # ax.set_xlim(left=0.)
                ax.grid()
                ax.set_title(self.zcl)
                ax.legend()
                plt.show()
             
            # acarctéristiques thermique des enveloppes qu'on supprime avec la réduction
            if True:
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                sns.histplot(ax=ax,x=self.technical_stock.index.get_level_values('Wall').map(str),weights=self.technical_stock[2018],stat='percent',fill=False,label='Full 2018')
                sns.histplot(ax=ax,x=self.technical_stock.index.get_level_values('Wall').map(str),weights=self.technical_stock[2050],stat='percent',fill=False,label='Full 2050')
                sns.histplot(ax=ax,x=reduced_stock.index.get_level_values('Wall').map(str),weights=reduced_stock[2018],stat='percent',fill=False,label='Reduced 2018')
                sns.histplot(ax=ax,x=reduced_stock.index.get_level_values('Wall').map(str),weights=reduced_stock[2050],stat='percent',fill=False,label='Reduced 2050')
                ax.legend()
                plt.show()
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                sns.histplot(ax=ax,x=self.technical_stock.index.get_level_values('Floor').map(str),weights=self.technical_stock[2018],stat='percent',fill=False,label='Full 2018')
                sns.histplot(ax=ax,x=self.technical_stock.index.get_level_values('Floor').map(str),weights=self.technical_stock[2050],stat='percent',fill=False,label='Full 2050')
                sns.histplot(ax=ax,x=reduced_stock.index.get_level_values('Floor').map(str),weights=reduced_stock[2018],stat='percent',fill=False,label='Reduced 2018')
                sns.histplot(ax=ax,x=reduced_stock.index.get_level_values('Floor').map(str),weights=reduced_stock[2050],stat='percent',fill=False,label='Reduced 2050')
                ax.legend()
                plt.show()
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                sns.histplot(ax=ax,x=self.technical_stock.index.get_level_values('Roof').map(str),weights=self.technical_stock[2018],stat='percent',fill=False,label='Full 2018')
                sns.histplot(ax=ax,x=self.technical_stock.index.get_level_values('Roof').map(str),weights=self.technical_stock[2050],stat='percent',fill=False,label='Full 2050')
                sns.histplot(ax=ax,x=reduced_stock.index.get_level_values('Roof').map(str),weights=reduced_stock[2018],stat='percent',fill=False,label='Reduced 2018')
                sns.histplot(ax=ax,x=reduced_stock.index.get_level_values('Roof').map(str),weights=reduced_stock[2050],stat='percent',fill=False,label='Reduced 2050')
                ax.legend()
                plt.show()
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                sns.histplot(ax=ax,x=self.technical_stock.index.get_level_values('Windows').map(str),weights=self.technical_stock[2018],stat='percent',fill=False,label='Full 2018')
                sns.histplot(ax=ax,x=self.technical_stock.index.get_level_values('Windows').map(str),weights=self.technical_stock[2050],stat='percent',fill=False,label='Full 2050')
                sns.histplot(ax=ax,x=reduced_stock.index.get_level_values('Windows').map(str),weights=reduced_stock[2018],stat='percent',fill=False,label='Reduced 2018')
                sns.histplot(ax=ax,x=reduced_stock.index.get_level_values('Windows').map(str),weights=reduced_stock[2050],stat='percent',fill=False,label='Reduced 2050')
                ax.legend()
                plt.show()
                
        
        reduced_stock_heater_agg = reduced_stock.copy()
        reduced_stock_heater_agg = reduced_stock_heater_agg.groupby([i for i in reduced_stock_heater_agg.index.names if i not in ['Heating system']]).sum()
        
        self.reduced_technical_stock = reduced_stock
        self.reduced_technical_stock_heater_agg = reduced_stock_heater_agg
        return 
    
    def compute_typologies(self):
        if self.reduced_technical_stock is None:
            self.compute_reduced_technical_stock()
        
        print('{} - Typologies creation...'.format(self.scenario_name))
        res_typologies = pd.DataFrame(index=self.reduced_technical_stock_heater_agg.index)
        typologies = [None]*len(res_typologies.index)
        saver = [None]*len(res_typologies.index)
        
        for num,idx in enumerate(res_typologies.index):
            bt, cs, wall, floor, roof, windows = idx
            
            typo = Typology(REFERENCE_TYPOLOGY_CODE.get(bt))
            typo.air_infiltration = 'medium'
            
            # ajustement des valeurs U
            typo.ceiling_U = roof
            typo.windows_U = windows
            typo.w0_insulation_thickness = typo.w0_insulation_thickness + INSULATION_LAYER.get(bt).get('wall').get(wall)
            typo.w1_insulation_thickness = typo.w1_insulation_thickness + INSULATION_LAYER.get(bt).get('wall').get(wall)
            typo.w2_insulation_thickness = typo.w2_insulation_thickness + INSULATION_LAYER.get(bt).get('wall').get(wall)
            typo.w3_insulation_thickness = typo.w3_insulation_thickness + INSULATION_LAYER.get(bt).get('wall').get(wall)
            typo.floor_insulation_thickness = typo.floor_insulation_thickness + INSULATION_LAYER.get(bt).get('floor').get(floor)
            
            # ajout d'un masquage si fenêtre rénovée et zcl focus
            if windows == 1.3: # level of renovated windows in Res-IRF
                focus_zcl_dict = {'NOF':NORTH_ZCL, 'REF':[], 'SOF':SOUTH_ZCL}
                focus_zcl_list = focus_zcl_dict.get(self.pz_scenario)
                if self.zcl in focus_zcl_list:
                    typo.solar_shader_length = 1. #m
            
            if cs == 'No AC':
                typo.cooler_maximum_power = 0.
                cooling_label = 'N'
            elif cs == 'Electricity-Portable unit':
                typo.cooler_maximum_power =  2500*typo.households # enquete CODA et scraping et leroy merlin
                cooling_label = 'P'
            else:
                typo.cooler_maximum_power =  100*typo.surface # enquete CODA et scraping et leroy merlin meh https://particuliers.engie.fr/economies-energie/conseils-equipements-chauffage/conseils-installation-climatisation/calcul-puissance-clim.html
                cooling_label = 'S'
            typo.heater_maximum_power = 100*typo.surface #meh https://www.clim-pac.fr/quelle-puissance-de-chauffage-par-m%C2%B2/
            
            compute_U_values(typo)
            typologies[num] = typo 
            saver[num] = '{}__segment__{}_cooling{}_uwall{}_ufloor{}_uroof{}_uwindows{}'.format(self.scenario_name, typo.code, cooling_label, wall, floor, roof, windows)
            
        res_typologies['typology'] = typologies
        res_typologies['save'] = saver
        
        self.typologies = res_typologies
        return     
    
    def compute_energy_needs(self, behaviour_str='conventionnel_th-bce_2020_gaussian_noise',
                             nocturnal_ventilation=True,force=False):
        if self.typologies is None:
            self.compute_typologies()
        
        print('{} - Behaviour initialization...'.format(self.scenario_name))
        # comportements d'usage 
        behaviour = Behaviour(behaviour_str)
        behaviour.nocturnal_ventilation = nocturnal_ventilation
        behaviour.update_name()
        
        print('{} - Weather initialization...'.format(self.scenario_name))
        # fichiers meteo
        weather_data_checkfile = ".weather_data_{}_{}_{}_explore2_mod{}".format(self.zcl_city,self.start_year,self.end_year,self.climate_model_number) + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_projected_weather_data(self.zcl, [self.start_year,self.end_year],nmod=self.climate_model_number)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            try:
                weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
            except pickle.UnpicklingError:
                weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
        
        # fichiers de sauvegarde
        extra_saver = '_bhv{}_start{}_end{}'.format(behaviour.full_name,self.start_year,self.end_year)
        typologies = self.typologies
        typologies['complete_save'] = ['{}_{}_daily.parquet'.format(s,extra_saver) for s in typologies.save]
        
        print('{} - Energy needs computation...'.format(self.scenario_name))
        list_runs = []
        for typ,sav in zip(typologies.typology,typologies.complete_save):
            if sav not in os.listdir(self.rc_path) or force:
                list_runs.append((typ,behaviour,weather_data,os.path.join(self.rc_path,sav)))
        
        if len(list_runs) > 0:
            nb_cpu = multiprocessing.cpu_count()-1
            pool = multiprocessing.Pool(nb_cpu)
            pool.starmap(run_and_save, list_runs)
        
        # simulation = pickle.load(open(os.path.join(os.path.join(output_path),'{}.pickle'.format(var_saver)), 'rb'))
        return 
    
    
    def add_level(self, df, ref, level, keep_distribution=False):
        # Add level to df respecting ref distribution (issue de Res-IRF)
        
        df = df.fillna(0)
        share = (ref.unstack(level).T / ref.unstack(level).sum(axis=1)).T.fillna(0)
        temp = pd.concat([df] * share.shape[1], keys=share.columns, names=share.columns.names, axis=1)
        share = reindex_mi(share, temp.columns, axis=1)
        share = reindex_mi(share, temp.index)
        if keep_distribution:
            df = (share * temp).stack(level).dropna()
        else:
            df = (temp).stack(level).dropna()
        return df
    
    
    def format_energy_needs(self):
        if self.typologies is None:
            self.compute_typologies()
            
        self.compute_energy_needs()
        
        energy_heating = self.reduced_technical_stock_heater_agg.copy()
        energy_cooling = self.reduced_technical_stock_heater_agg.copy()
        
        print('{} - Energy needs formatting...'.format(self.scenario_name))
        for typo_index,typo_save in zip(self.typologies.index,self.typologies.complete_save):
            needs = pd.read_parquet(os.path.join(self.rc_path,typo_save))
            needs = aggregate_resolution(needs[['heating_needs','cooling_needs']],resolution='YS', agg_method='sum')
            needs.index = needs.index.year
            
            energy_heating.loc[typo_index] = needs.heating_needs
            energy_cooling.loc[typo_index] = needs.cooling_needs
        
        self.energy_needs_heating = energy_heating
        self.energy_needs_cooling = energy_cooling
        
        # retablissement des systèmes de chauffage 
        # ne pas pondérer les consommations ! c'est unitaire ici
        for idx,y in enumerate(range(self.start_year, self.end_year+1)):
            if idx == 0:
                temp_energy_heating = pd.DataFrame(self.add_level(self.energy_needs_heating[y], self.reduced_technical_stock[y], 'Heating system',keep_distribution=False)).rename(columns={0:y})
                temp_energy_cooling = pd.DataFrame(self.add_level(self.energy_needs_cooling[y], self.reduced_technical_stock[y], 'Heating system',keep_distribution=False)).rename(columns={0:y})
            else: 
                temp_energy_heating[y] = self.add_level(self.energy_needs_heating[y], self.reduced_technical_stock[y], 'Heating system',keep_distribution=False)
                temp_energy_cooling[y] = self.add_level(self.energy_needs_cooling[y], self.reduced_technical_stock[y], 'Heating system',keep_distribution=False)
        temp_energy_heating = temp_energy_heating.loc[self.reduced_technical_stock.index]
        temp_energy_cooling = temp_energy_cooling.loc[self.reduced_technical_stock.index]
        
        self.energy_needs_heating = temp_energy_heating
        self.energy_needs_cooling = temp_energy_cooling
        
        return 
    
    
    def format_energy_needs_hourly(self, year=2020):
        if self.typologies is None:
            self.compute_typologies()

        self.compute_energy_needs()
        typo_number_list = self.reduced_technical_stock_heater_agg[year]
        
        needs = None
        print('{} - Energy needs formatting...'.format(self.scenario_name))
        for typo_index,typo_save,typo_number in zip(self.typologies.index,self.typologies.complete_save,typo_number_list.values):
            temp_needs = pd.read_parquet(os.path.join(self.rc_path,typo_save.replace('_daily.parquet','_hourly.parquet')))
            temp_needs = temp_needs[temp_needs.index.year==year][['heating_needs','cooling_needs']]
            temp_needs = temp_needs * typo_number
            
            if needs is None:
                needs = temp_needs
            else:
                needs.cooling_needs = needs.cooling_needs + temp_needs.cooling_needs
                needs.heating_needs = needs.heating_needs + temp_needs.heating_needs
        
        self.energy_needs_hourly = needs
        
        return 
    
    
    def compute_use_intensity(self, method='isoelastique'):
        if self.energy_consumption_heating_conv is None:
            self.compute_energy_consumption_conv()
            
        # calcul des intensité d'usage (energy bill, cf n-th best)
        if method == 'isoelastique':
            energy_price_data = get_energy_prices()*1e-3 #€/Wh
            energy_price_data['No AC'] = [0]*len(energy_price_data)
            
            energy_heating_bill = self.energy_consumption_heating_conv.copy()
            energy_cooling_bill = self.energy_consumption_cooling_conv.copy()
            energy_heating_vector = np.asarray([e.split('-')[0] for e in energy_heating_bill.index.get_level_values('Heating system')])
            energy_cooling_vector = np.asarray([e.split('-')[0] for e in energy_cooling_bill.index.get_level_values('Cooling system')])
            for y in range(self.start_year, self.end_year+1):
                energy_heating_bill[y] = np.asarray([energy_price_data.loc[y,e] for e in energy_heating_vector])
                energy_cooling_bill[y] = np.asarray([energy_price_data.loc[y,e] for e in energy_heating_vector]) + np.asarray([energy_price_data.loc[y,e] for e in energy_cooling_vector])
            energy_heating_bill = self.energy_consumption_heating_conv * energy_heating_bill
            energy_cooling_bill = self.energy_consumption_cooling_conv * energy_cooling_bill
            
            calibration = 0.5675
            zeta=5
            A = calibration**(-zeta)/((energy_heating_bill[2018]*self.reduced_technical_stock[2018]).sum()/self.reduced_technical_stock[2018].sum())
            # A = calibration**(-5)/energy_heating_bill[2018].mean()
            def use_intensity(bill, A, zeta):
                return (bill*A)**(-1/zeta)
            
            self.heating_use_intensity = use_intensity(energy_heating_bill,A,zeta)
            self.cooling_use_intensity = use_intensity(energy_cooling_bill,A,zeta)
            
        # calcul des intensité d'usage (jeanne astier, CAE données bancaires)
        elif method == 'cae':
            surface_dict = {k:Typology(v).surface/Typology(v).households for k,v in REFERENCE_TYPOLOGY_CODE.items()}
            surface = self.energy_consumption_heating_conv.index.get_level_values('Housing type').map(surface_dict.get).values
            energy_consumption_heating_conv_per_surface = self.energy_consumption_heating_conv.copy()
            energy_consumption_cooling_conv_per_surface = self.energy_consumption_cooling_conv.copy()
            for y in range(self.start_year, self.end_year+1):
                energy_consumption_heating_conv_per_surface[y] = energy_consumption_heating_conv_per_surface[y] / surface * 1e-3
                energy_consumption_cooling_conv_per_surface[y] = energy_consumption_cooling_conv_per_surface[y] / surface * 1e-3
            
            def use_intensity(cons_per_surface,a=0.3,b=123.7):
                return a + b/cons_per_surface
            
            self.heating_use_intensity = use_intensity(energy_consumption_heating_conv_per_surface)
            self.cooling_use_intensity = use_intensity(energy_consumption_cooling_conv_per_surface)
        return 
    
    
    def compute_energy_consumption(self):
        if self.heating_use_intensity is None:
            self.compute_use_intensity()
        if self.energy_consumption_heating_conv is None:
            self.compute_energy_consumption_conv()
            
        self.energy_consumption_heating = self.heating_use_intensity * self.energy_consumption_heating_conv
        self.energy_consumption_cooling = self.cooling_use_intensity * self.energy_consumption_cooling_conv
        # self.energy_consumption_heating = self.energy_consumption_heating_conv
        # self.energy_consumption_cooling = self.energy_consumption_cooling_conv
        return 
    
    
    def compute_energy_consumption_conv(self, force=False):
        save_name_energy_heating = '{}_energy_consumption_heating_conv.parquet'.format(self.scenario_name)
        save_name_energy_cooling = '{}_energy_consumption_cooling_conv.parquet'.format(self.scenario_name)
        
        if save_name_energy_heating in os.listdir(self.path) and not force:
            self.energy_consumption_heating_conv = pd.read_parquet(os.path.join(self.path,save_name_energy_heating))
            self.energy_consumption_cooling_conv = pd.read_parquet(os.path.join(self.path,save_name_energy_cooling))
            return 
        
        if self.energy_needs_heating is None:
            self.format_energy_needs()
            
        print('{} - Standard energy consumption computation...'.format(self.scenario_name))
        
        self.energy_consumption_heating_conv = self.energy_needs_heating.copy()
        self.energy_consumption_cooling_conv = self.energy_needs_cooling.copy()
        for idx in self.energy_consumption_heating_conv.index:
            cs,hs = idx[1], idx[-1]
            hs_eff = ENERGY_EFFICIENCY_HEATER.get(hs)
            cs_eff = ENERGY_EFFICIENCY_COOLER.get(cs)
            
            self.energy_consumption_heating_conv.loc[idx] = self.energy_needs_heating.loc[idx] / hs_eff
            self.energy_consumption_cooling_conv.loc[idx] = self.energy_needs_cooling.loc[idx] / cs_eff
        
        # self.energy_consumption_heating_conv.to_parquet(os.path.join(self.path,save_name_energy_heating))
        # self.energy_consumption_cooling_conv.to_parquet(os.path.join(self.path,save_name_energy_cooling))
        return
        
    
    def compute_stock_energy_needs(self,secondary_use_rate=1.):
        if self.energy_needs_heating is None:
            self.format_energy_needs()
        if self.reduced_technical_stock is None:
            self.compute_reduced_technical_stock()
        
        rts_after = self.compute_secondary_stock()
        
        self.stock_energy_needs_heating = self.energy_needs_heating * rts_after
        self.stock_energy_needs_cooling = self.energy_needs_cooling * rts_after
    
    
    def compute_stock_energy_consumption_conv(self,force=False):
        save_name_energy_heating = '{}_stock_energy_consumption_heating_conv.parquet'.format(self.scenario_name)
        save_name_energy_cooling = '{}_stock_energy_consumption_cooling_conv.parquet'.format(self.scenario_name)
        
        if save_name_energy_heating in os.listdir(self.path) and not force:
            self.stock_energy_consumption_heating_conv = pd.read_parquet(os.path.join(self.path,save_name_energy_heating))
            self.stock_energy_consumption_cooling_conv = pd.read_parquet(os.path.join(self.path,save_name_energy_cooling))
            return 
        
        if self.stock_energy_needs_heating is None or force:
            print('{} - Compute energy needs...'.format(self.scenario_name))
            self.compute_stock_energy_needs()
            
        print('{} - Standard energy consumption computation...'.format(self.scenario_name))
        
        self.stock_energy_consumption_heating_conv = self.stock_energy_needs_heating.copy()
        self.stock_energy_consumption_cooling_conv = self.stock_energy_needs_cooling.copy()
        for idx in self.stock_energy_consumption_heating_conv.index:
            cs,hs = idx[1], idx[-1]
            hs_eff = ENERGY_EFFICIENCY_HEATER.get(hs)
            cs_eff = ENERGY_EFFICIENCY_COOLER.get(cs)
            
            self.stock_energy_consumption_heating_conv.loc[idx] = self.stock_energy_needs_heating.loc[idx] / hs_eff
            self.stock_energy_consumption_cooling_conv.loc[idx] = self.stock_energy_needs_cooling.loc[idx] / cs_eff
        
        # self.stock_energy_consumption_heating_conv.to_parquet(os.path.join(self.path,save_name_energy_heating))
        # self.stock_energy_consumption_cooling_conv.to_parquet(os.path.join(self.path,save_name_energy_cooling))
        return
    
    
    def compute_secondary_stock(self):
        secondary = pd.read_csv(os.path.join('data','DPE','statistics_energy_systems_principal_secondary.csv'))
        secondary = secondary.set_index(['building_type','principal_system']).drop(columns='ratio')
        
        if self.reduced_technical_stock is None:
            self.compute_reduced_technical_stock()
            
        rts_before = self.reduced_technical_stock
        rts_after = rts_before.copy()*0
        s = 0
        for idx in rts_before.index:
            ht,ac,wa,fl,ro,wi,hs = idx
            
            sec_idx = secondary.loc[(ht,hs)].copy()
            sec_idx.loc[hs] = 1-sec_idx.sum()
            sec_idx = sec_idx.to_dict()
            
            for hss,tau in sec_idx.items():
                idx_hss = (ht,ac,wa,fl,ro,wi,hss)
                if hss == hs:
                    f = 1.
                else:
                    secondary_use_rate=1.
                    f = secondary_use_rate
                hss_vals = rts_before.loc[idx] * tau * f
                
                if idx_hss in rts_after.index:
                    rts_after.loc[idx_hss,rts_before.loc[idx].index.to_list()] = rts_after.loc[idx_hss] + hss_vals
                else:
                    s += hss_vals.loc[2018]
                    rts_after.loc[idx_hss,rts_before.loc[idx].index.to_list()] = hss_vals
        return rts_after
        
    
    def compute_stock_energy_consumption(self):
        if self.energy_consumption_heating is None:
            self.compute_energy_consumption()
        if self.reduced_technical_stock is None:
            self.compute_reduced_technical_stock()
        
        rts_after = self.compute_secondary_stock()
                
        self.stock_energy_consumption_heating = self.energy_consumption_heating * rts_after
        self.stock_energy_consumption_cooling = self.energy_consumption_cooling * rts_after
        return 

    
    def get_daily_DHI(self, energy_filter=None,income_filter=None,status_filter=None):
        save_name = '{}_daily_DHI.parquet'.format(self.scenario_name)
        if energy_filter is not None:
            save_name = save_name.replace('.parquet','_{}.parquet'.format(energy_filter))
        if income_filter is not None:
            save_name = save_name.replace('.parquet','_{}.parquet'.format(income_filter))
        if status_filter is not None:
            save_name = save_name.replace('.parquet','_{}.parquet'.format(status_filter))
            
        if save_name not in os.listdir(self.path):
            if self.typologies is None:
                self.compute_typologies()
                
            self.compute_energy_needs()
            
            # ajout du système de chauffage 
            temp_typologies = pd.DataFrame(self.add_level(self.typologies['complete_save'], self.reduced_technical_stock[2018], 'Heating system',keep_distribution=False)).rename(columns={0:'complete_save'})
            temp_typologies = temp_typologies.loc[self.reduced_technical_stock.index]
            
            flag = False
            
            filtered_stock = self.reduced_technical_stock.copy()
            
            # ajout du revenu locataire et filtre
            if income_filter is not None:
                flag = True
                income_tenant_stock = self.full_stock.groupby(self.technical_columns+['Income tenant'])[self.years].sum()
                for idx,y in enumerate(range(self.start_year, self.end_year+1)):
                    if idx == 0:
                        temp_stock = pd.DataFrame(self.add_level(filtered_stock[y], income_tenant_stock[y], 'Income tenant',keep_distribution=True)).rename(columns={0:y})
                    else: 
                        temp_stock[y] = self.add_level(filtered_stock[y], income_tenant_stock[y], 'Income tenant',keep_distribution=True)
                filtered_stock = temp_stock[temp_stock.index.get_level_values('Income tenant')==income_filter]
                filtered_stock = filtered_stock.groupby(self.technical_columns)[self.years].sum()
                
            # ajout du statut de propriété et filtre
            if status_filter is not None:
                if flag:
                    print('Caution, two filters is a non-tested feature.')
                status_stock = self.full_stock.groupby(self.technical_columns+['Occupancy status'])[self.years].sum()
                for idx,y in enumerate(range(self.start_year, self.end_year+1)):
                    if idx == 0:
                        temp_stock = pd.DataFrame(self.add_level(filtered_stock[y], status_stock[y], 'Occupancy status',keep_distribution=True)).rename(columns={0:y})
                    else: 
                        temp_stock[y] = self.add_level(filtered_stock[y], status_stock[y], 'Occupancy status',keep_distribution=True)
                filtered_stock = temp_stock[temp_stock.index.get_level_values('Occupancy status')==status_filter]
                filtered_stock = filtered_stock.groupby(self.technical_columns)[self.years].sum()

            
            DHI = None
            print('{} - Energy needs formatting...'.format(self.scenario_name))
            for typo_index,typo_save in zip(temp_typologies.index,temp_typologies.complete_save):
                needs = pd.read_parquet(os.path.join(self.rc_path,typo_save))
                needs['year'] = needs.index.year
                needs['nb'] = needs.year.map(filtered_stock.loc[typo_index].to_dict().get)
                needs['DHI_hot'] = needs.hot_DH * needs.nb
                needs['DHI_cold'] = needs.cold_DH * needs.nb
                
                if energy_filter == 'Electricity':
                    if typo_index[-1].split('-')[0] != energy_filter:
                        needs['DHI_cold'] = needs.DHI_cold*0
                
                if DHI is None:
                    DHI = needs[['temperature_2m', 'DHI_hot', 'DHI_cold','nb']]
                else:
                    DHI['DHI_hot'] = DHI['DHI_hot'] + needs['DHI_hot']
                    DHI['DHI_cold'] = DHI['DHI_cold'] + needs['DHI_cold']
                    DHI['nb'] = DHI['nb'] + needs['nb']
            DHI['zcl'] = [self.zcl]*len(DHI)
            DHI['climate_model'] = [self.climate_model]*len(DHI)
            DHI.to_parquet(os.path.join(self.path,save_name))
        
        DHI = pd.read_parquet(os.path.join(self.path,save_name))
        return DHI
    
    
    def get_daily_consumption(self, energy_filter=None):
        if energy_filter is not None:
            save_name = '{}_daily_consumption_{}.parquet'.format(self.scenario_name,energy_filter)
        else:
            save_name = '{}_daily_consumption.parquet'.format(self.scenario_name)
        if save_name not in os.listdir(self.path):
            if self.typologies is None:
                self.compute_typologies()
            if self.heating_use_intensity is None:
                self.compute_use_intensity()
                
            self.compute_energy_needs()
            rts = self.compute_secondary_stock()
            
            # ajout du système de chauffage 
            temp_typologies = pd.DataFrame(self.add_level(self.typologies['complete_save'], self.reduced_technical_stock[2018], 'Heating system',keep_distribution=False)).rename(columns={0:'complete_save'})
            temp_typologies = temp_typologies.loc[self.reduced_technical_stock.index]
            
            consumption = None
            print('{} - Energy needs formatting...'.format(self.scenario_name))
            for typo_index,typo_save in zip(temp_typologies.index,temp_typologies.complete_save):
                needs = pd.read_parquet(os.path.join(self.rc_path,typo_save))
                needs['year'] = needs.index.year
                needs['nb'] = needs.year.map(rts.loc[typo_index].to_dict().get)
                
                cs,hs = typo_index[1], typo_index[-1]
                hs_eff = ENERGY_EFFICIENCY_HEATER.get(hs)
                cs_eff = ENERGY_EFFICIENCY_COOLER.get(cs)
                
                needs['heating_cons_conv'] = needs.heating_needs / hs_eff
                needs['cooling_cons_conv'] = needs.cooling_needs / cs_eff
                needs['heating_ui'] = needs.year.map(self.heating_use_intensity.loc[typo_index].to_dict().get)
                needs['cooling_ui'] = needs.year.map(self.cooling_use_intensity.loc[typo_index].to_dict().get).replace([np.inf, -np.inf], 1)
                needs['heating_cons'] = needs.heating_cons_conv * needs.heating_ui
                needs['cooling_cons'] = needs.cooling_cons_conv * needs.cooling_ui.fillna(0)
                needs['heating_cons'] = needs.heating_cons * needs.nb
                needs['cooling_cons'] = needs.cooling_cons * needs.nb
                
                if all(needs.cooling_cons.isnull()):
                    print(typo_index)
                
                if energy_filter == 'Electricity':
                    if hs.split('-')[0] != energy_filter:
                        needs['heating_cons'] = needs.heating_cons*0
                        
                # calcul des émissions
                needs['heating_carbon_intensity'] = ENERGY_CARBON_INTENSITY[hs].loc[needs.year].values*1e-3 #kgCO2/Wh
                needs['cooling_carbon_intensity'] = ENERGY_CARBON_INTENSITY['Electricity-Direct electric'].loc[needs.year].values*1e-3 #kgCO2/Wh
                needs['heating_emissions'] = needs['heating_cons'] * needs['heating_carbon_intensity']
                needs['cooling_emissions'] = needs['cooling_cons'] * needs['cooling_carbon_intensity']
                needs['heating_carbon_intensity_update'] = ENERGY_CARBON_INTENSITY_UPDATE[hs].loc[needs.year].values*1e-3 #kgCO2/Wh
                needs['cooling_carbon_intensity_update'] = ENERGY_CARBON_INTENSITY_UPDATE['Electricity-Direct electric'].loc[needs.year].values*1e-3 #kgCO2/Wh
                needs['heating_emissions_update'] = needs['heating_cons'] * needs['heating_carbon_intensity_update']
                needs['cooling_emissions_update'] = needs['cooling_cons'] * needs['cooling_carbon_intensity_update']
                
                if consumption is None:
                    consumption = needs[['temperature_2m', 'heating_cons', 'cooling_cons','heating_emissions','cooling_emissions','heating_emissions_update','cooling_emissions_update']]
                else:
                    consumption['heating_cons'] = consumption['heating_cons'] + needs['heating_cons']
                    consumption['cooling_cons'] = consumption['cooling_cons'] + needs['cooling_cons']
                    consumption['heating_emissions'] = consumption['heating_emissions'] + needs['heating_emissions']
                    consumption['cooling_emissions'] = consumption['cooling_emissions'] + needs['cooling_emissions']
                    consumption['heating_emissions_update'] = consumption['heating_emissions_update'] + needs['heating_emissions_update']
                    consumption['cooling_emissions_update'] = consumption['cooling_emissions_update'] + needs['cooling_emissions_update']
                    
            consumption['total_cons'] = consumption.heating_cons + consumption.cooling_cons
            consumption['total_emissions'] = consumption['heating_emissions'] + needs['cooling_emissions']
            consumption['total_emissions_update'] = consumption['heating_emissions_update'] + needs['cooling_emissions_update']
            consumption['zcl'] = [self.zcl]*len(consumption)
            consumption['climate_model'] = [self.climate_model]*len(consumption)
            consumption.to_parquet(os.path.join(self.path,save_name))
        
        consumption = pd.read_parquet(os.path.join(self.path,save_name))
        return consumption
    
    
    def get_daily_power(self, energy_filter=None):
        if energy_filter is not None:
            save_name = '{}_daily_power_{}.parquet'.format(self.scenario_name,energy_filter)
        else:
            save_name = '{}_daily_power.parquet'.format(self.scenario_name)
        if save_name not in os.listdir(self.path):
            if self.typologies is None:
                self.compute_typologies()
            if self.heating_use_intensity is None:
                self.compute_use_intensity()
                
            self.compute_energy_needs()
            rts = self.compute_secondary_stock()
            
            # ajout du système de chauffage 
            temp_typologies = pd.DataFrame(self.add_level(self.typologies['complete_save'], self.reduced_technical_stock[2018], 'Heating system',keep_distribution=False)).rename(columns={0:'complete_save'})
            temp_typologies = temp_typologies.loc[self.reduced_technical_stock.index]
            
            power = None
            print('{} - Energy needs formatting...'.format(self.scenario_name))
            for typo_index,typo_save in zip(temp_typologies.index,temp_typologies.complete_save):
                needs = pd.read_parquet(os.path.join(self.rc_path,typo_save))
                needs['year'] = needs.index.year
                needs['nb'] = needs.year.map(rts.loc[typo_index].to_dict().get)
                
                cs,hs = typo_index[1], typo_index[-1]
                hs_eff = ENERGY_EFFICIENCY_HEATER.get(hs)
                cs_eff = ENERGY_EFFICIENCY_COOLER.get(cs)
                
                needs['heating_pmax_conv'] = needs.heating_pmax / hs_eff
                needs['cooling_pmax_conv'] = needs.cooling_pmax / cs_eff
                needs['heating_ui'] = needs.year.map(self.heating_use_intensity.loc[typo_index].to_dict().get)
                needs['cooling_ui'] = needs.year.map(self.cooling_use_intensity.loc[typo_index].to_dict().get).replace([np.inf, -np.inf], 1)
                needs['heating_pmax'] = needs.heating_pmax_conv * needs.heating_ui
                needs['cooling_pmax'] = needs.cooling_pmax_conv * needs.cooling_ui.fillna(0)
                needs['heating_pmax'] = needs.heating_pmax * needs.nb
                needs['cooling_pmax'] = needs.cooling_pmax * needs.nb
                
                if energy_filter == 'Electricity':
                    if hs.split('-')[0] != energy_filter:
                        needs['heating_pmax'] = needs.heating_pmax*0
                        
                if power is None:
                    power = needs[['temperature_2m', 'heating_pmax', 'cooling_pmax']]
                else:
                    power['heating_pmax'] = power['heating_pmax'] + needs['heating_pmax']
                    power['cooling_pmax'] = power['cooling_pmax'] + needs['cooling_pmax']
            power['total_pmax'] = power.heating_pmax + power.cooling_pmax
            power['zcl'] = [self.zcl]*len(power)
            power['climate_model'] = [self.climate_model]*len(power)
            power.to_parquet(os.path.join(self.path,save_name))
        
        power = pd.read_parquet(os.path.join(self.path,save_name))
        return power


def reindex_mi(df, mi_index, levels=None, axis=0):
    # Return re-indexed DataFrame based on miindex using only few labels. (issue de Res-IRF)

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


def define_DHI_threshold(theta, plot=False,save_fig=None):
    # https://www.novabuild.fr/sites/default/files/actualite/pdf/2021/08/re2020-arrete_du_4_aout_2021-annexe_3-th-bce_2020.pdf
    t = np.clip((theta-16)/3+26,a_min=26,a_max=None)
    
    if plot:
        X = np.linspace(5,30,100)
        Y = define_DHI_threshold(X,plot=False)
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.fill_between(X,[33]*len(X),Y,color='tab:red',label='Hot discomfort',alpha=0.8,ec='k')
        ax.fill_between(X,Y,[19]*len(X),color='w',label='Thermal comfort',zorder=-1,ec='k')
        ax.fill_between(X,[19]*len(X),[15]*len(X),color='tab:blue',label='Cold discomfort',alpha=0.8,ec='k')
        
        ax.set_ylabel('Internal temperature (°C)')
        ax.set_xlabel('Rolling average of external temperature - $\\theta$ (°C)')
        ax.legend()
        ax.set_xlim(X[0],X[-1])
        ax.set_ylim(15,33)
        if save_fig is not None:
            plt.savefig(os.path.join(save_fig,'incomfort_thresholds.png'), bbox_inches='tight')
        plt.show()
        
    return t


def compute_threshold(simulation_daily):
    # TODO : à vérifier/corriger
    # https://www.novabuild.fr/sites/default/files/actualite/pdf/2021/08/re2020-arrete_du_4_aout_2021-annexe_3-th-bce_2020.pdf
    simulation_daily['temperature_shifted'] = simulation_daily.temperature_2m.shift(-1).ffill()
    simulation_daily['theta'] = 0.8*simulation_daily.temperature_2m + 0.2*simulation_daily.temperature_shifted # TODO: à vérifier (0.8*theta(t-1))
    simulation_daily['hot_threshold'] = define_DHI_threshold(simulation_daily.theta.values)
    simulation_daily['cold_threshold'] = [19]*len(simulation_daily)
    return simulation_daily[['temperature_2m','hot_threshold','cold_threshold']]


def compute_incomfort_DH(simulation,simulation_daily):
    simulation = simulation.join(simulation_daily[['hot_threshold','cold_threshold']]).ffill()
    simulation['hot_DH'] = (simulation.internal_temperature-simulation.hot_threshold).clip(lower=0.)
    simulation['cold_DH'] = (simulation.cold_threshold-simulation.internal_temperature).clip(lower=0.)
    simulation_daily = simulation_daily.join(aggregate_resolution(simulation[['hot_DH','cold_DH']], resolution='D', agg_method='sum'))
    
    simulation = simulation[['temperature_2m', 'internal_temperature', 'heating_needs','cooling_needs']]
    simulation_daily = simulation_daily[['temperature_2m', 'hot_DH', 'cold_DH']]
    return simulation, simulation_daily


def compute_daily_simulation(simulation):
    simulation_daily = aggregate_resolution(simulation[['temperature_2m']], resolution='D', agg_method='mean')
    simulation_daily = compute_threshold(simulation_daily)
    
    smulation, simulation_daily = compute_incomfort_DH(simulation,simulation_daily)
    
    simulation_daily = simulation_daily.join(aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='D', agg_method='max').rename(columns={'heating_needs':'heating_pmax','cooling_needs':'cooling_pmax'}))
    simulation_daily = simulation_daily.join(aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='D', agg_method='sum'))
    return simulation_daily


def run_and_save(typo,behaviour,weather_data,save):
    simulation = run_thermal_model(typo, behaviour, weather_data, pmax_warning=False)
    simulation = simulation[['temperature_2m','internal_temperature','heating_needs','cooling_needs']]
    simulation = aggregate_resolution(simulation, resolution='h')
    
    # pour un seul logement
    simulation['heating_needs'] = simulation['heating_needs']/typo.households
    simulation['cooling_needs'] = simulation['cooling_needs']/typo.households
    
    simulation_daily = compute_daily_simulation(simulation)
    simulation_daily.to_parquet(save)
    
    simulation = simulation[simulation.index.year.isin([2020,2050])]
    simulation.to_parquet(save.replace('_daily.parquet','_hourly.parquet'))
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
    return ttc_prices


def piecewise_linear_cooling(T, Tc, kc):
    res = np.piecewise(T, [T<=Tc, T>Tc], [lambda T: 0, lambda T: kc*(T-Tc)])
    return res


def piecewise_linear_heating(T, Th, kh):
    res = np.piecewise(T, [T < Th, T>=Th], [lambda T: -kh*(T-Th), lambda T: 0])
    return res


def identify_thermal_sensitivity(temperature, consumption,k_init=1e10,cooling=False):
    temperature = np.asarray(temperature)
    consumption = np.asarray(consumption)

    # estimation initiale
    p0 = (15, k_init)
    
    # optimisation sur la fonction piecewise_linear
    if cooling:
        popt , e = curve_fit(piecewise_linear_cooling, temperature, consumption, p0=p0)
    else:
        popt , e = curve_fit(piecewise_linear_heating, temperature, consumption, p0=p0)
    pw_linear_consumption = piecewise_linear_cooling(temperature, *popt)
    r2_value = r2_score(consumption,pw_linear_consumption)
    
    Tc_opt, kc_opt = popt
    return Tc_opt, kc_opt, r2_value



def format_daily_consumption(ac_scenarios, pz_scenarios,climate_models_list,
                             output_folder,energy_filter=None):
    # TODO : à accélérer
    color_dict = {}
    
    df_consumption = None
    for acs in ac_scenarios:
        for pzs in pz_scenarios:
            # sce = '{}_{}'.format(acs,pzs)
            
            for cm in climate_models_list:
                for zcl in ZCL_LIST:
                    s = Stock(ac_scenario=acs,pz_scenario=pzs,zcl=zcl,climate_model=cm,folder=output_folder)
                    temp = s.get_daily_consumption(energy_filter=energy_filter)
                    temp['scenario'] = [s.ac_pz_scenario]*len(temp)
                    scenario = s.ac_pz_scenario
                    color_dict[scenario] = s.scenario_color
                    del s
                    if df_consumption is None:
                        df_consumption = temp
                    else:
                        df_consumption = pd.concat([df_consumption, temp])
    
    df_consumption_agg_zcl = df_consumption[['heating_cons','cooling_cons','total_cons',
                                             'heating_emissions','cooling_emissions','total_emissions',
                                             'heating_emissions_update','cooling_emissions_update','total_emissions_update',
                                             'climate_model','scenario']].reset_index().groupby(['index','climate_model','scenario'],as_index=False).sum()
    return df_consumption, df_consumption_agg_zcl, color_dict


def format_yearly_consumption(df_consumption_agg_zcl, climate_models_list):
    df_consumption_agg_zcl_yearly = None
    for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
        df_consumption_agg_zcl_sce = df_consumption_agg_zcl[df_consumption_agg_zcl.scenario==scenario].set_index('index')
        for cm in climate_models_list:
            temp = df_consumption_agg_zcl_sce[df_consumption_agg_zcl_sce.climate_model==cm]
            temp = aggregate_resolution(temp,'YS','sum')
            temp['climate_model'] = [cm]*len(temp)
            temp['scenario'] = [scenario]*len(temp)
            temp = temp.reset_index()
            if df_consumption_agg_zcl_yearly is None:
                df_consumption_agg_zcl_yearly = temp
            else:
                df_consumption_agg_zcl_yearly = pd.concat([df_consumption_agg_zcl_yearly,temp])
    return df_consumption_agg_zcl_yearly


def format_daily_power(ac_scenarios, pz_scenarios,climate_models_list,
                       output_folder,energy_filter='Electricity'):
    df_power = None
    color_dict = {}
    for acs in ac_scenarios:
        for pzs in pz_scenarios:
            for cm in climate_models_list:
                for zcl in ZCL_LIST:
                    s = Stock(ac_scenario=acs,pz_scenario=pzs,zcl=zcl,climate_model=cm,folder=output_folder)
                    temp = s.get_daily_power(energy_filter='Electricity')
                    temp['scenario'] = [s.ac_pz_scenario]*len(temp)
                    scenario = s.ac_pz_scenario
                    color_dict[scenario] = s.scenario_color
                    del s
                    if df_power is None:
                        df_power = temp
                    else:
                        df_power = pd.concat([df_power, temp])
    
    df_power_agg_zcl = df_power[['heating_pmax', 'cooling_pmax', 'total_pmax','climate_model','scenario']].reset_index().groupby(['index','climate_model','scenario'],as_index=False).sum()
    return df_power, df_power_agg_zcl, color_dict


def format_yearly_power(df_power_agg_zcl, climate_models_list):
    df_power_agg_zcl_yearly = None
    for scenario in list(set(df_power_agg_zcl.scenario.values)):
        df_power_agg_zcl_sce = df_power_agg_zcl[df_power_agg_zcl.scenario==scenario].set_index('index')
        for cm in climate_models_list:
            temp_heating = df_power_agg_zcl_sce[(df_power_agg_zcl_sce.climate_model==cm)&(df_power_agg_zcl_sce.index.month.isin([12,1,2]))]
            temp_cooling = df_power_agg_zcl_sce[(df_power_agg_zcl_sce.climate_model==cm)&(df_power_agg_zcl_sce.index.month.isin([6,7,8]))]
            
            temp_heating = aggregate_resolution(temp_heating[['heating_pmax', 'cooling_pmax']],'YS','max')
            temp_heating['climate_model'] = [cm]*len(temp_heating)
            temp_heating['scenario'] = [scenario]*len(temp_heating)
            temp_heating = temp_heating.reset_index()
            
            temp_cooling = aggregate_resolution(temp_cooling[['heating_pmax', 'cooling_pmax']],'YS','max')
            temp_cooling['climate_model'] = [cm]*len(temp_cooling)
            temp_cooling['scenario'] = [scenario]*len(temp_cooling)
            temp_cooling = temp_cooling.reset_index()
            
            temp_heating['cooling_pmax'] = temp_cooling.cooling_pmax
            if df_power_agg_zcl_yearly is None:
                df_power_agg_zcl_yearly = temp_heating
            else:
                df_power_agg_zcl_yearly = pd.concat([df_power_agg_zcl_yearly,temp_heating])
    return df_power_agg_zcl_yearly


def format_daily_dhi(ac_scenarios, pz_scenarios,climate_models_list,
                     output_folder,energy_filter=None,income_filter=None,
                     status_filter=None):
    df_dhi = None
    color_dict = {}
    for acs in ac_scenarios:
        for pzs in pz_scenarios:
            for cm in tqdm.tqdm(climate_models_list,desc='{}_{}'.format(acs,pzs)):
                for zcl in ZCL_LIST:
                    s = Stock(pz_scenario=pzs,zcl=zcl,ac_scenario=acs,climate_model=cm,folder=output_folder)
                    temp = s.get_daily_DHI(income_filter=income_filter,energy_filter=energy_filter,status_filter=status_filter)
                    temp['scenario'] = [s.ac_pz_scenario]*len(temp)
                    scenario = s.ac_pz_scenario
                    color_dict[scenario] = s.scenario_color
                    del s
                    if df_dhi is None:
                        df_dhi = temp
                    else:
                        df_dhi = pd.concat([df_dhi, temp])
    
    df_dhi_agg_zcl = df_dhi[['DHI_hot', 'DHI_cold','nb','climate_model','scenario']].reset_index().groupby(['index','climate_model','scenario'],as_index=False).sum()
    return df_dhi, df_dhi_agg_zcl, color_dict
    

def format_yearly_dhi(df_dhi_agg_zcl, climate_models_list):
    df_dhi_agg_zcl_yearly = None
    households = aggregate_resolution(df_dhi_agg_zcl[['index','nb']].set_index('index'),'YS','mean').values
    for scenario in list(set(df_dhi_agg_zcl.scenario.values)):
        df_dhi_agg_zcl_sce = df_dhi_agg_zcl[df_dhi_agg_zcl.scenario==scenario].set_index('index')
        for cm in climate_models_list:
            temp = df_dhi_agg_zcl_sce[(df_dhi_agg_zcl_sce.climate_model==cm)]
            temp = aggregate_resolution(temp[['DHI_hot', 'DHI_cold']],'YS','sum')
            temp['climate_model'] = [cm]*len(temp)
            temp['households'] = households
            temp['scenario'] = [scenario]*len(temp)
            temp = temp.reset_index()
    
            if df_dhi_agg_zcl_yearly is None:
                df_dhi_agg_zcl_yearly = temp
            else:
                df_dhi_agg_zcl_yearly = pd.concat([df_dhi_agg_zcl_yearly,temp])
    return df_dhi_agg_zcl_yearly


def get_resirf_output(ac_scenario,pz_scenario,climate_model,var,output_folder):
    resirf_df = None 
    for zcl in ZCL_LIST:
        stock_zcl = Stock(ac_scenario=ac_scenario,pz_scenario=pz_scenario,zcl=zcl,climate_model=climate_model,folder=output_folder)
        resirf_output_path = os.path.join(stock_zcl.stock_folder,'output.csv')
        resirf_output = pd.read_csv(resirf_output_path).rename(columns={'Unnamed: 0':'index'}).set_index('index').T
        resirf_output.index = resirf_output.index.map(int)
        resirf_output = resirf_output[[var]]
        resirf_output['scenario'] = [stock_zcl.ac_pz_scenario]*len(resirf_output)
        resirf_output['climate_model'] = [climate_model]*len(resirf_output)
        resirf_output['zcl'] = [zcl]*len(resirf_output)
        resirf_output = resirf_output.reset_index().rename(columns={'index':'year'})
        
        if resirf_df is None:
            resirf_df = resirf_output
        else:
            resirf_df = pd.concat([resirf_df,resirf_output],ignore_index=True)
                        
    sce = resirf_df.scenario.values[0]
    df = resirf_df[resirf_df.scenario==sce].groupby(['year','climate_model'],as_index=False)[var].sum()
        
    mean_cm = df.groupby(['year'],as_index=True)[var].mean()
    return mean_cm

#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_consumption'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie TODO
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
    
    
    #%% Vérification des effets de la réduction des stocks
    if False:
        cm = 'EC-EARTH_HadREM3-GA7'
        acs = 'REF' 
        pzs = "REF" 
        
        for zcl in ['H1a']:
            stock_zcl = Stock(ac_scenario=acs,pz_scenario=pzs,zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
            stock_zcl.compute_reduced_technical_stock(plot=True)
            
    #%% Calcul du coefficient de rigueur
    if False:
        # https://www.statistiques.developpement-durable.gouv.fr/indice-de-rigueur-degres-jours-unifies-aux-niveaux-national-regional-et-departemental
        
        # puis fct de la température annuelle extérieure ou HDD17, pondérée zcl pop
        # utiliser la conso corrigée du climat pour la calibration
        
        sdes_rigueur = pd.read_excel(os.path.join('data','SDES','dju_donnees_nationales_1970_2024_v4.xlsx'),sheet_name='DJU17').set_index('year')
        
        ref = sdes_rigueur['DJU de référence (moyenne sur la période 1991-2020)'].values[0]
        hdd17_models = None
        
        # climate_models_list = ['EC-EARTH_HadREM3-GA7'] 
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        for cm in climate_models_list:
            cmn = CLIMATE_MODELS_NUMBERS.get(cm)
            
            hdd17_models_zcl = None
            for zcl in ZCL_LIST:
                zcl_city = Climat(zcl).center_prefecture
                
                weather_data_checkfile = ".weather_data_{}_{}_{}_explore2_mod{}".format(zcl_city,2018,2050,cmn) + ".pickle"
                if weather_data_checkfile not in os.listdir():
                    weather_data = get_projected_weather_data(zcl, [2018,2050],nmod=cmn)
                    weather_data = refine_resolution(weather_data, resolution='600s')
                    pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
                else:
                    try:
                        weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
                    except pickle.UnpicklingError:
                        weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
                        
                weather_data = weather_data[['temperature_2m']]
                weather_data = aggregate_resolution(weather_data,'d','mean')
                weather_data[zcl] = (17-weather_data.temperature_2m).clip(lower=0.)
                weather_data = aggregate_resolution(weather_data[[zcl]],'YS','sum')   
                
                if hdd17_models_zcl is None:
                    hdd17_models_zcl = weather_data
                else:
                    hdd17_models_zcl = hdd17_models_zcl.join(weather_data)
            
            hdd17_models_zcl[cm] = [0]*len(hdd17_models_zcl)
            for zcl in ZCL_LIST: 
                hdd17_models_zcl[cm] = hdd17_models_zcl[cm] + hdd17_models_zcl[zcl] * ZCL_POPULATION_DISTRIBUTION.get(zcl)
                
            if hdd17_models is None:
                hdd17_models = hdd17_models_zcl[[cm]]
            else:
                hdd17_models = hdd17_models.join(hdd17_models_zcl[[cm]])
        hdd17_models.index = hdd17_models.index.year 
        
        hdd17_models = hdd17_models/ref
        hdd17_models.to_csv('rigueur.csv')
        
        color = plt.get_cmap('viridis')(0.5)
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(sdes_rigueur.index,sdes_rigueur['Indice de rigueur'],label='Reference',color='k')
        hdd17_models.mean(axis=1).plot(ax=ax,color=color,label='Projections')
        ax.fill_between(hdd17_models.index,hdd17_models.mean(axis=1)+hdd17_models.std(axis=1), hdd17_models.mean(axis=1)-hdd17_models.std(axis=1),color=color,alpha=0.27)
        ax.set_ylim(bottom=0.)
        ax.set_xlim(left=2000,right=2050)
        plt.show()
                
    
    #%% Caractérisation des scénarios, variables d'output
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        # climate_models_list = ['EC-EARTH_HadREM3-GA7'] 
        ac_scenarios = ['REF'] # ['ACM','REF','ACP']
        pz_scenarios = ["REF"] # ['NOF','REF','SOF']
        ac_scenarios = ['ACM','REF','ACP']
        # pz_scenarios = ['NOF','REF','SOF']
        
        # détail par zcl
        if False:
            to_plot = {
                       # 'ac_rate':{'ylabel':'AC equipment rate (ratio)',
                       #            'ylim':[0,1]},
                       'ac_stock':{'ylabel':'AC equipment stock (units)',
                                             'ylim':[0,None],
                                             'yfactor':1e6},
                       }
            
            output_vars = [
                           # 'Consumption (TWh)',
                           # 'Adoption cooler (Thousand households)'
                           # 'Renovation (Thousand households)',
                           # 'Consumption standard (TWh)',
                           # 'Surface Electricity-Direct electric (Million m2)',
                           # 'Surface Electricity-Heat pump air (Million m2)',
                           # 'Surface Electricity-Heat pump water (Million m2)',
                           # 'Surface Natural gas-Performance boiler (Million m2)',
                           # 'Cee Multi-family (Thousand households)',
                           # 'Retrofit (Thousand households)',
                           ]
            
            always_vars = ['Stock (Million)',
                           'Stock AC No AC (Million)']
            
            acs = 'REF'
            pzs = 'REF'
            
            resirf_df = None 
            for cm in climate_models_list:
                for zcl in ZCL_LIST:
                    stock_zcl = Stock(ac_scenario=acs,pz_scenario=pzs,zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
                    resirf_output_path = os.path.join(stock_zcl.stock_folder,'output.csv')
                    resirf_output = pd.read_csv(resirf_output_path).rename(columns={'Unnamed: 0':'index'}).set_index('index').T
                    resirf_output.index = resirf_output.index.map(int)
                    resirf_output = resirf_output[always_vars+output_vars]
                    resirf_output['scenario'] = [stock_zcl.ac_pz_scenario]*len(resirf_output)
                    resirf_output['climate_model'] = [cm]*len(resirf_output)
                    resirf_output['zcl'] = [zcl]*len(resirf_output)
                    resirf_output = resirf_output.reset_index().rename(columns={'index':'year'})
                    
                    if resirf_df is None:
                        resirf_df = resirf_output
                    else:
                        resirf_df = pd.concat([resirf_df,resirf_output],ignore_index=True)
            
            sce = resirf_df.scenario.values[0]
            for var in output_vars+list(to_plot.keys()):
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                
                if var == 'ac_rate':
                    df = resirf_df[resirf_df.scenario==sce].groupby(['year','zcl'],as_index=False)[['Stock AC No AC (Million)','Stock (Million)']].mean()
                    df[var] = (df['Stock (Million)']-df['Stock AC No AC (Million)'])/df['Stock (Million)']
                elif var == 'ac_stock':
                    df = resirf_df[resirf_df.scenario==sce].groupby(['year','zcl'],as_index=False)[['Stock AC No AC (Million)','Stock (Million)']].mean()
                    df[var] = (df['Stock (Million)']-df['Stock AC No AC (Million)'])
                else:
                    df = resirf_df[resirf_df.scenario==sce].groupby(['year','zcl'],as_index=False)[var].mean()
                    
                if var in output_vars:
                    yfactor = 1.
                else:
                    yfactor = to_plot.get(var).get('yfactor')
                df[var] = df[var] * yfactor
                
                print(df[df.year==2050]['ac_stock'].values/df[df.year==2017]['ac_stock'].values)
                # print(df[df.year==2050])
                
                low = None
                for zcl in ZCL_LIST:
                    color = plt.get_cmap('viridis')(ZCL_LIST.index(zcl)/len(ZCL_LIST))
                    zcl_df = df[df.zcl==zcl].set_index('year')[var]
                    
                    if low is None:
                        low = np.asarray([0]*len(zcl_df))
                    high = low + zcl_df.values
                    ax.fill_between(zcl_df.index,high,low,label=zcl,color=color,ec='k')
                    low = high
                    
                # mean_cm = df.groupby(['year'],as_index=True)[var].mean()
                # std_cm = df.groupby(['year'],as_index=True)[var].std()
                # color = get_scenarios_color().get(sce)
                
                # ax.plot(mean_cm.index,mean_cm,color='w',lw=3,zorder=0)
                # ax.plot(mean_cm.index,mean_cm,color=color,lw=2,label=sce.replace('_',' - '),zorder=1)
                # ax.fill_between(mean_cm.index,mean_cm+std_cm,mean_cm-std_cm,color=color,alpha=0.37,zorder=-1)
            
                if var in output_vars:
                    ylabel = var
                else:
                    ylabel = to_plot.get(var).get('ylabel')
                
                if var in output_vars:
                    ylims = 0, None
                else:
                    ylims = to_plot.get(var).get('ylim')
                    
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.set_ylim(bottom=ylims[0],top=ylims[1])
                # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
                ax.set_xlim([2018,2050])
                plt.savefig(os.path.join(figs_folder,'resirf_zcl_{}_{}_{}.png'.format(var,acs,pzs)), bbox_inches='tight')
                plt.show()
        
        
        # évolution des variables de sortie Res-IRF
        if True:
            always_vars = ['Stock (Million)',
                           'Stock AC No AC (Million)',
                           'Cooler tax gains (Billion euro)',
                           'Cooler subsidies (Billion euro)']
            
            output_vars = [
                           'Consumption (TWh)',
                           # 'Adoption cooler (Thousand households)',
                           # 'EOL cooler (Thousand households)',
                           # 'Renovation (Thousand households)',
                           # 'Consumption standard (TWh)',
                           # 'Surface Electricity-Direct electric (Million m2)',
                           # 'Surface Electricity-Heat pump air (Million m2)',
                           # 'Surface Electricity-Heat pump water (Million m2)',
                           # 'Surface Natural gas-Performance boiler (Million m2)',
                           # 'Cee Multi-family (Thousand households)',
                           # 'Cee (Billion euro)',
                           # 'Financing total (Billion euro)',
                           # 'Investment total (Billion euro)',
                           'Subsidies total (Billion euro)',
                           'Retrofit (Thousand households)',
                           'Renovation (Thousand households)',
                           'Emission (MtCO2)',
                           'Carbon footprint renovation (MtCO2)',
                           'Carbon footprint construction (MtCO2)',
                           'Consumption Electricity (TWh)',

                           ]
            
            to_plot = {
                       # 'ac_rate':{'ylabel':'AC equipment rate (ratio)',
                       #            'ylim':[0,1],
                       #            'yfactor':1.},
                       # 'ac_stock':{'ylabel':'AC equipment stock (units)',
                       #                       'ylim':[0,None],
                       #                       'yfactor':1e6},
                       # 'subsidies':{'ylabel':'Total subsidies (€)',
                       #                       'ylim':[0,None],
                       #                       'yfactor':1e9},
                       # 'Subsidies total (Billion euro)':{'ylabel':'Total subsidies (G€)',
                       #                                   # 'ylim':[0,5.5],
                       #                                   'ylim':[0,None],
                       #                                   'yfactor':1},
                       'Consumption (TWh)':{'ylabel':'Consumption (TWh)',
                                                         # 'ylim':[0,5.5],
                                                         'ylim':[0,None],
                                                         'yfactor':1},
                       # 'Retrofit (Thousand households)':{'ylabel':'Total retrofit (households)',
                       #                                   'ylim':[0,2.1e6],
                       #                                   'yfactor':1e3},
                       # 'Renovation (Thousand households)':{'ylabel':'Total renovation (households)',
                       #                                   'ylim':[0,0.75e6],
                       #                                   'yfactor':1e3},
                       # 'Emission (MtCO2)':{'ylabel':'Emission (MtCO2)',
                       #                     'ylim':[0,None],
                       #                     'yfactor':1},
                       # 'Carbon footprint renovation (MtCO2)':{'ylabel':'Carbon footprint renovation (MtCO2)',
                       #                                        'ylim':[0,None],
                       #                                        'yfactor':1},
                       # 'Carbon footprint construction (MtCO2)':{'ylabel':'Carbon footprint construction (MtCO2)',
                       #                                          'ylim':[0,None],
                       #                                          'yfactor':1},
                       # 'Consumption Electricity (TWh)':{'ylabel':'Consumption Electricity (TWh)',
                       #                                  'ylim':[0,None],
                       #                                  'yfactor':1},
                       }
            
            resirf_df = None 
            for acs in ac_scenarios:
                for pzs in pz_scenarios:
                    for cm in climate_models_list:
                        for zcl in ZCL_LIST:
                            stock_zcl = Stock(ac_scenario=acs,pz_scenario=pzs,zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
                            resirf_output_path = os.path.join(stock_zcl.stock_folder,'output.csv')
                            resirf_output = pd.read_csv(resirf_output_path).rename(columns={'Unnamed: 0':'index'}).set_index('index').T
                            resirf_output.index = resirf_output.index.map(int)
                            resirf_output = resirf_output[always_vars+output_vars]
                            resirf_output['scenario'] = [stock_zcl.ac_pz_scenario]*len(resirf_output)
                            resirf_output['climate_model'] = [cm]*len(resirf_output)
                            resirf_output['zcl'] = [zcl]*len(resirf_output)
                            resirf_output = resirf_output.reset_index().rename(columns={'index':'year'})
                            
                            if resirf_df is None:
                                resirf_df = resirf_output
                            else:
                                resirf_df = pd.concat([resirf_df,resirf_output],ignore_index=True)
                        
            # for var in output_vars+list(to_plot.keys()):
            for var in list(to_plot.keys()):
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                
                for sce in sorted(list(set(resirf_df.scenario.values))):
                    if var == 'ac_rate':
                        df = resirf_df[resirf_df.scenario==sce].groupby(['year','climate_model'],as_index=False)[['Stock AC No AC (Million)','Stock (Million)']].sum()
                        df[var] = (df['Stock (Million)']-df['Stock AC No AC (Million)'])/df['Stock (Million)']
                    elif var == 'ac_stock':
                        df = resirf_df[resirf_df.scenario==sce].groupby(['year','climate_model'],as_index=False)[['Stock AC No AC (Million)','Stock (Million)']].sum()
                        df[var] = (df['Stock (Million)']-df['Stock AC No AC (Million)'])
                    elif var == 'subsidies':
                        ac_price = 1580 #€ vérifié
                        factor = ac_price/2500
                        df = resirf_df[resirf_df.scenario==sce].groupby(['year','climate_model'],as_index=False)[['Cooler tax gains (Billion euro)','Cooler subsidies (Billion euro)','Subsidies total (Billion euro)']].sum()
                        df[var] = (df['Subsidies total (Billion euro)']+df['Cooler subsidies (Billion euro)']*factor-df['Cooler tax gains (Billion euro)']*factor)
                    else:
                        df = resirf_df[resirf_df.scenario==sce].groupby(['year','climate_model'],as_index=False)[var].sum()
                    
                    if var not in to_plot.keys():
                        yfactor = 1.
                    else:
                        yfactor = to_plot.get(var).get('yfactor')
                    df[var] = df[var] * yfactor
                        
                    mean_cm = df.groupby(['year'],as_index=True)[var].mean()
                    std_cm = df.groupby(['year'],as_index=True)[var].std()
                    color = get_scenarios_color().get(sce)
                    
                    # print(sce, mean_cm.mean())
                    # print(sce, mean_cm.sum())
                    print(sce, mean_cm.loc[list(range(2025,2031))].mean())
                    
                    ax.plot(mean_cm.index,mean_cm,color='w',lw=3,zorder=0)
                    ax.plot(mean_cm.index,mean_cm,color=color,lw=2,label=sce.replace('_',' - '),zorder=1)
                    ax.fill_between(mean_cm.index,mean_cm+std_cm,mean_cm-std_cm,color=color,alpha=0.37,zorder=-1)
                
                if var not in to_plot.keys():
                    ylabel = var
                else:
                    ylabel = to_plot.get(var).get('ylabel')
                
                if var not in to_plot.keys():
                    ylims = 0, None
                else:
                    ylims = to_plot.get(var).get('ylim')
                    
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.set_ylim(bottom=ylims[0],top=ylims[1])
                # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
                ax.set_xlim([2018,2050])
                plt.savefig(os.path.join(figs_folder,'resirf_{}.png'.format(var)), bbox_inches='tight')
                plt.show()
            
                # north south difference
                if False:
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                        
                    for sce in sorted(list(set(resirf_df.scenario.values))):
                        df_north = resirf_df[(resirf_df.scenario==sce)&(resirf_df.zcl.isin(NORTH_ZCL))]
                        df_south = resirf_df[(resirf_df.scenario==sce)&(resirf_df.zcl.isin(SOUTH_ZCL))]
                        
                        df_north = df_north.groupby(['year','climate_model'],as_index=False)[var].sum()
                        df_south = df_south.groupby(['year','climate_model'],as_index=False)[var].sum()
                        
                        if var not in to_plot.keys():
                            yfactor = 1.
                        else:
                            yfactor = to_plot.get(var).get('yfactor')
                        df_north[var] = df_north[var] * yfactor
                        df_south[var] = df_south[var] * yfactor
                        
                        mean_cm_north = df_north.groupby(['year'],as_index=True)[var].mean()
                        mean_cm_south = df_south.groupby(['year'],as_index=True)[var].mean()
                        
                        print(sce,'north', mean_cm_north.sum())
                        print(sce,'south', mean_cm_south.sum())
                        
                        color = get_scenarios_color().get(sce)
                        ax.plot(mean_cm_north.index,mean_cm_north,color=color,label=sce.replace('_',' - '),ls='-')
                        # ax.plot(mean_cm_north.index,mean_cm_north,color=color,label=sce.replace('_',' - ') + ' North',ls=':')
                        ax.plot(mean_cm_south.index,mean_cm_south,color=color,ls=':')
                    
                    if var not in to_plot.keys():
                        ylabel = var
                    else:
                        ylabel = to_plot.get(var).get('ylabel')
                    
                    if var not in to_plot.keys():
                        ylims = 0, None
                    else:
                        ylims = to_plot.get(var).get('ylim')
                    
                    ax.plot([0],[0],color='k',ls='-',label='North')
                    ax.plot([0],[0],color='k',ls=':',label='South')
                    
                    ax.set_ylabel(ylabel)
                    ax.legend()
                    ax.set_ylim(bottom=ylims[0],top=ylims[1])
                    # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
                    ax.set_xlim([2018,2050])
                    plt.savefig(os.path.join(figs_folder,'resirf_{}_north_south.png'.format(var)), bbox_inches='tight')
                    plt.show()
        
        
    #%% premier test
    if False:
        stock = Stock(zcl='H1a',climate_model='EC-EARTH_HadREM3-GA7',folder=os.path.join(output, folder))
        
        # /home/amounier/PycharmProjects/thermal/data/DPE/statistics_energy_systems_principal_secondary.csv
        
        stock.compute_reduced_technical_stock(plot=True) 
        
        print(len(stock.full_stock))
        print(len(stock.technical_stock))
        print(len(stock.reduced_technical_stock))
        print(len(stock.reduced_technical_stock_heater_agg))
        # print(stock.reduced_technical_stock)
        # print(stock.reduced_technical_stock_heater_agg)
        
        # stock.compute_typologies()
        # print(stock.typologies)
        
        # stock.compute_energy_needs()
        # stock.format_energy_needs()
        
        # stock.compute_stock_energy_consumption()
        # cons1 = stock.stock_energy_consumption_heating
        # cons1 = cons1.sum()
        # # dhi = stock.get_daily_DHI()
        # cons2 = stock.get_daily_consumption()
        # cons2 = aggregate_resolution(cons2,'YS','sum')
        # cons2 = cons2['heating_cons']
        # cons2.index = cons2.index.year
        
        # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        # ax.plot(cons1)
        # ax.plot(cons2)
        # plt.show()
        pass
        
    
    #%% lancement des calculs pour tous les modèles climatiques 
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        # climate_models_list = ['EC-EARTH_HadREM3-GA7'] 
        # ac_scenarios = ['REF'] # ['ACM','REF','ACP']
        ac_scenarios = ['ACM','REF','ACP']
        # pz_scenarios = ["REF"] # ['NOF','REF','SOF']
        pz_scenarios = ['NOF','REF','SOF']
        
        for acs in ac_scenarios:
            for pzs in pz_scenarios:
                sce = '{}_{}'.format(acs,pzs)
                
                # if sce not in ['REF_REF','ACM_NOF','ACP_NOF','ACM_SOF','ACP_SOF']:
                #     continue 
                
                for cm in climate_models_list:
                    dict_zcl_stock = {}
                    for zcl in ZCL_LIST:
                        dict_zcl_stock[zcl] = Stock(ac_scenario=acs,pz_scenario=pzs,zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
            
                    for zcl in ZCL_LIST:
                        dict_zcl_stock[zcl].compute_stock_energy_consumption()
        
        
    #%% Étude des consommations d'énergie annuelle et plus fine
    sdes_data_heating = pd.read_csv(os.path.join('data','SDES','consommations_energie_chauffage_residentiel.csv')).set_index('Heating system').rename(columns={str(y):y for y in [2020]})
    sdes_data_cooling = pd.read_csv(os.path.join('data','SDES','consommations_energie_climatisation_residentiel.csv')).set_index('Cooling system').rename(columns={str(y):y for y in [2020]})
    
    # données des chauffages secondaires
    secondary = pd.read_csv(os.path.join('data','DPE','statistics_energy_systems_principal_secondary.csv'))
    secondary = secondary.set_index(['building_type','principal_system']).drop(columns='ratio')
    
    # indices de rigueur
    sdes_rigueur = pd.read_excel(os.path.join('data','SDES','dju_donnees_nationales_1970_2024_v4.xlsx'),sheet_name='DJU17').set_index('year')
    proj_rigueur = pd.read_csv(os.path.join('data','rigueur.csv')).rename(columns={'Unnamed: 0':'year'}).set_index('year')
    
    # 'ENR PAC' sont inclues dans élec mais c'est pas tout à fait clair
    sdes_ref = pd.read_csv(os.path.join('data','SDES','sdes_climat_reel.csv')).set_index('year')
    sdes_ref = sdes_ref.join(sdes_rigueur[['Indice de rigueur']])
    sdes_ref['heating_climat_corr'] = sdes_ref['heating']/sdes_ref['Indice de rigueur']
    # add_elec = 17.955 # including PAC_ENR 
    sdes_ref['heating_electricity_climat_corr'] = (sdes_ref['heating_electricity'])/sdes_ref['Indice de rigueur']
    
    # Consommation initial de chauffage (better)
    if False:
        ref_init = sdes_data_heating[['2017','2018','2019']].mean(axis=1)
        ref_init_cooling = sdes_data_cooling[['2017','2018','2019']].mean(axis=1).sum()
        
        ref_init_climate_corrected = sdes_data_heating[['2017','2018','2019']].copy()
        rig = pd.DataFrame(sdes_rigueur.loc[[2017,2018,2019]]['Indice de rigueur']).T.rename(columns={y:str(y) for y in [2017,2018,2019]})
        for y in ['2017','2018','2019']:
            ref_init_climate_corrected[y] = ref_init_climate_corrected[y]/rig[y].values[0]
        ref_init_climate_corrected = ref_init_climate_corrected.mean(axis=1)
    
        print('Référence')
        print('\tClimat réel: {:.0f} TWh'.format(ref_init.sum()))
        print('\tCorrigé du climat: {:.0f} TWh'.format(ref_init_climate_corrected.sum()))
        print(ref_init_climate_corrected/sum(ref_init_climate_corrected)*100)
        print('\tRefroidissement: {:.2f} TWh'.format(ref_init_cooling))
        
        # for climate_model in CLIMATE_MODELS_NUMBERS.keys():
        for climate_model in ['EC-EARTH_HadREM3-GA7']:
            
            dict_zcl_stock = {}
            for zcl in ZCL_LIST:
                dict_zcl_stock[zcl] = Stock(zcl=zcl,climate_model=climate_model,folder=os.path.join(output, folder))
            
            for zcl in ZCL_LIST:
                # dict_zcl_stock[zcl].compute_stock_energy_consumption_conv()
                dict_zcl_stock[zcl].compute_stock_energy_consumption()
                
            heating = None
            cooling = None
            for zcl in ZCL_LIST:
                if heating is None:
                    heating = pd.DataFrame(dict_zcl_stock[zcl].stock_energy_consumption_heating[2018]).rename(columns={2018:zcl})
                    cooling = pd.DataFrame(dict_zcl_stock[zcl].stock_energy_consumption_cooling[2018]).rename(columns={2018:zcl})
                else:
                    heating = heating.join(pd.DataFrame(dict_zcl_stock[zcl].stock_energy_consumption_heating[2018]).rename(columns={2018:zcl}),how='outer')
                    cooling = cooling.join(pd.DataFrame(dict_zcl_stock[zcl].stock_energy_consumption_cooling[2018]).rename(columns={2018:zcl}),how='outer')
            heating = heating.fillna(0)
            cooling = cooling.fillna(0)
            
            heating_hex = heating.groupby(heating.index.get_level_values('Heating system')).sum().sum(axis=1)
            cooling_hex = cooling.groupby(heating.index.get_level_values('Cooling system')).sum().sum(axis=1)
            rig_cm = proj_rigueur.loc[2018][climate_model]
            
            print('\nModélisation ({})'.format(climate_model))
            print('\tClimat réel: {:.0f} TWh'.format(heating_hex.sum()*1e-12))
            print('\tCorrigé du climat: {:.0f} TWh'.format(heating_hex.sum()/rig_cm*1e-12))
            print((heating_hex/rig_cm)/sum(heating_hex/rig_cm)*100)
            print('\tRefroidissement: {:.2f} TWh'.format(cooling_hex.sum()*1e-12))
        
    # Vérification de la consommation nationale initiale (old)
    if False:
        year = 2020
        # sdes_data_2018 = {k:v*0.8 for k,v in sdes_data_2018.items()}
        
        # ademe_correction 
        # sdes_data_cooling = sdes_data_cooling * 2.634408602
        
        climate_model = 'EC-EARTH_HadREM3-GA7'
        
        dict_zcl_stock = {}
        for zcl in ZCL_LIST:
            dict_zcl_stock[zcl] = Stock(zcl=zcl,climate_model=climate_model,folder=os.path.join(output, folder))
        
        for zcl in ZCL_LIST:
            # dict_zcl_stock[zcl].compute_stock_energy_consumption_conv()
            dict_zcl_stock[zcl].compute_stock_energy_consumption()
            
        heating = None
        cooling = None
        for zcl in ZCL_LIST:
            if heating is None:
                heating = pd.DataFrame(dict_zcl_stock[zcl].stock_energy_consumption_heating[year]).rename(columns={year:zcl})
                cooling = pd.DataFrame(dict_zcl_stock[zcl].stock_energy_consumption_cooling[year]).rename(columns={year:zcl})
            else:
                heating = heating.join(pd.DataFrame(dict_zcl_stock[zcl].stock_energy_consumption_heating[year]).rename(columns={year:zcl}),how='outer')
                cooling = cooling.join(pd.DataFrame(dict_zcl_stock[zcl].stock_energy_consumption_cooling[year]).rename(columns={year:zcl}),how='outer')
        heating = heating.fillna(0)
        cooling = cooling.fillna(0)
        
        heaters = ['Electricity-Direct electric',
                   'Electricity-Heat pump air',
                   'Electricity-Heat pump water',
                   'Natural gas-Performance boiler',
                   'Oil fuel-Performance boiler',
                   'Wood fuel-Performance boiler',
                   'Heating-District heating',]
        
        f = 1. # part des conso pour le secondaire
        
        heating_principal = heating.groupby([heating.index.get_level_values('Housing type'),heating.index.get_level_values('Heating system')]).sum().sum(axis=1)
        heating_secondary = {h:0 for h in heaters}
        for (bt,hs),cons in zip(heating_principal.index,heating_principal.values):
            minus_cons = 0
            for sec_h in heaters:
                tau = secondary.loc[('Single-family',hs),sec_h]
                cons_sec_h = cons * tau * f
                heating_secondary[sec_h] += cons_sec_h
                minus_cons += cons_sec_h
            heating_secondary[hs] += cons - minus_cons
        heating_secondary = pd.DataFrame().from_dict({'Heating system':list(heating_secondary.keys()),'modelled':list(heating_secondary.values())}).set_index('Heating system')        
        # print(heating_secondary)
        
        heating = heating_secondary
        heating = heating * 1e-12
        heating = heating.join(pd.DataFrame(sdes_data_heating.mean(axis=1)).rename(columns={0:'ref'}),how='outer')
        heating.loc['Electricity-Heat pump','modelled'] = heating.loc['Electricity-Heat pump air','modelled'] + heating.loc['Electricity-Heat pump water','modelled']
        heating = heating.dropna(axis=0)
        heating['ratio'] = heating.ref/heating.modelled
        
        heating_climat_corrected = heating.copy()
        heating_climat_corrected['modelled'] = heating_climat_corrected['modelled']/proj_rigueur.loc[2020,climate_model]
        heating_climat_corrected['ref'] = heating_climat_corrected['ref']/sdes_rigueur.loc[2020,'Indice de rigueur']
        heating_climat_corrected['ratio'] = heating_climat_corrected.ref/heating_climat_corrected.modelled
        
        print('Heating consumption {}'.format(year))
        # print((heating/heating.sum(axis=0)*100)[['modelled','ref']])
        print(heating_climat_corrected)
        print(heating_climat_corrected[['modelled','ref']].sum())
        
        # cooling = pd.DataFrame(cooling.groupby(cooling.index.get_level_values('Cooling system')).sum().sum(axis=1)).rename(columns={0:'modelled'})
        # cooling = cooling * 1e-12
        # cooling = cooling.join(sdes_data_cooling[[year]].rename(columns={year:'ref'}),how='outer')
        # cooling.loc['Electricity-AC','modelled'] = cooling.loc['Electricity-Heat pump air','modelled'] + cooling.loc['Electricity-Portable unit','modelled']
        # cooling = cooling.dropna(axis=0)
        
        # print('Cooling consumption {}'.format(year))
        # print(cooling)
        pass
        
    #%% affichage des séries temporelles de consommations
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        # climate_models_list = ['EC-EARTH_HadREM3-GA7'] 
        # ac_scenarios = ['REF'] #['ACM','REF','ACP']
        ac_scenarios = ['ACM','REF','ACP']
        # pz_scenarios = ['REF'] #['NOF','REF','SOF']
        pz_scenarios = ['NOF','REF','SOF']
        
        save = 'df_consumption_agg_zcl_yearly.parquet'
        save_agg_zcl = 'df_consumption_agg_zcl.parquet'
        save_df = 'df_consumption.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format consumption...')
            df_consumption, df_consumption_agg_zcl, color_dict = format_daily_consumption(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder), energy_filter=None)
            df_consumption_agg_zcl_yearly = format_yearly_consumption(df_consumption_agg_zcl, climate_models_list)
            df_consumption.to_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly.to_parquet(os.path.join(output,folder,save))
            df_consumption_agg_zcl.to_parquet(os.path.join(output,folder,save_agg_zcl))
        else:
            df_consumption_agg_zcl = pd.read_parquet(os.path.join(output,folder,save_agg_zcl))
            df_consumption = pd.read_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly = pd.read_parquet(os.path.join(output,folder,save))
            
        save = 'df_consumption_agg_zcl_yearly_elec.parquet'
        save_agg_zcl = 'df_consumption_agg_zcl_elec.parquet'
        save_df = 'df_consumption_elec.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format consumption elec...')
            df_consumption_elec, df_consumption_agg_zcl_elec, color_dict_elec = format_daily_consumption(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder), energy_filter='Electricity')
            df_consumption_agg_zcl_yearly_elec = format_yearly_consumption(df_consumption_agg_zcl_elec, climate_models_list)
            df_consumption_elec.to_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly_elec.to_parquet(os.path.join(output,folder,save))
            df_consumption_agg_zcl_elec.to_parquet(os.path.join(output,folder,save_agg_zcl))
        else:
            df_consumption_agg_zcl_elec = pd.read_parquet(os.path.join(output,folder,save_agg_zcl))
            df_consumption_elec = pd.read_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly_elec = pd.read_parquet(os.path.join(output,folder,save))
            
        # # correction elec
        # df_consumption_agg_zcl_elec['heating_cons'] = df_consumption_agg_zcl_elec['heating_cons']/1.487
        # df_consumption_elec['heating_cons'] = df_consumption_elec['heating_cons']/1.487
        # df_consumption_agg_zcl_yearly_elec['heating_cons'] = df_consumption_agg_zcl_yearly_elec['heating_cons']/1.487
            
        df_consumption_agg_zcl_north = df_consumption[df_consumption.zcl.isin(NORTH_ZCL)][['heating_cons','cooling_cons','total_cons','climate_model','scenario']].reset_index().groupby(['index','climate_model','scenario'],as_index=False).sum()
        df_consumption_agg_zcl_yearly_north = format_yearly_consumption(df_consumption_agg_zcl_north, climate_models_list)
        df_consumption_agg_zcl_south = df_consumption[df_consumption.zcl.isin(SOUTH_ZCL)][['heating_cons','cooling_cons','total_cons','climate_model','scenario']].reset_index().groupby(['index','climate_model','scenario'],as_index=False).sum()
        df_consumption_agg_zcl_yearly_south = format_yearly_consumption(df_consumption_agg_zcl_south, climate_models_list)
        
        # effets réno et changement climatique 
        if True:
            init_years = list(range(2018,2018+6))
            end_years = list(range(2050,2050-6,-1))
            for scenario in list(set(df_consumption_agg_zcl_yearly.scenario.values)):
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly[df_consumption_agg_zcl_yearly.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                rigueur = pd.DataFrame(proj_rigueur.stack()).rename(columns={0:'rigueur'})
                rigueur.index = rigueur.index.set_names(['year','climate_model'])
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.join(rigueur)
                df_consumption_agg_zcl_yearly_sce['heating_cons_climate_adjusted'] = df_consumption_agg_zcl_yearly_sce['heating_cons']/df_consumption_agg_zcl_yearly_sce['rigueur']
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce[['heating_cons','heating_cons_climate_adjusted']]
                
                df_init = df_consumption_agg_zcl_yearly_sce[df_consumption_agg_zcl_yearly_sce.index.get_level_values('year').isin(init_years)]
                df_init = df_init.groupby(df_init.index.get_level_values('climate_model')).mean()
                
                df_end = df_consumption_agg_zcl_yearly_sce[df_consumption_agg_zcl_yearly_sce.index.get_level_values('year').isin(end_years)]
                df_end = df_end.groupby(df_end.index.get_level_values('climate_model')).mean()
                
                if scenario == 'REF_REF':
                    print(df_init.heating_cons.mean())
                    print(df_end.heating_cons.mean())
                    print((1-df_end.heating_cons/(df_init.heating_cons.mean())).mean(),(1-df_end.heating_cons/(df_init.heating_cons.mean())).std())
                
                reno = df_end.heating_cons_climate_adjusted.mean()/df_init.heating_cons_climate_adjusted.mean()
                climat = df_end.heating_cons.mean()/(df_init.heating_cons.mean()*reno)
            
            # data['label'] = ['1990','Pop.','Suff.','Eff.','Ren.','2019']
            # data['color'] = [(183/255,213/255,240/255),'tab:blue',(196/255,88/255,75/255),(229/255,182/255,90/255),(151/255,190/255,97/255),(183/255,213/255,240/255)]
            # data['top'] = data.CO2.cumsum()[:-1].to_list() + [data.CO2.values[-1]]
            # data['bottom'] = data.top.shift(1).fillna(0)
            # enumerate(zip(data.label,data.top,data.bottom,data.color,data.CO2)):
            
            data = pd.DataFrame().from_dict({'label':['2018\n2023','Retrofit','Climate','2045\n2050'],
                                             'top':[df_init.heating_cons.mean()*1e-12,df_init.heating_cons.mean()*1e-12,df_init.heating_cons.mean()*1e-12*reno,df_end.heating_cons.mean()*1e-12],
                                             'bottom':[0,df_init.heating_cons.mean()*1e-12*reno,df_init.heating_cons.mean()*1e-12*reno*climat,0],
                                             'color':[get_scenarios_color().get('REF_REF'),'tab:blue','tab:blue',get_scenarios_color().get('REF_REF')],
                                             'percent':[0,1-reno,1-climat,0],
                                             'err':[df_init.heating_cons.std()*1e-12,0,0,df_end.heating_cons.std()*1e-12]})
            
            fig,ax= plt.subplots(figsize=(5,5),dpi=300)
            (data.set_index('label').percent*0).plot(ax=ax,alpha=0.)
            # prev_l = None
            # prev_t = None
            for idx,(l,t,b,c,ch,err) in enumerate(zip(data.label,data.top,data.bottom,data.color,data.percent,data.err)):
                if l == '2019':
                    b = 0
                ec = None
                if idx in [0,3]:
                    ec = 'k'
                    alpha = 1
                    error = err
                else:
                    alpha = 0.7
                    error = None
                bar = ax.bar(l,t-b,bottom=b,color=c,width=0.9,ec=ec,alpha=alpha,yerr=error,error_kw={'capsize':3})
                if idx not in [0,3]:
                    ax.bar_label(bar, labels=['{:.0f}%'.format(ch*100)], label_type='edge', padding=3,color='k')
                    if ch<0:
                        ax.bar_label(bar, labels=['$\\blacktriangle$'], label_type='center',color='w',size=20)
                    else:
                        ax.bar_label(bar, labels=['$\\blacktriangledown$'], label_type='center',color='w',size=20)
                    # ax.plot([prev_l,l],[prev_t,b],color=c,zorder=-1,alpha=0.9)
                # prev_l = l
                # prev_t = t
            ax.set_xticks([0,1,2,3],labels=['2018\n2023','Retrofit','Climate','2045\n2050'])
            ax.set_xlabel('')
            ax.set_ylabel('Heating consumption (TWh.yr$^{-1}$)')
            ax.set_ylim(bottom=0.,top=380)
            plt.savefig(os.path.join(figs_folder,'heating_consumption_ref_ref_decomposition.png'), bbox_inches='tight')
            plt.show()
                
        # heating (hist+elec)
        if True:
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(sdes_ref.index, sdes_ref['heating_climat_corr'],label='historical',color='k')
            ax.plot(sdes_ref.index, sdes_ref['heating_electricity_climat_corr'],label='electricity',color='tab:red')
            
            print(sdes_ref.loc[list(range(2018,2024))]['heating_electricity_climat_corr'].mean())
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly[df_consumption_agg_zcl_yearly.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                rigueur = pd.DataFrame(proj_rigueur.stack()).rename(columns={0:'rigueur'})
                rigueur.index = rigueur.index.set_names(['year','climate_model'])
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.join(rigueur)
                
                for col in ['heating_cons']:
                    df_consumption_agg_zcl_yearly_sce[col] = df_consumption_agg_zcl_yearly_sce[col]/df_consumption_agg_zcl_yearly_sce['rigueur']
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].std()
                
                color = get_scenarios_color().get(scenario)
                ax.plot(heating_mean*1e-12,color=color)
                ax.fill_between(heating_std.index,heating_mean.values*1e-12+heating_std.values*1e-12,heating_mean.values*1e-12-heating_std.values*1e-12,alpha=0.5,color=color)
                
            for scenario in list(set(df_consumption_agg_zcl_elec.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_elec[df_consumption_agg_zcl_yearly_elec.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                rigueur = pd.DataFrame(proj_rigueur.stack()).rename(columns={0:'rigueur'})
                rigueur.index = rigueur.index.set_names(['year','climate_model'])
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.join(rigueur)
                
                for col in ['heating_cons']:
                    df_consumption_agg_zcl_yearly_sce[col] = df_consumption_agg_zcl_yearly_sce[col]/df_consumption_agg_zcl_yearly_sce['rigueur']
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].std()
                
                print(heating_mean.loc[list(range(2018,2024))].mean()*1e-12)
                
                color = get_scenarios_color().get(scenario)
                ax.plot(heating_mean*1e-12,color='tab:orange')
                ax.fill_between(heating_std.index,heating_mean.values*1e-12+heating_std.values*1e-12,heating_mean.values*1e-12-heating_std.values*1e-12,alpha=0.5,color='tab:orange')
            
            axin = ax.inset_axes([0.01, (1-0.305)/2, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Climate-adjusted heating consumption (TWh.yr$^{-1}$)')
            ax.legend(loc='center right')
            ax.set_ylim(bottom=0.,top=390)
            # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
            ax.set_xlim([1990,2050])
            plt.savefig(os.path.join(figs_folder,'heating_consumption_ref_ref.png'), bbox_inches='tight')
            plt.show()
            
        # heating (elec)
        if True:
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot(sdes_ref.index, sdes_ref['heating_climat_corr'],label='historical',color='k')
            # ax.plot(sdes_ref.index, sdes_ref['heating_electricity_climat_corr'],label='electricity',color='tab:red')
            
            for scenario in list(set(df_consumption_agg_zcl_elec.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_elec[df_consumption_agg_zcl_yearly_elec.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                rigueur = pd.DataFrame(proj_rigueur.stack()).rename(columns={0:'rigueur'})
                rigueur.index = rigueur.index.set_names(['year','climate_model'])
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.join(rigueur)
                
                # for col in ['heating_cons']:
                #     df_consumption_agg_zcl_yearly_sce[col] = df_consumption_agg_zcl_yearly_sce[col]/df_consumption_agg_zcl_yearly_sce['rigueur']
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].std()
                
                color = get_scenarios_color().get(scenario)
                ax.plot(heating_mean*1e-12,color=color)
                ax.fill_between(heating_std.index,heating_mean.values*1e-12+heating_std.values*1e-12,heating_mean.values*1e-12-heating_std.values*1e-12,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01,0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Electricity consumption (TWh.yr$^{-1}$)')
            # ax.legend(loc='center right')
            ax.set_ylim(bottom=0.,top=90)
            # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'heating_consumption_ref_ref_electricity.png'), bbox_inches='tight')
            plt.show()
            
        # cooling (hist)
        if True:
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(sdes_ref.index, sdes_ref['cooling'],label='historical',color='k')
            
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly[df_consumption_agg_zcl_yearly.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['year'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                # df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                rigueur = pd.DataFrame(proj_rigueur.stack()).rename(columns={0:'rigueur'})
                rigueur.index = rigueur.index.set_names(['year','climate_model'])
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.join(rigueur)
                
                for col in ['heating_cons']:
                    df_consumption_agg_zcl_yearly_sce[col] = df_consumption_agg_zcl_yearly_sce[col]/df_consumption_agg_zcl_yearly_sce['rigueur']
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['cooling_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['cooling_cons'].std()
                
                if scenario == 'REF_REF':
                    print('clim',heating_mean.loc[init_years].mean())
                    print('clim',heating_mean.loc[end_years].mean())
                    print((heating_mean.loc[end_years]/(heating_mean.loc[init_years].mean())).mean(),(heating_mean.loc[end_years]/(heating_mean.loc[init_years].mean())).std())
                
                
                color = get_scenarios_color().get(scenario)
                # ax.plot(heating_mean*1e-12,label=scenario.replace('_',' - '),color=color)
                ax.plot(heating_mean*1e-12,color=color)
                ax.fill_between(heating_std.index,heating_mean.values*1e-12+heating_std.values*1e-12,heating_mean.values*1e-12-heating_std.values*1e-12,alpha=0.5,color=color)
                
            # newax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor='NE', zorder=0)
            # newax.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            # newax.axis('off')
            
            axin = ax.inset_axes([0.01, 0.61, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')

            ax.set_ylabel('Cooling consumption (TWh.yr$^{-1}$)')
            ax.legend(loc='upper left')
            ax.set_ylim(bottom=0.,top=None)
            # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
            ax.set_xlim([1990,2050])
            plt.savefig(os.path.join(figs_folder,'cooling_consumption_ref_ref.png'), bbox_inches='tight')
            plt.show()
            
        # heating (9 scenarios) (relative start)
        if True:
            init_years = list(range(2018,2018+6))
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot(sdes_ref.index, sdes_ref['heating_climat_corr'],label='historical',color='k')
            # ax.plot(sdes_ref.index, sdes_ref['heating_electricity_climat_corr'],label='electricity',color='tab:red')
            
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly[df_consumption_agg_zcl_yearly.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                rigueur = pd.DataFrame(proj_rigueur.stack()).rename(columns={0:'rigueur'})
                rigueur.index = rigueur.index.set_names(['year','climate_model'])
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.join(rigueur)
                
                # for col in ['heating_cons']:
                #     df_consumption_agg_zcl_yearly_sce[col] = df_consumption_agg_zcl_yearly_sce[col]/df_consumption_agg_zcl_yearly_sce['rigueur']
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('HEX',heating_mean.loc[init_years].mean()*1e-12, heating_std.loc[init_years].mean()*1e-12)
                    
                color = get_scenarios_color().get(scenario)
                ax.plot(relative_mean,color=color)
                # ax.fill_between(heating_std.index,heating_mean.values*1e-12+heating_std.values*1e-12,heating_mean.values*1e-12-heating_std.values*1e-12,alpha=0.5,color=color)
                
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Relative change in heating consumption (ratio)')
            # ax.legend(loc='lower right')
            ax.set_ylim(bottom=None,top=None)
            # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'heating_consumption_relative.png'), bbox_inches='tight')
            plt.show()
        
        # cooling (9 scenarios) (relative start)
        if True:
            init_years = list(range(2018,2018+6))
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot(sdes_ref.index, sdes_ref['heating_climat_corr'],label='historical',color='k')
            # ax.plot(sdes_ref.index, sdes_ref['heating_electricity_climat_corr'],label='electricity',color='tab:red')
            
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly[df_consumption_agg_zcl_yearly.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                rigueur = pd.DataFrame(proj_rigueur.stack()).rename(columns={0:'rigueur'})
                rigueur.index = rigueur.index.set_names(['year','climate_model'])
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.join(rigueur)
                
                # for col in ['heating_cons']:
                #     df_consumption_agg_zcl_yearly_sce[col] = df_consumption_agg_zcl_yearly_sce[col]/df_consumption_agg_zcl_yearly_sce['rigueur']
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['cooling_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['cooling_cons'].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('HEX',heating_mean.loc[init_years].mean()*1e-12, heating_std.loc[init_years].mean()*1e-12)
                    
                color = get_scenarios_color().get(scenario)
                ax.plot(relative_mean,color=color)
                # ax.fill_between(heating_std.index,heating_mean.values*1e-12+heating_std.values*1e-12,heating_mean.values*1e-12-heating_std.values*1e-12,alpha=0.5,color=color)
                
            axin = ax.inset_axes([0.01, 0.685, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Relative change in cooling consumption (ratio)')
            # ax.legend(loc='lower right')
            ax.set_ylim(bottom=None,top=None)
            # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cooling_consumption_relative.png'), bbox_inches='tight')
            plt.show()

        # heating nord sud
        if True:
            init_years = list(range(2018,2018+6))
            end_years = list(range(2050,2050-6,-1))
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_north[df_consumption_agg_zcl_yearly_north.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('North',heating_mean.loc[init_years].mean()*1e-12, heating_std.loc[init_years].mean()*1e-12)
                if scenario in ['ACP_NOF','ACP_REF','ACP_SOF']:
                    print('North',scenario, relative_mean.loc[end_years].mean())
                
                color = get_scenarios_color().get(scenario)
                ax.plot(relative_mean,color=color)
                
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Relative change in heating consumption (ratio)')
            # ax.legend()
            ax.set_title('North')
            ax.set_ylim(bottom=None,top=None)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'heating_consumption_relative_north.png'), bbox_inches='tight')
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot([pd.to_datetime('{}-01-01'.format(y)) for y in sdes_data_heating.sum().index], sdes_data_heating.sum().values,label='SDES',color='k')
            
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_south[df_consumption_agg_zcl_yearly_south.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['heating_cons'].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('South',heating_mean.loc[init_years].mean()*1e-12, heating_std.loc[init_years].mean()*1e-12)
                if scenario in ['ACP_NOF','ACP_REF','ACP_SOF']:
                    print('South',scenario, relative_mean.loc[end_years].mean())
                    
                color = get_scenarios_color().get(scenario)
                ax.plot(relative_mean,color=color)
                
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Relative change in heating consumption (ratio)')
            # ax.legend()
            ax.set_title('South')
            ax.set_ylim(bottom=None,top=None)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'heating_consumption_relative_south.png'), bbox_inches='tight')
            plt.show()
            
        # cooling nord sud
        if True:
            init_years = list(range(2018,2018+6))
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_north[df_consumption_agg_zcl_yearly_north.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['cooling_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['cooling_cons'].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('North',heating_mean.loc[init_years].mean()*1e-12, heating_std.loc[init_years].mean()*1e-12)
                if scenario in ['ACP_SOF','ACP_REF','ACP_NOF']:
                    print('North',scenario, relative_mean.loc[end_years].mean())
                    
                color = get_scenarios_color().get(scenario)
                ax.plot(relative_mean,color=color)
                
            axin = ax.inset_axes([0.01, 0.685, 0.305, 0.305],zorder=0)      # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Relative change in cooling consumption (ratio)')
            # ax.legend()
            ax.set_title('North')
            ax.set_ylim(bottom=None,top=None)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cooling_consumption_relative_north.png'), bbox_inches='tight')
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot([pd.to_datetime('{}-01-01'.format(y)) for y in sdes_data_heating.sum().index], sdes_data_heating.sum().values,label='SDES',color='k')
            
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_south[df_consumption_agg_zcl_yearly_south.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')['cooling_cons'].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')['cooling_cons'].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('South',heating_mean.loc[init_years].mean()*1e-12, heating_std.loc[init_years].mean()*1e-12)
                if scenario in ['ACP_SOF','ACP_REF','ACP_NOF']:
                    print('South',scenario, relative_mean.loc[end_years].mean())
                    
                color = get_scenarios_color().get(scenario)
                ax.plot(relative_mean,color=color)
                
            axin = ax.inset_axes([0.01, 0.685, 0.305, 0.305],zorder=0)     # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Relative change in cooling consumption (ratio)')
            # ax.legend()
            ax.set_title('South')
            ax.set_ylim(bottom=None,top=None)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cooling_consumption_relative_south.png'), bbox_inches='tight')
            plt.show()
    
    
    #%% description des systèmes de chauffage
    if False: 
        # # test (ref)
        # years = list(range(2018,2051))
        # stock_heating_system_vars_output = {'joule':'Stock Direct electric (Million)',
        #                                     'pac':'Stock Heat pump (Million)',
        #                                     'oil':'Stock Oil fuel (Million)',
        #                                     'wood':'Stock Wood fuel (Million)',
        #                                     'gaz':'Stock Natural gas (Million)',
        #                                     'reseau':'Stock District heating (Million)'}
            
        # stock_heating_system_output = None
        # for k,v in stock_heating_system_vars_output.items():
        #     temp = pd.DataFrame(get_resirf_output('REF','REF','EC-EARTH_HadREM3-GA7',v,os.path.join(output,folder)).loc[years])
        #     if stock_heating_system_output is None:
        #         stock_heating_system_output = temp
        #     else:
        #         stock_heating_system_output = stock_heating_system_output.join(temp)
        
        # # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        # # ax.set_title('Res-IRF REF')
        # # stock_heating_system_output.plot.area(ax=ax)
        # # ax.set_xlabel('')
        # # plt.show()
        
        # # test (ref)
        # years = list(range(2018,2051))
        # stock_heating_system_vars_output = {'joule':'Stock Direct electric (Million)',
        #                                     'pac':'Stock Heat pump (Million)',
        #                                     'oil':'Stock Oil fuel (Million)',
        #                                     'wood':'Stock Wood fuel (Million)',
        #                                     'gaz':'Stock Natural gas (Million)',
        #                                     'reseau':'Stock District heating (Million)'}
            
        # stock_heating_system_output = None
        # for k,v in stock_heating_system_vars_output.items():
        #     temp = pd.DataFrame(get_resirf_output('ACP','REF','EC-EARTH_HadREM3-GA7',v,os.path.join(output,folder)).loc[years])
        #     if stock_heating_system_output is None:
        #         stock_heating_system_output = temp
        #     else:
        #         stock_heating_system_output = stock_heating_system_output.join(temp)
        
        # # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        # # ax.set_title('Res-IRF ACP')
        # # stock_heating_system_output.plot.area(ax=ax)
        # # ax.set_xlabel('')
        # # plt.show()
        
        # test (ref)
        years = list(range(2018,2051))
        stock_heating_system_vars_output = {'joule':'Stock Direct electric (Million)',
                                            'pac':'Stock Heat pump (Million)',
                                            'oil':'Stock Oil fuel (Million)',
                                            'wood':'Stock Wood fuel (Million)',
                                            'gaz':'Stock Natural gas (Million)',
                                            'reseau':'Stock District heating (Million)'}
            
        stock_heating_system_output = None
        for k,v in stock_heating_system_vars_output.items():
            temp = pd.DataFrame(get_resirf_output('ACM','REF','EC-EARTH_HadREM3-GA7',v,os.path.join(output,folder)).loc[years])
            if stock_heating_system_output is None:
                stock_heating_system_output = temp
            else:
                stock_heating_system_output = stock_heating_system_output.join(temp)
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.set_title('Res-IRF ACM')
        stock_heating_system_output.plot.area(ax=ax)
        ax.set_xlabel('')
        plt.show()
        
        # normalement identique 
        stock_heating_system = None
        for zcl in ZCL_LIST:
            stock = Stock(ac_scenario='REF',pz_scenario='REF',zcl=zcl,climate_model='EC-EARTH_HadREM3-GA7',folder=os.path.join(output,folder))
            stock.compute_reduced_technical_stock()
            stock_zcl = stock.reduced_technical_stock.groupby(stock.reduced_technical_stock.index.get_level_values('Heating system')).sum().T
            for v in ['Electricity-Direct electric', 'Electricity-Heat pump air', 'Electricity-Heat pump water', 'Heating-District heating','Natural gas-Performance boiler', 'Oil fuel-Performance boiler','Wood fuel-Performance boiler']:
                if v not in stock_zcl.columns:
                    stock_zcl[v] = [0]*len(stock_zcl)
            stock_zcl['Stock Direct electric (Million)'] = stock_zcl['Electricity-Direct electric']
            stock_zcl['Stock Heat pump (Million)'] = (stock_zcl['Electricity-Heat pump air'] + stock_zcl['Electricity-Heat pump water'])
            stock_zcl['Stock Oil fuel (Million)'] = stock_zcl['Oil fuel-Performance boiler']
            stock_zcl['Stock Wood fuel (Million)'] = stock_zcl['Wood fuel-Performance boiler']
            stock_zcl['Stock Natural gas (Million)'] = stock_zcl['Natural gas-Performance boiler']
            stock_zcl['Stock District heating (Million)'] = stock_zcl['Heating-District heating']
            stock_zcl = stock_zcl[list(stock_heating_system_vars_output.values())]
            if stock_heating_system is None:
                stock_heating_system = stock_zcl
            else:
                for v in list(stock_heating_system_vars_output.values()):
                    stock_heating_system[v] += stock_zcl[v]
        
        stock_heating_system.index = stock_heating_system.index.map(int)
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        df = stock_heating_system.rename(columns={k:k.replace('Stock ','').replace(' (Million)','') for k in stock_heating_system.columns})
        # .plot.area(ax=ax,cmap=plt.get_cmap('viridis'))
        ax.stackplot(df.index,
                     df.T.values,
                     labels=df.columns,
                     colors=[plt.get_cmap('viridis')(idx/len(df.columns)) for idx in range(len(df.columns))],
                     edgecolor='black',
                     linewidth=1)
        ax.set_title('Only principal heater')
        ax.set_xlabel('')
        ax.set_ylabel('Number of dwellings (units)')
        ax.set_xlim([2018,2050])
        ax.legend(ncols=2,loc='lower center')
        plt.savefig(os.path.join(figs_folder,'principal_heater_evolution.png'), bbox_inches='tight')
        plt.show()
        
        # prise en compte secondaire 
        stock_heating_system = None
        for zcl in ZCL_LIST:
            stock = Stock(ac_scenario='REF',pz_scenario='REF',zcl=zcl,climate_model='EC-EARTH_HadREM3-GA7',folder=os.path.join(output,folder))
            # stock.compute_reduced_technical_stock()
            stock_rts = stock.compute_secondary_stock()
            stock_zcl = stock_rts.groupby(stock_rts.index.get_level_values('Heating system')).sum().T
            for v in ['Electricity-Direct electric', 'Electricity-Heat pump air', 'Electricity-Heat pump water', 'Heating-District heating','Natural gas-Performance boiler', 'Oil fuel-Performance boiler','Wood fuel-Performance boiler']:
                if v not in stock_zcl.columns:
                    stock_zcl[v] = [0]*len(stock_zcl)
            stock_zcl['Stock Direct electric (Million)'] = stock_zcl['Electricity-Direct electric']
            stock_zcl['Stock Heat pump (Million)'] = (stock_zcl['Electricity-Heat pump air'] + stock_zcl['Electricity-Heat pump water'])
            stock_zcl['Stock Oil fuel (Million)'] = stock_zcl['Oil fuel-Performance boiler']
            stock_zcl['Stock Wood fuel (Million)'] = stock_zcl['Wood fuel-Performance boiler']
            stock_zcl['Stock Natural gas (Million)'] = stock_zcl['Natural gas-Performance boiler']
            stock_zcl['Stock District heating (Million)'] = stock_zcl['Heating-District heating']
            stock_zcl = stock_zcl[list(stock_heating_system_vars_output.values())]
            if stock_heating_system is None:
                stock_heating_system = stock_zcl
            else:
                for v in list(stock_heating_system_vars_output.values()):
                    stock_heating_system[v] += stock_zcl[v]
        
        stock_heating_system.index = stock_heating_system.index.map(int)
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.set_title('Secondary heating redistribution')
        df = stock_heating_system.rename(columns={k:k.replace('Stock ','').replace(' (Million)','') for k in stock_heating_system.columns})
        # ax = .plot.area(ax=ax,cmap=plt.get_cmap('viridis'))
        ax.stackplot(df.index,
                     df.T.values,
                     labels=df.columns,
                     colors=[plt.get_cmap('viridis')(idx/len(df.columns)) for idx in range(len(df.columns))],
                     edgecolor='black',
                     linewidth=1)
        ax.set_xlabel('')
        ax.set_ylabel('Number of dwellings (units)')
        ax.set_xlim([2018,2050])
        ax.legend(ncols=2,loc='lower center')
        plt.savefig(os.path.join(figs_folder,'principal_secondary_heater_evolution.png'), bbox_inches='tight')
        plt.show()
        
    
    #%% émissions de GES des consommations d'énergie
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        # climate_models_list = ['EC-EARTH_HadREM3-GA7'] 
        # ac_scenarios = ['REF'] #['ACM','REF','ACP']
        ac_scenarios = ['ACM','REF','ACP']
        # pz_scenarios = ['REF'] #['NOF','REF','SOF']
        pz_scenarios = ['NOF','REF','SOF']
        
        save = 'df_consumption_agg_zcl_yearly.parquet'
        save_agg_zcl = 'df_consumption_agg_zcl.parquet'
        save_df = 'df_consumption.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format consumption...')
            df_consumption, df_consumption_agg_zcl, color_dict = format_daily_consumption(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder), energy_filter=None)
            df_consumption_agg_zcl_yearly = format_yearly_consumption(df_consumption_agg_zcl, climate_models_list)
            df_consumption.to_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly.to_parquet(os.path.join(output,folder,save))
            df_consumption_agg_zcl.to_parquet(os.path.join(output,folder,save_agg_zcl))
        else:
            df_consumption_agg_zcl = pd.read_parquet(os.path.join(output,folder,save_agg_zcl))
            df_consumption = pd.read_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly = pd.read_parquet(os.path.join(output,folder,save))
            
        init_years = list(range(2018,2018+6))
        years = list(range(2018,2051))
        
        # stack ref absolute
        if True:
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            
            construction = get_resirf_output('REF','REF','EC-EARTH_HadREM3-GA7','Carbon footprint construction (MtCO2)',os.path.join(output,folder)).loc[years]
            renovation = get_resirf_output('REF','REF','EC-EARTH_HadREM3-GA7','Carbon footprint renovation (MtCO2)',os.path.join(output,folder)).loc[years]
            
            ax.fill_between(construction.index,construction,color=plt.get_cmap('viridis')(0.15),label='Construction',ec='k')
            ax.fill_between(renovation.index,construction+renovation,construction,color=plt.get_cmap('viridis')(0.6),label='Retrofit',ec='k')
            # var = 'total_emissions'
            var = 'total_emissions_update'
            # var = 'cooling_emissions'
            
            
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly[df_consumption_agg_zcl_yearly.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')[var].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')[var].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('HEX',heating_mean.loc[init_years].mean(), heating_std.loc[init_years].mean())
                    print('HEX',heating_mean.loc[list(range(2028,2028+6))].mean(), heating_mean.loc[list(range(2028,2028+6))].std())
                    
                color = get_scenarios_color().get(scenario)
                # ax.plot(relative_mean,color=color)
                ax.fill_between(heating_mean.index,construction+renovation+heating_mean*1e-9,construction+renovation,color=color,label='Energy',ec='k')
                # ax.fill_between(heating_std.index,heating_mean.values*1e-9+heating_std.values*1e-9,heating_mean.values*1e-9-heating_std.values*1e-9,alpha=0.5,color=color)
                
            # axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            # axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            # axin.axis('off')
            
            ax.set_ylabel('CO$_2$ emissions (MtCO$_2$eq.yr$^{-1}$)')
            ax.legend(loc='best')
            ax.set_ylim(bottom=0.,top=None)
            # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'{}_ref_ref_stack.png'.format(var)), bbox_inches='tight')
            plt.show()
            
            
        # ref absolute
        if True:
            # var = 'total_emissions'
            var = 'total_emissions_update'
            # var = 'cooling_emissions'
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly[df_consumption_agg_zcl_yearly.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')[var].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')[var].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                # if scenario == 'REF_REF':
                #     print('HEX',heating_mean.loc[init_years].mean(), heating_std.loc[init_years].mean())
                    
                color = get_scenarios_color().get(scenario)
                # ax.plot(relative_mean,color=color)
                ax.plot(heating_mean*1e-9,color=color)
                ax.fill_between(heating_std.index,heating_mean.values*1e-9+heating_std.values*1e-9,heating_mean.values*1e-9-heating_std.values*1e-9,alpha=0.5,color=color)
                
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Energy CO$_2$ emissions (MtCO$_2$eq.yr$^{-1}$)')
            # ax.legend(loc='lower right')
            ax.set_ylim(bottom=0.,top=None)
            # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'{}_ref_ref.png'.format(var)), bbox_inches='tight')
            plt.show()
        
        # relative
        if True:
            # var = 'total_emissions'
            var = 'total_emissions_update'
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_consumption_agg_zcl.scenario.values)):
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly[df_consumption_agg_zcl_yearly.scenario==scenario].copy()
                df_consumption_agg_zcl_yearly_sce['index'] = df_consumption_agg_zcl_yearly_sce['index'].dt.year
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_consumption_agg_zcl_yearly_sce = df_consumption_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_consumption_agg_zcl_yearly_sce.groupby('year')[var].mean()
                heating_std = df_consumption_agg_zcl_yearly_sce.groupby('year')[var].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('HEX',heating_mean.loc[init_years].mean(), heating_std.loc[init_years].mean())
                    
                color = get_scenarios_color().get(scenario)
                ax.plot(relative_mean,color=color)
                # ax.plot(heating_mean*1e-9,color=color)
                # ax.fill_between(heating_std.index,heating_mean.values*1e-9+heating_std.values*1e-9,heating_mean.values*1e-9-heating_std.values*1e-9,alpha=0.5,color=color)
                
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Relative energy CO$_2$ emissions (ratio)')
            # ax.legend(loc='lower right')
            ax.set_ylim(bottom=None,top=None)
            # ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'{}_relative.png'.format(var)), bbox_inches='tight')
            plt.show()
            
    
    #%% affichage des thermosensibilités
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        # climate_models_list = ['EC-EARTH_HadREM3-GA7'] 
        ac_scenarios = ['REF'] #['ACM','REF','ACP']
        # ac_scenarios = ['ACM','REF','ACP']
        pz_scenarios = ['REF'] #['NOF','REF','SOF']
        # pz_scenarios = ['NOF','REF','SOF']
        
        init_years = list(range(2018,2018+6))
        
        save = 'df_consumption_agg_zcl_yearly_elec.parquet'
        save_agg_zcl = 'df_consumption_agg_zcl_elec.parquet'
        save_df = 'df_consumption_elec.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format consumption elec...')
            df_consumption_elec, df_consumption_agg_zcl_elec, color_dict_elec = format_daily_consumption(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder), energy_filter='Electricity')
            df_consumption_agg_zcl_yearly_elec = format_yearly_consumption(df_consumption_agg_zcl_elec, climate_models_list)
            df_consumption_elec.to_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly_elec.to_parquet(os.path.join(output,folder,save))
            df_consumption_agg_zcl_elec.to_parquet(os.path.join(output,folder,save_agg_zcl))
        else:
            df_consumption_agg_zcl_elec = pd.read_parquet(os.path.join(output,folder,save_agg_zcl))
            df_consumption_elec = pd.read_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly_elec = pd.read_parquet(os.path.join(output,folder,save))
        
        # correction elec
        correction_elec = True
        if correction_elec:
            df_consumption_agg_zcl_elec['heating_cons'] = df_consumption_agg_zcl_elec['heating_cons']/1.487
            df_consumption_elec['heating_cons'] = df_consumption_elec['heating_cons']/1.487
            df_consumption_agg_zcl_yearly_elec['heating_cons'] = df_consumption_agg_zcl_yearly_elec['heating_cons']/1.487
            
        
        df_temperature = df_consumption_elec[['temperature_2m','zcl','climate_model','scenario']].copy().reset_index()
        df_temperature.loc[:,'ratio'] = df_temperature.zcl.map(ZCL_POPULATION_DISTRIBUTION.get)
        df_temperature['temperature'] = df_temperature.temperature_2m * df_temperature.ratio
        df_temperature = df_temperature[['temperature','climate_model','index','scenario']].groupby(['index','climate_model','scenario'],as_index=True).sum()
        df_consumption_agg_zcl = df_consumption_agg_zcl_elec.set_index(['index','climate_model','scenario']).join(df_temperature)
        
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        
        
        df_thermosensitivity_heating = {'scenario':[],'climate_model':[],'value':[]}
        df_thermosensitivity_cooling = {'scenario':[],'climate_model':[],'value':[]}
        
        for scenario in list(set(df_consumption_agg_zcl.index.get_level_values('scenario'))):
            th_list_2020 = []
            kh_list_2020 = []
            th_list_2050 = []
            kh_list_2050 = []
            for cm in climate_models_list:
                for y in [2023,2050]:
                    df = df_consumption_agg_zcl[(df_consumption_agg_zcl.index.get_level_values('index').year.isin(list(range(y-5,y+1))))&
                                                (df_consumption_agg_zcl.index.get_level_values('climate_model')==cm)&
                                                (df_consumption_agg_zcl.index.get_level_values('scenario')==scenario)]
    
                    th, kh, r2 = identify_thermal_sensitivity(df.temperature, df.heating_cons, cooling=False)
                    if y == 2023:
                        th_list_2020.append(th)
                        kh_list_2020.append(kh)
                    if y == 2050:
                        th_list_2050.append(th)
                        kh_list_2050.append(kh)
                        
                        df_thermosensitivity_heating['scenario'].append(scenario)
                        df_thermosensitivity_heating['climate_model'].append(cm)
                        df_thermosensitivity_heating['value'].append(kh)
            
            th_list_2020_mean = np.mean(th_list_2020)
            kh_list_2020_mean = np.mean(np.asarray(kh_list_2020)*1e-9/24)
            print('2020',scenario,'chauffage',kh_list_2020_mean)
            kh_list_2020_std = np.std(np.asarray(kh_list_2020)*1e-9/24)
            th_list_2050_mean = np.mean(th_list_2050)
            kh_list_2050_mean = np.mean(np.asarray(kh_list_2050)*1e-9/24)
            print('2050',scenario,'chauffage',kh_list_2050_mean)
            kh_list_2050_std = np.std(np.asarray(kh_list_2050)*1e-9/24)
            # print(scenario,'T chauffage',th_list_2050_mean)
            
            
            tc_list_2020 = []
            kc_list_2020 = []
            tc_list_2050 = []
            kc_list_2050 = []
            for cm in climate_models_list:
                for y in [2023,2050]:
                    df = df_consumption_agg_zcl[(df_consumption_agg_zcl.index.get_level_values('index').year.isin(list(range(y-5,y+1))))&
                                                (df_consumption_agg_zcl.index.get_level_values('climate_model')==cm)&
                                                (df_consumption_agg_zcl.index.get_level_values('scenario')==scenario)]
    
                    tc, kc, r2 = identify_thermal_sensitivity(df.temperature, df.cooling_cons, cooling=True)
                    if y == 2023:
                        tc_list_2020.append(tc)
                        kc_list_2020.append(kc)
                    if y == 2050:
                        tc_list_2050.append(tc)
                        kc_list_2050.append(kc)
                        
                        df_thermosensitivity_cooling['scenario'].append(scenario)
                        df_thermosensitivity_cooling['climate_model'].append(cm)
                        df_thermosensitivity_cooling['value'].append(kc)
            
            tc_list_2020_mean = np.mean(tc_list_2020)
            kc_list_2020_mean = np.mean(np.asarray(kc_list_2020)*1e-9/24)
            print('2020',scenario,'refroidissement',kc_list_2020_mean)
            kc_list_2020_std = np.std(np.asarray(kc_list_2020)*1e-9/24)
            tc_list_2050_mean = np.mean(tc_list_2050)
            kc_list_2050_mean = np.mean(np.asarray(kc_list_2050)*1e-9/24)
            print('2050',scenario,'refroidissement',kc_list_2050_mean)
            kc_list_2050_std = np.std(np.asarray(kc_list_2050)*1e-9/24)
            # print(scenario,'T refroidissement',tc_list_2050_mean)
            print()
            
            T = np.linspace(0,35)
            Yh_2020_mean = piecewise_linear_heating(T, th_list_2020_mean, kh_list_2020_mean)
            Yh_2020_upper = piecewise_linear_heating(T, th_list_2020_mean, kh_list_2020_mean+kh_list_2020_std)
            Yh_2020_lower = piecewise_linear_heating(T, th_list_2020_mean, kh_list_2020_mean-kh_list_2020_std)
            Yh_2050_mean = piecewise_linear_heating(T, th_list_2050_mean, kh_list_2050_mean)
            Yh_2050_upper = piecewise_linear_heating(T, th_list_2050_mean, kh_list_2050_mean+kh_list_2050_std)
            Yh_2050_lower = piecewise_linear_heating(T, th_list_2050_mean, kh_list_2050_mean-kh_list_2050_std)
            
            Yc_2020_mean = piecewise_linear_cooling(T, tc_list_2020_mean, kc_list_2020_mean)
            Yc_2020_upper = piecewise_linear_cooling(T, tc_list_2020_mean, kc_list_2020_mean+kc_list_2020_std)
            Yc_2020_lower = piecewise_linear_cooling(T, tc_list_2020_mean, kc_list_2020_mean-kc_list_2020_std)
            Yc_2050_mean = piecewise_linear_cooling(T, tc_list_2050_mean, kc_list_2050_mean)
            Yc_2050_upper = piecewise_linear_cooling(T, tc_list_2050_mean, kc_list_2050_mean+kc_list_2050_std)
            Yc_2050_lower = piecewise_linear_cooling(T, tc_list_2050_mean, kc_list_2050_mean-kc_list_2050_std)
            
            
            if scenario == 'REF_REF':
                label_2020 = '2018-2023'
                label_2050 = '2045-2050'
            else:
                label_2020 = None
                label_2050 = None
            
            if scenario == 'REF_REF':
                ax.plot(T,Yh_2020_mean,color='k',label=label_2020,zorder=-1)#,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
                ax.plot(T,Yc_2020_mean,color='k',zorder=-1)#,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
                
                # ax.fill_between(T,Yh_2020_upper,Yh_2020_lower,color='k',alpha=0.37,zorder=-1)
                # ax.fill_between(T,Yc_2020_upper,Yc_2020_lower,color='k',alpha=0.37,zorder=-1,)
                
                # ax.plot(T,Yh_2050_mean,color=get_scenarios_color().get(scenario),label=label_2050)#,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
                # ax.fill_between(T,Yh_2050_upper,Yh_2050_lower,color=get_scenarios_color().get(scenario),alpha=0.37)
                
                # ax.plot(T,Yc_2050_mean,color=get_scenarios_color().get(scenario))#,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
                # ax.fill_between(T,Yc_2050_upper,Yc_2050_lower,color=get_scenarios_color().get(scenario),alpha=0.37)
                
           
            ax.plot(T,Yh_2050_mean,color=get_scenarios_color().get(scenario),label=label_2050)#,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
            # ax.fill_between(T,Yh_2050_upper,Yh_2050_lower,color=color_dict_elec.get(scenario),alpha=0.37)
            
            ax.plot(T,Yc_2050_mean,color=get_scenarios_color().get(scenario))#,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
            # ax.fill_between(T,Yc_2050_upper,Yc_2050_lower,color=color_dict_elec.get(scenario),alpha=0.37)
            
            # add daily dots
            if False:
                for cm in climate_models_list:
                    for y in [2023,2050]:
                        if y == 2023:
                            color = 'k'
                        else: 
                            color = get_scenarios_color().get('REF_REF')
                        df = df_consumption_agg_zcl[(df_consumption_agg_zcl.index.get_level_values('index').year.isin(list(range(y-5,y+1))))&(df_consumption_agg_zcl.index.get_level_values('climate_model')==cm)]
                        df.loc[:,'total_cons'] = df.heating_cons + df.cooling_cons
                        sns.scatterplot(data=df,x='temperature',y='total_cons',alpha=0.005,ax=ax,color=color,zorder=-2)
                        
        ax.set_xlim([T[0],T[-1]])
        
        ax.legend(loc='upper right')
        axin = ax.inset_axes([1-0.305-0.025, 1-0.305-0.133, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
        axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
        axin.axis('off')
        
        if correction_elec:
            ax.set_ylabel('Daily corrected electricity consumption (GW)')
        else:
            ax.set_ylabel('Daily electricity consumption (GW)')
       
        ax.set_xlabel('Daily external temperature (°C)')
        ax.set_ylim(bottom=0.)
        if correction_elec:
            plt.savefig(os.path.join(figs_folder,'thermal_sensitivity_corrected_Electricity_consumption.png'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(figs_folder,'thermal_sensitivity_Electricity_consumption.png'), bbox_inches='tight')
        
        # plt.savefig(os.path.join(figs_folder,'thermal_sensitivity_corrected_Electricity_consumption_ref_ref.png'), bbox_inches='tight')
        plt.show()
        
        df_thermosensitivity_heating = pd.DataFrame().from_dict(df_thermosensitivity_heating)
        df_thermosensitivity_heating.to_parquet(os.path.join(output,folder,'thermal_sensitivity_heating'))
        df_thermosensitivity_cooling = pd.DataFrame().from_dict(df_thermosensitivity_cooling)
        df_thermosensitivity_cooling.to_parquet(os.path.join(output,folder,'thermal_sensitivity_cooling'))
        # print(df_thermosensitivity_heating)
        
        
        
    #%% affichage des séries temporelles de puissances
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        # climate_models_list = ['EC-EARTH_HadREM3-GA7'] 
        # ac_scenarios = ['REF'] #['ACM','REF','ACP']
        ac_scenarios = ['ACM','REF','ACP']
        # pz_scenarios = ['REF'] #['NOF','REF','SOF']
        pz_scenarios = ['NOF','REF','SOF']
        
        save = 'df_power_agg_zcl_yearly.parquet'
        save_df = 'df_power.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format power...')
            df_power, df_power_agg_zcl, color_dict = format_daily_power(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder), energy_filter='Electricity')
            df_power_agg_zcl_yearly = format_yearly_power(df_power_agg_zcl, climate_models_list)
            df_power_agg_zcl_yearly.to_parquet(os.path.join(output,folder,save))
            df_power.to_parquet(os.path.join(output,folder,save_df))
        else:
            df_power_agg_zcl_yearly = pd.read_parquet(os.path.join(output,folder,save))
            df_power = pd.read_parquet(os.path.join(output,folder,save_df))
        
        # correction elec
        correction_elec = True
        if correction_elec:
            df_power_agg_zcl_yearly['heating_pmax'] = df_power_agg_zcl_yearly['heating_pmax']/1.487
            df_power['heating_pmax'] = df_power['heating_pmax']/1.487
        
        end_years = list(range(2050,2050-6,-1))
        init_years = list(range(2018,2018+6))
        years = list(range(2018,2051))
        
        # ref absolute
        if True:
            #heating
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            
            for scenario in list(set(df_power_agg_zcl_yearly.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly[df_power_agg_zcl_yearly.scenario==scenario].copy()
                df_power_agg_zcl_yearly_sce['index'] = df_power_agg_zcl_yearly_sce['index'].dt.year
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_power_agg_zcl_yearly_sce.groupby('year')['heating_pmax'].mean()
                heating_std = df_power_agg_zcl_yearly_sce.groupby('year')['heating_pmax'].std()
            
                ax.plot(heating_mean*1e-9,color=get_scenarios_color().get(scenario))
                ax.fill_between(heating_std.index,heating_mean.values*1e-9+heating_std.values*1e-9,heating_mean.values*1e-9-heating_std.values*1e-9,alpha=0.5,color=get_scenarios_color().get(scenario))
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            if correction_elec:
                ax.set_ylabel('DJF Electricity corrected power peak for heating (GW)')
            else:
                ax.set_ylabel('DJF Electricity power peak for heating (GW)')
            # ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_xlim([2018,2050])
            if correction_elec:
                plt.savefig(os.path.join(figs_folder,'heating_corrected_Electricity_power.png'), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(figs_folder,'heating_Electricity_power.png'), bbox_inches='tight')
            
            plt.show()
            
            # cooling
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            
            for scenario in list(set(df_power_agg_zcl_yearly.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly[df_power_agg_zcl_yearly.scenario==scenario].copy()
                df_power_agg_zcl_yearly_sce['index'] = df_power_agg_zcl_yearly_sce['index'].dt.year
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                cooling_mean = df_power_agg_zcl_yearly_sce.groupby('year')['cooling_pmax'].mean()
                cooling_std = df_power_agg_zcl_yearly_sce.groupby('year')['cooling_pmax'].std()
                
                ax.plot(cooling_mean*1e-9,color=get_scenarios_color().get(scenario))
                ax.fill_between(cooling_std.index,cooling_mean.values*1e-9+cooling_std.values*1e-9,cooling_mean.values*1e-9-cooling_std.values*1e-9,alpha=0.5,color=get_scenarios_color().get(scenario))
            
            axin = ax.inset_axes([1-0.305-0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('JJA Electricity power peak for cooling (GW)')
            # ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cooling_Electricity_power.png'), bbox_inches='tight')
            plt.show()
        
        # absolute zcl ref ref 
        if True:
            df_power_yearly = None
            for scenario in list(set(df_power.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_power_sce = df_power[df_power.scenario==scenario]
                for cm in climate_models_list:
                    temp_heating = df_power_sce[(df_power_sce.climate_model==cm)&(df_power_sce.index.month.isin([12,1,2]))].copy()
                    temp_cooling = df_power_sce[(df_power_sce.climate_model==cm)&(df_power_sce.index.month.isin([6,7,8]))].copy()
                    
                    temp_heating_zcl = None
                    for zcl in ZCL_LIST:
                        temp = temp_heating[temp_heating.zcl==zcl].copy()
                        temp = aggregate_resolution(temp[['heating_pmax', 'cooling_pmax']],'YS','max')
                        temp['climate_model'] = [cm]*len(temp)
                        temp['scenario'] = [scenario]*len(temp)
                        temp['zcl'] = [zcl]*len(temp)
                        temp = temp.reset_index()
                        if temp_heating_zcl is None:
                            temp_heating_zcl = temp
                        else:
                            temp_heating_zcl = pd.concat([temp_heating_zcl,temp])
                    # temp_heating_zcl = temp_heating_zcl.set_index(['index','scenario','zcl','climate_model'])
                    
                    temp_cooling_zcl = None
                    for zcl in ZCL_LIST:
                        temp = temp_cooling[temp_cooling.zcl==zcl].copy()
                        temp = aggregate_resolution(temp[['heating_pmax', 'cooling_pmax']],'YS','max')
                        temp['climate_model'] = [cm]*len(temp)
                        temp['scenario'] = [scenario]*len(temp)
                        temp['zcl'] = [zcl]*len(temp)
                        temp = temp.reset_index()
                        if temp_cooling_zcl is None:
                            temp_cooling_zcl = temp
                        else:
                            temp_cooling_zcl = pd.concat([temp_cooling_zcl,temp])
                            
                    # temp_cooling_zcl = temp_cooling_zcl.set_index(['index','scenario','zcl','climate_model'])
                    
                    temp_heating_zcl['cooling_pmax'] = temp_cooling_zcl.cooling_pmax
                    if df_power_yearly is None:
                        df_power_yearly = temp_heating_zcl
                    else:
                        df_power_yearly = pd.concat([df_power_yearly,temp_heating_zcl])
            
            mean_heating = df_power_yearly.groupby(['index','zcl'],as_index=False)['heating_pmax'].mean()
            std_heating = df_power_yearly.groupby(['index','zcl'],as_index=False)['heating_pmax'].std()
            mean_cooling = df_power_yearly.groupby(['index','zcl'],as_index=False)['cooling_pmax'].mean()
            std_cooling = df_power_yearly.groupby(['index','zcl'],as_index=False)['cooling_pmax'].std()
            
            print('H1c','cooling',mean_cooling[(mean_cooling['index'].dt.year.isin(end_years))&(mean_cooling.zcl=='H1c')]['cooling_pmax'].mean())
            print('H2d','cooling',mean_cooling[(mean_cooling['index'].dt.year.isin(end_years))&(mean_cooling.zcl=='H2d')]['cooling_pmax'].mean())
            print('H3','cooling',mean_cooling[(mean_cooling['index'].dt.year.isin(end_years))&(mean_cooling.zcl=='H3')]['cooling_pmax'].mean())
            
            ls_list = ['-','-.','--']
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for idx,zcl in enumerate(ZCL_LIST):
                color = plt.get_cmap('viridis')(ZCL_LIST.index(zcl)/len(ZCL_LIST))
                to_plot = mean_heating[mean_heating.zcl==zcl]
                to_plot_std = std_heating[mean_heating.zcl==zcl]
                ax.plot(to_plot['index'].dt.year,to_plot['heating_pmax']*1e-9,label=zcl,color=color,ls=ls_list[idx%3])
                ax.fill_between(to_plot['index'].dt.year,to_plot['heating_pmax']*1e-9+to_plot_std['heating_pmax']*1e-9,to_plot['heating_pmax']*1e-9-to_plot_std['heating_pmax']*1e-9,color=color,alpha=0.37)
            ax.set_ylim(bottom=0.,top=7.)
            ax.set_xlim([2018,2050])
            ax.legend(ncols=1)
            if correction_elec:
                ax.set_ylabel('DJF Electricity corrected power peak for heating (GW)')
            else:
                ax.set_ylabel('DJF Electricity power peak for heating (GW)')
            if correction_elec:
                plt.savefig(os.path.join(figs_folder,'heating_corrected_Electricity_power_zcl.png'), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(figs_folder,'heating_Electricity_power_zcl.png'), bbox_inches='tight')
            plt.show()
            
            ls_list = ['-','-.','--']
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for idx,zcl in enumerate(ZCL_LIST):
                color = plt.get_cmap('viridis')(ZCL_LIST.index(zcl)/len(ZCL_LIST))
                to_plot = mean_cooling[mean_heating.zcl==zcl]
                to_plot_std = std_cooling[mean_heating.zcl==zcl]
                ax.plot(to_plot['index'].dt.year,to_plot['cooling_pmax']*1e-9,label=zcl,color=color,ls=ls_list[idx%3])
                ax.fill_between(to_plot['index'].dt.year,to_plot['cooling_pmax']*1e-9+to_plot_std['cooling_pmax']*1e-9,to_plot['cooling_pmax']*1e-9-to_plot_std['cooling_pmax']*1e-9,color=color,alpha=0.37)
            ax.set_ylabel('JJA Electricity power peak for cooling (GW)')
            ax.set_ylim(bottom=0.,top=7.)
            ax.set_xlim([2018,2050])
            ax.legend(ncols=1)
            plt.savefig(os.path.join(figs_folder,'cooling_Electricity_power_zcl.png'), bbox_inches='tight')
            plt.show()
            
            
        # relative
        if True:
            end_years = list(range(2050,2050-6,-1))
            
            #heating
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_power_agg_zcl_yearly.scenario.values)):
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly[df_power_agg_zcl_yearly.scenario==scenario].copy()
                df_power_agg_zcl_yearly_sce['index'] = df_power_agg_zcl_yearly_sce['index'].dt.year
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_power_agg_zcl_yearly_sce.groupby('year')['heating_pmax'].mean()
                heating_std = df_power_agg_zcl_yearly_sce.groupby('year')['heating_pmax'].std()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('HEX',heating_mean.loc[init_years].mean(), heating_std.loc[init_years].mean())
                    print('HEX','relative',relative_mean.loc[end_years].mean())
                if scenario == 'ACP_REF':
                    print('HEX','ACP relative',relative_mean.loc[end_years].mean())
                    
                ax.plot(relative_mean,color=get_scenarios_color().get(scenario))
                # ax.plot(heating_mean*1e-9,color=color_dict.get(scenario))
                # ax.fill_between(heating_std.index,heating_mean.values*1e-9+heating_std.values*1e-9,heating_mean.values*1e-9-heating_std.values*1e-9,alpha=0.5,color=color_dict.get(scenario))
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('DJF relative power peak for heating (ratio)')
            # ax.legend()
            # ax.set_ylim(bottom=0.)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'heating_Electricity_power_relative.png'), bbox_inches='tight')
            plt.show()
            
            # cooling
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_power_agg_zcl_yearly.scenario.values)):
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly[df_power_agg_zcl_yearly.scenario==scenario].copy()
                df_power_agg_zcl_yearly_sce['index'] = df_power_agg_zcl_yearly_sce['index'].dt.year
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_power_agg_zcl_yearly_sce = df_power_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                cooling_mean = df_power_agg_zcl_yearly_sce.groupby('year')['cooling_pmax'].mean()
                cooling_std = df_power_agg_zcl_yearly_sce.groupby('year')['cooling_pmax'].std()
                relative_mean = cooling_mean/cooling_mean.loc[init_years].mean()
                
                if scenario == 'REF_REF':
                    print('HEX',cooling_mean.loc[init_years].mean(), cooling_std.loc[init_years].mean())
                    print('HEX','relative',relative_mean.loc[end_years].mean())
                if scenario in ['ACP_REF','REF_REF','ACM_REF']:
                    print('HEX',scenario, 'relative',(relative_mean.loc[end_years].mean()))
                    
                ax.plot(relative_mean,color=get_scenarios_color().get(scenario))
                # ax.plot(cooling_mean*1e-9,color=color_dict.get(scenario))
                # ax.fill_between(cooling_std.index,cooling_mean.values*1e-9+cooling_std.values*1e-9,cooling_mean.values*1e-9-cooling_std.values*1e-9,alpha=0.5,color=color_dict.get(scenario))
            
            axin = ax.inset_axes([1-0.305-0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('JJA relative power peak for cooling (ratio)')
            # ax.legend()
            # ax.set_ylim(bottom=0.)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cooling_Electricity_power_relative.png'), bbox_inches='tight')
            plt.show()
        
        
    #%% affichage des séries temporelles des degrés heure d'inconfort
    if True:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        # climate_models_list = ['EC-EARTH_HadREM3-GA7'] 
        # ac_scenarios = ['REF'] #['ACM','REF','ACP']
        ac_scenarios = ['ACM','REF','ACP']
        # pz_scenarios = ['REF'] #['NOF','REF','SOF']
        pz_scenarios = ['NOF','REF','SOF']
        
        
        save = 'df_dhi_agg_zcl_yearly.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format DHI...')
            df_dhi, df_dhi_agg_zcl, color_dict = format_daily_dhi(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder))
            df_dhi_agg_zcl_yearly = format_yearly_dhi(df_dhi_agg_zcl, climate_models_list)
            df_dhi_agg_zcl_yearly.to_parquet(os.path.join(output,folder,save))
        else:
            df_dhi_agg_zcl_yearly = pd.read_parquet(os.path.join(output,folder,save))
        
        save = 'df_dhi_agg_zcl_yearly_C1.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format DHI C1...')
            df_dhi_C1, df_dhi_agg_zcl_C1, color_dict = format_daily_dhi(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder),income_filter='C1')
            df_dhi_agg_zcl_yearly_C1 = format_yearly_dhi(df_dhi_agg_zcl_C1, climate_models_list)
            df_dhi_agg_zcl_yearly_C1.to_parquet(os.path.join(output,folder,save))
        else:
            df_dhi_agg_zcl_yearly_C1 = pd.read_parquet(os.path.join(output,folder,save))
        
        save = 'df_dhi_agg_zcl_yearly_C5.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format DHI C5...')
            df_dhi_C5, df_dhi_agg_zcl_C5, color_dict = format_daily_dhi(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder),income_filter='C5')
            df_dhi_agg_zcl_yearly_C5 = format_yearly_dhi(df_dhi_agg_zcl_C5, climate_models_list)
            df_dhi_agg_zcl_yearly_C5.to_parquet(os.path.join(output,folder,save))
        else:
            df_dhi_agg_zcl_yearly_C5 = pd.read_parquet(os.path.join(output,folder,save))
        
        save = 'df_dhi_agg_zcl_yearly_renter.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format DHI Renters...')
            df_dhi_renter, df_dhi_agg_zcl_renter, color_dict = format_daily_dhi(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder),status_filter='Privately rented')
            df_dhi_agg_zcl_yearly_renter = format_yearly_dhi(df_dhi_agg_zcl_renter, climate_models_list)
            df_dhi_agg_zcl_yearly_renter.to_parquet(os.path.join(output,folder,save))
        else:
            df_dhi_agg_zcl_yearly_renter = pd.read_parquet(os.path.join(output,folder,save))
        
        save = 'df_dhi_agg_zcl_yearly_owner.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format DHI Owners...')
            df_dhi_owner, df_dhi_agg_zcl_owner, color_dict = format_daily_dhi(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder),status_filter='Owner-occupied')
            df_dhi_agg_zcl_yearly_owner = format_yearly_dhi(df_dhi_agg_zcl_owner, climate_models_list)
            df_dhi_agg_zcl_yearly_owner.to_parquet(os.path.join(output,folder,save))
        else:
            df_dhi_agg_zcl_yearly_owner = pd.read_parquet(os.path.join(output,folder,save))
        
        save = 'df_dhi_agg_zcl_yearly_C1_renter.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format DHI Renters C1...')
            df_dhi_C1_renter, df_dhi_agg_zcl_C1_renter, color_dict = format_daily_dhi(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder),income_filter='C1',status_filter='Privately rented')
            df_dhi_agg_zcl_yearly_C1_renter = format_yearly_dhi(df_dhi_agg_zcl_C1_renter, climate_models_list)
            df_dhi_agg_zcl_yearly_C1_renter.to_parquet(os.path.join(output,folder,save))
        else:
            df_dhi_agg_zcl_yearly_C1_renter = pd.read_parquet(os.path.join(output,folder,save))
        
        end_years = list(range(2050,2050-6,-1))
        init_years = list(range(2018,2018+6))
        years = list(range(2018,2051))
        
        # ref absolute
        if True:
            #heating
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly[df_dhi_agg_zcl_yearly.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                heating_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                color = get_scenarios_color().get(scenario)
                ax.plot(heating_mean,color=color)
                ax.fill_between(heating_std.index,heating_mean.values+heating_std.values,heating_mean.values-heating_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Cold incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cold_dhi_ref_ref.png'), bbox_inches='tight')
            plt.show()
            
            # cooling
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly.scenario.values)):
                if scenario != 'REF_REF':
                    continue
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly[df_dhi_agg_zcl_yearly.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                cooling_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                cooling_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                color = get_scenarios_color().get(scenario)
                ax.plot(cooling_mean,color=color)
                ax.fill_between(cooling_std.index,cooling_mean.values+cooling_std.values,cooling_mean.values-cooling_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([1-0.305-0.01, 1-0.305-0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Hot incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'hot_dhi_ref_ref.png'), bbox_inches='tight')
            plt.show()
            
            
        # realtive
        if True:
            #heating
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly.scenario.values)):
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly[df_dhi_agg_zcl_yearly.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                heating_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                relative_mean = heating_mean/heating_mean.loc[init_years].mean()
                
                # if scenario == 'REF_REF':
                #     print('HEX','heating',heating_mean.loc[init_years].mean(), heating_std.loc[init_years].mean())
                #     print('HEX','heating',heating_mean.loc[end_years].mean(), heating_std.loc[end_years].mean())
                
                color = get_scenarios_color().get(scenario)
                ax.plot(relative_mean,color=color)
                # ax.plot(heating_mean*1e-9,color=color_dict.get(scenario))
                # ax.fill_between(heating_std.index,heating_mean.values*1e-9+heating_std.values*1e-9,heating_mean.values*1e-9-heating_std.values*1e-9,alpha=0.5,color=color_dict.get(scenario))
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Relative cold incomfort (ratio)')
            # ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cold_dhi_relative.png'), bbox_inches='tight')
            plt.show()
            
            # cooling
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly.scenario.values)):
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly[df_dhi_agg_zcl_yearly.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                cooling_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                cooling_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                relative_mean = cooling_mean/cooling_mean.loc[init_years].mean()
                
                # if scenario == 'REF_REF':
                #     print('HEX','cooling',cooling_mean.loc[init_years].mean(), cooling_std.loc[init_years].mean())
                #     print('HEX','cooling',cooling_mean.loc[end_years].mean(), cooling_std.loc[end_years].mean())
                
                color = get_scenarios_color().get(scenario)
                ax.plot(relative_mean,color=color)
                # ax.plot(cooling_mean*1e-9,color=color_dict.get(scenario))
                # ax.fill_between(cooling_std.index,cooling_mean.values*1e-9+cooling_std.values*1e-9,cooling_mean.values*1e-9-cooling_std.values*1e-9,alpha=0.5,color=color_dict.get(scenario))
            
            axin = ax.inset_axes([1-0.305-0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Relative hot incomfort (ratio)')
            # ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'hot_dhi_relative.png'), bbox_inches='tight')
            plt.show()
        
        # ref absolute comparaison C1 C5
        if True:
            #heating
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly_C1.scenario.values)):
                
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_C1[df_dhi_agg_zcl_yearly_C1.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                heating_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                # if scenario == 'REF_REF':
                #     print('HEX','heating C1',heating_mean.loc[init_years].mean(), heating_std.loc[init_years].mean())
                #     print('HEX','heating C1',heating_mean.loc[end_years].mean(), heating_std.loc[end_years].mean())
                
                color = get_scenarios_color().get(scenario)
                ax.plot(heating_mean,color=color)
                # ax.fill_between(heating_std.index,heating_mean.values+heating_std.values,heating_mean.values-heating_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Cold incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_title('Income level - C1')
            ax.set_ylim(bottom=0.,top=5000)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cold_dhi_C1.png'), bbox_inches='tight')
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly_C5.scenario.values)):
                
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_C5[df_dhi_agg_zcl_yearly_C5.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                heating_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                # if scenario == 'REF_REF':
                #     print('HEX','heating C5',heating_mean.loc[init_years].mean(), heating_std.loc[init_years].mean())
                #     print('HEX','heating C5',heating_mean.loc[end_years].mean(), heating_std.loc[end_years].mean())
                
                color = get_scenarios_color().get(scenario)
                ax.plot(heating_mean,color=color)
                # ax.fill_between(heating_std.index,heating_mean.values+heating_std.values,heating_mean.values-heating_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Cold incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_title('Income level - C5')
            ax.set_ylim(bottom=0.,top=5000)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cold_dhi_C5.png'), bbox_inches='tight')
            plt.show()
            
            # cooling
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly_C1.scenario.values)):
                
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_C1[df_dhi_agg_zcl_yearly_C1.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                cooling_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                cooling_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                # if scenario == 'REF_REF':
                #     print('HEX','cooling C1',cooling_mean.loc[init_years].mean(), cooling_std.loc[init_years].mean())
                #     print('HEX','cooling C1',cooling_mean.loc[end_years].mean(), cooling_std.loc[end_years].mean())
                #     print('HEX','cooling C1 all',cooling_mean.loc[years].mean(), cooling_std.loc[years].mean())
            
                color = get_scenarios_color().get(scenario)
                ax.plot(cooling_mean,color=color)
                # ax.fill_between(cooling_std.index,cooling_mean.values+cooling_std.values,cooling_mean.values-cooling_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Hot incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_title('Income level - C1')
            ax.set_ylim(bottom=0.,top=3100)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'hot_dhi_C1.png'), bbox_inches='tight')
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly_C5.scenario.values)):
                
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_C5[df_dhi_agg_zcl_yearly_C5.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                cooling_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                cooling_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                # if scenario == 'REF_REF':
                #     print('HEX','cooling C5',cooling_mean.loc[init_years].mean(), cooling_std.loc[init_years].mean())
                #     print('HEX','cooling C5',cooling_mean.loc[end_years].mean(), cooling_std.loc[end_years].mean())
                #     print('HEX','cooling C5 all',cooling_mean.loc[years].mean(), cooling_std.loc[years].mean())
                    
                color = get_scenarios_color().get(scenario)
                ax.plot(cooling_mean,color=color)
                # ax.fill_between(cooling_std.index,cooling_mean.values+cooling_std.values,cooling_mean.values-cooling_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Hot incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_title('Income level - C5')
            ax.set_ylim(bottom=0.,top=3100)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'hot_dhi_C5.png'), bbox_inches='tight')
            plt.show()
            
            
        # ref absolute comparaison locataire proprio
        if True:
            #heating
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly_renter.scenario.values)):
                
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_renter[df_dhi_agg_zcl_yearly_renter.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                heating_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                # if scenario == 'REF_REF':
                #     print('HEX','heating renter',heating_mean.loc[init_years].mean(), heating_std.loc[init_years].mean())
                #     print('HEX','heating renter',heating_mean.loc[end_years].mean(), heating_std.loc[end_years].mean())
                
                color = get_scenarios_color().get(scenario)
                ax.plot(heating_mean,color=color)
                # ax.fill_between(heating_std.index,heating_mean.values+heating_std.values,heating_mean.values-heating_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Cold incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_title('Occupancy status - Privately rented')
            ax.set_ylim(bottom=0.,top=5000)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cold_dhi_renter.png'), bbox_inches='tight')
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly_owner.scenario.values)):
                
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_owner[df_dhi_agg_zcl_yearly_owner.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                heating_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                heating_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_cold'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                # if scenario == 'REF_REF':
                #     print('HEX','heating owner',heating_mean.loc[init_years].mean(), heating_std.loc[init_years].mean())
                #     print('HEX','heating owner',heating_mean.loc[end_years].mean(), heating_std.loc[end_years].mean())
                
                
                color = get_scenarios_color().get(scenario)
                ax.plot(heating_mean,color=color)
                # ax.fill_between(heating_std.index,heating_mean.values+heating_std.values,heating_mean.values-heating_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Cold incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_title('Occupancy status - Owner-occupied')
            ax.set_ylim(bottom=0.,top=5000)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'cold_dhi_owner.png'), bbox_inches='tight')
            plt.show()
            
            # cooling
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly_renter.scenario.values)):
                
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_renter[df_dhi_agg_zcl_yearly_renter.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                cooling_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                cooling_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                if scenario in ['REF_REF','ACP_REF','ACP_SOF','REF_SOF']:
                    # print('HEX','cooling renter',cooling_mean.loc[init_years].mean(), cooling_std.loc[init_years].mean())
                    # print('HEX','cooling renter',cooling_mean.loc[end_years].mean(), cooling_std.loc[end_years].mean())
                    print('HEX',scenario, 'cooling renter all',cooling_mean.loc[years].mean(), cooling_std.loc[years].mean())
                    
                color = get_scenarios_color().get(scenario)
                ax.plot(cooling_mean,color=color)
                # ax.fill_between(cooling_std.index,cooling_mean.values+cooling_std.values,cooling_mean.values-cooling_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01, 0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Hot incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_title('Occupancy status - Privately rented')
            ax.set_ylim(bottom=0.,top=3300)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'hot_dhi_renter.png'), bbox_inches='tight')
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            for scenario in list(set(df_dhi_agg_zcl_yearly_owner.scenario.values)):
                
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_owner[df_dhi_agg_zcl_yearly_owner.scenario==scenario].copy()
                df_dhi_agg_zcl_yearly_sce['index'] = df_dhi_agg_zcl_yearly_sce['index'].dt.year
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.rename(columns={'index':'year'})
                df_dhi_agg_zcl_yearly_sce = df_dhi_agg_zcl_yearly_sce.set_index(['year','climate_model'])
                
                cooling_mean = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].mean()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                cooling_std = df_dhi_agg_zcl_yearly_sce.groupby('year')['DHI_hot'].std()/df_dhi_agg_zcl_yearly_sce.groupby('year')['households'].mean()
                
                if scenario in ['REF_REF','ACP_REF','ACP_SOF','REF_SOF']:
                    # print('HEX','cooling owner',cooling_mean.loc[init_years].mean(), cooling_std.loc[init_years].mean())
                    # print('HEX','cooling owner',cooling_mean.loc[end_years].mean(), cooling_std.loc[end_years].mean())
                    print('HEX',scenario,'cooling owner all',cooling_mean.loc[years].mean(), cooling_std.loc[years].mean())
                    
                color = get_scenarios_color().get(scenario)
                ax.plot(cooling_mean,color=color)
                # ax.fill_between(cooling_std.index,cooling_mean.values+cooling_std.values,cooling_mean.values-cooling_std.values,alpha=0.5,color=color)
            
            axin = ax.inset_axes([0.01, 1-0.305-0.01, 0.305, 0.305],zorder=0)    # create new inset axes in data coordinates
            axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
            axin.axis('off')
            
            ax.set_ylabel('Hot incomfort (°C.h.household$^{-1}$)')
            # ax.legend()
            ax.set_title('Occupancy status - Owner-occupied')
            ax.set_ylim(bottom=0.,top=3300)
            ax.set_xlim([2018,2050])
            plt.savefig(os.path.join(figs_folder,'hot_dhi_owner.png'), bbox_inches='tight')
            plt.show()
    
    
    #%% SPIDER CHART (ou pas)
    if False:
        df_values = None
        
        save = 'df_consumption_agg_zcl_yearly.parquet'
        save_agg_zcl = 'df_consumption_agg_zcl.parquet'
        save_df = 'df_consumption.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format consumption...')
            df_consumption, df_consumption_agg_zcl, color_dict = format_daily_consumption(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder), energy_filter=None)
            df_consumption_agg_zcl_yearly = format_yearly_consumption(df_consumption_agg_zcl, climate_models_list)
            df_consumption.to_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly.to_parquet(os.path.join(output,folder,save))
            df_consumption_agg_zcl.to_parquet(os.path.join(output,folder,save_agg_zcl))
        else:
            df_consumption_agg_zcl = pd.read_parquet(os.path.join(output,folder,save_agg_zcl))
            df_consumption = pd.read_parquet(os.path.join(output,folder,save_df))
            df_consumption_agg_zcl_yearly = pd.read_parquet(os.path.join(output,folder,save))
        
        save = 'df_power_agg_zcl_yearly.parquet'
        save_df = 'df_power.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format power...')
            df_power, df_power_agg_zcl, color_dict = format_daily_power(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder), energy_filter='Electricity')
            df_power_agg_zcl_yearly = format_yearly_power(df_power_agg_zcl, climate_models_list)
            df_power_agg_zcl_yearly.to_parquet(os.path.join(output,folder,save))
            df_power.to_parquet(os.path.join(output,folder,save_df))
        else:
            df_power_agg_zcl_yearly = pd.read_parquet(os.path.join(output,folder,save))
            df_power = pd.read_parquet(os.path.join(output,folder,save_df))
            
        correction_elec = True
        if correction_elec:
            df_power_agg_zcl_yearly['heating_pmax'] = df_power_agg_zcl_yearly['heating_pmax']/1.487
            df_power['heating_pmax'] = df_power['heating_pmax']/1.487
        
        save = 'df_dhi_agg_zcl_yearly.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            print('Format DHI...')
            df_dhi, df_dhi_agg_zcl, color_dict = format_daily_dhi(ac_scenarios, pz_scenarios, climate_models_list, os.path.join(output,folder))
            df_dhi_agg_zcl_yearly = format_yearly_dhi(df_dhi_agg_zcl, climate_models_list)
            df_dhi_agg_zcl_yearly.to_parquet(os.path.join(output,folder,save))
        else:
            df_dhi_agg_zcl_yearly = pd.read_parquet(os.path.join(output,folder,save))
        df_dhi_agg_zcl_yearly['DHI_hot_per_household'] = df_dhi_agg_zcl_yearly.DHI_hot/df_dhi_agg_zcl_yearly.households
        df_dhi_agg_zcl_yearly['DHI_cold_per_household'] = df_dhi_agg_zcl_yearly.DHI_cold/df_dhi_agg_zcl_yearly.households
        
        
        end_years = list(range(2050,2050-6,-1))
        init_years = list(range(2018,2018+6))
        years = list(range(2018,2051))
        
        # total cost
            # subsidies
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        ac_scenarios = ['ACM','REF','ACP']
        pz_scenarios = ['NOF','REF','SOF']
        
        save = 'subsidies.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            subsidies_df = {'scenario':[],'climate_model':[],'value':[]}
            for acs in ac_scenarios:
                for pzs in pz_scenarios:
                    sce = '{}_{}'.format(acs,pzs)
                    for cm in tqdm.tqdm(climate_models_list,desc=sce):
                        renovation = get_resirf_output(acs,pzs,cm,'Subsidies total (Billion euro)',os.path.join(output,folder)).loc[years].sum()*1e9
                        subsidies_df['scenario'].append(sce)
                        subsidies_df['climate_model'].append(cm)
                        subsidies_df['value'].append(renovation)
            subsidies_df = pd.DataFrame().from_dict(subsidies_df)
            subsidies_df.to_parquet(os.path.join(output,folder,save))
        else:
            subsidies_df = pd.read_parquet(os.path.join(output,folder,save))
            
            # AC gains
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        ac_scenarios = ['ACM','REF','ACP']
        pz_scenarios = ['NOF','REF','SOF']
        
        save = 'ac_gains.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            ac_gains_df = {'scenario':[],'climate_model':[],'value':[]}
            for acs in ac_scenarios:
                for pzs in pz_scenarios:
                    sce = '{}_{}'.format(acs,pzs)
                    for cm in tqdm.tqdm(climate_models_list,desc=sce):
                        renovation = get_resirf_output(acs,pzs,cm,'Cooler tax gains (Billion euro)',os.path.join(output,folder)).loc[years].sum()*1e9
                        ac_gains_df['scenario'].append(sce)
                        ac_gains_df['climate_model'].append(cm)
                        ac_gains_df['value'].append(renovation)
            ac_gains_df = pd.DataFrame().from_dict(ac_gains_df)
            ac_gains_df.to_parquet(os.path.join(output,folder,save))
        else:
            ac_gains_df = pd.read_parquet(os.path.join(output,folder,save))
            
            # AC subsidies
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        ac_scenarios = ['ACM','REF','ACP']
        pz_scenarios = ['NOF','REF','SOF']
        
        save = 'ac_subsidies.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            ac_subsidies_df = {'scenario':[],'climate_model':[],'value':[]}
            for acs in ac_scenarios:
                for pzs in pz_scenarios:
                    sce = '{}_{}'.format(acs,pzs)
                    for cm in tqdm.tqdm(climate_models_list,desc=sce):
                        renovation = get_resirf_output(acs,pzs,cm,'Cooler subsidies (Billion euro)',os.path.join(output,folder)).loc[years].sum()*1e9
                        ac_subsidies_df['scenario'].append(sce)
                        ac_subsidies_df['climate_model'].append(cm)
                        ac_subsidies_df['value'].append(renovation)
            ac_subsidies_df = pd.DataFrame().from_dict(ac_subsidies_df)
            ac_subsidies_df.to_parquet(os.path.join(output,folder,save))
        else:
            ac_subsidies_df = pd.read_parquet(os.path.join(output,folder,save))
        
        
        subsidies_df = subsidies_df.set_index(['scenario','climate_model']).join(ac_gains_df.set_index(['scenario','climate_model']),rsuffix='_ac_gains').reset_index()
        subsidies_df = subsidies_df.set_index(['scenario','climate_model']).join(ac_subsidies_df.set_index(['scenario','climate_model']),rsuffix='_ac_subsidies').reset_index()
        subsidies_df['value'] = subsidies_df['value'] - subsidies_df['value_ac_gains'] + subsidies_df['value_ac_subsidies']
        subsidies_df['variable'] = ['subsidies_euro']*len(subsidies_df)
        subsidies_df = subsidies_df[['scenario','climate_model','variable','value']]
        if df_values is None:
            df_values = subsidies_df
        else:
            df_values = pd.concat([df_values,subsidies_df])
            
        # conso totale
        conso = df_consumption_agg_zcl_yearly[df_consumption_agg_zcl_yearly['index'].dt.year.isin(end_years)].groupby(['climate_model','scenario'],as_index=False).mean()[['climate_model','scenario','total_cons']]
        conso['variable'] = ['total_consumption_Wh']*len(conso)
        conso = conso.rename(columns={'total_cons':'value'})[['scenario','climate_model','variable','value']]
        if df_values is None:
            df_values = conso
        else:
            df_values = pd.concat([df_values,conso])
        
        # GES
            # construction idem pour tous 
        construction = get_resirf_output('REF','REF','EC-EARTH_HadREM3-GA7','Carbon footprint construction (MtCO2)',os.path.join(output,folder)).loc[years].sum()*1e9
        
            # renvoation diff par scéanrio
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        ac_scenarios = ['ACM','REF','ACP']
        pz_scenarios = ['NOF','REF','SOF']
    
        save = 'renovation.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            renovation_df = {'scenario':[],'climate_model':[],'value':[]}
            for acs in ac_scenarios:
                for pzs in pz_scenarios:
                    sce = '{}_{}'.format(acs,pzs)
                    for cm in tqdm.tqdm(climate_models_list,desc=sce):
                        renovation = get_resirf_output(acs,pzs,cm,'Carbon footprint renovation (MtCO2)',os.path.join(output,folder)).loc[years].sum()*1e9
                        renovation_df['scenario'].append(sce)
                        renovation_df['climate_model'].append(cm)
                        renovation_df['value'].append(renovation)
            renovation_df = pd.DataFrame().from_dict(renovation_df)
            renovation_df.to_parquet(os.path.join(output,folder,save))
        else:
            renovation_df = pd.read_parquet(os.path.join(output,folder,save))
            
            # ges ac
        save = 'ac_fixe.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            ac_fixe_df = {'scenario':[],'climate_model':[],'value':[]}
            for acs in ac_scenarios:
                for pzs in pz_scenarios:
                    sce = '{}_{}'.format(acs,pzs)
                    for cm in tqdm.tqdm(climate_models_list,desc=sce):
                        renovation = get_resirf_output(acs,pzs,cm,'Adoption cooler split system (Thousand households)',os.path.join(output,folder)).loc[years].sum()*1e3
                        ac_fixe_df['scenario'].append(sce)
                        ac_fixe_df['climate_model'].append(cm)
                        ac_fixe_df['value'].append(renovation)
            ac_fixe_df = pd.DataFrame().from_dict(ac_fixe_df)
            ac_fixe_df.to_parquet(os.path.join(output,folder,save))
        else:
            ac_fixe_df = pd.read_parquet(os.path.join(output,folder,save))
        ac_fixe_df['value'] *= 380 #kgCO2eq 
        
        save = 'ac_portable.parquet'
        if save not in os.listdir(os.path.join(output,folder)):
            ac_portable_df = {'scenario':[],'climate_model':[],'value':[]}
            for acs in ac_scenarios:
                for pzs in pz_scenarios:
                    sce = '{}_{}'.format(acs,pzs)
                    for cm in tqdm.tqdm(climate_models_list,desc=sce):
                        renovation = get_resirf_output(acs,pzs,cm,'Adoption cooler portable unit (Thousand households)',os.path.join(output,folder)).loc[years].sum()*1e3
                        ac_portable_df['scenario'].append(sce)
                        ac_portable_df['climate_model'].append(cm)
                        ac_portable_df['value'].append(renovation)
            ac_portable_df = pd.DataFrame().from_dict(ac_portable_df)
            ac_portable_df.to_parquet(os.path.join(output,folder,save))
        else:
            ac_portable_df = pd.read_parquet(os.path.join(output,folder,save))
        ac_portable_df['value'] *= 240 #kgCO2eq 
        
        ges = df_consumption_agg_zcl_yearly.groupby(['climate_model','scenario'],as_index=False)['total_emissions'].sum()[['climate_model','scenario','total_emissions']]
        ges['variable'] = ['total_emissions_kgCO2eq']*len(ges)
        ges = ges.rename(columns={'total_emissions':'value'})[['scenario','climate_model','variable','value']]
        ges['value'] += construction
        ges = ges.set_index(['scenario','climate_model']).join(renovation_df.set_index(['scenario','climate_model']),rsuffix='_renovation').reset_index()
        ges = ges.set_index(['scenario','climate_model']).join(ac_fixe_df.set_index(['scenario','climate_model']),rsuffix='_ac_fixe').reset_index()
        ges = ges.set_index(['scenario','climate_model']).join(ac_portable_df.set_index(['scenario','climate_model']),rsuffix='_ac_portable').reset_index()
        ges['value'] = ges['value'] + ges['value_renovation'] + ges['value_ac_fixe'] + ges['value_ac_portable']
        ges = ges[['scenario','climate_model','variable','value']]
        if df_values is None:
            df_values = ges
        else:
            df_values = pd.concat([df_values,ges])
            
        # pic DJF
        conso = df_power_agg_zcl_yearly[(df_power_agg_zcl_yearly['index'].dt.year.isin(end_years))].groupby(['climate_model','scenario'],as_index=False).mean()[['climate_model','scenario','heating_pmax']]
        conso['variable'] = ['heating_pmax_W']*len(conso)
        conso = conso.rename(columns={'heating_pmax':'value'})[['scenario','climate_model','variable','value']]
        if df_values is None:
            df_values = conso
        else:
            df_values = pd.concat([df_values,conso])
        
        # pic JJA
        conso = df_power_agg_zcl_yearly[(df_power_agg_zcl_yearly['index'].dt.year.isin(end_years))].groupby(['climate_model','scenario'],as_index=False).mean()[['climate_model','scenario','cooling_pmax']]
        conso['variable'] = ['cooling_pmax_W']*len(conso)
        conso = conso.rename(columns={'cooling_pmax':'value'})[['scenario','climate_model','variable','value']]
        if df_values is None:
            df_values = conso
        else:
            df_values = pd.concat([df_values,conso])
        
        # DHI chaud
        conso = df_dhi_agg_zcl_yearly[(df_dhi_agg_zcl_yearly['index'].dt.year.isin(end_years))].groupby(['climate_model','scenario'],as_index=False).mean()[['climate_model','scenario','DHI_hot_per_household']]
        conso['variable'] = ['DHI_hot_per_household_degChourhousehold']*len(conso)
        conso = conso.rename(columns={'DHI_hot_per_household':'value'})[['scenario','climate_model','variable','value']]
        if df_values is None:
            df_values = conso
        else:
            df_values = pd.concat([df_values,conso])
        
        # DHI froid
        conso = df_dhi_agg_zcl_yearly[(df_dhi_agg_zcl_yearly['index'].dt.year.isin(end_years))].groupby(['climate_model','scenario'],as_index=False).mean()[['climate_model','scenario','DHI_cold_per_household']]
        conso['variable'] = ['DHI_cold_per_household_degChourhousehold']*len(conso)
        conso = conso.rename(columns={'DHI_cold_per_household':'value'})[['scenario','climate_model','variable','value']]
        if df_values is None:
            df_values = conso
        else:
            df_values = pd.concat([df_values,conso])
        
        # thermosensible en refroidissement
        df_thermosensitivity_cooling = pd.read_parquet(os.path.join(output,folder,'thermal_sensitivity_cooling'))
        df_thermosensitivity_cooling['variable'] = ['thermal_sensitivity_cooling_WhdegC']*len(df_thermosensitivity_cooling)
        df_thermosensitivity_cooling = df_thermosensitivity_cooling[['scenario','climate_model','variable','value']]
        if df_values is None:
            df_values = df_thermosensitivity_cooling
        else:
            df_values = pd.concat([df_values,df_thermosensitivity_cooling])
        
        # thermosensible en chauffage
        df_thermosensitivity_heating = pd.read_parquet(os.path.join(output,folder,'thermal_sensitivity_heating'))
        df_thermosensitivity_heating['variable'] = ['thermal_sensitivity_heating_WhdegC']*len(df_thermosensitivity_heating)
        df_thermosensitivity_heating = df_thermosensitivity_heating[['scenario','climate_model','variable','value']]
        if df_values is None:
            df_values = df_thermosensitivity_heating
        else:
            df_values = pd.concat([df_values,df_thermosensitivity_heating])
        
        
        dict_vars_labels = {
                            'subsidies_euro':'Total subsidies\n(G€)',
                            'total_emissions_kgCO2eq':'Carbon emissions\n(MtCO$_2$eq)',
                            'total_consumption_Wh':'Energy consumption\n(TWh.yr$^{-1}$)',
                            'heating_pmax_W':'Electricity heating peak\n(GW)',
                            'thermal_sensitivity_heating_WhdegC':'Electricity heating sensitivity\n(GW.°C$^{-1}$)',
                            'DHI_cold_per_household_degChourhousehold':'Cold incomfort\n(°C.h.yr$^{-1}$)',
                            'cooling_pmax_W':'Electricity cooling peak\n(GW)',
                            'thermal_sensitivity_cooling_WhdegC':'Electricity cooling sensitivity\n(GW.°C$^{-1})$',
                            'DHI_hot_per_household_degChourhousehold':'Hot incomfort\n(°C.h.yr$^{-1}$)',
                            }
        
        shifter = 0.02
        shift_dict = {k:idx*shifter-4*shifter for idx,k in enumerate(['ACM_NOF', 'ACM_REF', 'ACM_SOF',
                                                                      'REF_NOF', 'REF_REF', 'REF_SOF',
                                                                      'ACP_NOF', 'ACP_REF', 'ACP_SOF', ])}
        
        
        # Source - https://stackoverflow.com/a
        # Posted by Matt Pitkin, modified by community. See post 'Timeline' for change history
        # Retrieved 2025-12-06, License - CC BY-SA 4.0
        
        class MinorSymLogLocator(mpl.ticker.Locator):
            """
            Dynamically find minor tick positions based on the positions of
            major ticks for a symlog scaling.
            """
            def __init__(self, linthresh, nints=10):
                """
                Ticks will be placed between the major ticks.
                The placement is linear for x between -linthresh and linthresh,
                otherwise its logarithmically. nints gives the number of
                intervals that will be bounded by the minor ticks.
                """
                self.linthresh = linthresh
                self.nintervals = nints
        
            def __call__(self):
                # Return the locations of the ticks
                majorlocs = self.axis.get_majorticklocs()
        
                if len(majorlocs) == 1:
                    return self.raise_if_exceeds(np.array([]))
        
                # add temporary major tick locs at either end of the current range
                # to fill in minor tick gaps
                dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
                dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end
        
                # add temporary major tick location at the lower end
                if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (dmlower == self.linthresh and majorlocs[0] < 0)):
                    majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
                else:
                    majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)
        
                # add temporary major tick location at the upper end
                if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (dmupper == self.linthresh and majorlocs[-1] > 0)):
                    majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
                else:
                    majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)
        
                # iterate through minor locs
                minorlocs = []
        
                # handle the lowest part
                for i in range(1, len(majorlocs)):
                    majorstep = majorlocs[i] - majorlocs[i-1]
                    if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                        ndivs = self.nintervals
                    else:
                        ndivs = self.nintervals - 1.
        
                    minorstep = majorstep / ndivs
                    locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
                    minorlocs.extend(locs)
        
                return self.raise_if_exceeds(np.array(minorlocs))
        
            def tick_values(self, vmin, vmax):
                raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))

        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        # pd.DataFrame().from_dict({k:[0] for k in list(dict_vars_labels.values())}).T.plot.barh(ax=ax,legend=False)
        
        extreme = 0
        for idx,(var,label) in enumerate(dict_vars_labels.items()):
            df_ref_var = df_values[(df_values.scenario=='REF_REF')&(df_values.variable==var)]
            
            for scenario in sorted(list(set(df_values.scenario.values))):
                df_sce_var = df_values[(df_values.scenario==scenario)&(df_values.variable==var)]
                value_sce_mean = df_sce_var.value.mean()
                value_sce_std = df_sce_var.value.std()
                value_sce_relative = value_sce_mean/df_ref_var.value.mean()
                value_sce_percent = (value_sce_relative - 1)*100
                # value_sce_relative_std_up = (value_sce_mean+value_sce_std)/df_ref_var.value.mean()
                # value_sce_relative_std_down = (value_sce_mean-value_sce_std)/df_ref_var.value.mean()
                value_sce_percent_up = np.abs(((value_sce_mean+value_sce_std)/df_ref_var.value.mean()-1)*100)
                value_sce_percent_down = np.abs(((value_sce_mean-value_sce_std)/df_ref_var.value.mean()-1)*100)
                
                # gap = np.abs(value_sce_relative-1)
                # extreme = max([np.abs(value_sce_percent+value_sce_percent_up),np.abs(value_sce_percent-value_sce_percent_down),extreme])
                extreme = max([np.abs(value_sce_percent),np.abs(value_sce_percent),extreme])
                
                mec = 'w'
                ms = 8
                zorder = 3
                if scenario == 'REF_REF':
                    zorder = 2
                ax.plot([value_sce_percent],[idx+shift_dict.get(scenario)],color=get_scenarios_color().get(scenario),ls='',marker='o',mec=mec,ms=ms,zorder=zorder)
                # ax.errorbar([value_sce_percent],[idx+shift_dict.get(scenario)],color=get_scenarios_color().get(scenario),ls='',marker='o',mec=mec,ms=ms,xerr=np.asarray([[value_sce_percent_down,value_sce_percent_up]]).T,capsize=3)
        
        axin = ax.inset_axes([0.0013*5, 1-0.305-0.0013*100, 0.305, 0.305],zorder=2.7)    # create new inset axes in data coordinates
        axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
        axin.axis('off')
        
        ylim = ax.get_ylim()
        ax.plot([0]*2,ylim,color='k',zorder=-1,alpha=0.7,lw=1)
        ax.fill_between([-1,1],[ylim[1]]*2,[ylim[0]]*2,alpha=0.2,color='grey',zorder=-2)
        ax.set_ylim(ylim)
        ax.grid(axis='y',zorder=-2)
        # ax.set_xscale('function', functions=(forward, inverse))
        ax.set_xscale('symlog')
        ax.yaxis.set_inverted(True) 
        ax.set_xlim([-extreme*1.1,extreme*1.1])
        # ax.set_xlim([-40,40])
        ax.set_xlim([-100,100])
        
        xaxis = plt.gca().xaxis
        xaxis.set_minor_locator(MinorSymLogLocator(1e-1))
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        
        for spine in ax.spines.values():
            spine.set_zorder(10)
        
        ax.set_xlabel('Relative change compared to reference scenario (%)')
        ax.set_yticks(list(range(len(list(dict_vars_labels.values())))), list(dict_vars_labels.values()))
        plt.savefig(os.path.join(figs_folder,'summary_2050_1.png'), bbox_inches='tight')
        plt.show()
        
        # une seule variable (meilleure échelle)
        if False:
            for variable in dict_vars_labels.keys():
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                # pd.DataFrame().from_dict({k:[0] for k in list(dict_vars_labels.values())}).T.plot.barh(ax=ax,legend=False)
                
                extreme = 0
                for idx,(var,label) in enumerate(dict_vars_labels.items()):
                    if var != variable:
                        continue
                    df_ref_var = df_values[(df_values.scenario=='REF_REF')&(df_values.variable==var)]
                    
                    for scenario in sorted(list(set(df_values.scenario.values))):
                        df_sce_var = df_values[(df_values.scenario==scenario)&(df_values.variable==var)]
                        value_sce_mean = df_sce_var.value.mean()
                        value_sce_std = df_sce_var.value.std()
                        value_sce_relative = value_sce_mean/df_ref_var.value.mean()
                        value_sce_percent = (value_sce_relative - 1)*100
                        # value_sce_relative_std_up = (value_sce_mean+value_sce_std)/df_ref_var.value.mean()
                        # value_sce_relative_std_down = (value_sce_mean-value_sce_std)/df_ref_var.value.mean()
                        value_sce_percent_up = np.abs(((value_sce_mean+value_sce_std)/df_ref_var.value.mean()-1)*100)
                        value_sce_percent_down = np.abs(((value_sce_mean-value_sce_std)/df_ref_var.value.mean()-1)*100)
                        
                        # gap = np.abs(value_sce_relative-1)
                        # extreme = max([np.abs(value_sce_percent+value_sce_percent_up),np.abs(value_sce_percent-value_sce_percent_down),extreme])
                        extreme = max([np.abs(value_sce_percent),np.abs(value_sce_percent),extreme])
                        
                        mec = 'w'
                        ms = 8
                        zorder = 3
                        if scenario == 'REF_REF':
                            zorder = 2
                        ax.plot([value_sce_percent],[idx+shift_dict.get(scenario)],color=get_scenarios_color().get(scenario),ls='',marker='o',mec=mec,ms=ms,zorder=zorder)
                        # ax.errorbar([value_sce_percent],[idx+shift_dict.get(scenario)],color=get_scenarios_color().get(scenario),ls='',marker='o',mec=mec,ms=ms,xerr=np.asarray([[value_sce_percent_down,value_sce_percent_up]]).T,capsize=3)
                
                axin = ax.inset_axes([1-0.305-0.0013, 1-0.305-0.0013, 0.305, 0.305],zorder=2.7)    # create new inset axes in data coordinates
                axin.imshow(plt.imread(get_sample_data('/home/amounier/PycharmProjects/thermal/data/scenarios_compact.png')))
                axin.axis('off')
                
                ylim = ax.get_ylim()
                ax.plot([0]*2,ylim,color='k',zorder=-1,alpha=0.7,lw=1)
                ax.set_ylim(ylim)
                ax.grid(axis='y',zorder=-2)
                # ax.set_xscale("log")
                ax.yaxis.set_inverted(True) 
                ax.set_xlim([1-extreme*1.1,1+extreme*1.1])
                ax.set_xlim([-extreme*1.1,extreme*1.1])
                ax.set_xlim([-100,100])
                # ax.patch.set_zorder(0)
                # for spine in ax.spines.values():
                #     spine.set_zorder(10)
                ax.set_xlabel('Relative change compared to reference scenario (%)')
                ax.set_yticks(list(range(len(list(dict_vars_labels.values())))), list(dict_vars_labels.values()))
                plt.savefig(os.path.join(figs_folder,'summary_2050_{}.png'.format(variable)), bbox_inches='tight')
                plt.show()
    
    #%% Autres graphes
    if False:
        
        # etude des besoins horaire par jour 
        if True:
            # reference 
            ninja = pd.read_excel("data/Ninja/41560_2023_1341_MOESM9_ESM_doubled.xlsx",sheet_name='Figure ED3')
            moreau = pd.read_csv('data/Res-IRF/hourly_profile_moreau_doubled.csv')
            moreau['value'] = moreau['value']/moreau['value'].mean()
            
            needs = None
            
            for zcl in ZCL_LIST:
                stock = Stock(zcl=zcl,climate_model='EC-EARTH_HadREM3-GA7',folder=os.path.join(output, folder))
                stock.format_energy_needs_hourly(year=2020)
                temp_needs = stock.energy_needs_hourly
                if needs is None:
                    needs = temp_needs
                else:
                    needs.heating_needs = needs.heating_needs + temp_needs.heating_needs
                    needs.cooling_needs = needs.cooling_needs + temp_needs.cooling_needs
            
            hours_list = np.asarray(list(range(0,25)))
            mean_cooling_list = []
            mean_heating_list = []
            std_cooling_list = []
            std_heating_list = []
            for h in hours_list[:-1]:
                mean_cooling_list.append(needs[(needs.index.hour==h)].cooling_needs.mean())
                mean_heating_list.append(needs[(needs.index.hour==h)].heating_needs.mean())
                std_cooling_list.append(needs[needs.index.hour==h].cooling_needs.std())
                std_heating_list.append(needs[needs.index.hour==h].heating_needs.std())
            mean_cooling_list.append(mean_cooling_list[0])
            mean_heating_list.append(mean_heating_list[0])
            std_cooling_list.append(std_cooling_list[0])
            std_heating_list.append(std_heating_list[0])
            
            mean_cooling_list = np.asarray(mean_cooling_list)
            mean_heating_list = np.asarray(mean_heating_list)
            std_cooling_list = np.asarray(std_cooling_list)
            std_heating_list = np.asarray(std_heating_list)
            
            # upper_cooling_list = (mean_cooling_list+std_cooling_list)/np.mean(mean_cooling_list)
            # lower_cooling_list = (mean_cooling_list-std_cooling_list)/np.mean(mean_cooling_list)
            # upper_heating_list = (mean_heating_list+std_heating_list)/np.mean(mean_heating_list)
            # lower_heating_list = (mean_heating_list-std_heating_list)/np.mean(mean_heating_list)
            
            mean_cooling_list = mean_cooling_list/np.mean(mean_cooling_list)
            mean_heating_list = mean_heating_list/np.mean(mean_heating_list)
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.plot(moreau.hour,moreau['value'],color='tab:red',ls=':')
            ax.plot(ninja.Hour,ninja['Heating (mean)'],color='tab:red',ls=':')
            ax.plot(ninja.Hour,ninja['Cooling (mean)'],color='tab:blue',ls=':')
            ax.plot(hours_list,mean_heating_list, color='tab:red',label='Heating')
            ax.plot(hours_list,mean_cooling_list, color='tab:blue',label='Cooling')
            # ax.fill_between(hours_list,upper_cooling_list,lower_cooling_list, color='tab:blue',alpha=0.2)
            ax.plot([-1],[0],color='k',label='Modelled')
            ax.plot([-1],[0],color='k',ls=':',label='Staffell et al. (2023)')
            # ax.fill_between(hours_list,upper_heating_list,lower_heating_list, color='tab:red',alpha=0.2)
            ax.legend()
            ax.set_ylim(bottom=0.)
            ax.set_xlim([0,24])
            ax.set_xlabel('Hours')
            ax.set_ylabel('Average daily profile (normalised)')
            plt.savefig(os.path.join(figs_folder,'daily_profile_conventionnal.png'), bbox_inches='tight')
            plt.show()
            
            
            
        # affichage des seuils d'inconfort
        if False:
            define_DHI_threshold(1.,plot=True,save_fig=figs_folder)
            
        # affichage d'intensité d'usage Astier
        if False:
            gap_dpe = pd.read_csv('data/CAE/use_intensity_dpe.csv')
            
            a,b = np.polyfit(gap_dpe.standard_consumption,gap_dpe.consumption, deg=1)
            Y_compare = gap_dpe.standard_consumption*a+b
            r2 = r2_score(gap_dpe.consumption, Y_compare)
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(gap_dpe.standard_consumption,gap_dpe.consumption,label='Astier et al. 2024',marker='o',ls='')
            
            xlims,ylims = ax.get_xlim(), ax.get_ylim()
            max_lim = np.max([np.max(xlims),np.max(ylims)])
            
            ax.plot([0,max_lim],[0,max_lim],color='k',zorder=-1)
            ax.set_xlim([0,max_lim])
            ax.set_ylim([0,max_lim])
            
            X = np.linspace(0,max_lim)
            Y = a*X+b
            ax.plot(X,Y,label='Linear regression (R$^2$={:.2f})\n(y = {:.1f}$\\cdot$x + {:.1f})'.format(r2,a,b),color='tab:blue',alpha=0.6)
            
            ax.set_xlabel('Conventional consumption (kWh.m$^{-2}$.yr$^{-1}$)')
            ax.set_ylabel('Actual consumption (kWh.m$^{-2}$.yr$^{-1}$)')
            ax.legend()
            plt.savefig(os.path.join(figs_folder,'consumption_astier.png'), bbox_inches='tight')
            plt.show()
            
            ui = gap_dpe.consumption/gap_dpe.standard_consumption
            
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(gap_dpe.standard_consumption,ui,label='Astier et al. 2024',marker='o',ls='')
            
            xlims,ylims = ax.get_xlim(), ax.get_ylim()
            # max_lim = np.max([np.max(xlims),np.max(ylims)])
            
            # ax.plot([0,max_lim],[0,max_lim],color='k',zorder=-1)
            ax.set_xlim([0,xlims[1]])
            ax.set_ylim([0,3])
            
            X = np.linspace(0,xlims[1])
            Y = a+b/X
            ax.plot(X,Y,label='Inverse regression (R$^2$={:.2f})\n(y = {:.1f} + {:.1f}/x)'.format(r2,a,b),color='tab:blue',alpha=0.6)
            
            ax.set_xlabel('Conventional consumption (kWh.m$^{-2}$.yr$^{-1}$)')
            ax.set_ylabel('Use intensity (ratio)')
            ax.legend()
            plt.savefig(os.path.join(figs_folder,'consumption_astier_inverse.png'), bbox_inches='tight')
            plt.show()
        
        
    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
    
if __name__ == '__main__':
    main()
        