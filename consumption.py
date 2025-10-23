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
import matplotlib.patheffects as pe

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
        
    def __str__(self):
        return self.scenario_name
    
    
    def compute_reduced_technical_stock(self, threshold=0.8, plot=False):
        print('{} - Stock reduction...'.format(self.scenario_name))
        # on garde les plus grands segments, représentant 80% du parc
        end_stock = self.technical_stock.sum(axis=1)
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
            plt.savefig(os.path.join(self.figs_path,'reduced_stock__{}.png'.format(self.scenario_name)), bbox_inches='tight')
            plt.show()
            
            if False:
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                plot_years = [self.start_year,self.end_year]
                for y in plot_years:
                    X = np.linspace(0,len(self.technical_stock),50)
                    Y = [0.]*len(X)
                    for idx,n_seg in enumerate(X):
                        largest = self.technical_stock[y].nlargest(int(n_seg))
                        Y[idx] = largest.sum()/self.technical_stock[y].sum()
                        
                    ax.plot(X,Y,label=y)
                    
                # ax.set_ylim(bottom=0.,top=1.)
                ax.set_xlim(left=0.)
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
        # ne pas pondérer les consommations ! c'est unitaire ici TODO
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
            
            def use_intensity(bill, A=0.02, zeta=5):
                return (bill*A)**(-1/zeta)
            
            self.heating_use_intensity = use_intensity(energy_heating_bill)
            self.cooling_use_intensity = use_intensity(energy_cooling_bill)
            
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
            self.compute_use_intensity(method='isoelastique')
        if self.energy_consumption_heating_conv is None:
            self.compute_energy_consumption_conv()
            
        self.energy_consumption_heating = self.heating_use_intensity * self.energy_consumption_heating_conv
        self.energy_consumption_cooling = self.cooling_use_intensity * self.energy_consumption_cooling_conv
        # self.energy_consumption_heating = self.energy_consumption_heating_conv
        # self.energy_consumption_cooling = self.energy_consumption_cooling_conv
        return 
    
    
    def compute_energy_consumption_conv(self):
        save_name_energy_heating = '{}_energy_consumption_heating_conv.parquet'.format(self.scenario_name)
        save_name_energy_cooling = '{}_energy_consumption_cooling_conv.parquet'.format(self.scenario_name)
        
        if save_name_energy_heating in os.listdir(self.path):
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
        
        self.energy_consumption_heating_conv.to_parquet(os.path.join(self.path,save_name_energy_heating))
        self.energy_consumption_cooling_conv.to_parquet(os.path.join(self.path,save_name_energy_cooling))
        return
        
    
    def compute_stock_energy_needs(self):
        if self.energy_needs_heating is None:
            self.format_energy_needs()
        if self.reduced_technical_stock is None:
            self.compute_reduced_technical_stock()
            
        self.stock_energy_needs_heating = self.energy_needs_heating * self.reduced_technical_stock
        self.stock_energy_needs_cooling = self.energy_needs_cooling * self.reduced_technical_stock
    
    
    def compute_stock_energy_consumption_conv(self):
        save_name_energy_heating = '{}_stock_energy_consumption_heating_conv.parquet'.format(self.scenario_name)
        save_name_energy_cooling = '{}_stock_energy_consumption_cooling_conv.parquet'.format(self.scenario_name)
        
        if save_name_energy_heating in os.listdir(self.path):
            self.stock_energy_consumption_heating_conv = pd.read_parquet(os.path.join(self.path,save_name_energy_heating))
            self.stock_energy_consumption_cooling_conv = pd.read_parquet(os.path.join(self.path,save_name_energy_cooling))
            return 
        
        if self.stock_energy_needs_heating is None:
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
        
        self.stock_energy_consumption_heating_conv.to_parquet(os.path.join(self.path,save_name_energy_heating))
        self.stock_energy_consumption_cooling_conv.to_parquet(os.path.join(self.path,save_name_energy_cooling))
        return
    
    
    def compute_stock_energy_consumption(self):
        if self.energy_consumption_heating is None:
            self.compute_energy_consumption()
        if self.reduced_technical_stock is None:
            self.compute_reduced_technical_stock()
            
        self.stock_energy_consumption_heating = self.energy_consumption_heating * self.reduced_technical_stock
        self.stock_energy_consumption_cooling = self.energy_consumption_cooling * self.reduced_technical_stock
        return 

    
    def get_daily_DHI(self, energy_filter=None):
        if energy_filter is not None:
            save_name = '{}_daily_DHI_{}.parquet'.format(self.scenario_name,energy_filter)
        else:
            save_name = '{}_daily_DHI.parquet'.format(self.scenario_name)
        if save_name not in os.listdir(self.path):
            if self.typologies is None:
                self.compute_typologies()
                
            self.compute_energy_needs()
            
            # ajout du système de chauffage 
            temp_typologies = pd.DataFrame(self.add_level(self.typologies['complete_save'], self.reduced_technical_stock[2018], 'Heating system',keep_distribution=False)).rename(columns={0:'complete_save'})
            temp_typologies = temp_typologies.loc[self.reduced_technical_stock.index]
            
            DHI = None
            print('{} - Energy needs formatting...'.format(self.scenario_name))
            for typo_index,typo_save in zip(temp_typologies.index,temp_typologies.complete_save):
                needs = pd.read_parquet(os.path.join(self.rc_path,typo_save))
                needs['year'] = needs.index.year
                needs['nb'] = needs.year.map(self.reduced_technical_stock.loc[typo_index].to_dict().get)
                needs['DHI_hot'] = needs.hot_DH * needs.nb
                needs['DHI_cold'] = needs.cold_DH * needs.nb
                
                if energy_filter == 'Electricity':
                    if typo_index[-1].split('-')[0] != energy_filter:
                        needs['DHI_cold'] = needs.DHI_cold*0
                
                if DHI is None:
                    DHI = needs[['temperature_2m', 'DHI_hot', 'DHI_cold']]
                else:
                    DHI['DHI_hot'] = DHI['DHI_hot'] + needs['DHI_hot']
                    DHI['DHI_cold'] = DHI['DHI_cold'] + needs['DHI_cold']
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
                self.compute_use_intensity(method='isoelastique')
                
            self.compute_energy_needs()
            
            # ajout du système de chauffage 
            temp_typologies = pd.DataFrame(self.add_level(self.typologies['complete_save'], self.reduced_technical_stock[2018], 'Heating system',keep_distribution=False)).rename(columns={0:'complete_save'})
            temp_typologies = temp_typologies.loc[self.reduced_technical_stock.index]
            
            consumption = None
            print('{} - Energy needs formatting...'.format(self.scenario_name))
            for typo_index,typo_save in zip(temp_typologies.index,temp_typologies.complete_save):
                needs = pd.read_parquet(os.path.join(self.rc_path,typo_save))
                needs['year'] = needs.index.year
                needs['nb'] = needs.year.map(self.reduced_technical_stock.loc[typo_index].to_dict().get)
                
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
                        
                if consumption is None:
                    consumption = needs[['temperature_2m', 'heating_cons', 'cooling_cons']]
                else:
                    consumption['heating_cons'] = consumption['heating_cons'] + needs['heating_cons']
                    consumption['cooling_cons'] = consumption['cooling_cons'] + needs['cooling_cons']
            consumption['total_cons'] = consumption.heating_cons + consumption.cooling_cons
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
                self.compute_use_intensity(method='isoelastique')
                
            self.compute_energy_needs()
            
            # ajout du système de chauffage 
            temp_typologies = pd.DataFrame(self.add_level(self.typologies['complete_save'], self.reduced_technical_stock[2018], 'Heating system',keep_distribution=False)).rename(columns={0:'complete_save'})
            temp_typologies = temp_typologies.loc[self.reduced_technical_stock.index]
            
            power = None
            print('{} - Energy needs formatting...'.format(self.scenario_name))
            for typo_index,typo_save in zip(temp_typologies.index,temp_typologies.complete_save):
                needs = pd.read_parquet(os.path.join(self.rc_path,typo_save))
                needs['year'] = needs.index.year
                needs['nb'] = needs.year.map(self.reduced_technical_stock.loc[typo_index].to_dict().get)
                
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
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    #%% premier test
    if False:
        stock = Stock(zcl='H1a',climate_model='EC-EARTH_HadREM3-GA7',folder=os.path.join(output, folder))
        

        
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
        
    
    #%% lancement des calculs pour tous les modèles climatiques 
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        
        for cm in climate_models_list:
            dict_zcl_stock = {}
            for zcl in ZCL_LIST:
                dict_zcl_stock[zcl] = Stock(zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
            
            for zcl in ZCL_LIST:
                dict_zcl_stock[zcl].compute_stock_energy_consumption()
        
        
    #%% Étude des consommations d'énergie annuelle et plus fine
    sdes_data_heating = pd.read_csv(os.path.join('data','SDES','consommations_energie_chauffage_residentiel.csv')).set_index('Heating system').rename(columns={str(y):y for y in range(2017,2024)})
    sdes_data_cooling = pd.read_csv(os.path.join('data','SDES','consommations_energie_climatisation_residentiel.csv')).set_index('Cooling system').rename(columns={str(y):y for y in range(2017,2024)})
    
    # Vérification de la consommation nationale initiale
    if False:
        year = 2020
        # sdes_data_2018 = {k:v*0.8 for k,v in sdes_data_2018.items()}
        
        # ademe_correction 
        # sdes_data_cooling = sdes_data_cooling * 2.634408602
        
        dict_zcl_stock = {}
        for zcl in ZCL_LIST:
            dict_zcl_stock[zcl] = Stock(zcl=zcl,climate_model='EC-EARTH_HadREM3-GA7',folder=os.path.join(output, folder))
        
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
        
        
        heating = pd.DataFrame(heating.groupby(heating.index.get_level_values('Heating system')).sum().sum(axis=1)).rename(columns={0:'modelled'})
        heating = heating * 1e-12
        heating = heating.join(sdes_data_heating[[year]].rename(columns={year:'ref'}),how='outer')
        heating.loc['Electricity-Heat pump','modelled'] = heating.loc['Electricity-Heat pump air','modelled'] + heating.loc['Electricity-Heat pump water','modelled']
        heating = heating.dropna(axis=0)
        heating['ratio'] = heating.ref/heating.modelled
        
        print('Heating consumption {}'.format(year))
        print(heating)
        print(heating.sum())
        
        # cooling = pd.DataFrame(cooling.groupby(cooling.index.get_level_values('Cooling system')).sum().sum(axis=1)).rename(columns={0:'modelled'})
        # cooling = cooling * 1e-12
        # cooling = cooling.join(sdes_data_cooling[[year]].rename(columns={year:'ref'}),how='outer')
        # cooling.loc['Electricity-AC','modelled'] = cooling.loc['Electricity-Heat pump air','modelled'] + cooling.loc['Electricity-Portable unit','modelled']
        # cooling = cooling.dropna(axis=0)
        
        # print('Cooling consumption {}'.format(year))
        # print(cooling)
        
    # affichage des séries temporelles de consommations
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        scenarios = None
        color = None
        
        df_consumption = None
        for cm in climate_models_list:
            for zcl in ZCL_LIST:
                s = Stock(zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
                temp = s.get_daily_consumption()
                if scenarios is None:
                    scenarios = s.ac_pz_scenario
                    color = s.scenario_color
                del s
                if df_consumption is None:
                    df_consumption = temp
                else:
                    df_consumption = pd.concat([df_consumption, temp])
        
        df_consumption_agg_zcl = df_consumption[['heating_cons','cooling_cons','total_cons','climate_model']].reset_index().groupby(['index','climate_model'],as_index=False).sum()
        
        df_consumption_agg_zcl_yearly = None
        for cm in climate_models_list:
            temp = df_consumption_agg_zcl[df_consumption_agg_zcl.climate_model==cm].set_index('index')
            temp = aggregate_resolution(temp,'YS','sum')
            temp['climate_model'] = [cm]*len(temp)
            temp = temp.reset_index()
            if df_consumption_agg_zcl_yearly is None:
                df_consumption_agg_zcl_yearly = temp
            else:
                df_consumption_agg_zcl_yearly = pd.concat([df_consumption_agg_zcl_yearly,temp])
        
        heating_mean = df_consumption_agg_zcl_yearly.groupby('index')['heating_cons'].mean()
        heating_std = df_consumption_agg_zcl_yearly.groupby('index')['heating_cons'].std()
        
        cooling_mean = df_consumption_agg_zcl_yearly.groupby('index')['cooling_cons'].mean()
        cooling_std = df_consumption_agg_zcl_yearly.groupby('index')['cooling_cons'].std()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot([pd.to_datetime('{}-01-01'.format(y)) for y in sdes_data_heating.sum().index], sdes_data_heating.sum().values,label='SDES',color='k')
        ax.plot(heating_mean*1e-12,label=scenarios.replace('_',' - '),color=color)
        ax.fill_between(heating_std.index,heating_mean.values*1e-12+heating_std.values*1e-12,heating_mean.values*1e-12-heating_std.values*1e-12,alpha=0.5,color=color)
        ax.set_ylabel('Annual heating consumption (TWh.yr$^{-1}$)')
        ax.legend()
        ax.set_ylim(bottom=0.)
        ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
        plt.savefig(os.path.join(figs_folder,'heating_consumption.png'), bbox_inches='tight')
        plt.show()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot([pd.to_datetime('{}-01-01'.format(y)) for y in sdes_data_cooling.sum().index], sdes_data_cooling.sum().values,label='SDES',color='k')
        ax.plot(cooling_mean*1e-12,label=scenarios.replace('_',' - '),color=color)
        ax.fill_between(cooling_std.index,cooling_mean.values*1e-12+cooling_std.values*1e-12,cooling_mean.values*1e-12-cooling_std.values*1e-12,alpha=0.5,color=color)
        ax.set_ylabel('Annual cooling consumption (TWh.yr$^{-1}$)')
        ax.legend()
        ax.set_ylim(bottom=0.)
        ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
        plt.savefig(os.path.join(figs_folder,'cooling_consumption.png'), bbox_inches='tight')
        plt.show()
        
        
        # juste électricité
        df_consumption = None
        for cm in climate_models_list:
            for zcl in ZCL_LIST:
                s = Stock(zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
                temp = s.get_daily_consumption(energy_filter='Electricity')
                temp['heating_cons'] = temp['heating_cons']*0.5
                if scenarios is None:
                    scenarios = s.ac_pz_scenario
                    color = s.scenario_color
                del s
                if df_consumption is None:
                    df_consumption = temp
                else:
                    df_consumption = pd.concat([df_consumption, temp])
        
        df_consumption_agg_zcl = df_consumption[['heating_cons','cooling_cons','total_cons','climate_model']].reset_index().groupby(['index','climate_model'],as_index=False).sum()
        
        df_consumption_agg_zcl_yearly = None
        for cm in climate_models_list:
            temp = df_consumption_agg_zcl[df_consumption_agg_zcl.climate_model==cm].set_index('index')
            temp = aggregate_resolution(temp,'YS','sum')
            temp['climate_model'] = [cm]*len(temp)
            temp = temp.reset_index()
            if df_consumption_agg_zcl_yearly is None:
                df_consumption_agg_zcl_yearly = temp
            else:
                df_consumption_agg_zcl_yearly = pd.concat([df_consumption_agg_zcl_yearly,temp])
        
        heating_mean = df_consumption_agg_zcl_yearly.groupby('index')['heating_cons'].mean()
        heating_std = df_consumption_agg_zcl_yearly.groupby('index')['heating_cons'].std()
        
        cooling_mean = df_consumption_agg_zcl_yearly.groupby('index')['cooling_cons'].mean()
        cooling_std = df_consumption_agg_zcl_yearly.groupby('index')['cooling_cons'].std()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        # ax.plot([pd.to_datetime('{}-01-01'.format(y)) for y in sdes_data_heating.sum().index], sdes_data_heating.sum().values,label='SDES',color='k')
        ax.plot(heating_mean*1e-12,label=scenarios.replace('_',' - '),color=color)
        ax.fill_between(heating_std.index,heating_mean.values*1e-12+heating_std.values*1e-12,heating_mean.values*1e-12-heating_std.values*1e-12,alpha=0.5,color=color)
        ax.set_ylabel('Annual heating consumption (TWh.yr$^{-1}$)')
        ax.legend()
        ax.set_ylim(bottom=0.)
        ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
        plt.savefig(os.path.join(figs_folder,'heating_Electricity_consumption.png'), bbox_inches='tight')
        plt.show()
    
    
    # affichage des thermosensibilités
    if True:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        scenarios = None
        color = None
        
        df_consumption = None
        for cm in climate_models_list:
            for zcl in ZCL_LIST:
                s = Stock(zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
                temp = s.get_daily_consumption(energy_filter='Electricity')
                temp['heating_cons'] = temp['heating_cons']*0.5 # TODO transferts de vecteurs à mieux faire !
                if scenarios is None:
                    scenarios = s.ac_pz_scenario
                    # color_sce = s.scenario_color
                    color_sce = plt.get_cmap('viridis')(0.5)
                del s
                if df_consumption is None:
                    df_consumption = temp
                else:
                    df_consumption = pd.concat([df_consumption, temp])
                    
        df_temperature = df_consumption[['temperature_2m','zcl','climate_model']].copy().reset_index()
        df_temperature.loc[:,'ratio'] = df_temperature.zcl.map(ZCL_POPULATION_DISTRIBUTION.get)
        df_temperature['temperature'] = df_temperature.temperature_2m * df_temperature.ratio
        df_temperature = df_temperature[['temperature','climate_model','index']].groupby(['index','climate_model'],as_index=True).sum()
        
        df_consumption_agg_zcl = df_consumption[['heating_cons','cooling_cons','total_cons','climate_model']].reset_index().groupby(['index','climate_model'],as_index=True).sum()
        df_consumption_agg_zcl = df_consumption_agg_zcl.join(df_temperature)
        
        
        th_list_2020 = []
        kh_list_2020 = []
        th_list_2050 = []
        kh_list_2050 = []
        for cm in climate_models_list:
            for y in [2023,2050]:
                df = df_consumption_agg_zcl[(df_consumption_agg_zcl.index.get_level_values('index').year.isin(list(range(y-5,y+1))))&(df_consumption_agg_zcl.index.get_level_values('climate_model')==cm)]

                th, kh, r2 = identify_thermal_sensitivity(df.temperature, df.heating_cons, cooling=False)
                if y == 2023:
                    th_list_2020.append(th)
                    kh_list_2020.append(kh)
                if y == 2050:
                    th_list_2050.append(th)
                    kh_list_2050.append(kh)
        
        th_list_2020_mean = np.mean(th_list_2020)
        kh_list_2020_mean = np.mean(kh_list_2020)
        kh_list_2020_std = np.std(kh_list_2020)
        th_list_2050_mean = np.mean(th_list_2050)
        kh_list_2050_mean = np.mean(kh_list_2050)
        kh_list_2050_std = np.std(kh_list_2050)
        
        
        tc_list_2020 = []
        kc_list_2020 = []
        tc_list_2050 = []
        kc_list_2050 = []
        for cm in climate_models_list:
            for y in [2023,2050]:
                df = df_consumption_agg_zcl[(df_consumption_agg_zcl.index.get_level_values('index').year.isin(list(range(y-5,y+1))))&(df_consumption_agg_zcl.index.get_level_values('climate_model')==cm)]

                tc, kc, r2 = identify_thermal_sensitivity(df.temperature, df.cooling_cons, cooling=True)
                if y == 2023:
                    tc_list_2020.append(tc)
                    kc_list_2020.append(kc)
                if y == 2050:
                    tc_list_2050.append(tc)
                    kc_list_2050.append(kc)
        
        tc_list_2020_mean = np.mean(tc_list_2020)
        kc_list_2020_mean = np.mean(kc_list_2020)
        kc_list_2020_std = np.std(kc_list_2020)
        tc_list_2050_mean = np.mean(tc_list_2050)
        kc_list_2050_mean = np.mean(kc_list_2050)
        kc_list_2050_std = np.std(kc_list_2050)
        
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
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        
        ax.plot(T,Yh_2020_mean,color='k',label='2018-2023',zorder=-1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        ax.fill_between(T,Yh_2020_upper,Yh_2020_lower,color='k',alpha=0.37,zorder=-1)
        ax.plot(T,Yh_2050_mean,color=color_sce,label='2045-2050',path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        ax.fill_between(T,Yh_2050_upper,Yh_2050_lower,color=color_sce,alpha=0.37)
        
        ax.plot(T,Yc_2020_mean,color='k',zorder=-1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        ax.fill_between(T,Yc_2020_upper,Yc_2020_lower,color='k',alpha=0.37,zorder=-1,)
        ax.plot(T,Yc_2050_mean,color=color_sce,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        ax.fill_between(T,Yc_2050_upper,Yc_2050_lower,color=color_sce,alpha=0.37)
        
        for cm in climate_models_list:
            for y in [2023,2050]:
                if y == 2023:
                    color = 'k'
                else: 
                    color = color_sce
                df = df_consumption_agg_zcl[(df_consumption_agg_zcl.index.get_level_values('index').year.isin(list(range(y-5,y+1))))&(df_consumption_agg_zcl.index.get_level_values('climate_model')==cm)]
                df.loc[:,'total_cons'] = df.heating_cons + df.cooling_cons
                sns.scatterplot(data=df,x='temperature',y='total_cons',alpha=0.005,ax=ax,color=color,zorder=-2)
        ax.set_xlim([T[0],T[-1]])
        
        ax.legend()
        ax.set_ylabel('Daily electricity consumption (Wh)')
        ax.set_xlabel('Daily external temperature (°C)')
        ax.set_ylim(bottom=0.)
        plt.savefig(os.path.join(figs_folder,'thermal_sensitivity_Electricity_consumption.png'), bbox_inches='tight')
        plt.show()
        
        
        
    # affichage des séries temporelles de puissances
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        scenarios = None
        color = None
        
        df_power = None
        for cm in climate_models_list:
            for zcl in ZCL_LIST:
                s = Stock(zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
                temp = s.get_daily_power(energy_filter='Electricity')
                if scenarios is None:
                    scenarios = s.ac_pz_scenario
                    color = s.scenario_color
                del s
                if df_power is None:
                    df_power = temp
                else:
                    df_power = pd.concat([df_power, temp])
        
        df_power_agg_zcl = df_power[['heating_pmax', 'cooling_pmax', 'total_pmax','climate_model']].reset_index().groupby(['index','climate_model'],as_index=False).sum()
        
        df_power_agg_zcl_yearly = None
        for cm in climate_models_list:
            temp_heating = df_power_agg_zcl[(df_power_agg_zcl.climate_model==cm)&(df_power_agg_zcl['index'].dt.month.isin([12,1,2]))].set_index('index')
            temp_cooling = df_power_agg_zcl[(df_power_agg_zcl.climate_model==cm)&(df_power_agg_zcl['index'].dt.month.isin([6,7,8]))].set_index('index')
            
            temp_heating = aggregate_resolution(temp_heating[['heating_pmax', 'cooling_pmax']],'YS','max')
            temp_heating['climate_model'] = [cm]*len(temp_heating)
            temp_heating = temp_heating.reset_index()
            
            temp_cooling = aggregate_resolution(temp_cooling[['heating_pmax', 'cooling_pmax']],'YS','max')
            temp_cooling['climate_model'] = [cm]*len(temp_cooling)
            temp_cooling = temp_cooling.reset_index()
            
            temp_heating['cooling_pmax'] = temp_cooling.cooling_pmax
            if df_power_agg_zcl_yearly is None:
                df_power_agg_zcl_yearly = temp_heating
            else:
                df_power_agg_zcl_yearly = pd.concat([df_power_agg_zcl_yearly,temp_heating])
        
        heating_mean = df_power_agg_zcl_yearly.groupby('index')['heating_pmax'].mean()
        heating_std = df_power_agg_zcl_yearly.groupby('index')['heating_pmax'].std()
        
        cooling_mean = df_power_agg_zcl_yearly.groupby('index')['cooling_pmax'].mean()
        cooling_std = df_power_agg_zcl_yearly.groupby('index')['cooling_pmax'].std()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(heating_mean*1e-9,label=scenarios.replace('_',' - '),color=color)
        ax.fill_between(heating_std.index,heating_mean.values*1e-9+heating_std.values*1e-9,heating_mean.values*1e-9-heating_std.values*1e-9,alpha=0.5,color=color)
        ax.set_ylabel('DJF Electricity power peak for heating (GW)')
        ax.legend()
        ax.set_ylim(bottom=0.)
        ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
        plt.savefig(os.path.join(figs_folder,'heating_Electricity_power.png'), bbox_inches='tight')
        plt.show()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(cooling_mean*1e-9,label=scenarios.replace('_',' - '),color=color)
        ax.fill_between(cooling_std.index,cooling_mean.values*1e-9+cooling_std.values*1e-9,cooling_mean.values*1e-9-cooling_std.values*1e-9,alpha=0.5,color=color)
        ax.set_ylabel('JJA Electricity power peak for cooling (GW)')
        ax.legend()
        ax.set_ylim(bottom=0.)
        ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
        plt.savefig(os.path.join(figs_folder,'cooling_Electricity_power.png'), bbox_inches='tight')
        plt.show()
        
        
    # affichage des séries temporelles des degrés heure d'inconfort
    if False:
        climate_models_list = list(CLIMATE_MODELS_NUMBERS.keys())
        scenarios = None
        color = None
        
        df_dhi = None
        for cm in climate_models_list:
            for zcl in ZCL_LIST:
                s = Stock(zcl=zcl,climate_model=cm,folder=os.path.join(output, folder))
                temp = s.get_daily_DHI()
                if scenarios is None:
                    scenarios = s.ac_pz_scenario
                    color = s.scenario_color
                del s
                if df_dhi is None:
                    df_dhi = temp
                else:
                    df_dhi = pd.concat([df_dhi, temp])
        
        df_dhi_agg_zcl = df_dhi[['DHI_hot', 'DHI_cold','climate_model']].reset_index().groupby(['index','climate_model'],as_index=False).sum()
        
        df_dhi_agg_zcl_yearly = None
        for cm in climate_models_list:
            temp = df_dhi_agg_zcl[(df_dhi_agg_zcl.climate_model==cm)].set_index('index')
            temp = aggregate_resolution(temp[['DHI_hot', 'DHI_cold']],'YS','sum')
            temp['climate_model'] = [cm]*len(temp)
            temp = temp.reset_index()

            if df_dhi_agg_zcl_yearly is None:
                df_dhi_agg_zcl_yearly = temp
            else:
                df_dhi_agg_zcl_yearly = pd.concat([df_dhi_agg_zcl_yearly,temp])
        
        heating_mean = df_dhi_agg_zcl_yearly.groupby('index')['DHI_cold'].mean()
        heating_std = df_dhi_agg_zcl_yearly.groupby('index')['DHI_cold'].std()
        
        cooling_mean = df_dhi_agg_zcl_yearly.groupby('index')['DHI_hot'].mean()
        cooling_std = df_dhi_agg_zcl_yearly.groupby('index')['DHI_hot'].std()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(heating_mean,label=scenarios.replace('_',' - '),color=color)
        ax.fill_between(heating_std.index,heating_mean.values+heating_std.values,heating_mean.values-heating_std.values,alpha=0.5,color=color)
        ax.set_ylabel('Cold incomfort (°C.h)')
        ax.legend()
        ax.set_ylim(bottom=0.)
        ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
        plt.savefig(os.path.join(figs_folder,'cold_dhi.png'), bbox_inches='tight')
        plt.show()
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(cooling_mean,label=scenarios.replace('_',' - '),color=color)
        ax.fill_between(cooling_std.index,cooling_mean.values+cooling_std.values,cooling_mean.values-cooling_std.values,alpha=0.5,color=color)
        ax.set_ylabel('Hot incomfort (°C.h)')
        ax.legend()
        ax.set_ylim(bottom=0.)
        ax.set_xlim([pd.to_datetime('{}-01-01'.format(y)) for y in [2018,2050]])
        plt.savefig(os.path.join(figs_folder,'hot_dhi.png'), bbox_inches='tight')
        plt.show()
    
        
    
    #%% Autres graphes
    if False:
        
        # affichage des seuils d'inconfort
        if True:
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
        