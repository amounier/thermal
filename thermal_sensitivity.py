#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:08:40 2024

@author: amounier
"""


import time 
import pandas as pd
from datetime import date
import os
import matplotlib.pyplot as plt
import numpy as np
# import tqdm
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import seaborn as sns

from meteorology import get_meteo_data
from utils import plot_timeserie

# Définition des dictionnaires administratifs de régions
dict_region_code_region_name = {11:'Île-de-France',
                                24:'Centre-Val de Loire',
                                27:'Bourgogne-Franche-Comté',
                                28:'Normandie',
                                32:'Hauts-de-France',
                                44:'Grand-Est',
                                52:'Pays de la Loire',
                                53:'Bretagne',
                                75:'Nouvelle-Aquitaine',
                                76:'Occitanie',
                                84:'Auvergne-Rhône-Alpes',
                                93:"Provence-Alpes-Côte d'Azur",
                                94:'Corse'}
dict_region_name_region_code = {v:k for k,v in dict_region_code_region_name.items()}

# pop = pd.read_excel('data/INSEE/base-ic-logement-2021.xlsx',sheet_name='IRIS',header=5)
# pop.groupby('REG')['P21_RP'].sum().to_dict()

dict_region_code_nb_log = {11: 5335031,
                           24: 1181699,
                           27: 1313611,
                           28: 1524634,
                           32: 2597977,
                           44: 2532350,
                           52: 1736156,
                           53: 1584802,
                           75: 2882774,
                           76: 2840001,
                           84: 3681163,
                           93: 2383101,
                           94: 156046}
    
dict_region_code_chef_lieu = {11:'Paris',
                              24:'Orléans',
                              27:'Dijon',
                              28:'Rouen',
                              32:'Lille',
                              44:'Strasbourg',
                              52:'Nantes',
                              53:'Rennes',
                              75:'Bordeaux',
                              76:'Toulouse',
                              84:'Lyon',
                              93:'Marseille',
                              94:'Ajaccio'}


def open_electricity_consumption(scale='national', force=False):
    enedis_data_folder = os.path.join('data','Enedis','202408_consommation_horaire')
    
    if scale == 'national':
        clean_data_file = 'national_electricity_consumption.csv'
        
        if clean_data_file not in os.listdir(enedis_data_folder) or force:
            # Données de consommation aux PDL de moins de 36 kVA (https://www.data.gouv.fr/fr/datasets/agregats-segmentes-de-consommation-electrique-au-pas-1-2-h-des-points-de-soutirage-36kva-maille-nationale-1/)
            # Il n'y a pas de résidentiel de plus de 36 kVA
            raw_data_file = 'conso-inf36.csv'
            raw_data = pd.read_csv(os.path.join(enedis_data_folder,raw_data_file), sep=';')
            
            raw_data = raw_data[['horodate', 'profil', 'plage_de_puissance_souscrite',
                                 'nb_points_soutirage', 'total_energie_soutiree_wh']]
                        
            # j'ai pas besoin des différences par puissances souscrites
            raw_data = raw_data[raw_data.plage_de_puissance_souscrite=='P0: Total <= 36 kVA']
            
            # je me focalise sur le résidentiel, et je ne m'intéresse pas aux différences de profils
            res_profiles = [c for c in set(raw_data.profil.values) if 'RES' in c]
            raw_data = raw_data[raw_data.profil.isin(res_profiles)]
            raw_data = raw_data.groupby(by='horodate')[['nb_points_soutirage', 'total_energie_soutiree_wh']].sum().reset_index()
            
            # enregistrement des données filtrées
            raw_data.to_csv(os.path.join(enedis_data_folder,clean_data_file),index=False)
        
        data = pd.read_csv(os.path.join(enedis_data_folder,clean_data_file))
        data = data.set_index('horodate')
        data.index = pd.to_datetime(data.index)
        data = data.groupby(pd.Grouper(freq='h')).sum()
        data.index = data.index.tz_localize(None)
    
    if scale == 'regional':
        clean_data_file  = 'regional_electricity_consumption.csv'
        
        if clean_data_file not in os.listdir(enedis_data_folder) or force:
            # Données de consommation aux PDL de moins de 36 kVA (https://www.data.gouv.fr/fr/datasets/agregats-segmentes-de-consommation-electrique-au-pas-1-2-h-des-points-de-soutirage-36kva-maille-nationale-1/)
            raw_data_file = 'conso-inf36-region_3.csv'
            raw_data = pd.read_csv(os.path.join(enedis_data_folder,raw_data_file), sep=';')
            
            raw_data = raw_data[['horodate', 'profil', 'plage_de_puissance_souscrite','region',
                                 'nb_points_soutirage', 'total_energie_soutiree_wh']]

            # j'ai pas besoin des différences par puissances souscrites
            raw_data = raw_data[raw_data.plage_de_puissance_souscrite=='P0: Total <= 36 kVA']
            
            # je me focalise sur le résidentiel, et je ne m'intéresse pas aux différences de profils
            res_profiles = [c for c in set(raw_data.profil.values) if 'RES' in c]
            raw_data = raw_data[raw_data.profil.isin(res_profiles)]
            raw_data = raw_data.groupby(['horodate','region'])[['nb_points_soutirage', 'total_energie_soutiree_wh']].sum().reset_index()
            
            # enregistrement des données filtrées
            raw_data.to_csv(os.path.join(enedis_data_folder,clean_data_file),index=False)
        
        data = pd.read_csv(os.path.join(enedis_data_folder,clean_data_file))
        
        reformatted_data = {'horodate':sorted(list(set(data.horodate)))}
        
        for reg in list(set(data.region.to_list())):
            if reg == 'Nouvelle Aquitaine':
                regcode = dict_region_name_region_code.get(reg.replace(' ','-'))
            else:
                regcode = dict_region_name_region_code.get(reg)
            data_reg = data[data.region==reg]
            for c in ['nb_points_soutirage', 'total_energie_soutiree_wh']:
                c_reg = c + '_reg_{}'.format(regcode)
                reformatted_data[c_reg] = data_reg[c].values
            
        data = pd.DataFrame().from_dict(reformatted_data)
        data = data.set_index('horodate')
        data.index = pd.to_datetime(data.index)
        data = data.groupby(pd.Grouper(freq='h')).sum()
        data.index = data.index.tz_localize(None)
        
        for c in data.columns:
            data[c] = data[c].replace({0:np.nan})
        
    return data


def piecewise_linear(T, Th, Tc, C0, kh, kc):
    # on force Tc à être supérieure à Th
    Tc = max(Tc,Th)
    Th = min(Tc,Th)
    res = np.piecewise(T, [T < Th, (T >= Th)&(T<=Tc), T>Tc], [lambda T: -kh*(T-Th) + C0, lambda T: C0, lambda T: kc*(T-Tc)+C0])
    return res


def identify_thermal_sensitivity(temperature, consumption,C0_init=200,k_init=1):
    temperature = np.asarray(temperature)
    consumption = np.asarray(consumption)

    # estimation initiale
    p0 = (10, 20, C0_init, k_init, k_init)
    
    # optimisation sur la fonction piecewise_linear
    popt , e = curve_fit(piecewise_linear, temperature, consumption, p0=p0)
    pw_linear_consumption = piecewise_linear(temperature, *popt)
    r2_value = r2_score(consumption,pw_linear_consumption)
    
    Th_opt, Tc_opt, C0_opt, kh_opt, kc_opt = popt
    Tc_opt = min(temperature.max(),Tc_opt)
    return Th_opt, Tc_opt, C0_opt, kh_opt, kc_opt, r2_value


def plot_thermal_sensitivity(temperature,consumption,figs_folder,reg_code,reg_name,year,
                             C0_init=200,k_init=1,ylabel=None,set_ylim=None):
    Th_opt, Tc_opt, C0_opt, kh_opt, kc_opt, r2_value = identify_thermal_sensitivity(temperature, consumption, C0_init, k_init)
    yd = piecewise_linear(temperature, *(Th_opt, Tc_opt, C0_opt, kh_opt, kc_opt))


    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
    # ax.plot(temperature,consumption,alpha=0.05, ls='',marker='.',label='Data',color='tab:blue')
    sns.scatterplot(x=temperature,y=consumption,marker='.',label='Data',color=plt.get_cmap('Blues')(0.66),ax=ax,alpha=0.5,linewidth=0.1)
    label_fit = 'Piecewise linear (R$^2$ = {:.2f})\n   $k_h$=-{:.1f} Wh/K\n   $k_c$={:.2f} Wh/K\n   $C_0$={:.2f} Wh'.format(r2_value,kh_opt,kc_opt,C0_opt)
    ax.plot(temperature,yd ,label=label_fit,color='k')
    
    ax.set_ylim(bottom=0.)
    ylim = ax.get_ylim()
    if set_ylim is not None:
        ylim = [0,set_ylim]
    
    ax.plot([Th_opt,Th_opt],ylim,color='k',alpha=0.4)
    ax.text(Th_opt,10,'{:.1f}°C '.format(Th_opt),horizontalalignment='right',verticalalignment='bottom')
    ax.plot([Tc_opt,Tc_opt],ylim,color='k',alpha=0.4)
    ax.text(Tc_opt,10,' {:.1f}°C'.format(Tc_opt),horizontalalignment='left',verticalalignment='bottom')
    
    ax.set_ylim(ylim)
    ax.set_xlabel('Outdoor temperature (°C)')
    if ylabel is None:
        ax.set_ylabel('Hourly electricity energy cons. (by PDL) (Wh)')
    else:
        ax.set_ylabel(ylabel)
    
    ax.set_title('{} ({})'.format(reg_name, year))
    ax.legend(loc='upper right')
    # plt.savefig(os.path.join(figs_folder,'{}.png'.format('thermosensibilite_reg{}_{}'.format(reg_code, year))),bbox_inches='tight')
    plt.show()
    return yd


def plot_daily_consumption(data,figs_folder,col_name=None,normalize=True):
    data['hour'] = data.index.hour
    hours = list(range(24))*7
    
    seasons = ['DJF','JJA']
    season_dict = {'JJA':[6,7,8],
                   'DJF':[12,1,2],
                   'MAM':[3,4,5],
                   'SON':[9,10,11]}
    colors = {'DJF':'tab:red','JJA':'tab:blue'}
    labels = {'DJF':'Heating','JJA':'Cooling'}
    
    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
    for season in seasons:
        if col_name is None:
            col = {'DJF':'heating_needs','JJA':'cooling_needs'}.get(season)
        else: 
            col = col_name
        weekly_cons = pd.DataFrame().from_dict({'hour':hours})
        
        mean_weekly = data[data.index.month.isin(season_dict.get(season))][['hour',col]].groupby(by=['hour']).mean()
        std_weekly = data[data.index.month.isin(season_dict.get(season))][['hour',col]].groupby(by=['hour']).std()
        
        mean_weekly = mean_weekly.rename(columns={col:'total_needs_mean'})
        std_weekly = std_weekly.rename(columns={col:'total_needs_std'})
        
        weekly_cons = weekly_cons.join(mean_weekly)
        weekly_cons = weekly_cons.join(std_weekly)
        
        
        ax.plot(weekly_cons.hour, weekly_cons['total_needs_mean'],
                label=labels.get(season),color=colors.get(season))
        ax.fill_between(weekly_cons.hour, 
                        weekly_cons['total_needs_mean']+weekly_cons['total_needs_std'],
                        weekly_cons['total_needs_mean']-weekly_cons['total_needs_std'],
                        alpha=0.2,color=colors.get(season))
    ylims = ax.get_ylim()
    ax.set_ylim(ylims)
    ax.set_xlim([0,24])
    ax.set_ylabel('Mean hourly consumption by connection point (Wh)')
    plt.legend()
    # plt.savefig(os.path.join(figs_folder,'{}.png'.format('hourly_consumption_over_week_regions_{}_season_{}'.format('-'.join(map(str, regions)),season))), bbox_inches='tight')
    plt.show()
    return 


def get_nationale_meteo(period=[2022,2024]):
    # meteo de la prefecture de chaque region, pondérée par la population régionale
    meteo_nationale = None
    for reg_code in dict_region_code_region_name.keys():
        city = dict_region_code_chef_lieu.get(reg_code)
        meteo_data = get_meteo_data(city,period,variables=['temperature_2m'])
        meteo_data = meteo_data.rename(columns={'temperature_2m':reg_code})
        
        if meteo_nationale is None:
            meteo_nationale = meteo_data
        else:
            meteo_nationale = meteo_nationale.join(meteo_data)
    
    pop_cols = []
    for reg_code in dict_region_code_region_name.keys():
        col = '{}_nb_log'.format(reg_code)
        pop_cols.append(col)
        meteo_nationale[col] =  meteo_nationale[reg_code]*dict_region_code_nb_log.get(reg_code)
            
    meteo_nationale['france'] = meteo_nationale[pop_cols].sum(axis=1)/sum(list(dict_region_code_nb_log.values()))
    meteo_nationale = meteo_nationale[['france']]
    meteo_nationale = meteo_nationale.rename(columns={'france':'temperature'})
        
    return meteo_nationale
    
#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_thermal_sensitivity'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    #--------------------------------------------------------------------------
        
    national_consumption_data = open_electricity_consumption('national')
    regional_consumption_data = open_electricity_consumption('regional')
    
    #%% Vérification de la somme des énergies consommées par région
    if False:
        sum_reg = regional_consumption_data.copy()
        sum_reg = sum_reg[[c for c in sum_reg.columns if c.startswith('total_energie_soutiree')]]
        sum_reg = pd.DataFrame(sum_reg.sum(axis=1)).rename(columns={0:'total_energie_soutiree_wh_sum_reg'})
        sum_reg['total_energie_soutiree_wh_sum_reg'] = sum_reg['total_energie_soutiree_wh_sum_reg'].replace({'0':np.nan, 0:np.nan})
        fig,ax = plot_timeserie(national_consumption_data[['total_energie_soutiree_wh']], figsize=(10,5),
                                figs_folder=figs_folder, save_fig='total_energie_soutiree_wh_national_enedis',
                                show=False, alpha=0.5)
        fig,ax = plot_timeserie(sum_reg, figax=(fig,ax),
                                figs_folder=figs_folder, save_fig='total_energie_soutiree_wh_national_enedis',
                                show=False, alpha=0.5)
        
        
    # plot_timeserie(national_consumption_data[['nb_points_soutirage']], figsize=(10,5),
    #                figs_folder=figs_folder, save_fig='nb_points_soutirage_national_enedis')
    # plot_timeserie(national_consumption_data[['total_energie_soutiree_wh']], figsize=(10,5),
    #                figs_folder=figs_folder, save_fig='total_energie_soutiree_wh_national_enedis')
    
    # reg = 93
    # plot_timeserie(regional_consumption_data[['total_energie_soutiree_wh_reg_{}'.format(reg)]], figsize=(10,5),
    #                figs_folder=figs_folder, save_fig='total_energie_soutiree_wh_reg{}_enedis'.format(reg))
    
    
    #%% Premiers tests de thermosensibilité
    if False:
        for reg_code in dict_region_code_region_name.keys():
            # la Corse n'est pas intégrée par Enedis
            if reg_code == 94:
                continue
            # reg_code = 76#11#93#76#93
            year = None
            city = dict_region_code_chef_lieu.get(reg_code)
            reg_name = dict_region_code_region_name.get(reg_code)
            
            if year is None:
                meteo_data = get_meteo_data(city,[2022,2024])
                year = '2022-2024'
            else:
                meteo_data = get_meteo_data(city,[year,year])
            data = meteo_data.join(regional_consumption_data,how='inner')
            
            # data['weekday'] = data.index.weekday
            # weekday_data = dict()
            # for i in range(0,7):
            #     weekday_data[i] = (data[data.weekday==i]).total_energie_soutiree_wh.values
                
            # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # ax.errorbar(list(weekday_data.keys()),[np.nanmean(weekday_data.get(i)) for i in range(0,7)], yerr = [np.nanstd(weekday_data.get(i)) for i in range(0,7)])
            
            
            data = data.dropna(axis=0)#[:20000]
            data_temperature_sorted = data.copy().sort_values(by='temperature_2m')
            
            x = data_temperature_sorted.temperature_2m.values
            y = data_temperature_sorted['total_energie_soutiree_wh_reg_{}'.format(reg_code)].values/data_temperature_sorted['nb_points_soutirage_reg_{}'.format(reg_code)].values
            
            plot_thermal_sensitivity(temperature=x,consumption=y,figs_folder=figs_folder,
                                     reg_code=reg_code,reg_name=reg_name,year=year,set_ylim=600)
            
            data['month'] = data.index.month
            data['total_energie_per_pdl_wh_reg{}'.format(reg_code)] = data['total_energie_soutiree_wh_reg_{}'.format(reg_code)].values/data['nb_points_soutirage_reg_{}'.format(reg_code)].values
            
            # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            # sns.scatterplot(data=data,x='temperature_2m',
            #                 y='total_energie_per_pdl_wh_reg{}'.format(reg_code),
            #                 hue='month',ax=ax,alpha=0.3)
            # ax.set_title('{} ({})'.format(reg_name, year))
            # ax.set_ylim(bottom=0.)
            # plt.show()        
    
    # thermosensibilité nationale 
    if True:
        meteo_nationale = get_nationale_meteo()
        data = meteo_nationale.join(national_consumption_data,how='inner')
        
        data = data.dropna(axis=0)#[:20000]
        data_temperature_sorted = data.copy().sort_values(by='temperature')
        
        x = data_temperature_sorted.temperature.values
        y = data_temperature_sorted['total_energie_soutiree_wh'].values/data_temperature_sorted['nb_points_soutirage'].values
        
        plot_thermal_sensitivity(temperature=x,consumption=y,figs_folder=figs_folder,
                                 reg_code='france',reg_name='France',year='2022-2024',set_ylim=600)
        
        
    #%% Vérification des courbes de charges hebdomadaires 
    if False:
        cons = open_electricity_consumption(scale='regional')
        list_cols = []
        for regcode in dict_region_code_chef_lieu.keys():
            if regcode == 94:
                continue
            new_col = 'total_energie_soutiree_wh_reg_{}_par_point_soutirage'.format(regcode)
            cons[new_col] = cons['total_energie_soutiree_wh_reg_{}'.format(regcode)] / cons['nb_points_soutirage_reg_{}'.format(regcode)]
            list_cols.append(new_col)
            
        cons['weekday'] = cons.index.dayofweek
        cons['hour'] = cons.index.hour
        list_cols += ['weekday','hour']
        
        weekdays = [x for xs in [[i]*24 for i in range(7)] for x in xs]
        hours = list(range(24))*7
        
        
        # seasons = ['DJF','MAM','JJA','SON']
        seasons = ['DJF','JJA']
        regions = [28,93]
        regions_color = {44:'tab:blue',
                         28:'tab:blue',
                         93:'tab:red'}
        
        season_dict = {'JJA':[6,7,8],
                       'DJF':[12,1,2],
                       'MAM':[3,4,5],
                       'SON':[9,10,11]}
        
        dayofweek_dict = {0:'Monday',
                          1:'Tuesday',
                          2:'Wednesday',
                          3:'Thursday',
                          4:'Friday',
                          5:'Saturday',
                          6:'Sunday'}
        
        for season in seasons:
            weekly_cons = pd.DataFrame().from_dict({'weekday':weekdays,'hour':hours}).set_index(['weekday','hour'])
            
            mean_col_dict = {'total_energie_soutiree_wh_reg_{}_par_point_soutirage'.format(regcode):'energie_soutiree_moyenne_wh_reg_{}'.format(regcode) for regcode in dict_region_code_chef_lieu.keys()}
            std_col_dict = {'total_energie_soutiree_wh_reg_{}_par_point_soutirage'.format(regcode):'energie_soutiree_std_wh_reg_{}'.format(regcode) for regcode in dict_region_code_chef_lieu.keys()}
            
            mean_weekly = cons[cons.index.month.isin(season_dict.get(season))][list_cols].groupby(by=['weekday','hour']).mean()
            std_weekly = cons[cons.index.month.isin(season_dict.get(season))][list_cols].groupby(by=['weekday','hour']).std()
            
            mean_weekly = mean_weekly.rename(columns=mean_col_dict)
            std_weekly = std_weekly.rename(columns=std_col_dict)
            
            weekly_cons = weekly_cons.join(mean_weekly)
            weekly_cons = weekly_cons.join(std_weekly)
            
            weekly_cons['weekday_hour'] = [hour + 24*dow for dow,hour in weekly_cons.index]
            
            fig,ax = plt.subplots(figsize=(15,5),dpi=300)
            
            for regcode in regions:
                if regcode == 94:
                    continue
                ax.plot(weekly_cons.weekday_hour, weekly_cons['energie_soutiree_moyenne_wh_reg_{}'.format(regcode)],
                        label=dict_region_code_region_name.get(regcode)+' ({})'.format(season),color=regions_color.get(regcode))
                ax.fill_between(weekly_cons.weekday_hour, 
                                weekly_cons['energie_soutiree_moyenne_wh_reg_{}'.format(regcode)]+weekly_cons['energie_soutiree_std_wh_reg_{}'.format(regcode)],
                                weekly_cons['energie_soutiree_moyenne_wh_reg_{}'.format(regcode)]-weekly_cons['energie_soutiree_std_wh_reg_{}'.format(regcode)],
                                alpha=0.2,color=regions_color.get(regcode))
            
            ylims = ax.get_ylim()
            ylims = (0.,cons[[e for e in cons.columns if 'par_point_soutirage' in e]].max().max())
            ylims = (0.,600)
            for e in range(1,7):
                ax.plot([e*24]*2,ylims,color='k',ls=':',zorder=-1)
            ax.set_ylim(ylims)
            
            xticks = list(range(0,weekly_cons['weekday_hour'].max()+6,6))
            ax.set_xlim([0,24*7])
            ax.set_xticks(xticks)
            ax.set_xticklabels([e%24 if e%24!=12 else '12\n{}'.format(dayofweek_dict.get(e//24)) for e in xticks])
            
            ax.set_ylabel('Mean hourly consumption by connection point (Wh)')
            plt.legend()
            # ax.set_ylim
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('hourly_consumption_over_week_regions_{}_season_{}'.format('-'.join(map(str, regions)),season))), bbox_inches='tight')
            plt.show()
    
    
    # profil des temperatures extérieures # TODO: à faire au niveau national
    if False:
        reg_code = 33 
        
        data = open_electricity_consumption(scale='regional')
        
        year = None
        city = dict_region_code_chef_lieu.get(reg_code)
        reg_name = dict_region_code_region_name.get(reg_code)
        
        if year is None:
            meteo_data = get_meteo_data(city,[2022,2024])
            year = '2022-2024'
        else:
            meteo_data = get_meteo_data(city,[year,year])
            
        data = meteo_data.join(data,how='inner')
        
        plot_daily_consumption(data, figs_folder=figs_folder, col_name='temperature_2m')
        
    # profil des consommations réelles
    if False:
        enedis = open_electricity_consumption(scale='national')
        enedis['total_energie_soutiree_wh_per_pdl'] = enedis.total_energie_soutiree_wh / enedis.nb_points_soutirage
        enedis_djf = enedis[enedis.index.month.isin([12,1,2])]
        enedis_jja = enedis[enedis.index.month.isin([6,7,8])]
        
        plot_daily_consumption(enedis, figs_folder=figs_folder, col_name='total_energie_soutiree_wh_per_pdl')
        
    # Profil de Valentin Moreau
    if False:
        ninja = pd.read_excel("data/Ninja/41560_2023_1341_MOESM9_ESM_doubled.xlsx",sheet_name='Figure ED3')
        moreau = pd.read_csv('data/Res-IRF/hourly_profile_moreau_doubled.csv')
        moreau['value'] = moreau['value']/moreau['value'].mean()
        
        
        
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(moreau.hour,moreau['value'],color='tab:red',ls=':')
        ax.plot(ninja.Hour,ninja['Heating (mean)'],color='tab:red')
        ax.fill_between(ninja.Hour,ninja['Heating (mean)']+ninja['Heating (stdev)'],ninja['Heating (mean)']-ninja['Heating (stdev)'],color='tab:red',alpha=0.2)
        ax.plot(ninja.Hour,ninja['Cooling (mean)'],color='tab:blue')
        ax.fill_between(ninja.Hour,ninja['Cooling (mean)']+ninja['Cooling (stdev)'],ninja['Cooling (mean)']-ninja['Cooling (stdev)'],color='tab:blue',alpha=0.2)
        ax.plot([-1],[0],color='k',ls=':',label='Moreau (2024)')
        ax.plot([-1],[0],color='k',ls='-',label='Staffell et al. (2023)')
        ax.fill_between([-1],[0],[0],color='tab:red',label='Heating')
        ax.fill_between([-1],[0],[0],color='tab:blue',label='Cooling')
        ax.set_xlim([0,24])
        ax.set_ylim(bottom=0.)
        ax.set_ylabel('Intensity of use (normalized)')
        ax.set_xlabel('Hour of the day')
        ax.legend()
        plt.show()
        
        
        
    
            
    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
if __name__ == '__main__':
    main()
