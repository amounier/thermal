#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:53:00 2025

@author: amounier
"""

import time
from datetime import date
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import pickle

from administrative import France, Climat, Departement, draw_departement_map, draw_climat_map
from climate_zone_characterisation import get_departement_temperature


def get_cee_statistics(fiches=['BAR-TH-104','BAR-TH-129','BAR-TH-150','BAR-TH-159'],force=False):
    """
    Récuparation des données de délivrance des CEE par semestre
    Données brutes disponibles ici : https://www.ecologie.gouv.fr/politiques-publiques/comites-pilotage-lettres-dinformation-statistiques-du-dispositif-certificats

    Parameters
    ----------
    force : boolean, optional
        Force la réouverture et le formatage du fichier. The default is False.

    Returns
    -------
    cee_data : pandas DataFrame
        données CEE.

    """
    save_name = 'statistiques_delivrance_cee.csv'
    
    if save_name not in os.listdir(os.path.join('data','CEE')) or force:
        
        france = France()
        departements = france.departements
        
        cee_folder = os.path.join('data','CEE',"SDB délivrances-date d'engagement-Standards-par semestre d'engagement")
        files = [e for e in sorted(os.listdir(cee_folder)) if not e.startswith('.')]
        
        files_date_dict = {f:pd.to_datetime(f.split('.')[0].split(' ')[-1]) for f in files}
        
        cee_data = {'date':[],'departement':[],'fiche':[],'volume_cee':[]}
        
        for file in tqdm.tqdm(files):
            cee_temp = pd.read_excel(os.path.join(cee_folder,file), sheet_name='Classique',usecols='A:DA')
            cee_temp = cee_temp.set_index('Numéro département')
            cee_temp = cee_temp.rename(columns={c:Departement(c).code for c in cee_temp.columns})
            
            for dep in departements:
                cee_dep_temp = cee_temp[[dep.code]]
                for f in fiches:
                    volume_cee = cee_dep_temp.loc[f].values[0]
            
                    cee_data['date'].append(files_date_dict.get(file))
                    cee_data['departement'].append(dep.code)
                    cee_data['fiche'].append(f)
                    cee_data['volume_cee'].append(volume_cee)
                
        cee_data = pd.DataFrame().from_dict(cee_data)
        cee_data['zcl'] = [Departement(d).climat for d in cee_data.departement]
        
        cee_data.to_csv(os.path.join('data','CEE',save_name),index=False)
        
    cee_data = pd.read_csv(os.path.join('data','CEE',save_name))
    cee_data['date'] = [pd.to_datetime(d) for d in cee_data.date]
    # cee_data = cee_data.set_index('date')
    
    return cee_data
    


#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_insulation_characterisation'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
    
    
    #%% Caractérisation des analyses géographiques présentées dans ONRE oct 2024
    # https://www.statistiques.developpement-durable.gouv.fr/les-renovations-energetiques-aidees-du-secteur-residentiel-entre-2016-et-2021?rubrique=&dossier=843982
    
    # moins de rénovations car climat doux dans le sud : vraiment ?
    if False:
        data = pd.read_excel(os.path.join('data','SDES','graphiques_onre_2016_2021.xlsx'), sheet_name='Carte1', skiprows=2)
        data['Département'] = [Departement(dep) for dep in data.Département]
        
        data['zcl'] = [dep.climat for dep in data.Département]
        data['zcl'] = pd.Categorical(data.zcl, France().climats)
        
        data['housing_surface'] = [np.nan]*len(data)
        data['households'] = [np.nan]*len(data)
        data['temperature_DJF'] = get_departement_temperature(period='DJF').temperature
        data['temperature_JJA'] = get_departement_temperature(period='JJA').temperature
        
        # carte des données étudiées
        if True:
            france = France()
            climats = [Climat(e) for e in france.climats]
            draw_climat_map({c:None for c in climats},zcl_label=True, 
                            figs_folder=figs_folder, save='zcl',
                            add_legend=False,lw=0.7)
            
            data_dict = {d:v for d,v in zip(data.Département,data['Intensité energétique'])}
            draw_departement_map(data_dict, figs_folder, automatic_cbar_values=True)
            
        # graphes des interactions avec le climat
        if False:
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.boxplot(data=data,x="zcl", y='Intensité energétique',ax=ax)
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.scatterplot(data=data,x="temperature_DJF", y='Intensité energétique',ax=ax,alpha=0.6)
            plt.show()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.scatterplot(data=data,x="temperature_JJA", y='Intensité energétique',ax=ax,alpha=0.6)
            plt.show()
            # TODO : ajouter d'autres variables tq : DPE, économies réelles
            
        
    #%% TREMI
    if True:
        tremi = pd.read_csv(os.path.join('data','TREMI','tremi_2020_metropole_opda.csv'), na_values=['_NC_', 'NC', '_NR_'], low_memory=False).dropna(axis=1, how='all')
        tremi = tremi[tremi.treg!=7]
        
        
        dict_tremi_vars_1 = pd.read_excel(os.path.join('data','TREMI','tremi_2020_dictionnaire_variables_opda.xlsx'), sheet_name='Indiv_Log',)
        dict_tremi_vars_2 = pd.read_excel(os.path.join('data','TREMI','tremi_2020_dictionnaire_variables_opda.xlsx'), sheet_name='Q1-Q4',)
        # dict_tremi_vars_3 = pd.read_excel(os.path.join('data','TREMI','tremi_2020_dictionnaire_variables_opda.xlsx'), sheet_name='Q10',)
        # dict_tremi_vars_4 = pd.read_excel(os.path.join('data','TREMI','tremi_2020_dictionnaire_variables_opda.xlsx'), sheet_name='Q20-Q30',)
        
        # dict_tremi_Q1 = dict_tremi_vars_2.iloc[0:24].set_index('Q1_1').to_dict().get('Unnamed: 1')
        dict_tremi_Q2 = dict_tremi_vars_2.iloc[26:39].set_index('Q1_1').to_dict().get('Unnamed: 1')
        dict_tremi_Q2 = {k:v for k,v in dict_tremi_Q2.items() if not np.isnan(k)}
        # dict_tremi_Q3 = dict_tremi_vars_2.iloc[41:50].set_index('Q1_1').to_dict().get('Unnamed: 1')
        # dict_tremi_Q4 = dict_tremi_vars_2.iloc[52:60].set_index('Q1_1').to_dict().get('Unnamed: 1')
        dict_tremi_treg = dict_tremi_vars_1.iloc[0:13].set_index('treg').to_dict().get('Unnamed: 1')
        dict_tremi_treg = {k:v.strip() for k,v in dict_tremi_treg.items() if k != 7}
        # dict_tremi_Q102 = dict_tremi_vars_1.iloc[68:76].set_index('treg').to_dict().get('Unnamed: 1')
        # dict_tremi_Q11 = dict_tremi_vars_3.iloc[6:9].set_index('Q10').to_dict().get('Unnamed: 1')
        # dict_tremi_Q14 = dict_tremi_vars_3.iloc[11:19].set_index('Q10').to_dict().get('Unnamed: 1')
        # dict_tremi_Q31 = dict_tremi_vars_4.iloc[39:49].set_index('Q20').to_dict().get('Unnamed: 1')
        
        # questions de l'enquete (en anglais)
        Q2_title_en = "Why didn't you carry out any renovation work?"
        Q3_title_en = "Which event triggered the work?"
        Q4_title_en = "What is the motivation?"
        
        # réponses de l'enquete en anglais
        dict_tremi_Q2_en = {1: "Not thought",
                            2: "Not concerned",
                            3: "No will",
                            4: "Financial situation",
                            5: 'Doubts about potential savings',
                            6: "No return on investment through savings",
                            7: 'Lack of knowledge',
                            8: 'Short-term occupancy',
                            9: 'Tenant',
                            10: "No need to",
                            11: 'Technical constraints',
                            12: 'Waiting for the right moment'}
        
        dict_tremi_Q3_en = {1: "Equipment breakdown or damage",
                            2: "Funding opportunity",
                            3: "Energy performance diagnosis",
                            4: "Other renovation work",
                            5: 'Successful example in my neighbourhood',
                            6: "Buying the home or moving in",
                            7: 'A good moment for',
                            8: 'No decision',
                            9: 'No particular event',}
        
        dict_tremi_Q4_en = {1: "Reducing energy bills",
                            2: "Enhancing asset value",
                            3: "Improving thermal comfort",
                            4: "Soundproofing",
                            5: 'Improving air quality',
                            6: "Environmental action",
                            7: 'Beautifying',
                            8: 'Redesigning',}
        
        dict_tremi_Q1_en = {11:'Roof insulation',
                            14:'Attic insulation',
                            15:'Flat roof insulation',
                            21:'Walls insulation (ITE)',
                            23:'Walls insulation (ITI)',
                            31:'Floor insulation',
                            41:'Windows replacement',
                            51:'Heater replacement',
                            53:'ECS replacement',
                            55:'Ventilation replacement',
                            56:'Cooler replacement',
                            57:'Chiller replacement'}
        
        dict_tremi_Q102_en = {1: '< 1948',
                              2: '1949-1974',
                              3: '1975-1981',
                              4: '1982-1989',
                              5: '1990-2000',
                              6: '2001-2011',
                              7: '> 2012',}
        
        dict_tremi_Q31_en = {1: 'Individual boiler',
                             2: 'Heat pump',
                             3: 'Electric system',
                             4: 'Wood system',
                             5: 'Solar thermal',
                             6: 'Hybrid heat pump',
                             7: 'District network',
                             97: 'None',
                             99: 'DNK'}
        
        dict_tremi_Q14_en = {1: 'Individual boiler',
                             2: 'Heat pump',
                             3: 'Electric system',
                             4: 'Wood system',
                             5: 'Solar thermal',
                             6: 'Hybrid heat pump',
                             7: 'District network',
                             98: 'Other'}
        
        # filtre des répondants TREMI pour ne garder que ceux qui effectuent des travux énergétiques
        # (définis ci-dessus)
        
        tremi_reno_path = os.path.join('data','TREMI')
        tremi_reno_file = 'tremi_isolation_filtered.pickle'
        force = False
        
        if tremi_reno_file not in os.listdir(tremi_reno_path) or force:
            filter_reno = [False]*len(tremi)
            number_actions_list = [0]*len(tremi)
            
            for i in tqdm.tqdm(range(len(tremi))):
                row = tremi.iloc[i]
                actions = [t in dict_tremi_Q1_en.keys() for t in row[[c for c in tremi.columns if c.startswith('Q1_')]]]
                f = any(actions)
                nb_actions = actions.count(True)
                
                filter_reno[i] = f
                number_actions_list[i] = nb_actions
            
            tremi['number_actions'] = number_actions_list
            filter_reno = np.asarray(filter_reno)
            tremi_insulation = tremi[filter_reno]
        
            pickle.dump(tremi_insulation, open(os.path.join(tremi_reno_path,tremi_reno_file), "wb"))
            
        else:
            tremi_insulation = pickle.load(open(os.path.join(tremi_reno_path,tremi_reno_file), 'rb'))
        
        tremi_reno = tremi[tremi.Q1_1!=97]
        tremi_noreno = tremi[tremi.Q1_1==97]
        
        ratio_reno = tremi_reno.wCal.sum() / tremi.wCal.sum()
        ratio_insulation = tremi_insulation.wCal.sum() / tremi.wCal.sum()
        
        print('Pourcentage de travaux : {:.1f}%'.format(ratio_reno*100))
        print('Pourcentage de rénovation énergétique : {:.1f}%'.format(ratio_insulation*100))
        
        
        # caractérisation de l'effet de la période de construction sur la réalisation de travaux 
        # (et en séparant par remplacement chauffage et d'isolation)
        if True:
            tremi['Q102'] = pd.Categorical([dict_tremi_Q102_en.get(e,np.nan) for e in tremi['Q102']], list(dict_tremi_Q102_en.values()))
            tremi_insulation['Q102'] = pd.Categorical([dict_tremi_Q102_en.get(e,np.nan) for e in tremi_insulation['Q102']], list(dict_tremi_Q102_en.values()))
            for nba in range(1,13):
                tremi_insulation['Q1_{}'.format(nba)] = pd.Categorical([dict_tremi_Q1_en.get(e,np.nan) for e in tremi_insulation['Q1_{}'.format(nba)]], list(dict_tremi_Q1_en.values()))
                
            list_ratio_reno = [0]*len(dict_tremi_Q102_en.keys())
            for idx,(pi,p) in enumerate(dict_tremi_Q102_en.items()):
                ratio_reno = tremi_insulation[tremi_insulation.Q102==p].wCal.sum() / tremi[tremi.Q102==p].wCal.sum()
                list_ratio_reno[idx] = ratio_reno*100
                
            # ajout :
                # dont travaux isolation
            
            filter_thermal_insulation = [False]*len(tremi_insulation)
            for i in tqdm.tqdm(range(len(tremi_insulation))):
                filter_thermal_insulation[i] = any(['insulation' in str(e) for e in tremi_insulation[['Q1_{}'.format(nba) for nba in range(1,13)]].iloc[i].to_list()])
            tremi_insulation_insulation = tremi_insulation[np.asarray(filter_thermal_insulation)]
            
            list_ratio_insulation = [0]*len(dict_tremi_Q102_en.keys())
            for idx,(pi,p) in enumerate(dict_tremi_Q102_en.items()):
                ratio_insu = tremi_insulation_insulation[tremi_insulation_insulation.Q102==p].wCal.sum() / tremi[tremi.Q102==p].wCal.sum()
                list_ratio_insulation[idx] = ratio_insu*100
                
            ratio_insulation_insulation = tremi_insulation_insulation.wCal.sum() / tremi.wCal.sum()
            
            fig,ax = plt.subplots(dpi=300,figsize=(10,5))
            ax.bar(list(dict_tremi_Q102_en.values()),list_ratio_reno,label='Energy efficiency renovation (total = {:.1f}%)'.format(ratio_insulation*100))
            ax.bar(list(dict_tremi_Q102_en.values()),list_ratio_insulation,label='including thermal insulation (total = {:.1f}%)'.format(ratio_insulation_insulation*100))
            ax.set_ylim(bottom=0.)
            ax.legend()
            ax.set_ylabel('Percentage of respondents (%)')
            ax.set_xlabel('Construction period')
            plt.savefig(os.path.join(figs_folder,'tremi_renovation_and_insulation_construction_period.png'), bbox_inches='tight')
            plt.show()
        
        # caractérisation des motivations de travaux ou non
        if False:
            pd.options.mode.chained_assignment = None  # default='warn'
            
            # pourcentage de non renovation (application des pondérations)
            if True:
                ratio_no_reno = tremi_noreno.wCal.sum() / tremi.wCal.sum()
                
                print("Pourcentage d'absence de travaux : {:.1f}%".format(ratio_no_reno*100))
                
                tremi_noreno['treg'] = pd.Categorical([dict_tremi_treg.get(e,np.nan) for e in tremi_noreno['treg']], list(dict_tremi_treg.values()))
                tremi_noreno['Q2_1'] = pd.Categorical([dict_tremi_Q2_en.get(e,np.nan) for e in tremi_noreno['Q2_1']], list(dict_tremi_Q2_en.values()))
                
                Q2_2_list = []
                for e in tremi_noreno['Q2_2'].values:
                    if isinstance(e, float) or isinstance(e, int):
                        Q2_2_list.append(e)
                    elif isinstance(e, str):
                        if '-' in e:
                            Q2_2_list.append(np.nan)
                        else:
                            Q2_2_list.append(int(e))
                tremi_noreno['Q2_2'] = pd.Categorical([dict_tremi_Q2_en.get(e,np.nan) for e in Q2_2_list], list(dict_tremi_Q2_en.values()))
                
                ratio_no_reno_1 = (tremi_noreno['Q2_1'].notna()*tremi_noreno.wCal).sum()/ tremi.wCal.sum()
                ratio_no_reno_2 = (tremi_noreno['Q2_2'].notna()*tremi_noreno.wCal).sum()/ tremi.wCal.sum()
                
                noreno_reason = {'reason':[],'Reason 1':[],'Reason 2':[]}
                weight_sum = tremi_noreno.wCal.sum()
                for reason in list(dict_tremi_Q2_en.values()):
                    noreno_1 = tremi_noreno[tremi_noreno['Q2_1']==reason]
                    noreno_2 = tremi_noreno[tremi_noreno['Q2_2']==reason]
                    noreno_reason['reason'].append(reason)
                    noreno_reason['Reason 1'].append(noreno_1.wCal.sum()/weight_sum*100)
                    noreno_reason['Reason 2'].append(noreno_2.wCal.sum()/weight_sum*100)
                    
                noreno_reason = pd.DataFrame().from_dict(noreno_reason)
                noreno_reason = noreno_reason.rename(columns={'Reason 1':"Reason 1 ({} - {:.0f}%)".format(sum(tremi_noreno['Q2_1'].notna()),ratio_no_reno_1*100),
                                                              'Reason 2':"Reason 2 ({} - {:.0f}%)".format(sum(tremi_noreno['Q2_2'].notna()),ratio_no_reno_2*100)})
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                noreno_reason.set_index('reason').plot.barh(stacked=True,ax=ax)
                ax.set_title(Q2_title_en)
                ax.set_ylabel('')
                ax.set_xlabel('Percentage of respondents (%)')
                plt.gca().invert_yaxis()
                ax.legend()
                plt.savefig(os.path.join(figs_folder,'tremi_no_renovation_reasons.png'),bbox_inches='tight')
                plt.show()
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                ax.set_title(Q2_title_en)
                
                sns.histplot(data=tremi_noreno[tremi_noreno.Q2_1.isin(['No need to'])], 
                             y='Q2_1', weights='wCal', hue='treg', multiple="dodge",
                             stat='count', ax=ax, palette='tab20',legend=False)
                # ax.set_xlabel('Percentage of respondents (%)')
                ax.set_ylabel('')
                plt.legend(title='Regions', loc='center left', bbox_to_anchor=(1, 0.5), labels=list(dict_tremi_treg.values()))
                # ax.legend(loc='center left', )
                ax.set_xlim(right=500000)
                plt.savefig(os.path.join(figs_folder,'tremi_no_renovation_no_need_region.png'),bbox_inches='tight')
                plt.show()
            
            
            # raisons de renovation (application des pondérations)
            if True:
                tremi_insulation = pickle.load(open(os.path.join(tremi_reno_path,tremi_reno_file), 'rb'))
                
                ratio_reno = tremi_insulation.wCal.sum() / tremi.wCal.sum()
                
                # print(ratio_reno)
                
                tremi_insulation['treg'] = pd.Categorical([dict_tremi_treg.get(e,np.nan) for e in tremi_insulation['treg']], list(dict_tremi_treg.values()))
                tremi_insulation['Q3_1'] = pd.Categorical([dict_tremi_Q3_en.get(e,np.nan) for e in tremi_insulation['Q3_1']], list(dict_tremi_Q3_en.values()))
                tremi_insulation['Q3_2'] = pd.Categorical([dict_tremi_Q3_en.get(e,np.nan) if not np.isnan(e) else e for e in tremi_insulation['Q3_2']], list(dict_tremi_Q3_en.values()))
                tremi_insulation['Q3_3'] = pd.Categorical([dict_tremi_Q3_en.get(e,np.nan) if not np.isnan(e) else e for e in tremi_insulation['Q3_3']], list(dict_tremi_Q3_en.values()))
                
                ratio_1 = (tremi_insulation['Q3_1'].notna()*tremi_insulation.wCal).sum()/ tremi.wCal.sum()
                ratio_2 = (tremi_insulation['Q3_2'].notna()*tremi_insulation.wCal).sum()/ tremi.wCal.sum()
                ratio_3 = (tremi_insulation['Q3_3'].notna()*tremi_insulation.wCal).sum()/ tremi.wCal.sum()
                
                reno = {'reason':[],'Event 1':[],'Event 2':[],'Event 3':[]}
                weight_sum = tremi_insulation.wCal.sum()
                for reason in list(dict_tremi_Q3_en.values()):
                    r1 = tremi_insulation[tremi_insulation['Q3_1']==reason]
                    r2 = tremi_insulation[tremi_insulation['Q3_2']==reason]
                    r3 = tremi_insulation[tremi_insulation['Q3_3']==reason]
                    reno['reason'].append(reason)
                    reno['Event 1'].append(r1.wCal.sum()/weight_sum*100)
                    reno['Event 2'].append(r2.wCal.sum()/weight_sum*100)
                    reno['Event 3'].append(r3.wCal.sum()/weight_sum*100)
                    
                reno = pd.DataFrame().from_dict(reno)
                reno = reno.rename(columns={'Event 1':"Event 1 ({} - {:.0f}%)".format(sum(tremi_insulation['Q3_1'].notna()),ratio_1*100),
                                            'Event 2':"Event 2 ({} - {:.0f}%)".format(sum(tremi_insulation['Q3_2'].notna()),ratio_2*100),
                                            'Event 3':"Event 3 ({} - {:.0f}%)".format(sum(tremi_insulation['Q3_3'].notna()),ratio_3*100)})
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                reno.set_index('reason').plot.barh(stacked=True,ax=ax)
                ax.set_title(Q3_title_en)
                ax.set_ylabel('')
                ax.set_xlabel('Percentage of respondents (%)')
                plt.gca().invert_yaxis()
                ax.legend()
                plt.savefig(os.path.join(figs_folder,'tremi_renovation_trigger_event.png'),bbox_inches='tight')
                plt.show()
                
                
                tremi_insulation['Q4_1'] = pd.Categorical([dict_tremi_Q4_en.get(e,np.nan) for e in tremi_insulation['Q4_1']], list(dict_tremi_Q4_en.values()))
                tremi_insulation['Q4_2'] = pd.Categorical([dict_tremi_Q4_en.get(e,np.nan) if not np.isnan(e) else e for e in tremi_insulation['Q4_2']], list(dict_tremi_Q4_en.values()))
                
                Q_list = []
                for e in tremi_insulation['Q4_3'].values:
                    if isinstance(e, float) or isinstance(e, int):
                        Q_list.append(e)
                    elif isinstance(e, str):
                        if '-' in e or '/' in e:
                            Q_list.append(np.nan)
                        else:
                            Q_list.append(int(e))
                tremi_insulation['Q4_3'] = pd.Categorical([dict_tremi_Q4_en.get(e,np.nan) for e in Q_list], list(dict_tremi_Q4_en.values()))
                
                
                ratio_1 = (tremi_insulation['Q4_1'].notna()*tremi_insulation.wCal).sum()/ tremi.wCal.sum()
                ratio_2 = (tremi_insulation['Q4_2'].notna()*tremi_insulation.wCal).sum()/ tremi.wCal.sum()
                ratio_3 = (tremi_insulation['Q4_3'].notna()*tremi_insulation.wCal).sum()/ tremi.wCal.sum()
                
                reno = {'reason':[],'Reason 1':[],'Reason 2':[],'Reason 3':[]}
                weight_sum = tremi_insulation.wCal.sum()
                for reason in list(dict_tremi_Q4_en.values()):
                    r1 = tremi_insulation[tremi_insulation['Q4_1']==reason]
                    r2 = tremi_insulation[tremi_insulation['Q4_2']==reason]
                    r3 = tremi_insulation[tremi_insulation['Q4_3']==reason]
                    reno['reason'].append(reason)
                    reno['Reason 1'].append(r1.wCal.sum()/weight_sum*100)
                    reno['Reason 2'].append(r2.wCal.sum()/weight_sum*100)
                    reno['Reason 3'].append(r3.wCal.sum()/weight_sum*100)
                    
                reno = pd.DataFrame().from_dict(reno)
                reno = reno.rename(columns={'Reason 1':"Reason 1 ({} - {:.0f}%)".format(sum(tremi_insulation['Q4_1'].notna()),ratio_1*100),
                                            'Reason 2':"Reason 2 ({} - {:.0f}%)".format(sum(tremi_insulation['Q4_2'].notna()),ratio_2*100),
                                            'Reason 3':"Reason 3 ({} - {:.0f}%)".format(sum(tremi_insulation['Q4_3'].notna()),ratio_3*100)})
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                reno.set_index('reason').plot.barh(stacked=True,ax=ax)
                ax.set_title(Q4_title_en)
                ax.set_ylabel('')
                ax.set_xlabel('Percentage of respondents (%)')
                plt.gca().invert_yaxis()
                ax.legend()
                plt.savefig(os.path.join(figs_folder,'tremi_renovation_motivation.png'),bbox_inches='tight')
                plt.show()
                
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                ax.set_title(Q4_title_en)
                
                sns.histplot(data=tremi_insulation[tremi_insulation.Q4_1.isin(['Improving thermal comfort'])], 
                             y='Q4_1', hue='treg', multiple="dodge", weights='wCal',
                             stat='count', ax=ax, palette='tab20',legend=False)
                # ax.set_xlabel('Percentage of respondents (%)')
                ax.set_ylabel('')
                plt.legend(title='Regions', loc='center left', bbox_to_anchor=(1, 0.5), labels=list(dict_tremi_treg.values()))
                # ax.legend(loc='center left', )
                # ax.set_xlim(right=500000)
                plt.savefig(os.path.join(figs_folder,'tremi_renovation_thermal_comfort_region.png'),bbox_inches='tight')
                plt.show()
            
        
        # Caractérisation des gestes de rénovations
        if False:
            tremi_insulation = pickle.load(open(os.path.join(tremi_reno_path,tremi_reno_file), 'rb'))
            pd.options.mode.chained_assignment = None  # default='warn'
            
            tremi_insulation['treg'] = pd.Categorical([dict_tremi_treg.get(e,np.nan) for e in tremi_insulation['treg']], list(dict_tremi_treg.values()))
            tremi_insulation['Q102'] = pd.Categorical([dict_tremi_Q102_en.get(e,np.nan) for e in tremi_insulation['Q102']], list(dict_tremi_Q102_en.values()))
            tremi_insulation['number_actions_cat'] = pd.Categorical(tremi_insulation['number_actions'], sorted(list(set(tremi_insulation['number_actions'].values))))
            tremi_insulation['Q31'] = pd.Categorical([dict_tremi_Q31_en.get(e,np.nan) for e in tremi_insulation['Q31']], list(dict_tremi_Q31_en.values()))
            tremi_insulation['Q14'] = pd.Categorical([dict_tremi_Q14_en.get(e,np.nan) for e in tremi_insulation['Q14']], list(dict_tremi_Q14_en.values()))
            
            # print(tremi_insulation.number_actions.mean())
            # histogramme du nombre d'actions
            if True:
                fig,ax= plt.subplots(figsize=(5,5),dpi=300)
                sns.histplot(tremi_insulation, x='number_actions_cat',ax=ax, 
                             stat='percent',multiple='dodge',weights='wCal',)
                ax.set_xlim(right=8)
                ax.set_xlabel('Number of renovation actions (#)')
                ax.set_ylabel('Percentage of respondents (%)')
                plt.savefig(os.path.join(figs_folder,'tremi_renovation_number_actions.png'),bbox_inches='tight')
                plt.show()
                
                # def weighted_boxplot(df, weight_col):
                
                # def reindex_df(df, weight_col):
                #     """expand the dataframe to prepare for resampling
                #     result is 1 row per count per sample"""
                #     df = df.reindex(df.index.repeat(df[weight_col]))
                #     df.reset_index(drop=True, inplace=True)
                #     return(df)
                
                nb_actions_dict = {}
                for reg in list(dict_tremi_treg.values()):
                    temp_reg = tremi_insulation[tremi_insulation.treg==reg]
                    nb_actions_dict['{} ({})'.format(reg,len(temp_reg))] = temp_reg.number_actions.values
                    
                fig,ax= plt.subplots(figsize=(5,5),dpi=300)
                ax.plot([l.mean() for l in nb_actions_dict.values()], nb_actions_dict.keys(),marker='o',ls='',color='tab:blue')
                for k,v in nb_actions_dict.items():
                    ax.plot([v.min(),v.max()], [k,k], marker='|',ls=':',color='tab:blue')
                
                
                # # sns.boxplot(x='treg',y='number_actions',data=reindex_df(tremi_insulation, weight_col='wCal'))
                # sns.histplot(tremi_insulation, hue='treg',ax=ax,# weights='wCal',
                #              x='number_actions_cat',multiple='dodge', stat='percent')
                ax.set_xlim([0.95,2])
                ax.set_xlabel('Mean number of renovation actions (#)')
                # # ax.set_ylabel('Percentage of respondents (%)')
                plt.savefig(os.path.join(figs_folder,'tremi_renovation_number_actions_region.png'),bbox_inches='tight')
                plt.show()
                
                actions = {'number_actions_cat':[]}
                for period in list(dict_tremi_Q102_en.values()):
                    actions[period] = []
                    
                weight_sum = tremi_insulation.wCal.sum()
                for nba in range(1,9):
                    
                    temp = tremi_insulation[tremi_insulation.number_actions==nba]
                    weight_sum_temp = temp.wCal.sum()
                    
                    actions['number_actions_cat'].append('{} ({})'.format(nba,len(temp)))
                    for period in list(dict_tremi_Q102_en.values()):
                        temp_period = temp[temp.Q102==period]
                        actions[period].append(temp_period.wCal.sum()/weight_sum_temp*100)
                    
                actions = pd.DataFrame().from_dict(actions)
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                actions.set_index('number_actions_cat').plot.barh(stacked=True,ax=ax)
                # ax.set_title(Q2_title_en)
                ax.set_xlim(right=100)
                ax.set_ylabel('Number of actions (#)')
                ax.set_xlabel('Distribution by construction period (%)')
                plt.gca().invert_yaxis()
                ax.legend()
                plt.savefig(os.path.join(figs_folder,'tremi_renovation_number_actions_by_period.png'),bbox_inches='tight')
                plt.show()
                
            
            # caractérisation des monogestes 
            if True:
                
                tremi_monogeste = tremi_insulation[tremi_insulation.number_actions==1].copy()
                for nba in range(1,13):
                    tremi_monogeste['Q1_{}'.format(nba)] = pd.Categorical([dict_tremi_Q1_en.get(e,np.nan) for e in tremi_monogeste['Q1_{}'.format(nba)]], list(dict_tremi_Q1_en.values()))
                
                # le premier geste n'est pas forcément un de rénovation énergétique
                monogeste_list = []
                for i in range(len(tremi_monogeste)):
                    idx = 1
                    while not isinstance(tremi_monogeste.iloc[i]['Q1_{}'.format(idx)],str):
                        idx +=1
                    monogeste_list.append(tremi_monogeste.iloc[i]['Q1_{}'.format(idx)])
                
                tremi_monogeste['monogeste'] = monogeste_list
                tremi_monogeste['monogeste'] = pd.Categorical(tremi_monogeste['monogeste'], list(dict_tremi_Q1_en.values()))
                
                fig,ax= plt.subplots(figsize=(5,5),dpi=300)
                sns.histplot(tremi_monogeste, y='monogeste',ax=ax, 
                             stat='percent',weights='wCal',)
                # ax.set_xlim(right=8)
                ax.set_xlabel('Percentage of respondents (among single actions) (%)')
                ax.set_ylabel('')
                plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions.png'),bbox_inches='tight')
                plt.show()
                
                
                # étudier les gains énergétiques
                #  par région (pas assez de points ? trop de bruit ?)
                #  par période de construction : ok
                
                actions = list(dict_tremi_Q1_en.values())
                reverse_actions_dict = {v:k for k,v in dict_tremi_Q1_en.items()}
                
                # graphes pour les actions de rénovations
                if True:
                    actions_insulation = [l for l in actions if 'insulation' in l]
                    for action in actions_insulation:
                        width_var = 'Q12_{}'.format(reverse_actions_dict.get(action))
                
                    
                        tremi_action = tremi_monogeste[tremi_monogeste.monogeste==action]
                        tremi_action['energy_gains_EP'] = (1-(tremi_action['CF_EP']/tremi_action['CI_EP']))*100
                        
                        # if action == 'Flat roof insulation':
                        #     print(tremi_action)
                        nb_res = len(tremi_action[width_var].dropna())
                            
                        # print(len(tremi_action))
                        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                        sns.histplot(data=tremi_action,x=width_var,stat='percent')
                        ax.set_title(action)
                        ax.set_xlabel('Insulation thickness (cm)')
                        ax.set_ylabel('Percentage of respondents ({}) (%)'.format(nb_res))
                        plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions_{}_hist.png'.format(action)),bbox_inches='tight')
                        plt.show()
                        
                        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                        sns.scatterplot(data=tremi_action,x=width_var,y='energy_gains_EP',hue='Q102',alpha=0.2,legend=False)
                        for p in list(dict_tremi_Q102_en.values()):
                            tremi_action_period = tremi_action[tremi_action.Q102==p]
                            # ax.plot([tremi_action_period[width_var].mean()],
                            #         [tremi_action_period['energy_gains_EP'].mean()],
                            #         marker='o',label=p,ls='',mec='w')
                            
                            ax.errorbar([tremi_action_period[width_var].mean()],
                                        [tremi_action_period['energy_gains_EP'].mean()],
                                        xerr=[tremi_action_period[width_var].std()],
                                        yerr=[tremi_action_period['energy_gains_EP'].std()],
                                        marker='o',label=p,ls='',mec='w')
                            
                        xlim = ax.get_xlim()
                        ax.plot(xlim,[0,0],color='k',ls=':')
                        ax.set_xlim(xlim)
                        ax.legend(title='Mean values')
                        ax.set_title(action)
                        ax.set_xlabel('Insulation thickness (cm)')
                        ax.set_ylabel('Energy savings in primary energy (%)')
                        plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions_{}_energy_savings.png'.format(action)),bbox_inches='tight')
                        plt.show()
                    
                # statistiques pour les actiosn de remplacement d'équipements 
                if True:
                    # actions_replacement = [l for l in actions if 'replacement' in l]
                    
                    # remplacement de chauffage
                    tremi_action = tremi_monogeste[tremi_monogeste.monogeste=='Heater replacement']
                    tremi_action = tremi_action[tremi_action.CF_EP.notna()]
                    tremi_action = tremi_action[tremi_action.Q31.notna()]
                    tremi_action = tremi_action[tremi_action.Q14.notna()]
                    
                    tremi_action['energy_gains_EP'] = (1-(tremi_action['CF_EP']/tremi_action['CI_EP']))*100
                    
                    # suppresion de deux outlier : -60 et - 8000
                    tremi_action = tremi_action[tremi_action['energy_gains_EP']>-40]
                    
                    tremi_action['heater_switch'] = [' $\\rightarrow$ '.join([hi,hf]) for hi,hf in zip(tremi_action.Q31,tremi_action.Q14)]
                    
                    main_replacement = tremi_action['heater_switch'].value_counts()
                    main_replacement = main_replacement.iloc[:10].index.to_list()
                    # print(main_replacement)
                    
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    sns.histplot(data=tremi_action[tremi_action.heater_switch.isin(main_replacement)],y='heater_switch',stat='count',ax=ax)
                    ax.set_ylabel('')
                    ax.set_xlabel('Number of respondents (out of {}) (#)'.format(len(tremi_action)))
                    plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions_heater_replacement_hist.png'),bbox_inches='tight')
                    plt.show()
                    
                    # gains énergétiques par type de rénovation
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    sns.boxplot(data=tremi_action[tremi_action.heater_switch.isin(main_replacement)],
                                    y='heater_switch',x='energy_gains_EP')
                    ylim = ax.get_ylim()
                    ax.plot([0]*(len(main_replacement)+2), list(range(-1,len(main_replacement)+1)),color='k',ls=':')
                    ax.set_ylim(ylim)
                    ax.set_ylabel('')
                    ax.set_xlabel('Energy savings in primary energy (%)')
                    plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions_heater_replacement_energy_savings.png'),bbox_inches='tight')
                    plt.show()
                    
                    
                    # remplacement de fenetres 
                    tremi_action = tremi_monogeste[tremi_monogeste.monogeste=='Windows replacement']
                    tremi_action = tremi_action[tremi_action.Q13_1.notna()]
                    tremi_action = tremi_action[tremi_action.Q13bis_1_1.notna()]
                    tremi_action = tremi_action[tremi_action.CF_EP.notna()]
                    
                    # pondération de 1 pour les porte-fenêtres ou baies vitrées
                    pond = 1
                    tremi_action['percentage_single_glazed_before'] = ((tremi_action.Q13_1 + pond*tremi_action.Q13_4) / ((tremi_action.Q13_1+tremi_action.Q13_2+tremi_action.Q13_3) + pond*(tremi_action.Q13_4+tremi_action.Q13_5+tremi_action.Q13_6)))*100
                    tremi_action['percentage_single_glazed_after'] = ((tremi_action.Q13_1 - tremi_action.Q13bis_1_2 - tremi_action.Q13bis_1_3 + pond*tremi_action.Q13_4 - pond*(tremi_action.Q13bis_2_2 + tremi_action.Q13bis_2_3)) / ((tremi_action.Q13_1+tremi_action.Q13_2+tremi_action.Q13_3) + 2.5*(tremi_action.Q13_4+tremi_action.Q13_5+tremi_action.Q13_6)))*100
                    tremi_action['percentage_single_glazed_after'] = tremi_action['percentage_single_glazed_after'].clip(lower=0)
                    tremi_action['energy_gains_EP'] = (1-(tremi_action['CF_EP']/tremi_action['CI_EP']))*100
                    
                    tremi_action['nb_windows_replaced'] = tremi_action.Q13bis_1_2 + tremi_action.Q13bis_1_3 + tremi_action.Q13bis_2_2 + tremi_action.Q13bis_2_3
                    tremi_action['percentage_windows_replaced'] = (tremi_action['nb_windows_replaced'] / ((tremi_action.Q13_1+tremi_action.Q13_2+tremi_action.Q13_3) + pond*(tremi_action.Q13_4+tremi_action.Q13_5+tremi_action.Q13_6)))*100
                    tremi_action['percentage_windows_replaced'] = tremi_action['percentage_windows_replaced'].clip(upper=100)
                    
                    # print(tremi_action['percentage_single_glazed_before'].mean())
                    # print(tremi_action['percentage_single_glazed_after'].mean())
                    
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    sns.histplot(data=tremi_action,x='percentage_single_glazed_before',
                                 binwidth=5,
                                 binrange=[0,100],
                                 stat='percent',ax=ax, weights='wCal',hue='Q102',
                                 multiple='stack',legend=False)
                    ax.set_ylabel('Percentage of respondents ({}) (%)'.format(len(tremi_action)))
                    ax.set_xlabel('Percentage of single-gazed windows (%)')
                    ax.legend(title='',labels=list(dict_tremi_Q102_en.values())[::-1],ncols=2)
                    plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions_single_glazed_hist.png'),bbox_inches='tight')
                    plt.show()
                    
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    sns.histplot(data=tremi_action[tremi_action.percentage_single_glazed_before>0.],x='percentage_single_glazed_before',
                                 binwidth=5,
                                 binrange=[0,100],
                                 stat='percent',ax=ax, weights='wCal',hue='Q102',
                                 multiple='stack',
                                 legend=False)
                    ax.set_ylabel('Percentage of respondents, among > 0% ({}) (%)'.format(len(tremi_action[tremi_action.percentage_single_glazed_before>0.])))
                    ax.set_xlabel('Percentage of single-gazed windows (%)')
                    ax.legend(title='',labels=list(dict_tremi_Q102_en.values())[::-1],ncols=2)
                    plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions_single_glazed_hist_zoom.png'),bbox_inches='tight')
                    plt.show()
                    
                    
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    sns.histplot(data=tremi_action,x='percentage_single_glazed_after',
                                 binwidth=5,
                                 binrange=[0,100],
                                 stat='percent',ax=ax, weights='wCal',hue='Q102',
                                 multiple='stack',legend=False)
                    ax.set_ylabel('Percentage of respondents ({}) (%)'.format(len(tremi_action)))
                    ax.set_xlabel('Percentage of single-gazed windows (%)')
                    ax.legend(title='',labels=list(dict_tremi_Q102_en.values())[::-1],ncols=2)
                    plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions_single_glazed_after_hist.png'),bbox_inches='tight')
                    plt.show()
                    
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    sns.histplot(data=tremi_action[tremi_action.percentage_single_glazed_after>0.],x='percentage_single_glazed_after',
                                 binwidth=5,
                                 binrange=[0,100],
                                 stat='percent',ax=ax, weights='wCal',hue='Q102',
                                 multiple='stack',
                                 legend=False)
                    ax.set_ylabel('Percentage of respondents, among > 0% ({}) (%)'.format(len(tremi_action[tremi_action.percentage_single_glazed_after>0.])))
                    ax.set_xlabel('Percentage of single-gazed windows (%)')
                    ax.legend(title='',labels=list(dict_tremi_Q102_en.values())[::-1],ncols=2)
                    plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions_single_glazed_after_hist_zoom.png'),bbox_inches='tight')
                    plt.show()
                    
                    
                    # gains énergétiques par période
                    # tremi_action = tremi_action[tremi_action.difference_percentage_single_glazed>0.]
                    
                    fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                    sns.scatterplot(data=tremi_action,x='percentage_windows_replaced',y='energy_gains_EP',hue='Q102',alpha=0.2,legend=False)
                    
                    for p in list(dict_tremi_Q102_en.values()):
                        tremi_action_period = tremi_action[tremi_action.Q102==p]
                        
                        ax.errorbar([tremi_action_period['percentage_windows_replaced'].mean()],
                                    [tremi_action_period['energy_gains_EP'].mean()],
                                    xerr=[tremi_action_period['percentage_windows_replaced'].std()],
                                    yerr=[tremi_action_period['energy_gains_EP'].std()],
                                    marker='o',label=p,ls='',mec='w')
                        
                    xlim = ax.get_xlim()
                    ax.plot(xlim,[0,0],color='k',ls=':')
                    ax.set_xlim(xlim)
                    # ax.set_xlim([0,15])
                    ax.set_ylim(bottom=-5)
                    ax.legend(title='Mean values',ncols=2)
                    ax.set_xlabel('Percentage of replaced windows (%)')
                    ax.set_ylabel('Energy savings in primary energy (%)')
                    plt.savefig(os.path.join(figs_folder,'tremi_renovation_mono_actions_windows_replacement_energy_savings.png'),bbox_inches='tight')
                    plt.show()
                
            
            # caractérisation des multigestes
            if True:
                tremi_insulation = pickle.load(open(os.path.join(tremi_reno_path,tremi_reno_file), 'rb'))
                
                for nb_actions in range(1,4):
                    # nb_actions = 1
                    tremi_multigestes = tremi_insulation[tremi_insulation.number_actions==nb_actions].copy()
                    
                    for nba in range(1,13):
                        tremi_multigestes['Q1_{}'.format(nba)] = pd.Categorical([dict_tremi_Q1_en.get(e,np.nan) for e in tremi_multigestes['Q1_{}'.format(nba)]], list(dict_tremi_Q1_en.values()))
                
                    multigestes_list = []
                    
                    for i in range(len(tremi_multigestes)):
                        actions_list = []
                        idx = 1
                        for act in range(1,nb_actions+1): 
                            action = tremi_multigestes.iloc[i]['Q1_{}'.format(idx)]
                            while not isinstance(action, str):
                                idx += 1
                                action = tremi_multigestes.iloc[i]['Q1_{}'.format(idx)]
                            actions_list.append(action)
                            idx += 1
                            
                        # action_2 = tremi_multigestes.iloc[i]['Q1_{}'.format(idx)]
                        # while not isinstance(action_2, str):
                        #     idx += 1
                        #     action_2 = tremi_multigestes.iloc[i]['Q1_{}'.format(idx)]
                        multigestes_list.append(', '.join(sorted(list(set(actions_list)))))
                    
                    tremi_multigestes['Q1'] = multigestes_list
                    
                    multi_actions = tremi_multigestes.groupby(by='Q1').wCal.sum().sort_values(ascending=False)
                    
                    multi_actions = multi_actions/multi_actions.sum()*100
                    multi_actions_cumsum = multi_actions.cumsum().reset_index()
                    
                    idx_th = 0
                    while multi_actions_cumsum.iloc[idx_th].wCal < 75:
                        idx_th += 1
                    
                    fig,ax= plt.subplots(figsize=(5,5),dpi=300)
                    multi_actions.iloc[:idx_th+1].plot.barh(ax=ax)
                    ax.set_ylabel('')
                    ax.set_xlabel('Percentage of respondents ({}) (%)'.format(len(tremi_multigestes)))
                    plt.gca().invert_yaxis()
                    plt.savefig(os.path.join(figs_folder,'tremi_renovation_multi_actions_{}.png'.format(nb_actions)),bbox_inches='tight')
                    plt.show()
                
        
        # Caractéeisation des aides obtenues
        if False:
            # TODO
            pass
            
    #%% CEE
    if False:
        # Pour les PAC air-air : BAR-TH-129 ( = clim fixe)
        # Pour les PAC air-eau ou eau-eau : BAR-TH-104 abrogée au 1er janvier 2024
        # PAC hybride individuelle (air-eau + gaz) : BAR-TH-159
        # PAC collectives à absorption air/eau : BAR-TH-150
        
        fiches = ['BAR-TH-104','BAR-TH-129','BAR-TH-150','BAR-TH-159']
        cee_data = get_cee_statistics(fiches=fiches, force=False)
        
        # données de 2023 par encore consolidées
        cee_data = cee_data[cee_data.date.dt.year<2023]
        
        print(cee_data)
        
        # affichage de l'évolution des 4 fiches à l'échelle nationale
        if False:
            cee_data_nat = cee_data.groupby(by=['date','fiche'])['volume_cee'].sum().reset_index()
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            sns.lineplot(data=cee_data_nat,x='date',y='volume_cee',hue='fiche',ax=ax)
            ax.set_ylim(bottom=0.)
            plt.show()
            
        if True:
            cee_data_nat = cee_data.groupby(by=['date','fiche','zcl'])['volume_cee'].sum().reset_index()
            
            for f in fiches:
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                sns.lineplot(data=cee_data_nat[cee_data_nat.fiche==f],x='date',y='volume_cee',hue='zcl',ax=ax)
                ax.set_ylim(bottom=0.)
                ax.set_title(f)
                plt.show()
                
        if True:
            year = 2021
            
            france = France()
            departements = france.departements
            for f in fiches[:2]:
                cee_dict = {d:cee_data[(cee_data.date.dt.year==year)&(cee_data.fiche==f)&(cee_data.departement==d.code)]['volume_cee'].values.sum() for d in departements}
                
                draw_departement_map(cee_dict, figs_folder, 
                                     automatic_cbar_values=True, map_title=f)
                plt.show()
                
        # TODO : passer des volumes_CEE en nombre de systèms (estimés)
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()