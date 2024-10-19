#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:25:15 2024

@author: amounier
"""

import os 
import time
from datetime import date, datetime
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pickle

rd.seed(0)

from bdnb_opener import get_bdnb
from administrative import France, Departement, draw_departement_map


def compute_dpe_representativity(dep_code,external_disk,output_folder):
    dep_stats_name = 'dep{}_representativity.csv'.format(dep_code)
    
    dpe, _, bgc = get_bdnb(dep=dep_code,external_disk=external_disk)
    dpe = dpe[dpe.type_dpe.isin(['dpe arrêté 2021 3cl logement','dpe arrêté 2021 re2020 logement'])]
    dpe = dpe[['type_batiment_dpe']].compute()
    bgc = bgc[['ffo_bat_nb_log']].compute()
    
    bgc_maison = bgc[bgc.ffo_bat_nb_log==1]
    bgc_appartement = bgc[bgc.ffo_bat_nb_log>1]
    bgc_appartement_counts = bgc_appartement.value_counts().reset_index().rename(columns={'count':'count_value'})
    nbl_maison_bdnb = len(bgc_maison.index)
    nbl_appartement_bdnb = (bgc_appartement_counts.ffo_bat_nb_log * bgc_appartement_counts.count_value).sum()
    
    dpe_maison = dpe[dpe.type_batiment_dpe=='maison']
    dpe_appartement = dpe[dpe.type_batiment_dpe=='appartement']
    nbl_maison_dpe = len(dpe_maison.index)
    nbl_appartement_dpe = len(dpe_appartement.index)
    
    dep_stats = pd.DataFrame().from_dict({'type_batiment_dpe':['maison','appartement'],
                                          'nb_logements_bdnb':[nbl_maison_bdnb, nbl_appartement_bdnb],
                                          'nb_logements_dpe':[nbl_maison_dpe, nbl_appartement_dpe],
                                          'departement':[dep_code, dep_code]})
    
    dep_stats.to_csv(os.path.join(output_folder, dep_stats_name),index=False)
    return


def compute_heater_statistics(dep_code,external_disk,output_folder):
    dep_stats_heater_all_file_name = 'dep{}_stats_heater_all.csv'.format(dep_code)
    dep_stats_heater_sec_file_name = 'dep{}_stats_heater_sec.csv'.format(dep_code)
    
    dpe, _, _ = get_bdnb(dep=dep_code,external_disk=external_disk)
    dpe = dpe[dpe.type_dpe.isin(['dpe arrêté 2021 3cl logement','dpe arrêté 2021 re2020 logement'])]
    
    variables_heater = ['type_batiment_dpe', 
                        'type_installation_chauffage', 
                        'type_energie_chauffage', 
                        'type_generateur_chauffage',]
    supp_variables_sec_heater = ['type_energie_chauffage_appoint',
                                 'type_generateur_chauffage_appoint']
    
    dpe = dpe[variables_heater+supp_variables_sec_heater].compute()
    
    dep_stats_heater_all = dpe.dropna(subset='type_batiment_dpe').fillna('aucun').value_counts().reset_index()
    dep_stats_heater_all['departement'] = [dep_code]*len(dep_stats_heater_all)
    
    dep_stats_heater_sec = dpe.dropna(subset='type_batiment_dpe').value_counts().reset_index()
    dep_stats_heater_sec['departement'] = [dep_code]*len(dep_stats_heater_sec)
    
    dep_stats_heater_all.to_csv(os.path.join(output_folder, dep_stats_heater_all_file_name),index=False)
    # dep_stats_heater_sec.to_csv(os.path.join(output_folder, dep_stats_heater_sec_file_name),index=False)
    
    return


def test_DPE_stats(dep_code,external_disk,output_folder,period='1948-1974'):
    #TODO à completer pour les MFH et AB
    dep_dpe_id_typologies_dict_name = 'dep{}_dpe_id_typologies_dict'.format(dep_code)
    dep_dpe_id_typologies_dict = {}
    
    dpe, rel, _ = get_bdnb(dep=dep_code,external_disk=external_disk)
    dpe = dpe[dpe.type_dpe.isin(['dpe arrêté 2021 3cl logement','dpe arrêté 2021 re2020 logement'])]
    
    variables = ['identifiant_dpe','type_batiment_dpe', 'surface_mur_deperditif', 'periode_construction_dpe']
    
    # supp_variables_sec_heater = ['type_energie_chauffage_appoint',
    #                              'type_generateur_chauffage_appoint']
    
    dpe = dpe[variables]
    dpe = dpe[(dpe.type_batiment_dpe=='appartement')&(dpe.periode_construction_dpe==period)]
    dpe = dpe.compute()
    
    rel = rel[['batiment_groupe_id', 'identifiant_dpe']].dropna().set_index('identifiant_dpe')['batiment_groupe_id'].compute().to_dict()
    
    for dpe_id, smd in zip(dpe.identifiant_dpe,dpe.surface_mur_deperditif):
        dep_dpe_id_typologies_dict[rel.get(dpe_id)] = smd
   
    # enregistrement comme pickle
    now_ts = datetime.now().strftime("%Y-%m-%dT%H")
    pickle_file_name = os.path.join(output_folder, dep_dpe_id_typologies_dict_name + period + ".pickle")
    pickle.dump((dep_dpe_id_typologies_dict), open(pickle_file_name, "wb"))
    
    # energy_needs_dict_1, energy_needs_dict_2 = pickle.load(open('energy_needs_2024-10-17T04.pickle', 'rb'))
    
    return


#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_statistics_building'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)

    external_disk_connection = 'MPBE' in os.listdir('/media/amounier/')
    
#%% Statistiques sur les systèmes de chauffage de la base DPE
    if False:
        
        # Génération des fichiers statistiques par départements
        if False and external_disk_connection:
            departements = France().departements
            
            for dep in tqdm.tqdm(departements):
                compute_heater_statistics(dep.code,
                                          external_disk=external_disk_connection, 
                                          output_folder=os.path.join(output,folder))
    
        # Représentativité de la base DPE 
        if False:
            # statistiques sur la BDNB 
            if False and external_disk_connection:
                departements = France().departements
                
                for dep in tqdm.tqdm(departements):
                    compute_dpe_representativity(dep.code,
                                                 external_disk=external_disk_connection,
                                                 output_folder=os.path.join(output,folder))
        
            # sorties en cartes
            if True:
                representativity_path = os.path.join('data','BDNB','representativity.csv')
                representativity = pd.read_csv(representativity_path)
                
                departements_dict_maison = {d:None for d in France().departements}
                departements_dict_appart = {d:None for d in France().departements}
                for dep in departements_dict_maison.keys():
                    representativity_dep = representativity[representativity.departement==dep.code]
                    
                    rep_maison = representativity_dep[representativity_dep.type_batiment_dpe=='maison']
                    rep_maison = (rep_maison.nb_logements_dpe.values[0]/rep_maison.nb_logements_bdnb.values[0])*100
                    rep_appart = representativity_dep[representativity_dep.type_batiment_dpe=='appartement']
                    rep_appart = (rep_appart.nb_logements_dpe.values[0]/rep_appart.nb_logements_bdnb.values[0])*100
                    
                    departements_dict_maison[dep] = rep_maison
                    departements_dict_appart[dep] = rep_appart
                    
                draw_departement_map(departements_dict_maison, figs_folder,
                                     map_title='Maisons',
                                     cbar_label='Représentativité de la base DPE (%)',
                                     automatic_cbar_values=True,
                                     # cbar_min=0,cbar_max=100.,
                                     save='carte_repr_maison_dpe-BDNB')
                plt.show()
                
                draw_departement_map(departements_dict_appart, figs_folder,
                                     map_title='Appartements',
                                     cbar_label='Représentativité de la base DPE (%)',
                                     automatic_cbar_values=True,
                                     # cbar_min=0,cbar_max=100.,
                                     save='carte_repr_appartement_dpe-BDNB')
                plt.show()
        
        # statistiques nationale (fusion des fichiers par csvstack sur terminal)
        if True:
            stats_heater_all_path = os.path.join('data','BDNB','stats_heater_all.csv')
            stats_heater_sec_path = os.path.join('data','BDNB','stats_heater_sec.csv')
            
            stats_heater_all = pd.read_csv(stats_heater_all_path)
            stats_heater_all = stats_heater_all.rename(columns={'count':'count_value'})
            stats_heater_all = stats_heater_all[stats_heater_all.count_value>10]
            
            stats_heater_sec = pd.read_csv(stats_heater_sec_path)
            stats_heater_sec = stats_heater_sec.rename(columns={'count':'count_value'})
            stats_heater_sec = stats_heater_sec[stats_heater_sec.count_value>10]
            
            test_all = os.path.join('data','BDNB','stats_heater.csv')
            test_all = pd.read_csv(test_all)
            test_all = test_all.rename(columns={'count':'count_value'})
            test_all = test_all[test_all.count_value>10]
            
            
            # carte des pourcentages de logements par dep chauffés au gaz dans la base BDNB-DPE
            if False:
                departements_dict_nblog = {d:None for d in France().departements}
                departements_dict_elec = {d:None for d in France().departements}
                departements_dict_gaz = {d:None for d in France().departements}
                departements_dict_bois = {d:None for d in France().departements}
                
                for dep in departements_dict_nblog.keys():
                    stats_heater_all_dep = stats_heater_all[stats_heater_all.departement==dep.code]
                    
                    nb_gaz_principal = (stats_heater_all_dep[stats_heater_all_dep.type_energie_chauffage=='gaz']).count_value.sum()
                    nb_elec_principal = (stats_heater_all_dep[stats_heater_all_dep.type_energie_chauffage=='electricite']).count_value.sum()
                    nb_bois_principal = (stats_heater_all_dep[stats_heater_all_dep.type_energie_chauffage=='bois']).count_value.sum()
                    nb_logement = stats_heater_all_dep.count_value.sum()
                    
                    departements_dict_nblog[dep] = nb_logement
                    departements_dict_elec[dep] = nb_elec_principal/nb_logement*100
                    departements_dict_gaz[dep] = nb_gaz_principal/nb_logement*100
                    departements_dict_bois[dep] = nb_bois_principal/nb_logement*100
                    
                draw_departement_map(departements_dict_nblog, figs_folder,
                                     map_title='Nombre de logement',
                                     cbar_label='Nombre de logement dans la base BDNB-DPE (%)',
                                     automatic_cbar_values=True,
                                     save='carte_nb_logement_dpe-BDNB')
                plt.show()
                
                draw_departement_map(departements_dict_gaz, figs_folder,
                                     map_title='Chauffage principal au gaz',
                                     cbar_label='Pourcentage de logement dans la base BDNB-DPE (%)',
                                     # cbar_min=0,cbar_max=100,
                                     automatic_cbar_values=True,
                                     save='carte_chauffage_principal_gaz')
                plt.show()
                
                draw_departement_map(departements_dict_elec, figs_folder,
                                     map_title='Chauffage principal électrique',
                                     cbar_label='Pourcentage de logement dans la base BDNB-DPE (%)',
                                     # cbar_min=0,cbar_max=100,
                                     automatic_cbar_values=True,
                                     save='carte_chauffage_principal_elec')
                plt.show()
                
                draw_departement_map(departements_dict_bois, figs_folder,
                                     map_title='Chauffage principal au bois',
                                     cbar_label='Pourcentage de logement dans la base BDNB-DPE (%)',
                                     # cbar_min=0,cbar_max=100,
                                     automatic_cbar_values=True,
                                     save='carte_chauffage_principal_bois')
                plt.show()  
                
            # Pie chart des installation de chauffage (à finir)
            if False:
                # vecteurs considérés, solaire et réseau de chaleur exclus
                type_chauffage_dict = {t:None for t in ['bois',
                                                        'charbon',
                                                        'electricite',
                                                        'fioul',
                                                        'gaz',
                                                        'gpl/butane/propane',
                                                        'reseau de chaleur',
                                                        'solaire']}
                
                for tc in type_chauffage_dict.keys():
                    stats_heater_all_tc = stats_heater_all[stats_heater_all.type_energie_chauffage==tc]
                    type_chauffage_dict[tc] = {}
                    
                    tgc_list = sorted(list(set(stats_heater_all_tc.type_generateur_chauffage.values)))
                    for tgc in tgc_list:
                        type_chauffage_dict[tc][tgc] = stats_heater_all_tc[stats_heater_all_tc.type_generateur_chauffage==tgc].count_value.sum()
                
                # print(type_chauffage_dict)
                
                type_chauffage_dict['reseau de chaleur'] = {'reseau de chaleur':sum(type_chauffage_dict['reseau de chaleur'].values())}
                
                del type_chauffage_dict['solaire']
                del type_chauffage_dict['gpl/butane/propane']
                del type_chauffage_dict['charbon']
                
                rd.seed(2)
                
                fig, ax = plt.subplots(figsize=(5,5),dpi=300)

                size = 0.4
                start_angle = 30
                
                vals = [list(v.values()) for v in type_chauffage_dict.values()]
                labels_cat = list(type_chauffage_dict.keys())
                vals_flat = [x for xs in vals for x in xs]
                percentage_vals_flat = [x/sum(xs)*100 for xs in vals for x in xs]
                unique_value = [float(len(xs)!=1) for xs in vals for x in xs]            
                labels_flat = [x for xs in [list(v.keys()) for v in type_chauffage_dict.values()] for x in xs]
                labels_flat = ['{} ({:.0f}%)'.format(l[0].upper() + l[1:],p) if bool(f) and p>4 else '' for l,f,p in zip(labels_flat, unique_value,percentage_vals_flat)]
                
                cmap = plt.colormaps["turbo"]
                cmap = plt.colormaps["viridis"]
                inner_colors = cmap(np.arange(len(vals)+1)/len(vals))
                outer_colors = [inner_colors[i] for i,xs in enumerate(vals) for x in xs]
                
                intensities = [rd.random()*2/3 + 0.33 for e in outer_colors]
                outer_colors = [(r,v,b,i*f) for (r,v,b,_),i,f in zip(outer_colors,intensities,unique_value)]
                
                ax.pie([sum(l) for l in vals], radius=1, colors=inner_colors,
                       labels=labels_cat,labeldistance=None,startangle=start_angle,
                       autopct='%1.0f%%',pctdistance=0.42,
                       wedgeprops=dict(width=size, edgecolor='w',))
                
                ax.pie(vals_flat, radius=1+size, colors=outer_colors,startangle=start_angle,
                       wedgeprops=dict(width=size, edgecolor='w'),labels=labels_flat)
                
                ax.set(aspect="equal")
                ax.legend(labels=labels_cat,)
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('statistiques_nationale_type_installation_par_type_energie')),bbox_inches='tight')
                plt.show()
        
            # statistiques sur les possesseurs de PAC en chauffage principal tous types de logements
            if False:
                pac_list = ['pac air/air','pac air/eau','pac eau/eau']
                # stats_heater_all_pac = stats_heater_all[stats_heater_all.type_generateur_chauffage.isin(pac_list)]
                # stats_heater_sec_pac = stats_heater_sec[stats_heater_sec.type_generateur_chauffage.isin(pac_list)]
                
                stats_heater_sec_pac = test_all[test_all.type_generateur_chauffage.isin(pac_list)]
                
                # nb_pac_chauffage_principal_France = stats_heater_all_pac.count_value.sum()
                nb_pac_chauffage_principal_sec_France = stats_heater_sec_pac.count_value.sum()
        
                nb_chauffage_appoint = stats_heater_sec_pac[['type_energie_chauffage_appoint','count_value']].groupby('type_energie_chauffage_appoint').sum()
                nb_chauffage_appoint = nb_chauffage_appoint.sort_values(by='count_value',ascending=False)
                
                nb_chauffage_appoint.loc['gaz'] = nb_chauffage_appoint.loc['gaz'] + nb_chauffage_appoint.loc['gpl/butane/propane']
                nb_chauffage_appoint = nb_chauffage_appoint.loc[['electricite', 'bois', 'gaz', 'fioul','aucun']]
                
                dict_format_type_energie = {'electricite':'Électricité', 
                                            'bois':'Bois', 
                                            'gaz':'Gaz', 
                                            'fioul':'Fioul', 
                                            'reseau de chaleur':'Réseau de chaleur',
                                            'gpl/butane/propane':'GPL, butane, propane'}
                
                fig, ax = plt.subplots(figsize=(5,5),dpi=300)

                size = 0.4
                start_angle = 0
                vals = [list(nb_chauffage_appoint.loc[['aucun']].values.T[0]), 
                        list(nb_chauffage_appoint.loc[['electricite', 'bois', 'gaz', 'fioul']].values.T[0])]
                percentage_vals_flat = [x/sum(xs)*100 for xs in vals for x in xs]
                unique_value = [float(len(xs)!=1) for xs in vals for x in xs]
                
                labels_cat = ["Pas de chauffage secondaire ({} logements)".format(sum(list(nb_chauffage_appoint.loc[['aucun']].values.T[0]))),
                              'Présence de chauffage secondaire ({} logements)'.format(sum(list(nb_chauffage_appoint.loc[['electricite', 'bois', 'gaz', 'fioul']].values.T[0])))]
                vals_flat = [x for xs in vals for x in xs]
                labels_flat = [''] + list(nb_chauffage_appoint.index.values)
                labels_flat = ['{} ({:.0f}%)'.format(dict_format_type_energie.get(l),p) if bool(f) else '' for l,f,p in zip(labels_flat, unique_value, percentage_vals_flat)]
                
                cmap = plt.colormaps["tab20c"]
                inner_colors = cmap(np.arange(len(vals))*4)
                outer_colors = [x for xs in [[cmap(i*4)]*len(l) for i,l in enumerate(vals)] for x in xs]
                
                intensities = [rd.random()*2/3 + 0.33 for e in outer_colors]
                outer_colors = [(r,v,b,i*f) for (r,v,b,_),i,f in zip(outer_colors,intensities,unique_value)]
                
                ax.pie([sum(l) for l in vals], radius=1, colors=inner_colors,
                       labels=labels_cat,labeldistance=None,startangle=start_angle,
                       autopct='%1.0f%%',pctdistance=0.42,
                       wedgeprops=dict(width=size, edgecolor='w',))
                
                ax.pie(vals_flat, radius=1+size, colors=outer_colors,startangle=start_angle,
                       wedgeprops=dict(width=size, edgecolor='w'),labels=labels_flat)
                
                ax.set(aspect="equal")
                ax.legend(labels=labels_cat, loc='upper center')
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('statistiques_nationale_type_energie_secondaire_pac_principal')),bbox_inches='tight')
                plt.show()
            
            # idem, pour les maisons seulement
            if False:
                pac_list = ['pac air/air','pac air/eau','pac eau/eau']
                # stats_heater_all_pac = stats_heater_all[(stats_heater_all.type_generateur_chauffage.isin(pac_list))&(stats_heater_all.type_batiment_dpe=='maison')]
                # stats_heater_sec_pac = stats_heater_sec[(stats_heater_sec.type_generateur_chauffage.isin(pac_list))&(stats_heater_sec.type_batiment_dpe=='maison')]
                
                stats_heater_sec_pac = test_all[(test_all.type_generateur_chauffage.isin(pac_list))&(test_all.type_batiment_dpe=='maison')]
                # nb_pac_chauffage_principal_France = stats_heater_all_pac.count_value.sum()
                nb_pac_chauffage_principal_sec_France = stats_heater_sec_pac.count_value.sum()
        
                nb_chauffage_appoint = stats_heater_sec_pac[['type_energie_chauffage_appoint','count_value']].groupby('type_energie_chauffage_appoint').sum()
                nb_chauffage_appoint = nb_chauffage_appoint.sort_values(by='count_value',ascending=False)
                
                nb_chauffage_appoint.loc['gaz'] = nb_chauffage_appoint.loc['gaz'] + nb_chauffage_appoint.loc['gpl/butane/propane']
                nb_chauffage_appoint = nb_chauffage_appoint.loc[['electricite', 'bois', 'gaz', 'fioul','aucun']]
                
                dict_format_type_energie = {'electricite':'Électricité', 
                                            'bois':'Bois', 
                                            'gaz':'Gaz', 
                                            'fioul':'Fioul', 
                                            'reseau de chaleur':'Réseau de chaleur',
                                            'gpl/butane/propane':'GPL, butane, propane'}
                
                fig, ax = plt.subplots(figsize=(5,5),dpi=300)

                size = 0.4
                start_angle = 0
                vals = [list(nb_chauffage_appoint.loc[['aucun']].values.T[0]), 
                        list(nb_chauffage_appoint.loc[['electricite', 'bois', 'gaz', 'fioul']].values.T[0])]
                percentage_vals_flat = [x/sum(xs)*100 for xs in vals for x in xs]
                unique_value = [float(len(xs)!=1) for xs in vals for x in xs]
                
                labels_cat = ["Pas de chauffage secondaire ({} logements)".format(sum(list(nb_chauffage_appoint.loc[['aucun']].values.T[0]))),
                              'Présence de chauffage secondaire ({} logements)'.format(sum(list(nb_chauffage_appoint.loc[['electricite', 'bois', 'gaz', 'fioul']].values.T[0])))]
                vals_flat = [x for xs in vals for x in xs]
                labels_flat = [''] + list(nb_chauffage_appoint.index.values)
                labels_flat = ['{} ({:.0f}%)'.format(dict_format_type_energie.get(l),p) if bool(f) else '' for l,f,p in zip(labels_flat, unique_value, percentage_vals_flat)]
                
                cmap = plt.colormaps["tab20c"]
                inner_colors = cmap(np.arange(len(vals))*4)
                outer_colors = [x for xs in [[cmap(i*4)]*len(l) for i,l in enumerate(vals)] for x in xs]
                
                intensities = [rd.random()*2/3 + 0.33 for e in outer_colors]
                outer_colors = [(r,v,b,i*f) for (r,v,b,_),i,f in zip(outer_colors,intensities,unique_value)]
                
                ax.pie([sum(l) for l in vals], radius=1, colors=inner_colors,
                       labels=labels_cat,labeldistance=None,startangle=start_angle,
                       autopct='%1.0f%%',pctdistance=0.42,
                       wedgeprops=dict(width=size, edgecolor='w',))
                
                ax.pie(vals_flat, radius=1+size, colors=outer_colors,startangle=start_angle,
                       wedgeprops=dict(width=size, edgecolor='w'),labels=labels_flat)
                
                ax.set(aspect="equal")
                ax.legend(labels=labels_cat, loc='upper center')
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('statistiques_nationale_maisons_type_energie_secondaire_pac_principal')),bbox_inches='tight')
                plt.show()
                
            # idem, pour les appartements seulement
            if False:
                pac_list = ['pac air/air','pac air/eau','pac eau/eau']
                # stats_heater_all_pac = stats_heater_all[(stats_heater_all.type_generateur_chauffage.isin(pac_list))&(stats_heater_all.type_batiment_dpe=='appartement')]
                # stats_heater_sec_pac = stats_heater_sec[(stats_heater_sec.type_generateur_chauffage.isin(pac_list))&(stats_heater_sec.type_batiment_dpe=='appartement')]
                
                stats_heater_sec_pac = test_all[(test_all.type_generateur_chauffage.isin(pac_list))&(test_all.type_batiment_dpe=='appartement')]
                
                # nb_pac_chauffage_principal_France = stats_heater_all_pac.count_value.sum()
                nb_pac_chauffage_principal_sec_France = stats_heater_sec_pac.count_value.sum()
        
                nb_chauffage_appoint = stats_heater_sec_pac[['type_energie_chauffage_appoint','count_value']].groupby('type_energie_chauffage_appoint').sum()
                nb_chauffage_appoint = nb_chauffage_appoint.sort_values(by='count_value',ascending=False)
                
                nb_chauffage_appoint.loc['gaz'] = nb_chauffage_appoint.loc['gaz'] + nb_chauffage_appoint.loc['gpl/butane/propane']
                nb_chauffage_appoint = nb_chauffage_appoint.loc[['electricite', 'bois', 'gaz', 'fioul','aucun']]
                
                dict_format_type_energie = {'electricite':'Électricité', 
                                            'bois':'Bois', 
                                            'gaz':'Gaz', 
                                            'fioul':'Fioul', 
                                            'reseau de chaleur':'Réseau de chaleur',
                                            'gpl/butane/propane':'GPL, butane, propane'}
                
                fig, ax = plt.subplots(figsize=(5,5),dpi=300)

                size = 0.4
                start_angle = 0
                vals = [list(nb_chauffage_appoint.loc[['aucun']].values.T[0]), 
                        list(nb_chauffage_appoint.loc[['electricite', 'bois', 'gaz', 'fioul']].values.T[0])]
                percentage_vals_flat = [x/sum(xs)*100 for xs in vals for x in xs]
                unique_value = [float(len(xs)!=1) for xs in vals for x in xs]
                
                labels_cat = ["Pas de chauffage secondaire ({} logements)".format(sum(list(nb_chauffage_appoint.loc[['aucun']].values.T[0]))),
                              'Présence de chauffage secondaire ({} logements)'.format(sum(list(nb_chauffage_appoint.loc[['electricite', 'bois', 'gaz', 'fioul']].values.T[0])))]
                vals_flat = [x for xs in vals for x in xs]
                labels_flat = [''] + list(nb_chauffage_appoint.index.values)
                labels_flat = ['{} ({:.0f}%)'.format(dict_format_type_energie.get(l),p) if bool(f) else '' for l,f,p in zip(labels_flat, unique_value, percentage_vals_flat)]
                
                cmap = plt.colormaps["tab20c"]
                inner_colors = cmap(np.arange(len(vals))*4)
                outer_colors = [x for xs in [[cmap(i*4)]*len(l) for i,l in enumerate(vals)] for x in xs]
                
                intensities = [rd.random()*2/3 + 0.33 for e in outer_colors]
                outer_colors = [(r,v,b,i*f) for (r,v,b,_),i,f in zip(outer_colors,intensities,unique_value)]
                
                ax.pie([sum(l) for l in vals], radius=1, colors=inner_colors,
                       labels=labels_cat,labeldistance=None,startangle=start_angle,
                       autopct='%1.0f%%',pctdistance=0.42,
                       wedgeprops=dict(width=size, edgecolor='w',))
                
                ax.pie(vals_flat, radius=1+size, colors=outer_colors,startangle=start_angle,
                       wedgeprops=dict(width=size, edgecolor='w'),labels=labels_flat)
                
                ax.set(aspect="equal")
                ax.legend(labels=labels_cat, loc='upper center')
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('statistiques_nationale_appartements_type_energie_secondaire_pac_principal')),bbox_inches='tight')
                plt.show()
                
            # statistiques pour d'autres chauffages principaux, tous types de logements
            if True:
                rd.seed(0)
                
                dict_type_generateur = {'PAC':['pac air/air','pac air/eau','pac eau/eau'],
                                        'Générateurs à effet joule':['generateurs a effet joule'],
                                        'Chaudière fioul':['chaudiere fioul standard','chaudiere fioul condensation','chaudiere fioul basse temperature'],
                                        'Chaudière gaz':['chaudiere gaz standard','chaudiere gaz condensation','chaudiere gaz basse temperature']}
                
                for tg, pac_list in dict_type_generateur.items():

                    # stats_heater_all_pac = stats_heater_all[stats_heater_all.type_generateur_chauffage.isin(pac_list)]
                    # stats_heater_sec_pac = stats_heater_sec[stats_heater_sec.type_generateur_chauffage.isin(pac_list)]
                    
                    stats_heater_sec_pac = test_all[test_all.type_generateur_chauffage.isin(pac_list)]
                    
                    # nb_pac_chauffage_principal_France = stats_heater_all_pac.count_value.sum()
                    nb_pac_chauffage_principal_sec_France = stats_heater_sec_pac.count_value.sum()
            
                    nb_chauffage_appoint = stats_heater_sec_pac[['type_energie_chauffage_appoint','count_value']].groupby('type_energie_chauffage_appoint').sum()
                    nb_chauffage_appoint = nb_chauffage_appoint.sort_values(by='count_value',ascending=False)
                    
                    # print(nb_chauffage_appoint)
                    
                    if 'gpl/butane/propane' in nb_chauffage_appoint.index:
                        nb_chauffage_appoint.loc['gaz'] = nb_chauffage_appoint.loc['gaz'] + nb_chauffage_appoint.loc['gpl/butane/propane']
                    
                    cols = [c for c in ['electricite', 'bois', 'gaz', 'fioul','aucun'] if c in nb_chauffage_appoint.index]
                    cols_without_aucun = [c for c in ['electricite', 'bois', 'gaz', 'fioul'] if c in nb_chauffage_appoint.index]
                    nb_chauffage_appoint = nb_chauffage_appoint.loc[cols]
                    
                    dict_format_type_energie = {'electricite':'Électricité', 
                                                'bois':'Bois', 
                                                'gaz':'Gaz', 
                                                'fioul':'Fioul', 
                                                'reseau de chaleur':'Réseau de chaleur',
                                                'gpl/butane/propane':'GPL, butane, propane'}
                    
                    fig, ax = plt.subplots(figsize=(5,5),dpi=300)
    
                    size = 0.4
                    start_angle = 0
                    vals = [list(nb_chauffage_appoint.loc[['aucun']].values.T[0]), 
                            list(nb_chauffage_appoint.loc[cols_without_aucun].values.T[0])]
                    percentage_vals_flat = [x/sum(xs)*100 for xs in vals for x in xs]
                    unique_value = [float(len(xs)!=1) for xs in vals for x in xs]
                    
                    labels_cat = ["Pas de chauffage secondaire ({} logements)".format(sum(list(nb_chauffage_appoint.loc[['aucun']].values.T[0]))),
                                  'Présence de chauffage secondaire ({} logements)'.format(sum(list(nb_chauffage_appoint.loc[cols_without_aucun].values.T[0])))]
                    vals_flat = [x for xs in vals for x in xs]
                    labels_flat = [''] + list(nb_chauffage_appoint.index.values)
                    labels_flat = ['{} ({:.0f}%)'.format(dict_format_type_energie.get(l),p) if bool(f) and p>1 else '' for l,f,p in zip(labels_flat, unique_value, percentage_vals_flat)]
                    
                    cmap = plt.colormaps["tab20c"]
                    inner_colors = cmap(np.arange(len(vals))*4)
                    outer_colors = [x for xs in [[cmap(i*4)]*len(l) for i,l in enumerate(vals)] for x in xs]
                    
                    intensities = [rd.random()*2/3 + 0.33 for e in outer_colors]
                    outer_colors = [(r,v,b,i*f) for (r,v,b,_),i,f in zip(outer_colors,intensities,unique_value)]
                    
                    ax.pie([sum(l) for l in vals], radius=1, colors=inner_colors,
                           labels=labels_cat,labeldistance=None,startangle=start_angle,
                           autopct='%1.0f%%',pctdistance=0.42,
                           wedgeprops=dict(width=size, edgecolor='w',))
                    
                    ax.pie(vals_flat, radius=1+size, colors=outer_colors,startangle=start_angle,
                           wedgeprops=dict(width=size, edgecolor='w'),labels=labels_flat)
                    
                    ax.set(aspect="equal")
                    ax.legend(labels=labels_cat, loc='upper center')
                    plt.savefig(os.path.join(figs_folder,'{}.png'.format('statistiques_nationale_type_energie_secondaire_{}_principal'.format(tg.replace(' ','_')))),bbox_inches='tight')
                    plt.show()

    #%% Identification des typologies de bâtiments dans la base DPE
    if True:
        departement = Departement(75)
        period_list = ['1948-1974','1989-2000']
        # for p in period_list:
        #     test_DPE_stats(dep_code = departement.code,
        #                    external_disk=external_disk_connection,
        #                    output_folder=os.path.join(output, folder),
        #                    period=p)
        
        import seaborn as sns
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        
        for p in period_list[1:]:
            dep_stats_dict = pickle.load(open(os.path.join(output,folder,'dep75_dpe_id_typologies_dict{}.pickle'.format(p)), 'rb'))
            dep_stats = pd.DataFrame().from_dict({'bg_id':dep_stats_dict.keys(),'surface_mur_deperditif_{}'.format(p):dep_stats_dict.values()}).set_index('bg_id')
            dep_stats = dep_stats[dep_stats['surface_mur_deperditif_{}'.format(p)]<100]
        
        # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.histplot(dep_stats['surface_mur_deperditif_{}'.format(p)] ax=ax,stat='percent')
            ax.set_xlabel('Surface déperditive de murs (m$^2$)')
        plt.show()
        
        # from bdnb_opener import draw_city_map
        
        # draw_city_map(dep_stats, external_disk=external_disk_connection,
        #               cbar_label='Surface déperditive de murs (m$^2$)',
        #               figsize=10)
        # plt.show()
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__=='__main__':
    main()
