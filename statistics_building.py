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
import seaborn as sns
from scipy import stats
from scipy import signal
import matplotlib

rd.seed(0)

from bdnb_opener import get_bdnb, plot_dpe_distribution, neighbourhood_map
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



def concatenate_dpe_statistics(dep_code,external_disk):
    reformat_bdnb_dpe_file = 'dpe_statistics.parquet'
    data_path = os.path.join('data','BDNB')
    
    _, _, bgc = get_bdnb(dep=dep_code,external_disk=external_disk)
    # dpe = dpe[dpe.type_dpe.isin(['dpe arrêté 2021 3cl logement','dpe arrêté 2021 re2020 logement'])]
    
    
    variables = ['batiment_groupe_id',
                 'ffo_bat_nb_log',
                 
                 'dpe_mix_arrete_identifiant_dpe',
                 'dpe_mix_arrete_type_batiment_dpe',
                 'dpe_mix_arrete_periode_construction_dpe',
                 'dpe_mix_arrete_annee_construction_dpe',
                 'dpe_mix_arrete_nombre_niveau_logement',
                 'dpe_mix_arrete_nombre_niveau_immeuble',
                 'dpe_mix_arrete_surface_habitable_immeuble',
                 'dpe_mix_arrete_surface_habitable_logement',
                 'dpe_mix_arrete_classe_bilan_dpe',
                 'dpe_mix_arrete_classe_emission_ges',
                 'dpe_mix_arrete_conso_5_usages_ep_m2',
                 'dpe_mix_arrete_conso_5_usages_ef_m2',
                 
                 'dpe_mix_arrete_type_installation_chauffage',
                 'dpe_mix_arrete_type_energie_chauffage',
                 'dpe_mix_arrete_type_generateur_chauffage',
                 'dpe_mix_arrete_type_energie_chauffage_appoint',
                 'dpe_mix_arrete_type_generateur_chauffage_appoint',
                 'dpe_mix_arrete_type_energie_climatisation',
                 'dpe_mix_arrete_type_generateur_climatisation',
                 'dpe_mix_arrete_type_ventilation',
                 'dpe_mix_arrete_plusieurs_facade_exposee',
                 
                 'dpe_mix_arrete_surface_vitree_nord',
                 'dpe_mix_arrete_surface_vitree_sud',
                 'dpe_mix_arrete_surface_vitree_ouest',
                 'dpe_mix_arrete_surface_vitree_est',
                 'dpe_mix_arrete_surface_vitree_horizontal',
                 'dpe_mix_arrete_traversant',
                 'dpe_mix_arrete_uw',
                 'dpe_mix_arrete_facteur_solaire_baie_vitree',
                 
                 'dpe_mix_arrete_epaisseur_isolation_mur_exterieur_estim',
                 'dpe_mix_arrete_materiaux_structure_mur_exterieur',
                 'dpe_mix_arrete_epaisseur_structure_mur_exterieur',
                 'dpe_mix_arrete_type_isolation_mur_exterieur',
                 'dpe_mix_arrete_surface_mur_deperditif',
                 'dpe_mix_arrete_u_mur_exterieur',
                 'dpe_mix_arrete_l_orientation_mur_exterieur',
                 
                 'dpe_mix_arrete_type_isolation_plancher_bas',
                 'dpe_mix_arrete_type_plancher_bas_deperditif',
                 'dpe_mix_arrete_u_plancher_bas_final_deperditif',
                 'dpe_mix_arrete_surface_plancher_bas_totale',
                 'dpe_mix_arrete_surface_plancher_bas_deperditif',
                 
                 'dpe_mix_arrete_type_isolation_plancher_haut',
                 'dpe_mix_arrete_type_plancher_haut_deperditif',
                 'dpe_mix_arrete_surface_plancher_haut_totale',
                 'dpe_mix_arrete_surface_plancher_haut_deperditif',
                 'dpe_mix_arrete_u_plancher_haut_deperditif',
                 'dpe_mix_arrete_surface_porte',
                 'dpe_mix_arrete_u_porte',
                 
                 'dpe_mix_arrete_nb_classe_bilan_dpe_a',
                 'dpe_mix_arrete_nb_classe_bilan_dpe_b',
                 'dpe_mix_arrete_nb_classe_bilan_dpe_c',
                 'dpe_mix_arrete_nb_classe_bilan_dpe_d',
                 'dpe_mix_arrete_nb_classe_bilan_dpe_e',
                 'dpe_mix_arrete_nb_classe_bilan_dpe_f',
                 'dpe_mix_arrete_nb_classe_bilan_dpe_g',
                 ]
    
    bgc = bgc[variables]
    bgc = bgc[bgc.dpe_mix_arrete_identifiant_dpe.notnull()]
    bgc = bgc.compute()
    
    bgc['mitoyennete'] = [not str(e).startswith('(4') for e in bgc.dpe_mix_arrete_l_orientation_mur_exterieur]
    bgc['departement'] = [dep_code]*len(bgc)
    
    tabula_construction_period_dict = {'avant 1914':[-np.inf,1914],
                                       '1914-1948':[1914,1948],
                                       '1948-1967':[1948,1967],
                                       '1967-1974':[1967,1974],
                                       '1974-1981':[1974,1981],
                                       '1981-1989':[1981,1989],
                                       '1989-1999':[1989,1999],
                                       '1999-2005':[1999,2005],
                                       '2005-2012':[2005,2012],
                                       '2012-2021':[2012,2021],
                                       'après 2021':[2021,np.inf],}
    
    period_construction_list = [0]*len(bgc)
    for idx,cy in enumerate(bgc.dpe_mix_arrete_annee_construction_dpe.values):
        if np.isnan(cy):
            period_construction_list[idx] = np.nan
            continue
        
        for k,(y0,y1) in tabula_construction_period_dict.items():
            if cy >= y0 and cy < y1:
                period_construction_list[idx] = k
                
    bgc['periode_construction_tabula'] = period_construction_list
    # print(dpe.periode_construction_tabula.value_counts())
    
    bgc = bgc.reset_index().drop(columns=['index'])
    
    if reformat_bdnb_dpe_file in os.listdir(data_path):
        bgc.to_parquet(os.path.join(data_path,reformat_bdnb_dpe_file), engine='fastparquet', append=True)
    else:
        bgc.to_parquet(os.path.join(data_path,reformat_bdnb_dpe_file), engine='fastparquet')
                    
    
    return 
    
    
    

# def test_DPE_stats(dep_code,external_disk,output_folder,period='1948-1974'):
#     #TODO à completer pour les MFH et AB
#     dep_dpe_id_typologies_dict_name = 'dep{}_dpe_id_typologies_dict'.format(dep_code)
#     dep_dpe_id_typologies_dict = {}
    
#     dpe, rel, _ = get_bdnb(dep=dep_code,external_disk=external_disk)
#     dpe = dpe[dpe.type_dpe.isin(['dpe arrêté 2021 3cl logement','dpe arrêté 2021 re2020 logement'])]
    
#     variables = ['identifiant_dpe','type_batiment_dpe', 'surface_mur_deperditif', 'periode_construction_dpe']
    
#     # supp_variables_sec_heater = ['type_energie_chauffage_appoint',
#     #                              'type_generateur_chauffage_appoint']
    
#     dpe = dpe[variables]
#     dpe = dpe[(dpe.type_batiment_dpe=='appartement')&(dpe.periode_construction_dpe==period)]
#     dpe = dpe.compute()
    
#     rel = rel[['batiment_groupe_id', 'identifiant_dpe']].dropna().set_index('identifiant_dpe')['batiment_groupe_id'].compute().to_dict()
    
#     for dpe_id, smd in zip(dpe.identifiant_dpe,dpe.surface_mur_deperditif):
#         dep_dpe_id_typologies_dict[rel.get(dpe_id)] = smd
   
#     # enregistrement comme pickle
#     now_ts = datetime.now().strftime("%Y-%m-%dT%H")
#     pickle_file_name = os.path.join(output_folder, dep_dpe_id_typologies_dict_name + period + ".pickle")
#     pickle.dump((dep_dpe_id_typologies_dict), open(pickle_file_name, "wb"))
    
#     # energy_needs_dict_1, energy_needs_dict_2 = pickle.load(open('energy_needs_2024-10-17T04.pickle', 'rb'))
    
#     return


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
    if False:
        departement = Departement(75)
        period_list = ['1948-1974','1989-2000']
        # for p in period_list:
        #     test_DPE_stats(dep_code = departement.code,
        #                    external_disk=external_disk_connection,
        #                    output_folder=os.path.join(output, folder),
        #                    period=p)
        
        
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        
        for p in period_list[1:]:
            dep_stats_dict = pickle.load(open(os.path.join(output,folder,'dep75_dpe_id_typologies_dict{}.pickle'.format(p)), 'rb'))
            dep_stats = pd.DataFrame().from_dict({'bg_id':dep_stats_dict.keys(),'surface_mur_deperditif_{}'.format(p):dep_stats_dict.values()}).set_index('bg_id')
            dep_stats = dep_stats[dep_stats['surface_mur_deperditif_{}'.format(p)]<100]
        
        # fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.hist(dep_stats['surface_mur_deperditif_{}'.format(p)])
        ax.set_xlabel('Surface déperditive de murs (m$^2$)')
        plt.show()
        
        # from bdnb_opener import draw_city_map
        
        # draw_city_map(dep_stats, external_disk=external_disk_connection,
        #               cbar_label='Surface déperditive de murs (m$^2$)',
        #               figsize=10)
        # plt.show()
        
        
        
    #%% Statistiques DPE par typologies 
    if True:
        
        reformat_bdnb_dpe_file = 'dpe_statistics.parquet'
        
        if reformat_bdnb_dpe_file not in os.listdir(os.path.join('data','BDNB')) and external_disk_connection:
            france = France()
            for dep in tqdm.tqdm(france.departements):
                concatenate_dpe_statistics(dep.code, 
                                           external_disk=external_disk_connection)
                
        dpe = pd.read_parquet(os.path.join('data','BDNB',reformat_bdnb_dpe_file))
        
        mask = dpe["dpe_mix_arrete_surface_habitable_logement"] > 1e3
        dpe.loc[mask, 'dpe_mix_arrete_surface_habitable_logement'] = np.nan
        
        mask = dpe["dpe_mix_arrete_epaisseur_isolation_mur_exterieur_estim"] > 50
        dpe.loc[mask, 'dpe_mix_arrete_epaisseur_isolation_mur_exterieur_estim'] = np.nan
        
        dpe = dpe.fillna(value=np.nan)
        
        # représentativité et écarts par rapport à la distribution par étiquette
        if True:
            # cartes 
            if True:
                # TODO : representativity à recalculer
                representativity_path = os.path.join('data','BDNB','representativity.csv')
                representativity = pd.read_csv(representativity_path)
                representativity['dep_code'] = [Departement(d).code for d in representativity.departement]
                
                departements_dict_maison = {}
                departements_dict_appart = {}
                
                for dep in France().departements:
                    representativity_dep = representativity[representativity.dep_code==dep.code]
                    
                    rep_maison = representativity_dep[representativity_dep.type_batiment_dpe=='maison']
                    rep_maison = (rep_maison.nb_logements_dpe.values[0]/rep_maison.nb_logements_bdnb.values[0])*100
                    rep_appart = representativity_dep[representativity_dep.type_batiment_dpe=='appartement']
                    rep_appart = (rep_appart.nb_logements_dpe.values[0]/rep_appart.nb_logements_bdnb.values[0])*100
                    
                    departements_dict_maison[dep] = rep_maison
                    departements_dict_appart[dep] = rep_appart
                
                rep_maison_france = representativity[representativity.type_batiment_dpe=='maison'].nb_logements_dpe.sum()/representativity[representativity.type_batiment_dpe=='maison'].nb_logements_bdnb.sum()
                rep_appart_france = representativity[representativity.type_batiment_dpe=='appartement'].nb_logements_dpe.sum()/representativity[representativity.type_batiment_dpe=='appartement'].nb_logements_bdnb.sum()
                
                draw_departement_map(departements_dict_maison, figs_folder,
                                     map_title='Single-family (mean : {:.1f}%)'.format(rep_maison_france*100),
                                     cbar_label='Representativity (%)',
                                     automatic_cbar_values=True,
                                     # cbar_min=0,cbar_max=50.,
                                     save='carte_repr_maison_dpe-BDNB')
                plt.show()
                
                draw_departement_map(departements_dict_appart, figs_folder,
                                     map_title='Multi-family (mean : {:.1f}%)'.format(rep_appart_france*100),
                                     cbar_label='Representativity (%)',
                                     automatic_cbar_values=True,
                                     # cbar_min=0,cbar_max=50.,
                                     save='carte_repr_appartement_dpe-BDNB')
                plt.show()
            
            # histogramme etiquette
            if True:
                etiquette_colors_dict = {'A':(0, 156, 109),'B':(82, 177, 83),'C':(120, 189, 118),'D':(244, 231, 15),'E':(240, 181, 15),'F':(235, 130, 53),'G':(215, 34, 31)}
                etiquette_colors_dict = {k: tuple(map(lambda x: x/255, v)) for k,v in etiquette_colors_dict.items()}
                
                sdes_eti = pd.read_excel(os.path.join('data','SDES','parc_logements_dpe_2023.xlsx'),sheet_name='Graphique_1',skipfooter=8,skiprows=1)
                sdes_eti = sdes_eti.rename(columns={'N':'households','%':'percentage'})
                sdes_eti['Source'] = ['SDES 2023']*len(sdes_eti)
                
                # nb_logements = dpe.ffo_bat_nb_log.sum()
                
                bdnb_dpe = []
                for eti in sdes_eti['Étiquette DPE']:
                    number_eti = dpe['dpe_mix_arrete_nb_classe_bilan_dpe_{}'.format(eti.lower())].sum()
                    bdnb_dpe.append(number_eti)
                    
                bdnb_dpe = np.asarray(bdnb_dpe)
                sdes_eti['percentage_dpe'] = bdnb_dpe/sum(bdnb_dpe)*100
                sdes_eti['source_dpe'] = ['BDNB DPE']*len(sdes_eti)
                
                plotter = pd.concat([sdes_eti[['Étiquette DPE','percentage','Source']],
                                     sdes_eti[['Étiquette DPE','percentage_dpe','source_dpe']].rename(columns={'percentage_dpe':'percentage','source_dpe':'Source'})])
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                sns.barplot(plotter,x='Étiquette DPE',y='percentage',hue='Source',
                            ax=ax,palette='viridis')
                # sdes_eti[['Étiquette DPE','percentage','percentage_dpe']].plot.bar('Étiquette DPE',ax=ax)
                # plt.xticks(rotation=90)
                ax.set_xlabel('EPC label')
                ax.set_ylabel('Representation percentage (%)')
                plt.savefig(os.path.join(figs_folder,'DPE_distribution_dpe_france.png'),bbox_inches='tight')
                plt.show()
                
                
        # Statistiques des typologies en France
        # TODO à refaire avec les données complètes
        if True:
            
            dpe['periode_construction_tabula'] = dpe.periode_construction_tabula.replace({'avant 1914':'before 1914',
                                                                                          'après 2021':'after 2021'})
            
            dict_tabula_period = {'before 1914':1,
                                  '1914-1948':2,
                                  '1948-1967':3,
                                  '1967-1974':4,
                                  '1974-1981':5,
                                  '1981-1989':6,
                                  '1989-1999':7,
                                  '1999-2005':8,
                                  '2005-2012':9,
                                  '2012-2021':10,
                                  'after 2021':11,}
            
            typo_list = [np.nan]*len(dpe)
            group_typo_list = [np.nan]*len(dpe)
            
            for idx,(cpt, nbl, trav, mit) in tqdm.tqdm(enumerate(zip(dpe.periode_construction_tabula, dpe.ffo_bat_nb_log, dpe.dpe_mix_arrete_traversant, dpe.mitoyennete)),total=len(dpe)):
                if pd.isnull(cpt) or pd.isnull(trav):
                    continue
                if nbl == 1.:
                    if trav in ['tout venant','tout azimut','tout venant (faible)'] or not mit:
                        typo = 'FR.N.SFH.{:02d}.Gen'.format(dict_tabula_period.get(cpt))
                        gt = 'SFH'
                    else:
                        typo = 'FR.N.TH.{:02d}.Gen'.format(dict_tabula_period.get(cpt))
                        gt = 'TH'
                if nbl > 1. and nbl <= 10.:
                    typo = 'FR.N.MFH.{:02d}.Gen'.format(dict_tabula_period.get(cpt))
                    gt = 'MFH'
                if nbl > 10.:
                    typo = 'FR.N.AB.{:02d}.Gen'.format(dict_tabula_period.get(cpt))
                    gt = 'AB'
                    
                typo_list[idx] = typo
                group_typo_list[idx] = gt
                
            dpe['typology'] = typo_list
            dpe['group_typology'] = group_typo_list
            
            dpe = dpe[dpe.typology.notnull()]
            
            typology_category_list = ['SFH','TH','MFH','AB']
            tabula_period_list = list(dict_tabula_period.keys())
            
            if True:
                dpe['group_typology_cat'] = pd.Categorical(dpe['group_typology'], typology_category_list)
                # dpe['typology_cat'] = pd.Categorical(dpe['typology'], sorted(list(set(typo_list))))
                dpe['periode_construction_tabula_cat'] = pd.Categorical(dpe['periode_construction_tabula'], tabula_period_list)
                
                
                
                matrix_typo_repartition_logements = np.zeros((len(typology_category_list),len(tabula_period_list)))
                matrix_typo_repartition_batiments = np.zeros((len(typology_category_list),len(tabula_period_list)))
                
                for i,cpt in enumerate(tabula_period_list):
                    for j,typ in enumerate(typology_category_list):
                        typo_comp = 'FR.N.{}.{:02d}.Gen'.format(typ,dict_tabula_period.get(cpt))
                        count_logements = dpe[dpe['typology']==typo_comp].ffo_bat_nb_log.sum()
                        count_batiments = (dpe['typology']==typo_comp).sum()
                        matrix_typo_repartition_logements[j,i] = count_logements
                        matrix_typo_repartition_batiments[j,i] = count_batiments
                        
                matrix_typo_repartition_logements = matrix_typo_repartition_logements/matrix_typo_repartition_logements.sum().sum()*100
                matrix_typo_repartition_batiments = matrix_typo_repartition_batiments/matrix_typo_repartition_batiments.sum().sum()*100
                
                df_typo_repartition_logements = pd.DataFrame(data=matrix_typo_repartition_logements,
                                                             index=typology_category_list,
                                                             columns=tabula_period_list)
                
                df_typo_repartition_batiments = pd.DataFrame(data=matrix_typo_repartition_batiments,
                                                             index=typology_category_list,
                                                             columns=tabula_period_list)
                
                dict_sum_repartition_logements = df_typo_repartition_logements.sum(axis=1).to_dict()
                dict_sum_repartition_batiments = df_typo_repartition_batiments.sum(axis=1).to_dict()
                
                fig,ax = plt.subplots(figsize=(5*(len(dict_tabula_period.keys())/4),5), dpi=300)
                sns.heatmap(df_typo_repartition_logements, annot=True, fmt=".1f",cmap='viridis',cbar=False)
                for j,typ in enumerate(typology_category_list):
                    ax.text(len(tabula_period_list)+0.5,j+0.5,'{:.1f}%'.format(dict_sum_repartition_logements.get(typ)),
                            ha='right',va='center')
                ax.set_title('Distribution of households (%)')
                plt.savefig(os.path.join(figs_folder,'distribution_tabula_households_ponderated.png'), bbox_inches='tight')
                plt.show()
                
                fig,ax = plt.subplots(figsize=(5*(len(dict_tabula_period.keys())/4),5), dpi=300)
                sns.heatmap(df_typo_repartition_batiments, annot=True, fmt=".1f",cmap='viridis',cbar=False)
                for j,typ in enumerate(typology_category_list):
                    ax.text(len(tabula_period_list)+0.5,j+0.5,'{:.1f}%'.format(dict_sum_repartition_batiments.get(typ)),
                            ha='right',va='center')
                ax.set_title('Distribution of buildings (%)')
                plt.savefig(os.path.join(figs_folder,'distribution_tabula_buildings_ponderated.png'), bbox_inches='tight')
                plt.show()


            # statistiques sur les valeurs U par typologie
            if True:
                for typo_group in ['SFH', 'TH', 'MFH', 'AB']:
                # typo_group = 'AB' # SFH, TH, MFH, AB
                    hue_order = ['FR.N.{}.{:02d}.Gen'.format(typo_group,n) for n in range(1,12)]
                    
                    # sns.histplot(dpe_typology_group, x='dpe_mix_arrete_u_mur_exterieur',
                    #              binwidth=0.05, hue='typology',kde=True,stat='density',
                    #              palette='viridis',hue_order=hue_order, common_norm=False,
                    #              ec=None,legend=False)
                    
                    ls_dict = ['solid','dotted','dashed','dashdot',]
                    cmap = matplotlib.colormaps.get_cmap('viridis')
                    
                    dict_variables = {'dpe_mix_arrete_u_mur_exterieur':{'peak_height':0.15,
                                                                        'label':'Walls U-value (W.m$^{-2}$.K$^{-1}$)'},
                                      'dpe_mix_arrete_uw':{'peak_height':0.15,
                                                           'label':'Windows U-value (W.m$^{-2}$.K$^{-1}$)'},
                                      'dpe_mix_arrete_u_plancher_bas_final_deperditif':{'peak_height':0.15,
                                                                                        'label':'Floor U-value (W.m$^{-2}$.K$^{-1}$)'},
                                      'dpe_mix_arrete_u_plancher_haut_deperditif':{'peak_height':0.15,
                                                                                   'label':'Roof U-value (W.m$^{-2}$.K$^{-1}$)'},}
                    
                    for var in dict_variables.keys():
                        fig,ax = plt.subplots(figsize=(10,5),dpi=300)
                        
                        for i,ty in enumerate(hue_order):
                            values = dpe[dpe.typology==ty][var].dropna().values
                            kde_estim = stats.gaussian_kde(values, bw_method='scott')
                            
                            X0, X1 = 0,dpe[var].quantile(0.99)
                            X = np.linspace(X0,X1*1.5,300)
                            Y = kde_estim(X)
                            
                            peaks_x,_ = signal.find_peaks(Y, height=dict_variables.get(var).get('peak_height'))
                            
                            ax.plot(X,Y, label=ty, color=cmap(i/len(hue_order)), ls=ls_dict[i%4])
                            ax.plot(X[peaks_x],Y[peaks_x],marker='v',ls='',color=cmap(i/len(hue_order)))
        
                        ax.legend(ncol=2)
                        ax.set_xlim([X0,X1])
                        ax.set_xlabel(dict_variables.get(var).get('label'))
                        ax.set_ylabel('Density')
                        plt.savefig(os.path.join(figs_folder,'bdnb_distribution_{}_{}.png'.format(typo_group,var)), bbox_inches='tight')
                        plt.show()
                    
            if True:
                for typo_group in ['SFH', 'TH', 'MFH', 'AB']:
                # typo_group = 'AB' # SFH, TH, MFH, AB
                    hue_order = ['FR.N.{}.{:02d}.Gen'.format(typo_group,n) for n in range(1,12)]
                    
                    ls_dict = ['solid','dotted','dashed','dashdot',]
                    cmap = matplotlib.colormaps.get_cmap('viridis')
                    
                    dict_variables = {'dpe_mix_arrete_epaisseur_isolation_mur_exterieur_estim':{'label':'Walls insulation thickness (cm)'},}
                                      # 'dpe_mix_arrete_epaisseur_isolation_mur_exterieur_estim':{'label':'Floor insulation thickness (cm)'},
                                      # 'dpe_mix_arrete_epaisseur_isolation_mur_exterieur_estim':{'label':'Roof insulation thickness (cm)'}}
                    
                    for var in dict_variables.keys():
                        fig,ax = plt.subplots(figsize=(10,5),dpi=300)
                        
                        for i,ty in enumerate(hue_order):
                            # sns.histplot(dpe[dpe.typology==ty], x=var,
                            #              binwidth=1,color=cmap(i/len(hue_order)),
                            #              ls = ls_dict[i%4],
                            #              ax=ax,
                            #              element="poly", fill=False,cumulative=True, 
                            #              stat="density", common_norm=False,label=ty)
                            ax.hist(dpe[dpe.typology==ty][var], bins=np.arange(0,50,1), density=True, histtype="step", cumulative=-1,
                                    color=cmap(i/len(hue_order)),ls=ls_dict[i%4],label=ty)
                        ax.legend(ncol=1)
                        ax.set_xlim([0,dpe[var].quantile(0.995)])
                        ax.set_ylabel('Reverse cumulative density')
                        ax.set_xlabel(dict_variables.get(var).get('label'))
                        plt.savefig(os.path.join(figs_folder,'bdnb_distribution_{}_{}.png'.format(typo_group,var)), bbox_inches='tight')
                        plt.show()
                
                # print(dpe[dpe.group_typology==typo_group].dpe_mix_arrete_epaisseur_isolation_mur_exterieur_estim.value_counts())
                
                
            
        # carte des alentours d'un batiment
        if False:
            neighbourhood_map('bdnb-bg-117G-LB5M-XT4U', path=None, save=False)
        
    
    #%% Lien entre les étiquettes DPE et la surface des logements, lien entre la période de construction et la surface des logements 
    # Regarder les trois éléments les uns les autres
    if False:
        departement = Departement(24)
        max_energy_cons = 750
        
        # histogramme des consommations dans la base DPE
        if True:
            plot_dpe_distribution(dep=departement.code, path=os.path.join(output,folder),max_xlim=max_energy_cons, external_disk=external_disk_connection)
        
        # tous les grapĥes 
        if True:
            dpe,_,_ = get_bdnb(departement.code, external_disk=external_disk_connection)
            
            dpe = dpe[dpe.type_dpe=='dpe arrêté 2021 3cl logement'][['conso_5_usages_ep_m2','conso_5_usages_ef_m2','periode_construction_dpe','surface_habitable_logement']].compute()
            dpe = dpe.dropna()
            
            dpe_surfplot = dpe[np.abs(stats.zscore(dpe['surface_habitable_logement']))<3]
            dpe_surfplot = dpe_surfplot[np.abs(stats.zscore(dpe_surfplot['conso_5_usages_ep_m2']))<3]
            
            # dpe = dpe[dpe.conso_5_usages_ep_m2<dpe.conso_5_usages_ep_m2.quantile(0.95)]
            
            order = ['avant 1948','1948-1974','1975-1977','1978-1982','1983-1988','1989-2000','2001-2005','2006-2012','2013-2021','après 2021',]
            
            etiquette_colors_dict = {'A':(0, 156, 109),'B':(82, 177, 83),'C':(120, 189, 118),'D':(244, 231, 15),'E':(240, 181, 15),'F':(235, 130, 53),'G':(215, 34, 31)}
            etiquette_colors_dict = {k: tuple(map(lambda x: x/255, v)) for k,v in etiquette_colors_dict.items()}
            etiquette_ep_dict = {'A':[0,70],'B':[70,110],'C':[110,180],'D':[180,250],'E':[250,330],'F':[330,420],'G':[420,np.inf]}
            
            # les outliers sont enlevées pour des zscore < 3 : source
            
            
            fig,ax = plt.subplots(figsize=(5,5), dpi=300)
            
            sns.boxplot(x='periode_construction_dpe',y='conso_5_usages_ep_m2',data=dpe_surfplot, 
                        ax=ax, order=order, color='w',linecolor='k')
            ax.set_ylabel('Consommation annuelle en énergie primaire (kWh.m$^{-2}$)')
            ax.set_xlabel('Période de construction ({})'.format(departement.name))
            xlims = ax.get_xlim()
            ax.set_ylim([0,max_energy_cons])
            
            for letter,(y_low,y_high) in etiquette_ep_dict.items():
                y_max = max(ax.get_ylim())+1
                y_high = min(y_high, y_max)
                ax.fill_between(xlims, [y_high]*2, [y_low]*2, color=etiquette_colors_dict.get(letter),alpha=0.42)
            
            ax.set_xlim(xlims)
            plt.xticks(rotation=90)
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('consumption_period_construction_dep{}'.format(departement.code))), bbox_inches='tight')
            plt.show()
            
        
            fig,ax = plt.subplots(figsize=(5,5), dpi=300)
            sns.boxplot(x='periode_construction_dpe',y='surface_habitable_logement',data=dpe_surfplot, 
                        ax=ax, order=order, color='lightgrey',linecolor='k')
            ax.set_ylabel('Surface du logement (m$^{2}$)')
            ax.set_xlabel('Période de construction ({})'.format(departement.name))
            ax.set_ylim([0,150])
            plt.xticks(rotation=90)
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('surface_period_construction_dep{}'.format(departement.code))), bbox_inches='tight')
            plt.show()
            
            
            
            add_DPE_lines = True
            add_DPE_lines_small_surface_update = True
            
            
            max_energy_cons = min(max_energy_cons, dpe_surfplot.conso_5_usages_ep_m2.max())
            
            cmap = matplotlib.colormaps.get_cmap('viridis')
            
            fig,ax = plt.subplots(figsize=(5,5), dpi=300)
            h,_,_,_ = ax.hist2d(dpe_surfplot.surface_habitable_logement,dpe_surfplot.conso_5_usages_ep_m2,density=True,vmin=0,
                                bins=[np.asarray(range(int(dpe_surfplot.surface_habitable_logement.max()))),np.asarray(range(int(dpe_surfplot.conso_5_usages_ep_m2.max())))])
            
            h,_,_,_ = ax.hist2d(dpe_surfplot.surface_habitable_logement,dpe_surfplot.conso_5_usages_ep_m2,density=True,vmin=0,vmax=np.quantile(h,0.99),cmap=cmap,
                                 bins=[np.asarray(range(int(dpe_surfplot.surface_habitable_logement.max()))),np.asarray(range(int(dpe_surfplot.conso_5_usages_ep_m2.max())))])
            ax.set_xlabel('Surface du logement (m$^{2}$)')
            ax.set_ylabel('Consommation annuelle en énergie primaire (kWh.m$^{-2}$)')
            xlims = [0,150]
            
            if add_DPE_lines:
                for letter,(y_low,y_high) in etiquette_ep_dict.items():
                    y_max = max(ax.get_ylim())+1
                    y_high = min(y_high, y_max)
                    # ax.plot(xlims,[y_low]*2, color=etiquette_colors_dict.get(letter),lw=0.5,label=letter,ls=':') 
                    ax.plot(xlims,[y_high]*2, color=etiquette_colors_dict.get(letter),lw=1,ls=':') 
            
            if add_DPE_lines_small_surface_update:
                dpe_threshold = pd.read_csv(os.path.join('data','INSEE','DPE_petites_surfaces.csv')).set_index('S_REF')
                surface_list = list(range(151))
                
                for letter,(y_low,y_high) in etiquette_ep_dict.items():
                    if letter == 'G':
                        continue 
                    
                    dpe_energy_limits = []
                    for s in surface_list:
                        if s < 8:
                            s = 8
                        
                        if s < 40:
                            energy_lim = dpe_threshold.loc[s]['CEP_{}'.format(letter)]
                        else:
                            energy_lim = y_high
                        dpe_energy_limits.append(energy_lim)
                            
                    ax.plot(surface_list,dpe_energy_limits, color=etiquette_colors_dict.get(letter),lw=1,ls='-') 
            
            
            cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
            posn = ax.get_position()
            cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
            norm = matplotlib.colors.Normalize(vmin=0, vmax=np.quantile(h,0.99))
            mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            
            cbar_label_var = 'Density'
            _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend='max', extendfrac=0.02)
            
            # ax.legend(loc='upper right')
            ax.set_xlim(xlims)
            ax.set_ylim([0,max_energy_cons])
            # plt.xticks(rotation=90)
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('surface_consumption_dep{}'.format(departement.code))), bbox_inches='tight')
            plt.show()
    
    #%% Statistiques des besoins de chauffage DPE par "typologies"
    if False:
        dpe = pd.read_csv(os.path.join('data','DPE','dpe-v2-logements-existants.csv'))
        dpe['Bch'] = dpe.Besoin_chauffage/dpe.Surface_habitable_logement/1000
        dpe['Bfr'] = np.asarray([e if isinstance(e,float) else  float(e.replace(',','.')) for e in dpe.Besoin_refroidissement])/dpe.Surface_habitable_logement
        dpe = dpe[['Période_construction','Type_bâtiment','Zone_climatique_','Bfr','Bch']]
        
        # for p in list(set(dpe.Période_construction.values)):
        #     for bt in ['appartement','maison']:
        #         dpe_filtered = dpe[(dpe.Période_construction==p)&(dpe.Type_bâtiment==bt)&(dpe.Zone_climatique_=='H1a')]
        
        # Bch_corr = []
        # for bch, p in zip(dpe.Bch,dpe.Période_construction):
        #     if p == '1948-1974':
        #         # bch /= 10
        #     Bch_corr.append(bch)
        # dpe['Bch'] = Bch_corr
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        sns.histplot(dpe[dpe.Zone_climatique_=='H1a'],x='Bch',hue='Période_construction',ax=ax,stat='percent')
        ax.set_xlim([0,200])
        plt.show()
        # stats_dpe = dpe.groupby(by=['Période_construction','Zone_climatique_','Type_bâtiment'])[['Bch','Bfr']].mean()
        
        
        
        # print(stats_dpe)
        
        # GET https://data.ademe.fr/data-fair/api/v1/datasets/dpe-v2-logements-existants/lines?select=Surface_habitable_logement%2CBesoin_chauffage%2CType_b%C3%A2timent%2CP%C3%A9riode_construction%2CZone_climatique_%2CBesoin_refroidissement    


        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__=='__main__':
    main()
