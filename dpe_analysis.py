#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 22:53:52 2025

@author: amounier
"""

import time 
import os
from datetime import date
import pandas as pd
import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt

from bdnb_opener import get_bdnb, plot_dpe_distribution
from administrative import draw_departement_map, France, Departement, Climat

etiquette_colors_dict = {'A':(0, 156, 109),'B':(82, 177, 83),'C':(120, 189, 118),'D':(244, 231, 15),'E':(240, 181, 15),'F':(235, 130, 53),'G':(215, 34, 31)}
etiquette_colors_dict = {k: tuple(map(lambda x: x/255, v)) for k,v in etiquette_colors_dict.items()}
etiquette_ep_dict = {'A':[0,70],'B':[70,110],'C':[110,180],'D':[180,250],'E':[250,330],'F':[330,420],'G':[420,np.inf]}

def get_dpe_conso(dep, external_disk):
    dpe_data, _ , _ = get_bdnb(dep,external_disk=external_disk)
    dpe_data = dpe_data[dpe_data.type_dpe=='dpe arrêté 2021 3cl logement'][['conso_5_usages_ep_m2','conso_5_usages_ef_m2']].compute() 
    return dpe_data

# =============================================================================
# script 
# =============================================================================
def main():
    tic = time.time()
    
    # définition de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # définition du fichier de sortie
    output_folder = os.path.join('output')
    folder = '{}_DPE_analysis'.format(today)
    if folder not in os.listdir(output_folder):
        os.mkdir(os.path.join(output_folder,folder))
    if 'figs' not in os.listdir(os.path.join(output_folder, folder)):
        os.mkdir(os.path.join(output_folder,folder,'figs'))
    
    output_path = os.path.join(output_folder,folder)
    figs_folder = os.path.join(output_path, 'figs')
    external_disk_connection = 'MPBE' in os.listdir('/media/amounier/')
    
    #%% Étude des distribution de DPE aux frontières des zones climatiques
    if True:# and external_disk_connection:
        
        # idée : expliquer le shift entre les distributions des zones climatiques
        # plot_dpe_distribution(dep='84', path=output_path,max_xlim=600, external_disk=external_disk_connection)
        # plot_dpe_distribution(dep='13', path=output_path,max_xlim=600, external_disk=external_disk_connection)
        
        # TODO : vérifier les paramètres de U value, et de systèmes de chauffage pour vérifier la contribution principale de la zone climatique
        
        
        # définition des départements frontières
        H1_border12_dep_code = ['14','61','28','45','58','03','23','87','19','15','43','42','38','05']
        H2_border12_dep_code = ['50','53','72','41','18','36','86','16','24','46','12','48','07','26','04']
        H2_border23_dep_code = ['09','81','12','48','07','84','04'] # excluded 31 due to a lot of weird B
        H3_border23_dep_code = ['66','11','34','30','13','83','06']
        
        file_names = ['H1_border12','H2_border12','H2_border23','H3_border23']
    
        H2d_border32d_dep_code = ['48','07','84','04'] 
        H3_border32d_dep_code = ['30','13','83','06']
        H2c_border32c_dep_code = ['09','81','12'] # excluded 31 due to a lot of weird B
        H3_border32c_dep_code = ['66','11','34','30']
        
        # H1_border12_dep_code = H2d_border32d_dep_code
        # H2_border12_dep_code = H3_border32d_dep_code
        # H2_border23_dep_code = H2c_border32c_dep_code
        # H3_border23_dep_code = H3_border32c_dep_code
        
        # file_names = ['H3_border32c','H3_border32d','H2c_border32c','H2d_border32d']
        
        total_dep = {}
        
        # caractérisation du 31 exclu:
        if False:
            plot_dpe_distribution(dep='31', path=output_path,max_xlim=600, external_disk=external_disk_connection)
            
            
        # étude des effets des zones climatiques
        if True:
            reformat_bdnb_dpe_file_light = 'dpe_statistics_light_heater.parquet'
            dpe = pd.read_parquet(os.path.join('data','BDNB',reformat_bdnb_dpe_file_light))
            
            dpe = dpe[~dpe.dpe_mix_arrete_type_generateur_chauffage.isnull()]
            
            for v in ['dpe_mix_arrete_surface_vitree_nord','dpe_mix_arrete_surface_vitree_sud',
                      'dpe_mix_arrete_surface_vitree_ouest','dpe_mix_arrete_surface_vitree_est',
                      'dpe_mix_arrete_surface_vitree_horizontal']:
                dpe[v] = dpe[v].fillna(0)
                
            dpe['dpe_mix_arrete_surface_vitree'] = dpe['dpe_mix_arrete_surface_vitree_nord'] + dpe['dpe_mix_arrete_surface_vitree_sud'] + dpe['dpe_mix_arrete_surface_vitree_ouest'] + dpe['dpe_mix_arrete_surface_vitree_est'] + dpe['dpe_mix_arrete_surface_vitree_horizontal']
            dpe['Uw'] = dpe.dpe_mix_arrete_uw * dpe.dpe_mix_arrete_surface_vitree
            dpe['Um'] = dpe.dpe_mix_arrete_u_mur_exterieur * dpe.dpe_mix_arrete_surface_mur_deperditif
            dpe['Uph'] = dpe.dpe_mix_arrete_u_plancher_haut_deperditif * dpe.dpe_mix_arrete_surface_plancher_haut_deperditif
            dpe['Upb'] = dpe.dpe_mix_arrete_u_plancher_bas_final_deperditif * dpe.dpe_mix_arrete_surface_plancher_bas_deperditif
            dpe['U'] = dpe['Uw'] + dpe['Um'] + dpe['Uph'] + dpe['Upb']
            
            dpe = dpe.rename(columns={'dpe_mix_arrete_conso_5_usages_ep_m2':'conso_5_usages_ep_m2',
                                      'dpe_mix_arrete_conso_5_usages_ef_m2':'conso_5_usages_ef_m2'})
            
            
            for name, border_dep_list in zip(file_names,[H1_border12_dep_code,H2_border12_dep_code,H2_border23_dep_code,H3_border23_dep_code]):
                counter = None
                for dep in tqdm.tqdm(border_dep_list,desc=name):
                    # dpe_data = get_dpe_conso(dep,external_disk_connection)
                    dpe_data = dpe[dpe.departement==dep]
                    dpe_data = dpe_data.dropna()
                    dpe_data['conso_5_usages_ep_m2'] = dpe_data.conso_5_usages_ep_m2.map(int)
                    dpe_data['conso_5_usages_ef_m2'] = dpe_data.conso_5_usages_ef_m2.map(int)
                    
                    total_dep.update({dep:len(dpe_data)})
                    
                    if counter is None:
                        counter = pd.DataFrame(dpe_data.conso_5_usages_ep_m2.value_counts())
                    else:
                        counter = counter.join(pd.DataFrame(dpe_data.conso_5_usages_ep_m2.value_counts()).rename(columns={'count':'supp'}),how='outer')
                        counter = counter.fillna(0)
                        counter['count'] = counter['count'] + counter['supp']
                        counter = counter[['count']]
                pickle.dump(counter, open(os.path.join(output_path,'{}.pickle'.format(name)), "wb"))
                pickle.dump(total_dep, open(os.path.join(output_path,'nb_dpe_departement.pickle'), "wb"))
        
        total_dep = pickle.load(open(os.path.join(output_path,'nb_dpe_departement.pickle'), 'rb'))
        
        # border12
        h1_b12 = pickle.load(open(os.path.join(output_path,'{}.pickle'.format('H1_border12')), 'rb'))
        h1_b12['ratio'] = h1_b12['count']/h1_b12['count'].sum()
        mean_h1_b12 = sum(h1_b12['count'] * h1_b12.index) / h1_b12['count'].sum()
        
        h2_b12 = pickle.load(open(os.path.join(output_path,'{}.pickle'.format('H2_border12')), 'rb'))
        h2_b12['ratio'] = h2_b12['count']/h2_b12['count'].sum()
        mean_h2_b12 = sum(h2_b12['count'] * h2_b12.index) / h2_b12['count'].sum()
        
        # border23
        h2_b23 = pickle.load(open(os.path.join(output_path,'{}.pickle'.format('H2_border23')), 'rb'))
        h2_b23['ratio'] = h2_b23['count']/h2_b23['count'].sum()
        mean_h2_b23 = sum(h2_b23['count'] * h2_b23.index) / h2_b23['count'].sum()
        
        h3_b23 = pickle.load(open(os.path.join(output_path,'{}.pickle'.format('H3_border23')), 'rb'))
        h3_b23['ratio'] = h3_b23['count']/h3_b23['count'].sum()
        mean_h3_b23 = sum(h3_b23['count'] * h3_b23.index) / h3_b23['count'].sum()
        
        # borders H3-H2c
        h3_b2c3 = pickle.load(open(os.path.join(output_path,'{}.pickle'.format('H3_border32c')), 'rb'))
        h3_b2c3['ratio'] = h3_b2c3['count']/h3_b2c3['count'].sum()
        mean_h3_b2c3 = sum(h3_b2c3['count'] * h3_b2c3.index) / h3_b2c3['count'].sum()
        
        h2c_b2c3 = pickle.load(open(os.path.join(output_path,'{}.pickle'.format('H2c_border32c')), 'rb'))
        h2c_b2c3['ratio'] = h2c_b2c3['count']/h2c_b2c3['count'].sum()
        mean_h2c_b2c3 = sum(h2c_b2c3['count'] * h2c_b2c3.index) / h2c_b2c3['count'].sum()
        
        # border23
        h3_b2d3 = pickle.load(open(os.path.join(output_path,'{}.pickle'.format('H3_border32d')), 'rb'))
        h3_b2d3['ratio'] = h3_b2d3['count']/h3_b2d3['count'].sum()
        mean_h3_b2d3 = sum(h3_b2d3['count'] * h3_b2d3.index) / h3_b2d3['count'].sum()
        
        h2d_b2d3 = pickle.load(open(os.path.join(output_path,'{}.pickle'.format('H2d_border32d')), 'rb'))
        h2d_b2d3['ratio'] = h2d_b2d3['count']/h2d_b2d3['count'].sum()
        mean_h2d_b2d3 = sum(h2d_b2d3['count'] * h2d_b2d3.index) / h2d_b2d3['count'].sum()
        
        # france = France()
        
        # dh19 = dict(pd.read_csv('data/DPE/DH19_0-400.csv')[france.climats].sum())
        
        # h1b12 = pd.DataFrame().from_dict({'dep':H1_border12_dep_code})
        # h1b12['zcl'] = [Departement(dep).climat for dep in h1b12.dep]
        # h1b12['dh19'] = [dh19.get(zcl) for zcl in h1b12.zcl]
        # h1b12['dpe'] = [total_dep.get(dep) for dep in h1b12.dep]
        # mean_dh19_h1b12 = sum(h1b12.dpe * h1b12.dh19) / h1b12.dpe.sum()
        
        # h2b12 = pd.DataFrame().from_dict({'dep':H2_border12_dep_code})
        # h2b12['zcl'] = [Departement(dep).climat for dep in h2b12.dep]
        # h2b12['dh19'] = [dh19.get(zcl) for zcl in h2b12.zcl]
        # h2b12['dpe'] = [total_dep.get(dep) for dep in h2b12.dep]
        # mean_dh19_h2b12 = sum(h2b12.dpe * h2b12.dh19) / h2b12.dpe.sum()
        
        # h2b23 = pd.DataFrame().from_dict({'dep':H2_border23_dep_code})
        # h2b23['zcl'] = [Departement(dep).climat for dep in h2b23.dep]
        # h2b23['dh19'] = [dh19.get(zcl) for zcl in h2b23.zcl]
        # h2b23['dpe'] = [total_dep.get(dep) for dep in h2b23.dep]
        # mean_dh19_h2b23 = sum(h2b23.dpe * h2b23.dh19) / h2b23.dpe.sum()
        
        # h3b23 = pd.DataFrame().from_dict({'dep':H3_border23_dep_code})
        # h3b23['zcl'] = [Departement(dep).climat for dep in h3b23.dep]
        # h3b23['dh19'] = [dh19.get(zcl) for zcl in h3b23.zcl]
        # h3b23['dpe'] = [total_dep.get(dep) for dep in h3b23.dep]
        # mean_dh19_h3b23 = sum(h3b23.dpe * h3b23.dh19) / h3b23.dpe.sum()
        
        # print('Effet H1-H2:\n\t-Cep : {:+.1f}% (en faveur de H1)\n\t-DH19: {:+.1f}% (en faveur de H1)'.format((mean_h1_b12/mean_h2_b12-1)*100,(mean_dh19_h1b12/mean_dh19_h2b12-1)*100))
        # print('Effet H2-H3:\n\t-Cep : {:+.1f}% (en faveur de H2)\n\t-DH19: {:+.1f}% (en faveur de H2)'.format((mean_h2_b23/mean_h3_b23-1)*100,(mean_dh19_h2b23/mean_dh19_h3b23-1)*100))
        
        # graphes 
        if True:
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(h1_b12.index, h1_b12.ratio, color='tab:blue',label='H1 ($\\mu$={:.0f} '.format(mean_h1_b12)+'kWh.m$^{-2}$)',zorder=4)
            ax.plot(h2_b12.index, h2_b12.ratio, color='k',label='H2 ($\\mu$={:.0f} '.format(mean_h2_b12)+'kWh.m$^{-2}$)')
            
            xmax = 600.
            ymax = ax.get_ylim()[-1]
            ax.set_xlim(left=0.,right=xmax)
            ax.set_ylim(bottom=0.)
            
            for eti,(ep0,ep1) in etiquette_ep_dict.items():
                if ep1 == float("inf"):
                    ep1 = xmax
                ax.fill_between([ep0,ep1],[ymax]*2,[0]*2,color=etiquette_colors_dict.get(eti),alpha=0.42)
            ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [xmax])
            
            ax.set_xlabel("Annual primary energy consumption (kWh.m$^{-2}$)")
            ax.set_ylabel("Density")
            ax.legend()
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('dpe_distribution_border12')),bbox_inches='tight')
            plt.show()
        
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(h2_b23.index, h2_b23.ratio, color='k',label='H2 ($\\mu$={:.0f} '.format(mean_h2_b23)+'kWh.m$^{-2}$)')
            ax.plot(h3_b23.index, h3_b23.ratio, color='tab:blue',label='H3 ($\\mu$={:.0f} '.format(mean_h3_b23)+'kWh.m$^{-2}$)',zorder=4)
            
            xmax = 600.
            ymax = ax.get_ylim()[-1]
            ax.set_xlim(left=0.,right=xmax)
            ax.set_ylim(bottom=0.)
            
            for eti,(ep0,ep1) in etiquette_ep_dict.items():
                if ep1 == float("inf"):
                    ep1 = xmax
                ax.fill_between([ep0,ep1],[ymax]*2,[0]*2,color=etiquette_colors_dict.get(eti),alpha=0.42)
            ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [xmax])
            
            ax.set_xlabel("Annual primary energy consumption (kWh.m$^{-2}$)")
            ax.set_ylabel("Density")
            ax.legend()
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('dpe_distribution_border23')),bbox_inches='tight')
            plt.show()
            
        # graphes 
        if True:
            cmap = plt.get_cmap('viridis')
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(h2c_b2c3.index, h2c_b2c3.ratio, color=cmap(0.5),label='H2c ($\\mu$={:.0f} '.format(mean_h2c_b2c3)+'kWh.m$^{-2}$)',zorder=4)
            ax.plot(h3_b2c3.index, h3_b2c3.ratio, color='k',label='H3 ($\\mu$={:.0f} '.format(mean_h3_b2c3)+'kWh.m$^{-2}$)')
            
            xmax = 600.
            ymax = ax.get_ylim()[-1]
            ax.set_xlim(left=0.,right=xmax)
            ax.set_ylim(bottom=0.)
            
            for eti,(ep0,ep1) in etiquette_ep_dict.items():
                if ep1 == float("inf"):
                    ep1 = xmax
                ax.fill_between([ep0,ep1],[ymax]*2,[0]*2,color=etiquette_colors_dict.get(eti),alpha=0.42)
            ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [xmax])
            
            ax.set_xlabel("Annual primary energy consumption (kWh.m$^{-2}$)")
            ax.set_ylabel("Density")
            ax.legend()
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('dpe_distribution_border12')),bbox_inches='tight')
            plt.show()
        
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(h2d_b2d3.index, h2d_b2d3.ratio, color=cmap(0.5),label='H2d ($\\mu$={:.0f} '.format(mean_h2d_b2d3)+'kWh.m$^{-2}$)',zorder=4)
            ax.plot(h3_b2d3.index, h3_b2d3.ratio, color='k',label='H3 ($\\mu$={:.0f} '.format(mean_h3_b2d3)+'kWh.m$^{-2}$)')
            
            xmax = 600.
            ymax = ax.get_ylim()[-1]
            ax.set_xlim(left=0.,right=xmax)
            ax.set_ylim(bottom=0.)
            
            for eti,(ep0,ep1) in etiquette_ep_dict.items():
                if ep1 == float("inf"):
                    ep1 = xmax
                ax.fill_between([ep0,ep1],[ymax]*2,[0]*2,color=etiquette_colors_dict.get(eti),alpha=0.42)
            ax.set_xticks(ticks=[int(x) for x in list(set(list(np.asarray(list(etiquette_ep_dict.values())).flatten()))) if not np.isinf(x)] + [xmax])
            
            ax.set_xlabel("Annual primary energy consumption (kWh.m$^{-2}$)")
            ax.set_ylabel("Density")
            ax.legend()
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('dpe_distribution_border23')),bbox_inches='tight')
            plt.show()
        
        # cartes
        if True:
            # france = France()
            
            # dep_dict = {}
            # for dep in france.departements:
            #     if dep.code in H1_border12_dep_code:
            #         dep_dict[dep] = 0.2
            #     elif dep.code in H2_border12_dep_code:
            #         dep_dict[dep] = 0.5
            #     else:
            #         dep_dict[dep] = np.nan
           
            # fig,ax = draw_departement_map(dep_dict, figs_folder=figs_folder, save='border_zcl_12', 
            #                               hide_cbar=True,cmap=None,automatic_cbar_values=False,alpha=None)
            
            # dep_dict = {}
            # for dep in france.departements:
            #     if dep.code in H2_border23_dep_code:
            #         dep_dict[dep] = 0.5
            #     elif dep.code in H3_border23_dep_code:
            #         dep_dict[dep] = 0.9
            #     else:
            #         dep_dict[dep] = np.nan
           
            # fig,ax = draw_departement_map(dep_dict, figs_folder=figs_folder, save='border_zcl_23',
            #                               hide_cbar=True,cmap=None,automatic_cbar_values=False,alpha=None)
            
            france = France()
            
            dep_dict = {}
            for dep in france.departements:
                if dep.code in H3_border32d_dep_code:
                    dep_dict[dep] = 0.2
                elif dep.code in H2d_border32d_dep_code:
                    dep_dict[dep] = 0.5
                else:
                    dep_dict[dep] = np.nan
           
            fig,ax = draw_departement_map(dep_dict, figs_folder=figs_folder, save='border_zcl_H3H2d', 
                                          hide_cbar=True,cmap=None,automatic_cbar_values=False,alpha=None)
            
            dep_dict = {}
            for dep in france.departements:
                if dep.code in H3_border32c_dep_code:
                    dep_dict[dep] = 0.5
                elif dep.code in H2c_border32c_dep_code:
                    dep_dict[dep] = 0.9
                else:
                    dep_dict[dep] = np.nan
           
            fig,ax = draw_departement_map(dep_dict, figs_folder=figs_folder, save='border_zcl_H3H2c',
                                          hide_cbar=True,cmap=None,automatic_cbar_values=False,alpha=None)
                

        
            
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()