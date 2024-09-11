#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:34:48 2024

@author: amounier
"""

import time 
import pandas as pd
from datetime import date
import os
import matplotlib.pyplot as plt

from bdnb_opener import get_bdnb
from administrative import Departement


# dict_angle_orientation = {i*45:o for i,o in enumerate(['N','NE','E','SE','S','SW','W','NW'])}
# dict_orientation_angle = {v:k for k,v in dict_angle_orientation.items()}

# TODO : créer uen classe typology ? Surement

def open_tabula_typologies():
    path = os.path.join('data','TABULA','TABULA_typologies.csv')
    data = pd.read_csv(path)
    return data


def identify_typologies(dep='75'):
    lazy_dpe_logement, lazy_rel_batiment_groupe_dpe_logement, lazy_bdnb_batiment_groupe_compile = get_bdnb(dep=dep)
    # TODO : ouvrir la bdnb et compter le nombre de chaue typologie tabula
    # TODO : vérifier les données TABULA, caractériser les isolations
    
    return 


def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_typologies'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    #%% -----------------------------------------------------------------------
    
    # typologies = open_tabula_typologies()
    
    
    
    #%% Graphe des nombre de logements par bâtiment 
    if False:
        dep = Departement(75)
        insee_url = 'https://www.insee.fr/fr/statistiques/2011101?geo=DEP-{}#chiffre-cle-3'.format(dep.code)
        print(insee_url)
        
        
        _,_,bgc = get_bdnb(dep=dep.code)
        
        # on ne sélectionne que le sbâtiments présentant au moins un logement
        bgc = bgc[bgc.ffo_bat_nb_log>=1]
        dict_nb_households_bat = bgc.ffo_bat_nb_log.dropna().value_counts().compute().to_dict()

        counter = 0
        for i in range(int(max(list(dict_nb_households_bat.keys())))):
            end_tail = i
            if dict_nb_households_bat.get(i) == 1:
                counter += 1
                if counter == 10:
                    break
                
            
        nb_households = int(sum([k*v for k,v in dict_nb_households_bat.items()]))
        
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ax.set_title('{} ({}) - {:,} logements'.format(dep.name,dep.code,nb_households).replace(',',' '))
        ax.bar(dict_nb_households_bat.keys(),dict_nb_households_bat.values(),width=1)
        ax.set_yscale("log", nonpositive='clip')
        ax.set_ylim(bottom=0.5)
        ax.set_ylabel('# of buildings')
        ax.set_xlabel('# of households per building')
        ax.set_xlim(left=0., right=end_tail)
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('hist_nb_logements_bat_dep{}'.format(dep.code))),bbox_inches='tight')
        plt.show()
        
    #%% Graphe des périodes de construction
    if False:
        dep = Departement(75)
        insee_url = 'https://www.insee.fr/fr/statistiques/2011101?geo=DEP-{}#chiffre-cle-3'.format(dep.code)
        # print(insee_url)
        
        
        _,_,bgc = get_bdnb(dep=dep.code)  
        
        # on ne sélectionne que les bâtiments contenant des logements
        bgc = bgc[bgc.ffo_bat_nb_log>=1]
        dict_construction_bat = bgc.ffo_bat_annee_construction.dropna().value_counts().compute().to_dict() # 5.6s pour Paris

        # test_formatted = sorted([x for xs in [[int(k)]*v for k,v in test.items() if k != 0.] for x in xs])
        
        # counter = 0
        # for i in range(int(max(list(dict_nb_households_bat.keys())))):
        #     end_tail = i
        #     if dict_nb_households_bat.get(i) == 1:
        #         counter += 1
        #         if counter == 10:
        #             break
                
            
        # nb_households = int(sum([k*v for k,v in dict_nb_households_bat.items()]))
        
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ax.set_title('{} ({})'.format(dep.name,dep.code).replace(',',' '))
        ax.bar(dict_construction_bat.keys(),dict_construction_bat.values(),width=1)
        # ax.set_yscale("log", nonpositive='clip')
        ax.set_ylim(bottom=0.5)
        ax.set_ylabel('# of buildings')
        ax.set_xlabel('Construction year')
        ax.set_xlim(left=1690, right=int(today[:4]))
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('hist_construction_bat_dep{}'.format(dep.code))),bbox_inches='tight')
        plt.show()
    
    
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__=='__main__':
    main()