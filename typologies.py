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
import tqdm
import numpy as np

from bdnb_opener import get_bdnb, neighbourhood_map
from administrative import France, Departement, add_departement_map


dict_angle_orientation = {i*45:o for i,o in enumerate(['N','NE','E','SE','S','SW','W','NW'])}
dict_orientation_angle = {v:k for k,v in dict_angle_orientation.items()}

# TODO : completer la classe typology
# TODO : completer la classe material 


def open_materials_characteristics():
    path = os.path.join('data','Materials','materials.csv')
    data = pd.read_csv(path)
    data = data.set_index('material')
    return data


class Material():
    """
    Sources principales : 
        https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/2-fascicule_materiaux.pdf
        https://perso.univ-lemans.fr/~bcasta/Cours%20L3%20Echanges%20thermiques/Thermique%20de%20l'inge%C3%8C%C2%81nieur,%20Annexes.pdf
        
    """
    def __init__(self,name):
        self.name = name
        
        characteristics = open_materials_characteristics().loc[self.name].to_dict()
        self.thermal_conductivity = characteristics.get('thermal_conductivity') # en W/(m.K)
        self.density = characteristics.get('density') # en kg/m3
        self.thermal_capacity = characteristics.get('thermal_capacity') # en J/(kg.K)
        
        
    def __str__(self):
        return self.name
    
    
def open_tabula_typologies():
    path = os.path.join('data','TABULA','TABULA_typologies.csv')
    data = pd.read_csv(path)
    data = data.set_index('building_type_id')
    return data


class Typology():
    def __init__(self,code):
        self.code = code
        
        params = open_tabula_typologies().loc[self.code].to_dict()
        self.desc = params.get('building_name')
        
        # orientation des murs
        self.w0_orientation = params.get('building_orientation')
        self.w1_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w0_orientation)+90)%360)
        self.w2_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w1_orientation)+90)%360)
        self.w3_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w2_orientation)+90)%360)
        
        # paramètres géométriques
        self.ground_surface = params.get('building_ground_surface')
        self.surface = params.get('building_surface')
        self.roof_surface = self.ground_surface
        self.levels = params.get('building_levels')
        self.height = params.get('floor_height')
        self.volume = self.surface*self.height
        self.form_factor = params.get('building_form_factor')
        self.w0_length = np.sqrt(self.ground_surface/self.form_factor)
        self.w1_length = self.w0_length*self.form_factor
        self.perimeter = 2*(self.w0_length + self.w1_length)
        
        # caractéristiques inférieures et supérieurs 
        self.basement = bool(params.get('building_basement'))
        self.converted_attic = bool(params.get('building_converted_attic'))
        
        # caractéristiques des ventilations et infiltrations
        self.air_infiltration = params.get('air_infiltration')
        self.ventilation_efficiency = params.get('ventilation_efficiency')
        
        # caractéristiques du toit
        self.roof_color = params.get('building_roof_color')
        self.roof_U = params.get('building_roof_U')
        self.ceiling_U = params.get('building_ceiling_U')
        
        # caractéristiques des vitrages
        self.h_windows_surface = params.get('building_horizontal_windows_surface')
        self.w0_windows_surface = params.get('building_wall0_windows_surface')
        self.w1_windows_surface = params.get('building_wall1_windows_surface')
        self.w2_windows_surface = params.get('building_wall2_windows_surface')
        self.w3_windows_surface = params.get('building_wall3_windows_surface')
        self.windows_U = params.get('building_windows_U')
        self.windows_Ug = params.get('building_windows_Ug')
        
        # caractérisation de la porte
        self.door_U = params.get('building_door_U')
        self.door_surface = 2 #m2
        
        # caractéristiques des murs
        # TODO définir self.walls et mettre toutes variables dans ce dictionnaire (ou pas)
        self.w0_surface = self.w0_length*self.height*self.levels - self.w0_windows_surface
        self.w1_surface = self.w1_length*self.height*self.levels - self.w1_windows_surface
        self.w2_surface = self.w0_length*self.height*self.levels - self.w2_windows_surface
        self.w3_surface = self.w1_length*self.height*self.levels - self.w3_windows_surface
        
        self.w0_structure_material = Material(params.get('building_wall0_structure_material'))
        self.w1_structure_material = Material(params.get('building_wall1_structure_material'))
        self.w2_structure_material = Material(params.get('building_wall2_structure_material'))
        self.w3_structure_material = Material(params.get('building_wall3_structure_material'))
        self.w0_structure_thickness = params.get('building_wall0_structure_thickness')
        self.w1_structure_thickness = params.get('building_wall1_structure_thickness')
        self.w2_structure_thickness = params.get('building_wall2_structure_thickness')
        self.w3_structure_thickness = params.get('building_wall3_structure_thickness')
        
        self.w0_insulation_material = Material(params.get('building_wall0_insulation_material'))
        self.w1_insulation_material = Material(params.get('building_wall1_insulation_material'))
        self.w2_insulation_material = Material(params.get('building_wall2_insulation_material'))
        self.w3_insulation_material = Material(params.get('building_wall3_insulation_material'))
        self.w0_insulation_thickness = params.get('building_wall0_insulation_thickness')
        self.w1_insulation_thickness = params.get('building_wall1_insulation_thickness')
        self.w2_insulation_thickness = params.get('building_wall2_insulation_thickness')
        self.w3_insulation_thickness = params.get('building_wall3_insulation_thickness')
        self.w0_insulation_position = params.get('building_wall0_insulation_position')
        self.w1_insulation_position = params.get('building_wall1_insulation_position')
        self.w2_insulation_position = params.get('building_wall2_insulation_position')
        self.w3_insulation_position = params.get('building_wall3_insulation_position')
        
        # caractéristiques du sol
        # cf Rantala and Leivo 2006 et Skotnicova and Lausova (2016).
        # cf Thbat parois opaques p21
        # TODO à clarifier dans le cas 2D
        self.floor_ground_depth = params.get('building_depth')
        self.floor_ground_distance = np.sqrt((self.w1_length/3)**2 + self.floor_ground_depth**2)
        self.ground_depth = self.floor_ground_depth + 1 # à justifier
        self.ground_section = self.perimeter * self.ground_depth
        self.ground_volume = self.ground_surface * self.ground_depth
        self.floor_structure_material = Material(params.get('building_floor_structure_material'))
        self.floor_structure_thickness = params.get('building_floor_structure_thickness')
        self.floor_insulation_material = Material(params.get('building_floor_insulation_material'))
        self.floor_insulation_thickness = params.get('building_floor_insulation_thickness')
        
        # print(params)
        self.heater_maximum_power= 10000 # W
        self.cooler_maximum_power= 10000 # W

    def __str__(self):
        return self.code
    
    def update_orientation(self):
        self.w1_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w0_orientation)+90)%360)
        self.w2_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w1_orientation)+90)%360)
        self.w3_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w2_orientation)+90)%360)


# Peut-etre à bouger dans un nouveau fichier identification (ou pas, à voir)
# TODO : stocker les statistiques de typologies dans les départements (dans Departement)
# TODO : vérifier les données TABULA, caractériser les isolations


def identify_typologies(dep, external_disk=True, verbose=True):
    if verbose:
        print('Opening BDNB ({})...'.format(dep))
    
    _, _, bgc = get_bdnb(dep=dep,external_disk=external_disk)
    # TODO : à refaire de manière optimisée

    bgc = bgc[bgc.ffo_bat_nb_log>=1][['building_type_id',
                                      'building_type',
                                      'building_type_construction_year_start',
                                      'building_type_construction_year_end',
                                      'building_type_min_hh',
                                      'building_type_max_hh',]]
    bgc = bgc.compute()
    
    # rel = rel.compute()
    # rel = rel.set_index('batiment_groupe_id')
    
    dict_list_bg_id = {}
    typologies = open_tabula_typologies()
    for i in range(len(typologies)):
        # caractéristiques de typologies
        typo_id, typo_cat, typo_cys, typo_cye, typo_min_hh, typo_max_hh = typologies.iloc[i][['building_type_id',
                                                                                              'building_type',
                                                                                              'building_type_construction_year_start',
                                                                                              'building_type_construction_year_end',
                                                                                              'building_type_min_hh',
                                                                                              'building_type_max_hh',]]
        
        # filtre de la BDNB
        bgc_typo = bgc[(bgc.ffo_bat_nb_log>typo_min_hh)&
                       (bgc.ffo_bat_nb_log<=typo_max_hh)&
                       (bgc.ffo_bat_annee_construction>=typo_cys)&
                       (bgc.ffo_bat_annee_construction<=typo_cye)]
        
        # répartition des maisons individuelles par la liste des orientations des murs ext
        # TODO à faire avec les simulations DPE quand disponibles
        if typo_cat == 'TH':
            list_orientation = ['(4:est,nord,ouest,sud)', '(5:est,horizontal,nord,ouest,sud)']
            bgc_typo = bgc_typo[(~bgc_typo.dpe_mix_arrete_l_orientation_mur_exterieur.isin(list_orientation))|(bgc_typo.dpe_mix_arrete_l_orientation_mur_exterieur.isna())]
            
        elif typo_cat == 'SFH':
            list_orientation = ['(4:est,nord,ouest,sud)', '(5:est,horizontal,nord,ouest,sud)']
            bgc_typo = bgc_typo[(bgc_typo.dpe_mix_arrete_l_orientation_mur_exterieur.isin(list_orientation))&(~bgc_typo.dpe_mix_arrete_l_orientation_mur_exterieur.isna())]
        
        # number_typo = bgc_typo.shape[0].compute()
        dict_list_bg_id[typo_id] = bgc_typo.batiment_groupe_id.to_list()
        
    # compilation du nombre de logements
    bgc = bgc.set_index('batiment_groupe_id')
    dict_number_hh = {}
    for k,v in dict_list_bg_id.items():
        nb_hh = 0
        for bg_id in v:
            nb_hh += bgc.loc[bg_id].ffo_bat_nb_log
        dict_number_hh[k] = nb_hh
    
    
    del bgc # TODO vérifier que ça marche bien
    del bgc_typo
    return dict_list_bg_id, dict_number_hh




def stats_typologies_dep(dep,external_disk=True):
    # TODO très temporaire, à refaire 
    _, dep.typologies_households_number = identify_typologies(dep=dep.code,external_disk=external_disk,verbose=False)
    
    return dep

#%%============================================================================
# Script principal
# =============================================================================
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
        
    external_disk_connection = 'MPBE' in os.listdir('/media/amounier/')
    
    
    #%% Graphe des nombre de logements par bâtiment 
    if False:
        departement = Departement(75)
        insee_url = 'https://www.insee.fr/fr/statistiques/2011101?geo=DEP-{}#chiffre-cle-3'.format(departement.code)
        # print(insee_url)
        
        
        _,_,bgc = get_bdnb(dep=departement.code, external_disk=external_disk_connection)
        
        # on ne sélectionne que les bâtiments présentant au moins un logement
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
        # print(nb_households)
        
        fig,ax = plt.subplots(figsize=(5,5), dpi=300)
        ax.set_title('{} ({}) - {:,} logements'.format(departement.name,departement.code,nb_households).replace(',',' '))
        ax.bar(dict_nb_households_bat.keys(),dict_nb_households_bat.values(),width=1)
        ax.set_yscale("log", nonpositive='clip')
        ax.set_ylim(bottom=0.5)
        ax.set_ylabel('# of buildings')
        ax.set_xlabel('# of households per building')
        ax.set_xlim(left=0., right=end_tail)
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('hist_nb_logements_bat_dep{}'.format(departement.code))),bbox_inches='tight')
        plt.show()
        
    #%% Graphe des périodes de construction
    if False:
        dep = Departement(75)
        insee_url = 'https://www.insee.fr/fr/statistiques/2011101?geo=DEP-{}#chiffre-cle-3'.format(dep.code)
        # print(insee_url)
        
        
        _,_,bgc = get_bdnb(dep=dep.code, external_disk=external_disk_connection)  
        
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
    
    #%% Identification des typologies dans la BDNB
    if False:
        # typologies = open_tabula_typologies()
        
        departement = Departement(69)
        departement.typologies_batiments_groupe, departement.typologies_households_number = identify_typologies(dep=departement.code,external_disk=external_disk_connection)
        
        total_dep_hh = sum(departement.typologies_households_number.values())
        number_hh_typo_categories = {'SFH':0,'TH':0,'MFH':0,'AB':0}
        for k in number_hh_typo_categories.keys():
            for ty,nb in departement.typologies_households_number.items():
                if k in ty:
                    number_hh_typo_categories[k] += nb
        percent_hh_typo_categories = {k:v/total_dep_hh for k,v in number_hh_typo_categories.items()}
        
        for k in number_hh_typo_categories.keys():
            add_departement_map({departement:percent_hh_typo_categories.get(k)},figs_folder=figs_folder,cbar_label='{} ratio by department'.format(k))
    
    # Téléchargement des départements
    if False:
        if external_disk_connection:
            print('Téléchargement de la BDNB sur disque local.')
            # list_dep_code = ['2A']
            for d in tqdm.tqdm(France().departements):
                get_bdnb(d.code,external_disk=external_disk_connection)
    
    # Carte des stats de type de catégories (SFH,TH,MFH,AB) par départements # TODO en multithreads
    if False:
        stats = {}
        
        # list_dep_code = ['75','2A']
        for dep in tqdm.tqdm(France().departements):
            
            _, dep.typologies_households_number = identify_typologies(dep=dep.code,external_disk=external_disk_connection,verbose=False)
            
            total_dep_hh = sum(dep.typologies_households_number.values())
            number_hh_typo_categories = {'SFH':0,'TH':0,'MFH':0,'AB':0}
            for k in number_hh_typo_categories.keys():
                for ty,nb in dep.typologies_households_number.items():
                    if k in ty:
                        number_hh_typo_categories[k] += nb
            percent_hh_typo_categories = {k:v/total_dep_hh for k,v in number_hh_typo_categories.items()}
            stats[dep] = percent_hh_typo_categories
            
        dict_type_house = {'Multi-family':['MFH','AB'], 'Single-family':['SFH','TH']}
        for th,typos in dict_type_house.items():
            stats_type = {}
            for d,ratio_typo in stats.items():
                stats_type[d] = sum([ratio_typo.get(t) for t in typos])
            # stats_typo = {e:v.get(k) for e,v in stats.items()}
            add_departement_map(stats_type,figs_folder=figs_folder,save='{}_ratio_dep'.format(th),cbar_label='{} ratio by department'.format(th))
    
    # Carte des stats en multithreading # trop long, saturation en mémoire à comprendre
    if False:
        from multiprocessing import Pool, cpu_count
        
        departements = France().departements
        
        pool = Pool(processes=cpu_count()//2, maxtasksperchild=1)  # set the processes to half of total, maxetc à tester
        # departements = list(tqdm.tqdm(pool.imap(stats_typologies_dep, departements), total=len(departements)))
        departements = pool.map_async(stats_typologies_dep, departements)
        pool.close()
        pool.join()
    
        stats = {}
        for dep in tqdm.tqdm(departements):
            
            total_dep_hh = sum(dep.typologies_households_number.values())
            number_hh_typo_categories = {'SFH':0,'TH':0,'MFH':0,'AB':0}
            for k in number_hh_typo_categories.keys():
                for ty,nb in dep.typologies_households_number.items():
                    if k in ty:
                        number_hh_typo_categories[k] += nb
            percent_hh_typo_categories = {k:v/total_dep_hh for k,v in number_hh_typo_categories.items()}
            stats[dep] = percent_hh_typo_categories
        
        dict_type_house = {'Multi-family':['MFH','AB'], 'Single-family':['SFH','TH']}
        for th,typos in dict_type_house.items():
            stats_type = {}
            for d,ratio_typo in stats.items():
                stats_type[d] = sum([ratio_typo.get(t) for t in typos])
            # stats_typo = {e:v.get(k) for e,v in stats.items()}
            add_departement_map(stats_type,figs_folder=figs_folder,save='{}_ratio_dep'.format(th),cbar_label='{} ratio by department'.format(th))
            
        
            
        
    
    #%% Carte des alentours d'un bâtiment (pour vérification)
    if False:
        bg_id = 'bdnb-bg-YMYY-W2JJ-CKSG' # FR.N.SFH.01.Gen
        bg_id = 'bdnb-bg-YSXY-UB8R-38W2'
        bg_id = 'bdnb-bg-ZJZ7-6DEV-UX2U'
        
        bg_id = 'bdnb-bg-WYJM-98E5-ECML' # FR.N.TH.06.Gen # CAF du 13e hmm
        bg_id = 'bdnb-bg-TABY-KG2C-PL1Q' # hotel lacordaire - cercle natinal des armées
        bg_id = 'bdnb-bg-YXVC-FQCT-P4A1'
        neighbourhood_map(bg_id, figs_folder)
        
        _, _, bgc = get_bdnb(dep=Departement(75).code,external_disk=external_disk_connection)
        bgc = bgc[bgc.batiment_groupe_id==bg_id].compute()
        print(bgc.dpe_mix_arrete_l_orientation_mur_exterieur, bgc.ffo_bat_nb_log)
        
        
    #%% Tests de la classe Typology
    if False:
        code = 'FR.N.TH.06.Gen'
        typo = Typology(code)
        print(typo)
        
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__=='__main__':
    main()