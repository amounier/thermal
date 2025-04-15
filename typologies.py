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
import matplotlib

from bdnb_opener import get_bdnb, neighbourhood_map
from administrative import France, Departement, draw_departement_map


dict_angle_orientation = {i*45:o for i,o in enumerate(['N','NE','E','SE','S','SW','W','NW'])}
dict_orientation_angle = {v:k for k,v in dict_angle_orientation.items()}

#  completer la classe typology
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
        https://perso.univ-lemans.fr/~bcasta/Cours%20L3%20Echanges%20thermiques/Thermique%20de%20l'inge%c3%8c%c2%81nieur,%20Annexes.pdf
        http://pigo.free.fr/_media/re-caracteristiques-thermiques.pdf
        + parpaing : Sassine et al 2022 (10.1007/s42452-020-03881-x) + https://www.professionnels-isolation.com/performances-energetiques-parpaing/
        
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
    def __init__(self,code,level='initial'):
        """
        Initialisation de la typologie à partir des paramètres TABULA

        Parameters
        ----------
        code : str
            Code TABULA.
        level : str, optional
            Niveau d'isolation, parmi : 'initial', 'standard', 'advanced'. The default is 'initial'.

        Returns
        -------
        None.

        """
        self.code = code
        self.insulation_level = level
        
        params = open_tabula_typologies().loc[self.code].to_dict()
        # self.desc = params.get('building_name')
        self.type = params.get('building_type')
        # print(params.keys())
        
        # orientation des murs
        self.w0_orientation = params.get('building_orientation')
        self.w1_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w0_orientation)+90)%360)
        self.w2_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w1_orientation)+90)%360)
        self.w3_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w2_orientation)+90)%360)
        
        # paramètres géométriques
        self.surface = params.get('building_surface')
        self.levels = params.get('building_levels')
        self.ground_surface = self.surface/self.levels
        self.roof_ratio = params.get('building_roof_ratio')
        self.roof_surface = self.ground_surface * self.roof_ratio
        
        # paramètres d'habitation
        self.rdc = bool(params.get('building_rdc_level'))
        self.households = int(params.get('building_households'))
        
        # caractéristiques inférieures et supérieurs 
        self.basement = bool(params.get('building_basement'))
        self.converted_attic = bool(params.get('building_converted_attic'))
        
        # je considère les RDC LNC comme des caves
        if self.type in ['SFH'] and not self.rdc:
            # self.levels = self.levels - 0.5
            self.basement = True
        
        if self.type in ['TH'] and not self.rdc:
            # self.levels = self.levels + 1 
            self.basement = True
            
        # caractérisation de la mitoyenneté
        self.nb_non_detached = params.get('building_semi_detached')
        self.w0_adiabatic = False
        self.w1_adiabatic = False
        self.w2_adiabatic = False
        self.w3_adiabatic = False
        self.floor_adiabatic = False
        self.ceiling_adiabatic = False
        
        if self.nb_non_detached == 1:
            self.w1_adiabatic = True
        if self.nb_non_detached == 2:
            self.w1_adiabatic = True
            self.w3_adiabatic = True
        if self.nb_non_detached == 3:
            self.w1_adiabatic = True
            self.w2_adiabatic = True
            self.w3_adiabatic = True
            
        if self.type in ['AB','MFH'] and not self.rdc:
            self.floor_adiabatic = True
            self.levels = self.levels - 1
            
        
        
        self.height = params.get('building_floor_height')
        self.volume = self.surface*self.height
        self.form_factor = params.get('building_form_factor')
        self.w0_length = np.sqrt(self.ground_surface/self.form_factor)
        self.w1_length = self.w0_length*self.form_factor
        self.perimeter = 2*(self.w0_length + self.w1_length)
        
        # caractéristiques des ventilations et infiltrations
        self.air_infiltration = params.get('{}_air_infiltration'.format(level))
         
        ventilation_translator_dict = {'natural':'natural', 
                                       'SF indiv':'Individual MV',
                                       'SF collective':'Collective MV',
                                       'SF hygro B indiv':'Individual DCV',
                                       'SF hygro B collective':'Collective DCV',
                                       'DF indiv':'Individual HRV',
                                       'DF collective':'Collective HRV'}
        
        self.ventilation = ventilation_translator_dict.get(params.get('{}_ventilation'.format(level)))
        self.ventilation_efficiency = self.get_ventilation_efficiency()
        
        
        ventilation_night_over_list = ['Individual HRV','Collective HRV']
        if self.ventilation in ventilation_night_over_list:
            self.ventilation_night_over = True
        else:
            self.ventilation_night_over = False
        
        # caractéristiques du toit
        self.roof_color = params.get('{}_roof_color'.format(level))
        # self.roof_U = params.get('{}_roof_U'.format(level))
        # self.ceiling_U = params.get('{}_ceiling_U'.format(level))
        self.roof_U = params.get('{}_Uph'.format(level))
        self.ceiling_U = params.get('{}_Uph'.format(level))
        self.ceiling_supplementary_insulation_thickness = 0
        self.ceiling_supplementary_insulation_material = Material('laine minerale')
        
        self.ceiling_structure_material = Material(params.get('building_ceiling_structure_material'))
        self.ceiling_structure_thickness = params.get('building_ceiling_structure_thickness')
        
        # caractéristiques des vitrages
        self.windows_frame_ratio = 0.3 # valeur tabula
        self.h_windows_surface = params.get('building_horizontal_windows_surface')*(1-self.windows_frame_ratio)
        self.w0_windows_surface = (params.get('building_wall0_windows_surface')+0.001)*(1-self.windows_frame_ratio)
        self.w1_windows_surface = (params.get('building_wall1_windows_surface')+0.001)*(1-self.windows_frame_ratio)
        self.w2_windows_surface = (params.get('building_wall2_windows_surface')+0.001)*(1-self.windows_frame_ratio)
        self.w3_windows_surface = (params.get('building_wall3_windows_surface')+0.001)*(1-self.windows_frame_ratio)
        self.windows_U = params.get('{}_Uw'.format(level))
        self.windows_Ug = None
        
        self.windows_height = 1.5 #m
        self.solar_shader_height = 0.3 #m
        self.solar_shader_length = 0. #m
        
        # caractérisation de la porte
        self.door_U = params.get('{}_door_U'.format(level))
        self.door_surface = 2 #m2
        
        # caractéristiques des murs
        # définir self.walls et mettre toutes variables dans ce dictionnaire (ou pas)
        self.w0_surface = self.w0_length*self.height*self.levels - self.w0_windows_surface
        self.w1_surface = self.w1_length*self.height*self.levels - self.w1_windows_surface
        self.w2_surface = self.w0_length*self.height*self.levels - self.w2_windows_surface
        self.w3_surface = self.w1_length*self.height*self.levels - self.w3_windows_surface
        
        self.w0_color = params.get('{}_walls_color'.format(level))
        self.w1_color = params.get('{}_walls_color'.format(level))
        self.w2_color = params.get('{}_walls_color'.format(level))
        self.w3_color = params.get('{}_walls_color'.format(level))
        
        self.w0_structure_material = Material(params.get('building_wall0_structure_material'))
        self.w1_structure_material = Material(params.get('building_wall1_structure_material'))
        self.w2_structure_material = Material(params.get('building_wall2_structure_material'))
        self.w3_structure_material = Material(params.get('building_wall3_structure_material'))
        self.w0_structure_thickness = params.get('building_wall0_structure_thickness')
        self.w1_structure_thickness = params.get('building_wall1_structure_thickness')
        self.w2_structure_thickness = params.get('building_wall2_structure_thickness')
        self.w3_structure_thickness = params.get('building_wall3_structure_thickness')
        
        self.w0_insulation_material = Material(params.get('{}_wall0_insulation_material'.format(level)))
        self.w1_insulation_material = Material(params.get('{}_wall1_insulation_material'.format(level)))
        self.w2_insulation_material = Material(params.get('{}_wall2_insulation_material'.format(level)))
        self.w3_insulation_material = Material(params.get('{}_wall3_insulation_material'.format(level)))
        self.w0_insulation_thickness = params.get('{}_wall0_insulation_thickness'.format(level))
        self.w1_insulation_thickness = params.get('{}_wall1_insulation_thickness'.format(level))
        self.w2_insulation_thickness = params.get('{}_wall2_insulation_thickness'.format(level))
        self.w3_insulation_thickness = params.get('{}_wall3_insulation_thickness'.format(level))
        self.w0_insulation_position = params.get('{}_wall0_insulation_position'.format(level))
        self.w1_insulation_position = params.get('{}_wall1_insulation_position'.format(level))
        self.w2_insulation_position = params.get('{}_wall2_insulation_position'.format(level))
        self.w3_insulation_position = params.get('{}_wall3_insulation_position'.format(level))
        
        # caractéristiques du sol
        # cf Rantala and Leivo 2006 et Skotnicova and Lausova (2016).
        # cf Thbat parois opaques p21
        #  clarifier dans le cas 2D
        # 0.3 + floor height si cave 
        
        self.floor_ground_depth = 0.3
        if self.basement:
            self.floor_ground_depth += 3
        
        self.floor_ground_distance = self.get_floor_ground_distance()
        self.ground_depth = self.floor_ground_depth + self.floor_ground_distance 
        self.ground_section = self.perimeter * self.ground_depth
        self.ground_volume = self.ground_surface * self.ground_depth
        self.floor_structure_material = Material(params.get('building_floor_structure_material'))
        self.floor_structure_thickness = params.get('building_floor_structure_thickness')
            
        self.floor_insulation_material = Material(params.get('{}_floor_insulation_material'.format(level)))
        self.floor_insulation_thickness = params.get('{}_floor_insulation_thickness'.format(level))
        self.floor_insulation_position = params.get('{}_floor_insulation_position'.format(level))
        
        # ajout des "défauts de rénovation" de la base TABULA (pour les niveaux standard et advanced)
        self.defects_U = params.get('{}_retrofit_defects'.format(level))
        
        if self.defects_U > 0.:
            self.roof_U += self.defects_U
            self.ceiling_U += self.defects_U
            self.w0_insulation_thickness = max(0,(self.w0_insulation_material.thermal_conductivity*self.w0_insulation_thickness)/(self.w0_insulation_thickness*self.defects_U + self.w0_insulation_material.thermal_conductivity))
            self.w1_insulation_thickness = max(0,(self.w1_insulation_material.thermal_conductivity*self.w1_insulation_thickness)/(self.w1_insulation_thickness*self.defects_U + self.w1_insulation_material.thermal_conductivity))
            self.w2_insulation_thickness = max(0,(self.w2_insulation_material.thermal_conductivity*self.w2_insulation_thickness)/(self.w2_insulation_thickness*self.defects_U + self.w2_insulation_material.thermal_conductivity))
            self.w3_insulation_thickness = max(0,(self.w3_insulation_material.thermal_conductivity*self.w3_insulation_thickness)/(self.w3_insulation_thickness*self.defects_U + self.w3_insulation_material.thermal_conductivity))
            self.floor_insulation_thickness = max(0,(self.floor_insulation_material.thermal_conductivity*self.floor_insulation_thickness)/(self.floor_insulation_thickness*self.defects_U + self.floor_insulation_material.thermal_conductivity))
        
        # puissance maximale des émetteurs
        self.heater_maximum_power = 10000*self.households # W
        self.cooler_maximum_power = 10000*self.households # W
        
        # besoins de chauffage TABULA
        self.tabula_heating_needs = params.get('{}_heating_needs_tabula'.format(level)) # kWh/m2/yr
        
        # comparaison des valeurs U
        self.tabula_Uph = params.get('{}_Uph'.format(level)) # W/m2/K
        self.tabula_Umur = params.get('{}_Umur'.format(level)) # W/m2/K
        self.tabula_Uw = params.get('{}_Uw'.format(level)) # W/m2/K
        self.tabula_Upb = params.get('{}_Upb'.format(level)) # W/m2/K
        
        # valeurs U calculées lors de la résolution du modèle thermique
        self.modelled_Uph = None
        self.modelled_Upb = None
        self.modelled_Umur = None
        self.modelled_Uw = None
        

    def __str__(self):
        return self.code
    
    def update_orientation(self):
        self.w1_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w0_orientation)+90)%360)
        self.w2_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w1_orientation)+90)%360)
        self.w3_orientation = dict_angle_orientation.get((dict_orientation_angle.get(self.w2_orientation)+90)%360)

    # TODO : créer une fonction qui update toutes les variables (protégées ?)
    
    def get_floor_ground_distance(self,nb_discretize=50):
        X,Y = np.linspace(0, self.w0_length, nb_discretize), np.linspace(0, self.w1_length, nb_discretize)
        distance = np.zeros((len(X),len(Y)))
        for i,x in enumerate(X):
            for j,y in enumerate(Y):
                distance[i,j] = min(min(x,self.w0_length-x),min(y,self.w1_length-y))
                
        horizontal_distance = np.mean(distance)
        floor_ground_distance = np.sqrt(horizontal_distance**2 + self.floor_ground_depth**2)
        return floor_ground_distance
    
    def get_ventilation_efficiency(self):
        ventilation_efficiency_dict = {'natural':0,
                                       'Individual MV':0,
                                       'Collective MV':0.1,
                                       'Individual DCV':0.2,
                                       'Collective DCV':0.3,
                                       'Individual HRV':0.7,
                                       'Collective HRV':0.8}
        
        return ventilation_efficiency_dict.get(self.ventilation)
    

# Peut-etre à bouger dans un nouveau fichier identification (ou pas, à voir)
# : stocker les statistiques de typologies dans les départements (dans Departement)
# : vérifier les données TABULA, caractériser les isolations


# def identify_typologies(dep, external_disk=True, verbose=True):
#     if verbose:
#         print('Opening BDNB ({})...'.format(dep))
    
#     _, _, bgc = get_bdnb(dep=dep,external_disk=external_disk)
#     # : à refaire de manière optimisée

#     bgc = bgc[bgc.ffo_bat_nb_log>=1][['building_type_id',
#                                       'building_type',
#                                       'building_type_construction_year_start',
#                                       'building_type_construction_year_end',
#                                       'building_type_min_hh',
#                                       'building_type_max_hh',]]
#     bgc = bgc.compute()
    
#     # rel = rel.compute()
#     # rel = rel.set_index('batiment_groupe_id')
    
#     dict_list_bg_id = {}
#     typologies = open_tabula_typologies()
#     for i in range(len(typologies)):
#         # caractéristiques de typologies
#         typo_id, typo_cat, typo_cys, typo_cye, typo_min_hh, typo_max_hh = typologies.iloc[i][['building_type_id',
#                                                                                               'building_type',
#                                                                                               'building_type_construction_year_start',
#                                                                                               'building_type_construction_year_end',
#                                                                                               'building_type_min_hh',
#                                                                                               'building_type_max_hh',]]
        
#         # filtre de la BDNB
#         bgc_typo = bgc[(bgc.ffo_bat_nb_log>typo_min_hh)&
#                        (bgc.ffo_bat_nb_log<=typo_max_hh)&
#                        (bgc.ffo_bat_annee_construction>=typo_cys)&
#                        (bgc.ffo_bat_annee_construction<=typo_cye)]
        
#         # répartition des maisons individuelles par la liste des orientations des murs ext
#         # à faire avec les simulations DPE quand disponibles
#         if typo_cat == 'TH':
#             list_orientation = ['(4:est,nord,ouest,sud)', '(5:est,horizontal,nord,ouest,sud)']
#             bgc_typo = bgc_typo[(~bgc_typo.dpe_mix_arrete_l_orientation_mur_exterieur.isin(list_orientation))|(bgc_typo.dpe_mix_arrete_l_orientation_mur_exterieur.isna())]
            
#         elif typo_cat == 'SFH':
#             list_orientation = ['(4:est,nord,ouest,sud)', '(5:est,horizontal,nord,ouest,sud)']
#             bgc_typo = bgc_typo[(bgc_typo.dpe_mix_arrete_l_orientation_mur_exterieur.isin(list_orientation))&(~bgc_typo.dpe_mix_arrete_l_orientation_mur_exterieur.isna())]
        
#         # number_typo = bgc_typo.shape[0].compute()
#         dict_list_bg_id[typo_id] = bgc_typo.batiment_groupe_id.to_list()
        
#     # compilation du nombre de logements
#     bgc = bgc.set_index('batiment_groupe_id')
#     dict_number_hh = {}
#     for k,v in dict_list_bg_id.items():
#         nb_hh = 0
#         for bg_id in v:
#             nb_hh += bgc.loc[bg_id].ffo_bat_nb_log
#         dict_number_hh[k] = nb_hh
    
    
#     del bgc # vérifier que ça marche bien
#     del bgc_typo
#     return dict_list_bg_id, dict_number_hh




# def stats_typologies_dep(dep,external_disk=True):
#     # très temporaire, à refaire 
#     _, dep.typologies_households_number = identify_typologies(dep=dep.code,external_disk=external_disk,verbose=False)
    
#     return dep

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
        if False:
            # TODO à recoder dans statistics_building
            # typologies = open_tabula_typologies()
            
            departement = Departement(75)
            # departement.typologies_batiments_groupe, departement.typologies_households_number = identify_typologies(dep=departement.code,external_disk=external_disk_connection)
            
            # total_dep_hh = sum(departement.typologies_households_number.values())
            # number_hh_typo_categories = {'SFH':0,'TH':0,'MFH':0,'AB':0}
            # for k in number_hh_typo_categories.keys():
            #     for ty,nb in departement.typologies_households_number.items():
            #         if k in ty:
            #             number_hh_typo_categories[k] += nb
            # percent_hh_typo_categories = {k:v/total_dep_hh for k,v in number_hh_typo_categories.items()}
            
            # for k in number_hh_typo_categories.keys():
            #     draw_departement_map({departement:percent_hh_typo_categories.get(k)},figs_folder=figs_folder,cbar_label='{} ratio by department'.format(k))
        
        # Téléchargement des départements
        if False:
            if external_disk_connection:
                print('Téléchargement de la BDNB sur disque local.')
                # list_dep_code = ['2A']
                for d in tqdm.tqdm(France().departements):
                    get_bdnb(d.code,external_disk=external_disk_connection)
        
        # Carte des stats de type de catégories (SFH,TH,MFH,AB) par départements 
        # TODO à recoder dans statistics_building
        if False:
            stats = {}
            
            # # list_dep_code = ['75','2A']
            # for dep in tqdm.tqdm(France().departements):
                
            #     _, dep.typologies_households_number = identify_typologies(dep=dep.code,external_disk=external_disk_connection,verbose=False)
                
            #     total_dep_hh = sum(dep.typologies_households_number.values())
            #     number_hh_typo_categories = {'SFH':0,'TH':0,'MFH':0,'AB':0}
            #     for k in number_hh_typo_categories.keys():
            #         for ty,nb in dep.typologies_households_number.items():
            #             if k in ty:
            #                 number_hh_typo_categories[k] += nb
            #     percent_hh_typo_categories = {k:v/total_dep_hh for k,v in number_hh_typo_categories.items()}
            #     stats[dep] = percent_hh_typo_categories
                
            # dict_type_house = {'Multi-family':['MFH','AB'], 'Single-family':['SFH','TH']}
            # for th,typos in dict_type_house.items():
            #     stats_type = {}
            #     for d,ratio_typo in stats.items():
            #         stats_type[d] = sum([ratio_typo.get(t) for t in typos])
            #     # stats_typo = {e:v.get(k) for e,v in stats.items()}
            #     draw_departement_map(stats_type,figs_folder=figs_folder,save='{}_ratio_dep'.format(th),cbar_label='{} ratio by department'.format(th))
        
        # Carte des stats en multithreading # trop long, saturation en mémoire à comprendre
        # TODO à faire dans statistics_building
        if False:
            pass
            # from multiprocessing import Pool, cpu_count
            
            # departements = France().departements
            
            # pool = Pool(processes=cpu_count()//2, maxtasksperchild=1)  # set the processes to half of total, maxetc à tester
            # # departements = list(tqdm.tqdm(pool.imap(stats_typologies_dep, departements), total=len(departements)))
            # departements = pool.map_async(stats_typologies_dep, departements)
            # pool.close()
            # pool.join()
        
            # stats = {}
            # for dep in tqdm.tqdm(departements):
                
            #     total_dep_hh = sum(dep.typologies_households_number.values())
            #     number_hh_typo_categories = {'SFH':0,'TH':0,'MFH':0,'AB':0}
            #     for k in number_hh_typo_categories.keys():
            #         for ty,nb in dep.typologies_households_number.items():
            #             if k in ty:
            #                 number_hh_typo_categories[k] += nb
            #     percent_hh_typo_categories = {k:v/total_dep_hh for k,v in number_hh_typo_categories.items()}
            #     stats[dep] = percent_hh_typo_categories
            
            # dict_type_house = {'Multi-family':['MFH','AB'], 'Single-family':['SFH','TH']}
            # for th,typos in dict_type_house.items():
            #     stats_type = {}
            #     for d,ratio_typo in stats.items():
            #         stats_type[d] = sum([ratio_typo.get(t) for t in typos])
            #     # stats_typo = {e:v.get(k) for e,v in stats.items()}
            #     draw_departement_map(stats_type,figs_folder=figs_folder,save='{}_ratio_dep'.format(th),cbar_label='{} ratio by department'.format(th))
                
        pass
            
    
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
        code = 'FR.N.TH.03.Gen'
        typo = Typology(code)
        print(typo)
        
    #%% Étude de la distance au bord de plaque
    if False:
    
        code = 'FR.N.SFH.01.Gen'
        typo = Typology(code)
        
        print(typo.floor_ground_distance)
        
        L0 = typo.w0_length
        L1 = typo.w1_length
        
        X,Y = np.linspace(0, L0, 50), np.linspace(0, L1, 50)
        distance = np.zeros((len(X),len(Y)))
        for i,x in enumerate(X):
            for j,y in enumerate(Y):
                distance[i,j] = min(min(x,L0-x),min(y,L1-y))
        
        X, Y = np.meshgrid(X, Y)
        
        fig, ax = plt.subplots(dpi=300,figsize=(5,5*typo.form_factor))
        cs = ax.contourf(X, Y, distance.T)
        # cbar = fig.colorbar(cs)
        ax.set_xlabel('W$_0$ length (m)')
        ax.set_ylabel('W$_1$ length (m)')
        ax.axis('equal')
        
        cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
        norm = matplotlib.colors.Normalize(vmin=np.min(distance), vmax=np.max(distance))
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.colormaps.get_cmap('viridis'))
        
        cbar_label_var = 'Distance from edge (m) - Mean value = {:.1f}m'.format(np.mean(distance))
        # _ = plt.colorbar(cs, cax=cbar_ax, label=cbar_label_var, extend='neither', extendfrac=0.02)
        _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend='neither', extendfrac=0.02)
        
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('distance_from_edge')),bbox_inches='tight')
        
        plt.show()
        
    #%% Comparaisons entre typologies 
    if False:
        building_type = 'SFH'
        # building_type = 'TH'
        # building_type = 'MFH'
        # building_type = 'AB'
        
        heating_needs = {}
        for i in range(1,11):
            code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)

            for level in ['initial','standard','advanced']:
                typo = Typology(code,level)
                
                heating_needs[(code,level)] = typo.tabula_heating_needs
            
        # print(heating_needs)
        
        fig,ax = plt.subplots(figsize=(15,5),dpi=300)
        for i in range(1,11):
            j = i*7
            X = [j,j+2,j+4]
            Y = [heating_needs.get(('FR.N.{}.{:02d}.Gen'.format(building_type,i),e)) for e in ['initial','standard','advanced']]
            
            if i == 1:
                ax.plot(X,Y,color='k',ls=':',marker='o',label='TABULA')
            else:
                ax.plot(X,Y,color='k',ls=':',marker='o')
                
        ax.set_ylim(bottom=0.)
        ax.set_ylabel('Heating needs (kWh.m$^{-2}$.yr$^{-1}$)')
        ax.legend()
        ax.set_xticks([(i*7)+2 for i in range(1,11)],['{}.{:02d}'.format(building_type,i) for i in range(1,11)])
        
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('{}_TABULA_consumption_tabula_only'.format(building_type))),bbox_inches='tight')
        
    # Comparaison des valeurs U
    if False:
        building_type = 'SFH'
        element = 'Umur'
        
        building_type = 'TH'
        # building_type = 'MFH'
        # building_type = 'AB'
        
        U_values_dict = {}
        for i in range(1,11):
            code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)

            for level in ['initial','standard','advanced']:
                typo = Typology(code,level)
                
                tabula_element_dict = {'Umur':typo.tabula_Umur,
                                       'Uph':typo.tabula_Uph,
                                       'Upb':typo.tabula_Upb,
                                       'Uw':typo.tabula_Uw}
                
                U_values_dict[(code,level)] = tabula_element_dict.get(element)
            
        # print(U_values_dict)
        
        fig,ax = plt.subplots(figsize=(15,5),dpi=300)
        for i in range(1,11):
            j = i*7
            X = [j,j+2,j+4]
            Y = [U_values_dict.get(('FR.N.{}.{:02d}.Gen'.format(building_type,i),e)) for e in ['initial','standard','advanced']]
            
            if i == 1:
                ax.plot(X,Y,color='k',ls=':',marker='o',label='TABULA')
            else:
                ax.plot(X,Y,color='k',ls=':',marker='o')
        
        element_label_dict = {'Umur':'Walls U-value (W.m$^{-2}$.K$^{-1}$)',
                              'Uph':'Roof U-value (W.m$^{-2}$.K$^{-1}$)',
                              'Upb':'Floor U-value (W.m$^{-2}$.K$^{-1}$)',
                              'Uw':'Windows U-value (W.m$^{-2}$.K$^{-1}$)'}
        
        ax.set_ylim(bottom=0.)
        ax.set_ylabel(element_label_dict.get(element))
        ax.legend()
        ax.set_xticks([(i*7)+2 for i in range(1,11)],['{}.{:02d}'.format(building_type,i) for i in range(1,11)])
        
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('{}_TABULA_Umur_tabula_only'.format(building_type))),bbox_inches='tight')
                
    #%% Statistiques TABULA
    if True:

        building_type ='AB'
        formated_dict_data = ['Category','Variable','Unit'] + ['{}.{:02d}'.format(building_type,i) for i in range(1,11)]#,'standard','advanced']]
        data = pd.DataFrame().from_dict({e:[] for e in formated_dict_data})
        
        for i in range(1,11):
            for level in ['initial']:##,'standard','advanced']:
                
                code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
                typo = Typology(code,level)
                
                dict_data = [['Building','Households surface','\\SI{}{\\square\\meter}',typo.surface],
                             ['Building','Households levels','-',typo.levels],
                             ['Building','Basement','boolean',typo.basement],
                             ['Building','Converted attic','boolean',typo.converted_attic],
                             ['Building','\\# semi-detached','-',typo.nb_non_detached],
                             ['Building','Building orientation','-',typo.w0_orientation],
                             ['Building','Windows surface','\\SI{}{\\square\\meter}',typo.w0_windows_surface+typo.w1_windows_surface+typo.w2_windows_surface+typo.w3_windows_surface],
                             ['Insulation','Wall insulation thickness','\\SI{}{\\centi\\meter}',typo.w0_insulation_thickness*100],
                             ['Insulation','Floor insulation thickness','\\SI{}{\\centi\\meter}',typo.floor_insulation_thickness*100],
                             ['Insulation','Windows U-value','\\SI{}{\\watt\\per\\square\\meter\\kelvin}',typo.windows_U],
                             ['Insulation','Roof U-value','\\SI{}{\\watt\\per\\square\\meter\\kelvin}',typo.ceiling_U],
                             ['Energy needs','Heating needs','\\SI{}{\\kilo\\watthour\\per\\square\\meter\\year}',typo.tabula_heating_needs]
                             ]
                
                if i == 1 and level =='initial':
                    data['Category'] = [e[0] for e in dict_data]
                    data['Variable'] = [e[1] for e in dict_data]
                    data['Unit'] = [e[2] for e in dict_data]
                    
                data['{}.{:02d}'.format(building_type,i)] = [e[3] for e in dict_data]
                
        # data = data.set_index(['Category','Variable','Unit'])
        printer_data = data.to_latex(float_format='%.1f',index=False)
        # printer_data = data.split()
        printer_data = printer_data.replace("True",'\\checkmark').replace('False','')
        print(printer_data)
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__=='__main__':
    main()