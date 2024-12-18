#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:28:45 2024

@author: amounier
"""

import time
from datetime import date, datetime
import os
import pandas as pd
import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# import warnings
import pickle

from utils import plot_timeserie
from meteorology import get_init_ground_temperature, get_historical_weather_data
from typologies import Typology, dict_orientation_angle
from behaviour import Behaviour
from thermal_sensitivity import plot_thermal_sensitivity
from future_meteorology import get_projected_weather_data
from administrative import Departement, Climat, France, draw_climat_map


AIR_THERMAL_CAPACITY = 1000 # J/(kg.K)
AIR_DENSITY = 1.2 # kg/m3

GROUND_THERMAL_CAPACITY = 1000 # J/(kg.K)
GROUND_DENSITY = 2500 # kg/m3
GROUND_THERMAL_CONDUCTIVITY = 1.5 # W/(m.K)


def get_P_heater(Ti, Ti_min, Pmax, method='all_or_nothing'):
    """
    Renvoie la puissance des équipemetns de chauffage

    Parameters
    ----------
    Ti : float
        DESCRIPTION.
    Ti_min : float
        DESCRIPTION.
    Pmax : float
        DESCRIPTION.
    method : str, optional
        DESCRIPTION. The default is 'all_or_nothing'.

    Returns
    -------
    P_heater : float
        DESCRIPTION.

    """
    if method == 'all_or_nothing':
        if Ti < Ti_min:
            P_heater = Pmax
        else:
            P_heater = 0
            
    elif method == 'linear_tolerance':
        tolerance = 1 #°C
        if Ti < Ti_min - tolerance:
            P_heater = Pmax
        elif Ti >= Ti_min + tolerance:
            P_heater = 0
        else:
            P_heater = (-(Ti-(Ti_min-tolerance))/(2*tolerance)+1)*Pmax
    return P_heater


def get_P_cooler(Ti, Ti_max, Pmax, method='all_or_nothing'):
    """
    Renvoie la puissance des équipements de refroidissement

    Parameters
    ----------
    Ti : float
        DESCRIPTION.
    Ti_max : float
        DESCRIPTION.
    Pmax : float
        DESCRIPTION.
    method : str, optional
        DESCRIPTION. The default is 'all_or_nothing'.

    Returns
    -------
    P_cooler : float
        DESCRIPTION.

    """
    if method == 'all_or_nothing':
        if Ti > Ti_max:
            P_cooler = Pmax
        else:
            P_cooler = 0
            
    elif method == 'linear_tolerance':
         tolerance = 1 #°C
         if Ti > Ti_max + tolerance:
             P_cooler = Pmax
         elif Ti <= Ti_max - tolerance:
             P_cooler = 0
         else:
             P_cooler = ((Ti-(Ti_max-tolerance))/(2*tolerance))*Pmax
    
    # flux de refroidissement négatif
    P_cooler = - P_cooler
    return P_cooler


def get_P_vmeca(Ti,Te,P_heater,P_cooler,typology):
    U_air = get_ventilation_minimum_air_flow(typology) * AIR_THERMAL_CAPACITY * (1-typology.ventilation_efficiency)
    
    # bypass et surventilation nocturne
    f_sv = 1
    if typology.ventilation_efficiency > 0:
        if P_heater == 0:
            f_sv = f_sv/(1-typology.ventilation_efficiency)
        if P_cooler > 0:
            f_sv = f_sv * 2
            
    P_vmeca = U_air * (Te-Ti)
    return P_vmeca


def get_P_vnat(Ti,Te,typology,behaviour):
    # TODO : comportements de ventilation manuelles par ouverture des fenêtres
    P_vnat = 0
    return P_vnat


def dot3(A,B,C):
    return np.dot(A,np.dot(B,C))


def get_ventilation_minimum_air_flow(typology, plot=False, figs_folder=None):
    """
    https://www.legifrance.gouv.fr/loda/id/JORFTEXT000000862344/2021-01-08/
    Et hypothèse de 20m2 par pièce

    """
    ventilation_minimum_air_flow = typology.surface * 0.8 + 24.3 # m3/h
    ventilation_minimum_air_flow = AIR_DENSITY*ventilation_minimum_air_flow/3600 # kg/s
    
    # Affichage des données légales et de la modélisation associée
    if plot or figs_folder is not None:
        nb_rooms = [1,2,3,4,5,6,7,]# m3/h
        surface = [nbr*20 for nbr in nb_rooms]
        minimal_air_flow = [35,60,75,90,105,120,135,]
        
        a,b = np.polyfit(surface,minimal_air_flow,deg=1)
        minimal_air_flow_hat = [a*s+b for s in surface]
        r2 = r2_score(minimal_air_flow, minimal_air_flow_hat)

        label_hat = 'Linear fit (R$^2$='+'{:.2f})\n({:.2f} S + {:.1f})'.format(r2,a,b)
        
        fig,ax = plt.subplots(figsize=(5,5),dpi=300)
        ax.plot(surface,minimal_air_flow,ls='',marker='o',color='tab:blue',label='Legal data')
        ax.plot(surface,minimal_air_flow_hat,color='k',label=label_hat)
        ax.set_ylabel('Minimal ventilation air flow (m$^3$.h$^{-1}$)')
        ax.set_xlabel('Household surface (m$^2$)')
        ax.set_ylim(bottom=0)
        ax.legend()
        
        if figs_folder is not None:
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('minimal_ventilation_air_flow')),bbox_inches='tight')
        plt.show()
        
    return ventilation_minimum_air_flow

def get_infiltration_air_flow(typology):
    air_infiltration_dict = {'minimal': 0.05,
                             'low': 0.1,
                             'medium': 0.2,
                             'high': 0.5}
    
    air_infiltration = air_infiltration_dict.get(typology.air_infiltration,typology.air_infiltration)
    air_infiltration = air_infiltration*typology.volume # m3/h
    air_infiltration = AIR_DENSITY*air_infiltration/3600 # kg/s
    
    return air_infiltration


def compute_R_air(typology):
    # total_air_flow = get_infiltration_air_flow(typology) + get_ventilation_minimum_air_flow(typology)
    U_air = get_infiltration_air_flow(typology) * AIR_THERMAL_CAPACITY + get_ventilation_minimum_air_flow(typology) * AIR_THERMAL_CAPACITY * (1-typology.ventilation_efficiency)
    R_air = 1/U_air
    return R_air


def compute_R_inf(typology):
    U_air = get_infiltration_air_flow(typology) * AIR_THERMAL_CAPACITY
    R_inf = 1/U_air
    return R_inf


def get_external_convection_heat_transfer(wind_speed=5,method='th-bat',plot=False,figs_folder=None):
    """
    Supposition d'un vent moyen de 5m/s, à corriger avec des données météo ?

    """
    if method=='wiki':
        # https://fr.wikipedia.org/wiki/Coefficient_de_convection_thermique
        # pas trouvé de source :(
        h = 10.45 - wind_speed + 10*np.sqrt(wind_speed) # W/(m2.K)
    
    elif method=='th-bat':
        # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf p15
        h = 4 + 4*wind_speed # W/(m2.K)
    
    if plot:
        wind_speed_list = np.linspace(0,10,50)
        h_wiki = [get_external_convection_heat_transfer(ws,method='wiki',plot=False) for ws in wind_speed_list]
        h_thbat = [get_external_convection_heat_transfer(ws,method='th-bat',plot=False) for ws in wind_speed_list]
        fig,ax = plt.subplots(dpi=300, figsize=(5,5))
        ax.plot(wind_speed_list,h_thbat,label='Th-bat method')
        ax.plot(wind_speed_list,h_wiki,label='Alternative method')
        ax.legend()
        ax.set_ylim(bottom=0.)
        ax.set_ylabel('External convection heat transfer (W.m$^{-2}$.K$^{-1}$)')
        ax.set_xlabel('Near surface wind speed (m.s$^{-1}$)')
        
        if figs_folder is not None:
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('minimal_ventilation_air_flow')),bbox_inches='tight')
        plt.show()
    return h

def get_external_radiation_heat_transfer(Tm=None, method='cste',plot=False,figs_folder=None):
    """
    À corriger dynamiquement ?

    """
    if method=='cste':
        # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf p15
        # pour une température moyenne de 10°C
        h = get_external_radiation_heat_transfer(Tm=10, method='th-bat')
    
    elif method=='th-bat':
        # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf p15
        epsilon = 0.9
        sigma_SB = 5.67e-8 # W/(m2.K4)
        hro = 4 * sigma_SB  * (Tm+273.15)**3
        h = epsilon * hro # W/(m2.K)
    
    return h


def compute_R_uih(typology):
    # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf
    hi = 2.5 # W/(m2.K)
    R_uih = 1/(hi * typology.roof_surface) # K/W
    return R_uih


def compute_R_ueh(typology):
    h_ext_conv = get_external_convection_heat_transfer(wind_speed=5)
    h_ext_rad = get_external_radiation_heat_transfer()
    h_ext = h_ext_conv + h_ext_rad
    R_ueh = 1/(h_ext * typology.roof_surface) # K/W
    return R_ueh

def compute_R_w0eh(typology):
    h_ext_conv = get_external_convection_heat_transfer()
    h_ext_rad = get_external_radiation_heat_transfer()
    h_ext = h_ext_conv + h_ext_rad
    R_w0eh = 1/(h_ext * typology.w0_surface) # K/W
    return R_w0eh

def compute_R_w1eh(typology):
    h_ext_conv = get_external_convection_heat_transfer()
    h_ext_rad = get_external_radiation_heat_transfer()
    h_ext = h_ext_conv + h_ext_rad
    R_w1eh = 1/(h_ext * typology.w1_surface) # K/W
    return R_w1eh

def compute_R_w2eh(typology):
    h_ext_conv = get_external_convection_heat_transfer()
    h_ext_rad = get_external_radiation_heat_transfer()
    h_ext = h_ext_conv + h_ext_rad
    R_w2eh = 1/(h_ext * typology.w2_surface) # K/W
    return R_w2eh

def compute_R_w3eh(typology):
    h_ext_conv = get_external_convection_heat_transfer()
    h_ext_rad = get_external_radiation_heat_transfer()
    h_ext = h_ext_conv + h_ext_rad
    R_w3eh = 1/(h_ext * typology.w3_surface) # K/W
    return R_w3eh


def compute_Rw0i(typology):
    R_w0wall = typology.w0_structure_thickness/(typology.w0_structure_material.thermal_conductivity * typology.w0_surface)
    
    if typology.w0_insulation_position == 'ITI':
        # penser à rajouter les ponts thermiques plus précisément
        R_w0iso_in = typology.w0_insulation_thickness/(typology.w0_insulation_material.thermal_conductivity * typology.w0_surface)
        R_w0iso_in = R_w0iso_in*0.73
    else: 
        R_w0iso_in = 0
    
    # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf
    hi = 2.5 # W/(m2.K)
    R_w0ih = 1/(hi * typology.w0_surface) # K/W
    
    R_w0i = R_w0wall/2 + R_w0iso_in + R_w0ih
    return R_w0i


def compute_Rw0e(typology):
    R_w0wall = typology.w0_structure_thickness/(typology.w0_structure_material.thermal_conductivity * typology.w0_surface)
    
    if typology.w0_insulation_position == 'ITE':
        R_w0iso_out = typology.w0_insulation_thickness/(typology.w0_insulation_material.thermal_conductivity * typology.w0_surface)
    else: 
        R_w0iso_out = 0
    
    R_w0e = R_w0wall/2 + R_w0iso_out 
    return R_w0e


def compute_Rw1i(typology):
    R_w1wall = typology.w1_structure_thickness/(typology.w1_structure_material.thermal_conductivity * typology.w1_surface)
    
    if typology.w1_insulation_position == 'ITI':
        # penser à rajouter les ponts thermiques plus précisément
        R_w1iso_in = typology.w1_insulation_thickness/(typology.w1_insulation_material.thermal_conductivity * typology.w1_surface)
        R_w1iso_in = R_w1iso_in*0.73
    else: 
        R_w1iso_in = 0
    
    # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf
    hi = 2.5 # W/(m2.K)
    R_w1ih = 1/(hi * typology.w1_surface) # K/W
    
    R_w1i = R_w1wall/2 + R_w1iso_in + R_w1ih
    return R_w1i

def compute_Rw1e(typology):
    R_w1wall = typology.w1_structure_thickness/(typology.w1_structure_material.thermal_conductivity * typology.w1_surface)
    
    if typology.w1_insulation_position == 'ITE':
        R_w1iso_out = typology.w1_insulation_thickness/(typology.w1_insulation_material.thermal_conductivity * typology.w1_surface)
    else: 
        R_w1iso_out = 0
    
    R_w1e = R_w1wall/2 + R_w1iso_out 
    return R_w1e

def compute_Rw2i(typology):
    R_w2wall = typology.w2_structure_thickness/(typology.w2_structure_material.thermal_conductivity * typology.w2_surface)
    
    if typology.w2_insulation_position == 'ITI':
        # penser à rajouter les ponts thermiques plus précisément
        R_w2iso_in = typology.w2_insulation_thickness/(typology.w2_insulation_material.thermal_conductivity * typology.w2_surface)
        R_w2iso_in = R_w2iso_in*0.73
    else: 
        R_w2iso_in = 0
    
    # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf
    hi = 2.5 # W/(m2.K)
    R_w2ih = 1/(hi * typology.w2_surface) # K/W
    
    R_w2i = R_w2wall/2 + R_w2iso_in + R_w2ih
    return R_w2i

def compute_Rw2e(typology):
    R_w2wall = typology.w2_structure_thickness/(typology.w2_structure_material.thermal_conductivity * typology.w2_surface)
    
    if typology.w2_insulation_position == 'ITE':
        R_w2iso_out = typology.w2_insulation_thickness/(typology.w2_insulation_material.thermal_conductivity * typology.w2_surface)
    else: 
        R_w2iso_out = 0
    
    R_w2e = R_w2wall/2 + R_w2iso_out 
    return R_w2e

def compute_Rw3i(typology):
    R_w3wall = typology.w3_structure_thickness/(typology.w3_structure_material.thermal_conductivity * typology.w3_surface)
    
    if typology.w3_insulation_position == 'ITI':
        # penser à rajouter les ponts thermiques plus précisément
        R_w3iso_in = typology.w3_insulation_thickness/(typology.w3_insulation_material.thermal_conductivity * typology.w3_surface)
        R_w3iso_in = R_w3iso_in*0.73
    else: 
        R_w3iso_in = 0
    
    # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf
    hi = 2.5 # W/(m2.K)
    R_w3ih = 1/(hi * typology.w3_surface) # K/W
    
    R_w3i = R_w3wall/2 + R_w3iso_in + R_w3ih
    return R_w3i

def compute_Rw3e(typology):
    R_w3wall = typology.w3_structure_thickness/(typology.w3_structure_material.thermal_conductivity * typology.w3_surface)
    
    if typology.w3_insulation_position == 'ITE':
        R_w3iso_out = typology.w3_insulation_thickness/(typology.w3_insulation_material.thermal_conductivity * typology.w3_surface)
    else: 
        R_w3iso_out = 0
    
    R_w3e = R_w3wall/2 + R_w3iso_out 
    return R_w3e

def compute_C_c(typology):
    volume_c = typology.ceiling_structure_thickness * typology.roof_surface # m3
    mass_c = volume_c * typology.ceiling_structure_material.density # kg
    C_c = typology.ceiling_structure_material.thermal_capacity * mass_c # J/K
    return C_c

def compute_C_w0(typology):
    volume_w0 = typology.w0_structure_thickness * typology.w0_surface # m3
    mass_w0 = volume_w0 * typology.w0_structure_material.density # kg
    C_w0 = typology.w0_structure_material.thermal_capacity * mass_w0 # J/K
    return C_w0

def compute_C_w1(typology):
    volume_w1 = typology.w1_structure_thickness * typology.w1_surface # m3
    mass_w1 = volume_w1 * typology.w1_structure_material.density # kg
    C_w1 = typology.w1_structure_material.thermal_capacity * mass_w1 # J/K
    return C_w1

def compute_C_w2(typology):
    volume_w2 = typology.w2_structure_thickness * typology.w2_surface # m3
    mass_w2 = volume_w2 * typology.w2_structure_material.density # kg
    C_w2 = typology.w2_structure_material.thermal_capacity * mass_w2 # J/K
    return C_w2

def compute_C_w3(typology):
    volume_w3 = typology.w3_structure_thickness * typology.w3_surface # m3
    mass_w3 = volume_w3 * typology.w3_structure_material.density # kg
    C_w3 = typology.w3_structure_material.thermal_capacity * mass_w3 # J/K
    return C_w3

def compute_C_f(typology):
    volume_f = typology.floor_structure_thickness * typology.ground_surface # m3
    mass_f = volume_f * typology.floor_structure_material.density # kg
    C_f = typology.floor_structure_material.thermal_capacity * mass_f # J/K
    return C_f

def compute_Rfi(typology):
    R_df = typology.floor_structure_thickness/(typology.floor_structure_material.thermal_conductivity * typology.ground_surface)
    
    # ajouter les ponts thermiques
    # Pour l'instant, juste une diminution de l'efficacité (ie de l'epaisseur)
    if typology.floor_insulation_position == 'ITI':
        R_diso_in = typology.floor_insulation_thickness/(typology.floor_insulation_material.thermal_conductivity * typology.ground_surface)
        R_diso_in = R_diso_in*0.73
    else:
        R_diso_in = 0
        
    # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf p16
    hi = 2.9 # W/(m2.K)
    R_dih = 1/(hi * typology.ground_surface) # K/W
    
    R_di = R_df/2 + R_diso_in + R_dih
    return R_di


def compute_Rfg(typology):
    R_df = typology.floor_structure_thickness/(typology.floor_structure_material.thermal_conductivity * typology.ground_surface)
    
    if typology.floor_insulation_position == 'ITE':
        R_diso_out = typology.floor_insulation_thickness/(typology.floor_insulation_material.thermal_conductivity * typology.ground_surface)
    else:
        R_diso_out = 0
    
    R_di = R_df/2 + R_diso_out
    return R_di

def compute_R_df(typology):
    R_df = typology.floor_structure_thickness/(typology.floor_structure_material.thermal_conductivity * typology.ground_surface)
    return R_df

def compute_C_i(typology):
    mass_air = AIR_DENSITY * typology.volume
    C_air = AIR_THERMAL_CAPACITY * mass_air
    
    # estimations au doigt mouillé : cf Antonopoulos and Koronaki (1999) 
    #TODO à rafiner
    C_internal_partitions = 10*C_air
    C_mobilier = 0.2*C_internal_partitions 
    
    C_i = C_air + C_mobilier + C_internal_partitions
    return C_i


def compute_C_u(typology):
    mass_air = AIR_DENSITY * typology.roof_surface/2 * typology.height
    C_air = AIR_THERMAL_CAPACITY * mass_air
    
    # estimations au doigt mouillé : cf Antonopoulos and Koronaki (1999) 
    #TODO à rafiner
    C_internal_partitions = 2*C_air
    C_mobilier = 0.2*C_internal_partitions 
    
    C_u = C_air + C_mobilier + C_internal_partitions
    return C_u


def compute_C_d(typology):
    mass_air = AIR_DENSITY * typology.ground_surface * typology.height
    C_air = AIR_THERMAL_CAPACITY * mass_air
    
    # estimations au doigt mouillé : cf Antonopoulos and Koronaki (1999) 
    #TODO à rafiner
    C_internal_partitions = 2*C_air
    C_mobilier = 0.2*C_internal_partitions 
    
    C_d = C_air + C_mobilier + C_internal_partitions
    return C_d


def compute_R_g(typology):
    R_g = typology.floor_ground_distance/(GROUND_THERMAL_CONDUCTIVITY * typology.ground_section)
    return R_g

def compute_C_g(typology):
    C_g = GROUND_THERMAL_CAPACITY * GROUND_DENSITY * typology.ground_volume
    return C_g


def get_solar_absorption_coefficient(typology,wall):
    dict_color_absorption = {'light':0.4,
                             'medium':0.6,
                             'dark':0.8,
                             'black':1.}
    
    if wall == 'roof':
        absorption_coefficient = dict_color_absorption.get(typology.roof_color)
    else:
        color = {0:typology.w0_color,
                 1:typology.w1_color,
                 2:typology.w2_color,
                 3:typology.w3_color,}.get(wall)
        absorption_coefficient = dict_color_absorption.get(color)
    return absorption_coefficient


def compute_external_Phi(typology, weather_data, wall):
    # coefficient d'absorption du flux solaire
    # TODO : à préciser pour les masquages et clarifier les flux solaires diffus
    # alpha = get_solar_absorption_coefficient(typology)
    
    # orientation de la paroi
    if wall == 'roof':
        orientation = 'H'
        surface = typology.roof_surface
        alpha = get_solar_absorption_coefficient(typology,wall)
    else:
        orientation = {0:typology.w0_orientation,
                       1:typology.w1_orientation,
                       2:typology.w2_orientation,
                       3:typology.w3_orientation,}.get(wall)
        surface = {0:typology.w0_surface,
                   1:typology.w1_surface,
                   2:typology.w2_surface,
                   3:typology.w3_surface,}.get(wall)
        alpha = get_solar_absorption_coefficient(typology,wall)
        
    sun_radiation = weather_data['direct_sun_radiation_{}'.format(orientation)].values #+ weather_data['diffuse_sun_radiation_{}'.format(orientation)].values
    Phi_se = sun_radiation * surface * alpha
    return Phi_se


def get_solar_transmission_factor(typology,weather_data,wall):
    # Dans les règles Th-bat : voir norme NF P50 777, puis norme NF EN 410 
    # TODO à raffiner selon le nombre de couches principalement (et peut-être l'angle d'incidence ?)

    wall_orientation = {0:typology.w0_orientation,
                        1:typology.w1_orientation,
                        2:typology.w2_orientation,
                        3:typology.w3_orientation,}.get(wall)
    valid_orientations = ['N','NE','E','SE','S','SW','W','NW']
    dict_angle_orientation = {i*45:o for i,o in enumerate(valid_orientations)}
    dict_orientation_angle = {v:k for k,v in dict_angle_orientation.items()}
    
    wall_angle = dict_orientation_angle.get(wall_orientation)
    
    sun_angle = np.abs(weather_data.sun_azimuth.values - wall_angle)
    sun_alt = np.asarray([max(e,0) for e in weather_data.sun_altitude.values])
    sun_angle = sun_angle + sun_alt
    sun_angle = np.where(sun_alt==0, 90,sun_angle)
    
    solar_factor = np.maximum(np.cos(np.deg2rad(sun_angle)),0)
    return solar_factor


def get_elements_masking(typology,weather_data,wall):
    # TODO : masquage des éléments architecturaux 
    return 1


def get_environment_masking(typology,weather_data,wall,minimal_altitude=10):
    # TODO : masquage de l'environnement
    sun_alt = np.asarray([max(e,0) for e in weather_data.sun_altitude.values])
    env_mask = np.where(sun_alt>minimal_altitude, 1,0)
    
    # plt.plot(weather_data.index[:300],sun_alt[:300])
    # plt.plot(weather_data.index[:300],env_mask[:300])
    return env_mask


def compute_internal_Phi(typology, weather_data, wall):
    # TODO : à préciser
    # coefficient d'absorption du flux solaire
    Ug = typology.windows_Ug
    solar_factor = get_solar_transmission_factor(typology,weather_data,wall)
    
    solar_env_mask = get_environment_masking(typology,weather_data,wall)
    solar_elem_mask = get_elements_masking(typology,weather_data,wall)
    
    # orientation de la paroi
    if wall == 'roof':
        orientation = 'H'
        surface = typology.h_windows_surface
    else:
        orientation = {0:typology.w0_orientation,
                       1:typology.w1_orientation,
                       2:typology.w2_orientation,
                       3:typology.w3_orientation,}.get(wall)
        surface = {0:typology.w0_windows_surface,
                   1:typology.w1_windows_surface,
                   2:typology.w2_windows_surface,
                   3:typology.w3_windows_surface,}.get(wall)
    
    sun_radiation = weather_data['direct_sun_radiation_{}'.format(orientation)].values# + weather_data['diffuse_sun_radiation_{}'.format(orientation)].values
    Phi_si = sun_radiation * solar_factor * solar_env_mask * solar_elem_mask * surface * Ug
    return Phi_si



def SFH_test_model(typology, behaviour, weather_data, progressbar=False):
    """
    Maison individuelle détachée (SFH), sans cave et avec des combles aménagées
    Une seule zone thermique.
    
    La ventilation va devoir être constante dans ce premier modèle.
    Idem pour les coefficient de transfert surfaciques :
        (rayonnement (absent pour l'instant) et convection)

    """

    # Variables thermiques de ventilation et infiltrations
    R_air = compute_R_air(typology)
    
    # Variables thermiques vers le haut
    # R_uw = None # on considère qu'il n'y en a pas
    R_ueh = compute_R_ueh(typology)
    R_ui = 1/(typology.roof_U * typology.roof_surface)
    
    # Variables thermiques des murs latéraux 
    R_w0w = 1/(typology.windows_U * typology.w0_windows_surface + typology.door_U * typology.door_surface)
    R_w0i = compute_Rw0i(typology)
    R_w0e = compute_Rw0e(typology)
    R_w0eh = compute_R_w0eh(typology)
    C_w0 = compute_C_w0(typology)
    
    R_w1w = 1/(typology.windows_U * typology.w1_windows_surface)
    R_w1i = compute_Rw1i(typology)
    R_w1e = compute_Rw1e(typology)
    R_w1eh = compute_R_w1eh(typology)
    C_w1 = compute_C_w1(typology)
    
    R_w2w = 1/(typology.windows_U * typology.w2_windows_surface)
    R_w2i = compute_Rw2i(typology)
    R_w2e = compute_Rw2e(typology)
    R_w2eh = compute_R_w2eh(typology)
    C_w2 = compute_C_w2(typology)
    
    R_w3w = 1/(typology.windows_U * typology.w3_windows_surface)
    R_w3i = compute_Rw3i(typology)
    R_w3e = compute_Rw3e(typology)
    R_w3eh = compute_R_w3eh(typology)
    C_w3 = compute_C_w3(typology)
    
    # Variables thermiques vers le bas
    R_di = compute_Rfi(typology)
    R_df = compute_R_df(typology)
    C_f = compute_C_f(typology)
    
    # Variables thermiques internes
    C_i = compute_C_i(typology)
    
    # Variables thermiques du sol
    R_g = compute_R_g(typology)
    C_g = compute_C_g(typology)
    foundation_depth = typology.floor_ground_distance
    
    # Autres variables 
    Ti_setpoint_winter, Ti_setpoint_summer = behaviour.get_set_point_temperature(weather_data)
    P_max_heater = typology.heater_maximum_power
    P_max_cooler = typology.cooler_maximum_power
    internal_thermal_gains = behaviour.get_internal_gains(typology.surface,weather_data)
    
    time_ = np.asarray(weather_data.index)
    delta_t = (time_[1]-time_[0]) / np.timedelta64(1, 's')
    
    # Définition de la matrice A
    A = np.zeros((7,7))
    A[0,0] = 1/C_i * (- 1/R_air 
                      # - 1/R_uw 
                      + R_ueh/(R_ui*(R_ui+R_ueh)) - 1/R_ui
                      - 1/R_w0w - 1/R_w0i
                      - 1/R_w1w - 1/R_w1i
                      - 1/R_w2w - 1/R_w2i
                      - 1/R_w3w - 1/R_w3i
                      - 1/R_di) 
    
    A[0,1] = 1/C_i * 1/R_w0i
    A[0,2] = 1/C_i * 1/R_w1i
    A[0,3] = 1/C_i * 1/R_w2i
    A[0,4] = 1/C_i * 1/R_w3i
    A[0,5] = 1/C_i * 1/R_di
    
    A[1,0] = 1/C_w0 * 1/R_w0i
    A[2,0] = 1/C_w1 * 1/R_w1i
    A[3,0] = 1/C_w2 * 1/R_w2i
    A[4,0] = 1/C_w3 * 1/R_w3i
    
    A[1,1] = 1/C_w0 * (-1/R_w0e -1/R_w0i + R_w0eh/(R_w0e*(R_w0eh+R_w0e)))
    A[2,2] = 1/C_w1 * (-1/R_w1e -1/R_w1i + R_w1eh/(R_w1e*(R_w1eh+R_w1e)))
    A[3,3] = 1/C_w2 * (-1/R_w2e -1/R_w2i + R_w2eh/(R_w2e*(R_w2eh+R_w2e)))
    A[4,4] = 1/C_w3 * (-1/R_w3e -1/R_w3i + R_w3eh/(R_w3e*(R_w3eh+R_w3e)))
    
    A[5,0] = 1/C_f * 1/R_di
    A[5,5] = 1/C_f * (-1/(R_df/2) - 1/R_di)
    A[5,6] = 1/C_f * 1/(R_df/2)
    
    A[6,5] = 1/C_g * 1/(R_df/2)
    A[6,6] = 1/C_g * (-1/R_g - 1/(R_df/2))
    
    # Définition de la matrice B
    B = np.zeros((7,13))
    B[0,0] = 1/C_i * (1/R_air 
                      # +1/R_uw
                      +1/(R_ui+R_ueh)
                      +1/R_w0w
                      +1/R_w1w
                      +1/R_w2w
                      +1/R_w3w)
    
    B[0,1] = 1/C_i * R_ueh/(R_ui+R_ueh)
    B[0,2] = 1/C_i
    B[0,4] = 1/C_i
    B[0,6] = 1/C_i
    B[0,8] = 1/C_i
    B[0,10] = 1/C_i
    B[0,11] = 1/C_i
    B[0,12] = 1/C_i
    
    B[1,0] = 1/C_w0 * 1/(R_w0eh+R_w0e)
    B[2,0] = 1/C_w1 * 1/(R_w1eh+R_w1e)
    B[3,0] = 1/C_w2 * 1/(R_w2eh+R_w2e)
    B[4,0] = 1/C_w3 * 1/(R_w3eh+R_w3e)
    
    B[1,3] = 1/C_w0 * R_w0eh/(R_w0eh+R_w0e)
    B[2,5] = 1/C_w1 * R_w1eh/(R_w1eh+R_w1e)
    B[3,7] = 1/C_w2 * R_w2eh/(R_w2eh+R_w2e)
    B[4,9] = 1/C_w3 * R_w3eh/(R_w3eh+R_w3e)
    
    B[6,0] = 1/C_g * 1/R_g
    
    
    pickle.dump(A, open('.A_SFH.pickle', "wb"))
    
    # Matrices discretisées 
    F = expm(A * delta_t)
    G = dot3(inv(A), F-np.eye(A.shape[0]), B)
    
    # État initial
    X = np.zeros((len(time_), 7))
    U = np.zeros((len(time_), 13))
    
    U[:,0] = weather_data.temperature_2m
    U[:,1] = compute_external_Phi(typology, weather_data, wall='roof') # Phi_sue
    U[:,2] = [0]*len(weather_data) # Phi_sui
    U[:,3] = compute_external_Phi(typology, weather_data, wall=0) # Phi_sw0e
    U[:,4] = compute_internal_Phi(typology, weather_data, wall=0) # Phi_sw0i
    U[:,5] = compute_external_Phi(typology, weather_data, wall=1) # Phi_sw1e
    U[:,6] = compute_internal_Phi(typology, weather_data, wall=1) # Phi_sw1i
    U[:,7] = compute_external_Phi(typology, weather_data, wall=2) # Phi_sw2e
    U[:,8] = compute_internal_Phi(typology, weather_data, wall=2) # Phi_sw2i
    U[:,9] = compute_external_Phi(typology, weather_data, wall=3) # Phi_sw3e
    U[:,10] = compute_internal_Phi(typology, weather_data, wall=3) # Phi_sw3i
    U[:,12] = np.asarray(internal_thermal_gains)
    
    X[0,0] = Ti_setpoint_winter[0]
    X[0,1] = 1/(R_w0e+R_w0eh+R_w0i) * (R_w0i * U[0,0] + (R_w0e+R_w0eh) * X[0,0])
    X[0,2] = 1/(R_w1e+R_w1eh+R_w1i) * (R_w1i * U[0,0] + (R_w1e+R_w1eh) * X[0,0])
    X[0,3] = 1/(R_w2e+R_w2eh+R_w2i) * (R_w2i * U[0,0] + (R_w2e+R_w2eh) * X[0,0])
    X[0,4] = 1/(R_w3e+R_w3eh+R_w3i) * (R_w3i * U[0,0] + (R_w3e+R_w3eh) * X[0,0])
    X[0,6] = get_init_ground_temperature(foundation_depth, weather_data)
    X[0,5] = 1/(R_df/2+R_di) * (R_di * X[0,6] + (R_df/2) * X[0,0])
    
    # Simulation
    heating_needs = [0]*len(time_)
    cooling_needs = [0]*len(time_)
    
    if progressbar:
        iterator = tqdm.tqdm(range(1,len(time_)), total=len(time_)-1)
    else:
        iterator = range(1,len(time_))
        
    for i in iterator:
        # Te = U[i,0]
        Ti = X[i-1,0]
        
        Ts_heater = Ti_setpoint_winter[i-1]
        Ts_cooler = Ti_setpoint_summer[i-1]
        
        # Te = X[i-1,0]
        P_heater = get_P_heater(Ti, Ti_min=Ts_heater, Pmax=P_max_heater, method='linear_tolerance')
        P_cooler = get_P_cooler(Ti, Ti_max=Ts_cooler, Pmax=P_max_cooler, method='linear_tolerance')
        
        heating_needs[i-1] = P_heater
        cooling_needs[i-1] = -P_cooler
        
        U[i,11] = P_heater + P_cooler # i-1 ou i # à vérifier Rouchier, Madsen : peu de différence en tout cas
        
        X[i] = np.dot(F,X[i-1]) + np.dot(G, U[i].T)
    
    heating_needs[-1] = get_P_heater(X[i,0], Ti_min=Ti_setpoint_winter[i], Pmax=P_max_heater, method='linear_tolerance')
    cooling_needs[-1] = get_P_cooler(X[i,0], Ti_max=Ti_setpoint_summer[i], Pmax=P_max_cooler, method='linear_tolerance')
    
    weather_data['internal_temperature'] = X[:,0]
    weather_data['w0_internal_temperature'] = X[:,1]
    weather_data['w1_internal_temperature'] = X[:,2]
    weather_data['w2_internal_temperature'] = X[:,3]
    weather_data['w3_internal_temperature'] = X[:,4]
    weather_data['ground_temperature'] = X[:,6]
    
    weather_data['heating_needs'] = heating_needs
    weather_data['cooling_needs'] = cooling_needs
    
    return weather_data


def run_thermal_model(typology, behaviour, weather_data, progressbar=False):
    """
    Modélisation thermique RC

    Parameters
    ----------
    typology : TYPE
        DESCRIPTION.
    behaviour : TYPE
        DESCRIPTION.
    weather_data : TYPE
        DESCRIPTION.
    progressbar : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    # Variables thermiques internes
    C_i = compute_C_i(typology)
    
    # Variables thermiques d'infiltrations
    R_inf = compute_R_inf(typology)
    R_door = 1/(typology.door_U * typology.door_surface)
    
    # Variables thermiques vers le haut
    R_uih = compute_R_uih(typology)
    R_ucr = (1/(typology.ceiling_U * typology.roof_surface) + 1/(typology.roof_U * typology.roof_surface))
    R_uceiling = R_ucr/2
    R_uroof = R_ucr/2
    R_ueh = compute_R_ueh(typology)
    R_uhceiling = R_uih
    R_uhroof = R_uih
    C_c = compute_C_c(typology)
    C_u = compute_C_u(typology)
    
    # TODO à ajouter pour les 4 composantes
    typology.modelled_Uph = 1/(R_uih + R_uceiling + R_uroof + R_ueh)
    
    # Variables thermiques des murs latéraux 
    R_w0w = 1/(typology.windows_U * typology.w0_windows_surface)
    R_w0i = compute_Rw0i(typology)
    R_w0e = compute_Rw0e(typology)
    R_w0eh = compute_R_w0eh(typology)
    C_w0 = compute_C_w0(typology)
    
    R_w1w = 1/(typology.windows_U * typology.w1_windows_surface)
    R_w1i = compute_Rw1i(typology)
    R_w1e = compute_Rw1e(typology)
    R_w1eh = compute_R_w1eh(typology)
    C_w1 = compute_C_w1(typology)
    
    R_w2w = 1/(typology.windows_U * typology.w2_windows_surface)
    R_w2i = compute_Rw2i(typology)
    R_w2e = compute_Rw2e(typology)
    R_w2eh = compute_R_w2eh(typology)
    C_w2 = compute_C_w2(typology)
    
    R_w3w = 1/(typology.windows_U * typology.w3_windows_surface)
    R_w3i = compute_Rw3i(typology)
    R_w3e = compute_Rw3e(typology)
    R_w3eh = compute_R_w3eh(typology)
    C_w3 = compute_C_w3(typology)
    
    typology.modelled_Umur = 1/(R_w0i + R_w0e + R_w0eh + R_w1i + R_w1e + R_w1eh + R_w2i + R_w3e + R_w3eh + R_w3i + R_w3e + R_w3eh)
    
    typology.modelled_Uw = 1/(R_w0w + R_w1w + R_w2w + R_w3w)
    
    # Variables thermiques vers le bas
    R_fi = compute_Rfi(typology)
    R_dfh = R_uih
    R_fg = compute_Rfg(typology)
    R_dgh = R_uih
    C_f = compute_C_f(typology)
    C_d = compute_C_d(typology)
    
    typology.modelled_Upb = 1/(R_dgh + R_dfh + R_fg + R_fi)
    
    # Variables thermiques du sol
    R_g = compute_R_g(typology)
    C_g = compute_C_g(typology)
    foundation_depth = typology.floor_ground_distance
    
    # Autres variables 
    Ti_setpoint_winter, Ti_setpoint_summer = behaviour.get_set_point_temperature(weather_data)
    P_max_heater = typology.heater_maximum_power
    P_max_cooler = typology.cooler_maximum_power
    internal_thermal_gains = behaviour.get_internal_gains(typology.surface,weather_data)
    
    
    time_ = np.asarray(weather_data.index)
    delta_t = (time_[1]-time_[0]) / np.timedelta64(1, 's')
    
    # Définition de la matrice A
    A = np.zeros((10,10))
    
    # R_door (collectif) ?
    A[0,0] = 1/C_i * (-1/R_inf
                      -1/R_uih
                      -1/R_w0w - 1/R_w0i - 1/R_door
                      -1/R_w1w - 1/R_w1i
                      -1/R_w2w - 1/R_w2i
                      -1/R_w3w - 1/R_w3i
                      -1/R_fi)
    
    A[0,1] = 1/C_i * (1/R_w0i)
    A[0,2] = 1/C_i * (1/R_w1i)
    A[0,3] = 1/C_i * (1/R_w2i)
    A[0,4] = 1/C_i * (1/R_w3i)
    A[0,5] = 1/C_i * (1/R_uih)
    A[0,7] = 1/C_i * (1/R_fi)
    
    A[1,0] = 1/C_w0 * (1/R_w0i)
    A[2,0] = 1/C_w1 * (1/R_w1i)
    A[3,0] = 1/C_w2 * (1/R_w2i)
    A[4,0] = 1/C_w3 * (1/R_w3i)
    
    # Remplacement de Te par Ti pour les parois adiabatiques
    if typology.w3_adiabatic:
        A[4,0] = 1/C_w3 * (1/R_w3i) + 1/C_w3 * (1/(R_w3eh+R_w3e))
    if typology.w2_adiabatic:
        A[3,0] = 1/C_w2 * (1/R_w2i) + 1/C_w2 * (1/(R_w2eh+R_w2e))
    if typology.w1_adiabatic:
        A[2,0] = 1/C_w1 * (1/R_w1i) + 1/C_w1 * (1/(R_w1eh+R_w1e))
    
    A[1,1] = 1/C_w0 * (-1/R_w0e -1/R_w0i + R_w0eh/(R_w0e*(R_w0eh+R_w0e)))
    A[2,2] = 1/C_w1 * (-1/R_w1e -1/R_w1i + R_w1eh/(R_w1e*(R_w1eh+R_w1e)))
    A[3,3] = 1/C_w2 * (-1/R_w2e -1/R_w2i + R_w2eh/(R_w2e*(R_w2eh+R_w2e)))
    A[4,4] = 1/C_w3 * (-1/R_w3e -1/R_w3i + R_w3eh/(R_w3e*(R_w3eh+R_w3e)))
    A[5,0] = 1/C_c * (1/R_uih) 
    
    
    if typology.converted_attic:
        A[5,5] = 1/C_c * (-1/(R_uceiling + R_uroof)
                          -1/R_uih
                          +R_ueh/((R_uceiling+R_uroof)*(R_uroof+R_uceiling+R_ueh)))
        
    else:
        A[5,5] = 1/C_c * (-1/(R_uhceiling+R_uceiling) -1/R_uih)
        A[5,6] = 1/C_c * (1/(R_uhceiling+R_uceiling))
        A[6,6] = 1/C_u * (-1/(R_uroof+R_uhroof)
                          -1/(R_uhceiling+R_uceiling)
                          +R_ueh/((R_uroof+R_uhroof)*(R_uroof+R_uhroof+R_ueh)))
        A[6,5] = 1/C_u * (1/(R_uhceiling+R_uceiling))
        
    A[7,0] = 1/C_f * (1/R_fi)
    
    if typology.basement:
        A[7,7] = 1/C_f * (-1/(R_dfh+R_fg) - 1/R_fi)
        A[7,8] = 1/C_f * (1/(R_dfh+R_fg))
        A[8,8] = 1/C_d * (-1/R_dgh - 1/(R_dfh+R_fg))
        A[8,7] = 1/C_d * (1/(R_dfh+R_fg))
        A[8,9] = 1/C_d * (1/R_dgh)
        A[9,9] = 1/C_g * (-1/R_g - 1/R_dgh)
        A[9,8] = 1/C_g * (1/R_dgh)
        
        # remplacement de Tf par Ti
        if typology.floor_adiabatic:
            A[7,8] = 0
            A[7,0] = 1/C_f * (1/R_fi) + 1/C_f * (1/(R_dfh+R_fg))
        
    else:
        A[7,7] = 1/C_f * (-1/R_fg - 1/R_fi)
        A[7,9] = 1/C_f * (1/R_fg)
        A[9,9] = 1/C_g * (-1/R_g - 1/R_fg)
        A[9,7] = 1/C_g * (1/R_fg)
        
        # remplacement de Tg par Ti
        if typology.floor_adiabatic:
            A[7,9] = 0
            A[7,0] = 1/C_f * (1/R_fi) + 1/C_f * (1/R_fg)
    
    # Définition de la matrice B
    B = np.zeros((10,15))
    
    B[0,0] = 1/C_i * (1/R_inf
                      +1/R_w0w
                      +1/R_w1w
                      +1/R_w2w
                      +1/R_w3w
                      +1/R_door)
    
    B[0,2] = 1/C_i
    B[0,4] = 1/C_i
    B[0,6] = 1/C_i
    B[0,8] = 1/C_i
    B[0,10] = 1/C_i
    B[0,11] = 1/C_i
    B[0,12] = 1/C_i
    B[0,13] = 1/C_i
    B[0,14] = 1/C_i
    
    B[1,0] = 1/C_w0 * (1/(R_w0eh+R_w0e))
    B[2,0] = 1/C_w1 * (1/(R_w1eh+R_w1e))
    B[3,0] = 1/C_w2 * (1/(R_w2eh+R_w2e))
    B[4,0] = 1/C_w3 * (1/(R_w3eh+R_w3e))
    
    # Remplacement de Te par Ti pour les parois adiabatiques
    if typology.w3_adiabatic:
        B[4,0] = 0
    if typology.w2_adiabatic:
        B[3,0] = 0
    if typology.w1_adiabatic:
        B[2,0] = 0
    
    B[1,3] = 1/C_w0 * R_w0eh/(R_w0eh+R_w0e)
    B[2,5] = 1/C_w1 * R_w1eh/(R_w1eh+R_w1e)
    B[3,7] = 1/C_w2 * R_w2eh/(R_w2eh+R_w2e)
    B[4,9] = 1/C_w3 * R_w3eh/(R_w3eh+R_w3e)
    
    B[9,0] = 1/C_g * (1/R_g)
    
    if typology.converted_attic:
        B[5,0] = 1/C_c * (1/(R_uroof+R_uceiling+R_ueh))
        B[5,1] = 1/C_c * (R_ueh/(R_uroof+R_uceiling+R_ueh))
    else:
        B[6,0] = 1/C_u * (1/(R_uroof+R_uhroof+R_ueh))
        B[6,1] = 1/C_u * (R_ueh/(R_uroof+R_uhroof+R_ueh))
    
    # Suppression des parties vides de la matrice
    if not typology.basement:
        A = np.delete(A, 8, 0)
        A = np.delete(A, 8, 1)
        B = np.delete(B, 8, 0)
        
    if typology.converted_attic:
        A = np.delete(A, 6, 0)
        A = np.delete(A, 6, 1)
        B = np.delete(B, 6, 0)
    
    # sauvegarde de A pour analyse
    pickle.dump(A, open('.A_GENMOD.pickle', "wb"))
    
    # Matrices discretisées 
    F = expm(A * delta_t)
    G = dot3(inv(A), F-np.eye(A.shape[0]), B)
    
    # État initial
    X = np.zeros((len(time_), 10))
    U = np.zeros((len(time_), 15))
    
    U[:,0] = weather_data.temperature_2m
    U[:,1] = compute_external_Phi(typology, weather_data, wall='roof') # Phi_sue
    U[:,2] = [0]*len(weather_data) # Phi_sui
    U[:,3] = compute_external_Phi(typology, weather_data, wall=0) # Phi_sw0e
    U[:,4] = compute_internal_Phi(typology, weather_data, wall=0) # Phi_sw0i
    U[:,5] = compute_external_Phi(typology, weather_data, wall=1) # Phi_sw1e
    U[:,6] = compute_internal_Phi(typology, weather_data, wall=1) # Phi_sw1i
    U[:,7] = compute_external_Phi(typology, weather_data, wall=2) # Phi_sw2e
    U[:,8] = compute_internal_Phi(typology, weather_data, wall=2) # Phi_sw2i
    U[:,9] = compute_external_Phi(typology, weather_data, wall=3) # Phi_sw3e
    U[:,10] = compute_internal_Phi(typology, weather_data, wall=3) # Phi_sw3i
    # U[:,11] Phi_hc
    U[:,12] = np.asarray(internal_thermal_gains)
    # U[:,13] Phi_vmeca
    # U[:,14] Phi_vnat
    
    # Remplacement de Te par Ti pour les parois adiabatiques
    if typology.w3_adiabatic:
        U[:,9] = compute_external_Phi(typology, weather_data, wall=3)*0 # Phi_sw3e
        U[:,10] = compute_internal_Phi(typology, weather_data, wall=3)*0 # Phi_sw3i
    if typology.w2_adiabatic:
        U[:,7] = compute_external_Phi(typology, weather_data, wall=2)*0 # Phi_sw2e
        U[:,8] = compute_internal_Phi(typology, weather_data, wall=2)*0 # Phi_sw2i
    if typology.w1_adiabatic:
        U[:,5] = compute_external_Phi(typology, weather_data, wall=1)*0 # Phi_sw1e
        U[:,6] = compute_internal_Phi(typology, weather_data, wall=1)*0 # Phi_sw1i
    
    X[0,0] = Ti_setpoint_winter[0] # Ti
    X[0,1] = 1/(R_w0e+R_w0eh+R_w0i) * (R_w0i * U[0,0] + (R_w0e+R_w0eh) * X[0,0]) # Tw0
    X[0,2] = 1/(R_w1e+R_w1eh+R_w1i) * (R_w1i * U[0,0] + (R_w1e+R_w1eh) * X[0,0]) # Tw1
    X[0,3] = 1/(R_w2e+R_w2eh+R_w2i) * (R_w2i * U[0,0] + (R_w2e+R_w2eh) * X[0,0]) # Tw2
    X[0,4] = 1/(R_w3e+R_w3eh+R_w3i) * (R_w3i * U[0,0] + (R_w3e+R_w3eh) * X[0,0]) # Tw3
    X[0,5] = 1/(R_ueh+R_uroof+R_uhroof+R_uhceiling+R_uceiling+R_uih) * (R_uih * U[0,0] + (R_ueh+R_uroof+R_uhroof+R_uhceiling+R_uceiling) * X[0,0]) # Tc
    X[0,6] = 1/(R_ueh+R_uroof+R_uhroof+R_uhceiling+R_uceiling+R_uih) * ((R_uih+R_uceiling+R_uhceiling) * U[0,0] + (R_ueh+R_uroof+R_uhroof) * X[0,0]) # Tu
    X[0,9] = get_init_ground_temperature(foundation_depth, weather_data) # Tg
    X[0,7] = 1/(R_dgh+R_dfh+R_fg+R_fi) * (R_fi * X[0,9] + (R_dgh+R_dfh+R_fg) * X[0,0]) # Tf
    X[0,8] = 1/(R_dgh+R_dfh+R_fg+R_fi) * ((R_fi+R_fg+R_dfh) * X[0,9] + (R_dgh) * X[0,0]) # Td
    
    # Suppression des parties vides de la matrice
    if not typology.basement:
        X = np.delete(X, 8, 1)
    if typology.converted_attic:
        X = np.delete(X, 6, 1)
    
    # Remplacement de Te par Ti pour les parois adiabatiques
    if typology.w3_adiabatic:
        X[0,4] = X[0,0]
    if typology.w2_adiabatic:
        X[0,3] = X[0,0]
    if typology.w1_adiabatic:
        X[0,2] = X[0,0]
    if typology.floor_adiabatic:
        X[0,7] = X[0,0]
    
        
    # Simulation
    heating_needs = [0]*len(time_)
    cooling_needs = [0]*len(time_)
    
    if progressbar:
        iterator = tqdm.tqdm(range(1,len(time_)), total=len(time_)-1)
    else:
        iterator = range(1,len(time_))
        
    for i in iterator:
        Te = U[i,0]
        Ti = X[i-1,0]
        
        Ts_heater = Ti_setpoint_winter[i-1]
        Ts_cooler = Ti_setpoint_summer[i-1]
        
        P_heater = get_P_heater(Ti, Ti_min=Ts_heater, Pmax=P_max_heater, method='linear_tolerance')
        P_cooler = get_P_cooler(Ti, Ti_max=Ts_cooler, Pmax=P_max_cooler, method='linear_tolerance')
        
        P_vmeca = get_P_vmeca(Ti,Te,P_heater,P_cooler,typology)
        P_vnat = get_P_vnat(Ti,Te,typology,behaviour)
        
        heating_needs[i-1] = P_heater
        cooling_needs[i-1] = -P_cooler
        
        U[i,11] = P_heater + P_cooler # i-1 ou i # à vérifier Rouchier, Madsen : peu de différence en tout cas
        U[i,13] = P_vmeca
        U[i,14] = P_vnat
        
        X[i] = np.dot(F,X[i-1]) + np.dot(G, U[i].T)
    
    heating_needs[-1] = get_P_heater(X[i,0], Ti_min=Ti_setpoint_winter[i], Pmax=P_max_heater, method='linear_tolerance')
    cooling_needs[-1] = get_P_cooler(X[i,0], Ti_max=Ti_setpoint_summer[i], Pmax=P_max_cooler, method='linear_tolerance')
    
    weather_data['internal_temperature'] = X[:,0]
    weather_data['heating_needs'] = heating_needs
    weather_data['cooling_needs'] = cooling_needs
    
    return weather_data

    
    
def refine_resolution(data, resolution):
    # interpolated = pd.DataFrame(index=pd.date_range(data.index[0],data.index[-1],freq=resolution))
    # interpolated = interpolated.join(data,how='left')
    # interpolated = interpolated.interpolate()
    upsampled = data.resample(resolution)
    interpolated = upsampled.interpolate(method='linear')
    return interpolated


def aggregate_resolution(data, resolution='h', agg_method='mean'):
    agg_data = data.groupby(pd.Grouper(freq=resolution)).agg(func=agg_method)
    return agg_data


#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_thermal_model'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
    
    # warnings.simplefilter("ignore") # à cause du calcul de l'azimuth et de l'altitude du soleil 
    
    #%% Test de vitesse de calcul en fonction de la résolution
    if False:
        details = False
    
        def divisors(n):
            divs = [1]
            for i in range(2,int(np.sqrt(n))+1):
                if n%i == 0:
                    divs.extend([i,n/i])
            divs.extend([n])
            res = list(map(int, sorted(list(set(divs)))))
            return res

        # définition des listes à affficher
        resolution_list = divisors(3600)[:-3]
        power_max_heating, power_max_cooling = [],[]
        heating_needs_list, cooling_needs_list = [],[]
        time_compute = []
        
        
        # Définition de la typologie
        typo_name = 'FR.N.SFH.01.Test'
        typo = Typology(typo_name)
        
        typo.roof_U = 0.36
        
        no_insulation = True
        
        if no_insulation:
            typo.w0_insulation_thickness = 0
            typo.w1_insulation_thickness = 0
            typo.w2_insulation_thickness = 0
            typo.w3_insulation_thickness = 0
            
            # typo.roof_U = 1.35
            # typo.floor_insulation_thickness = 0
        
        # typo.heater_maximum_power = 0
        # typo.cooler_maximum_power = 0
        
        # typo.w0_orientation = 'E'
        # typo.update_orientation()
        
        # Génération du fichier météo
        city = 'Paris'
        period = [2020]*2
        principal_orientation = typo.w0_orientation
        
        # Définition des habitudes
        conventionnel = Behaviour('conventionnel_th-bce_2020')
        conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
        conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
        
        weather_data = get_historical_weather_data(city,period,principal_orientation)
        
        for res in tqdm.tqdm(resolution_list):
            weather_data_fine_res = refine_resolution(weather_data.copy(), resolution='{:.0f}s'.format(res))
            
            year = period[0]
            
            start_compute = time.time()
            
            simulation_data = run_thermal_model(typo, conventionnel, weather_data_fine_res,progressbar=False)
            simulation_data = aggregate_resolution(simulation_data, resolution='h')
            
            end_compute = time.time()
            
            # fig,ax = plot_timeserie(simulation_data[['temperature_2m','internal_temperature']],figsize=(15,5),
            #                xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],ylabel='Temperature (°C)',
            #                figs_folder=figs_folder, save_fig='thermal_model_temperature_{}_{}_{}'.format(city,year,typo_name),show=False)
            # ax.set_title('{:.0f}s'.format(res))
            # plt.show()
            
            # plot_timeserie(simulation_data[['heating_needs','cooling_needs']],figsize=(15,5),
            #                xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],ylabel='Energy needs (Wh)',
            #                figs_folder=figs_folder, save_fig='thermal_model_energy_needs_{}_{}_{}'.format(city,year,typo_name),show=False)
            # ax.set_title('{:.0f}s'.format(res))
            # plt.show()
            
            annual_heating_consumption = simulation_data.heating_needs.sum()/1000
            surface_annual_heating_consumption = annual_heating_consumption/typo.surface
            
            annual_cooling_consumption = simulation_data.cooling_needs.sum()/1000
            surface_annual_cooling_consumption = annual_cooling_consumption/typo.surface
            
            power_max_heating.append(simulation_data.heating_needs.max())
            power_max_cooling.append(simulation_data.cooling_needs.max())
            heating_needs_list.append(surface_annual_heating_consumption)
            cooling_needs_list.append(surface_annual_cooling_consumption)
            time_compute.append(end_compute-start_compute)
            
            if details:
                print('Besoins annuels de chauffage à {} en {}: {:.0f} kWh/an'.format(city,year, annual_heating_consumption))
                print('Besoins annuels de chauffage à {} en {}: {:.0f} kWh/(m2.an)'.format(city, year, surface_annual_heating_consumption))
                print('Besoins annuels de refroidissement à {} en {}: {:.0f} kWh/an'.format(city, year, annual_cooling_consumption))
                print('Besoins annuels de refroidissement à {} en {}: {:.0f} kWh/(m2.an)'.format(city, year, surface_annual_cooling_consumption))
        
        power_max_heating = np.asarray(power_max_heating)/power_max_heating[0]
        power_max_cooling = np.asarray(power_max_cooling)/power_max_cooling[0]
        heating_needs_list = np.asarray(heating_needs_list)/heating_needs_list[0]
        cooling_needs_list = np.asarray(cooling_needs_list)/cooling_needs_list[0]
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.semilogy(resolution_list, time_compute,color='k',marker='o')
        ax.set_ylabel('Computation time for one year simulation (s)')
        # ax.set_ylim(bottom=0.)
        ax.set_xlabel('Temporal resolution time step (s)')
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('resolution_effect_time_computation_{}_{}'.format(city,period[0]))),bbox_inches='tight')
        plt.show()
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(resolution_list, heating_needs_list,color='tab:red',label='Heating',marker='o')
        ax.plot(resolution_list, cooling_needs_list,color='tab:blue',label='Cooling',marker='o')
        ax.set_ylabel('Annual energy needs (kWh.m$^{-2}$.yr$^{-1}$)')
        # ax.set_ylim(bottom=0.)
        ax.legend()
        ax.set_xlabel('Temporal resolution time step (s)')
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('resolution_effect_energy_needs_{}_{}'.format(city,period[0]))),bbox_inches='tight')
        plt.show()
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(resolution_list, power_max_heating,color='tab:red',label='Heating',marker='o')
        ax.plot(resolution_list, power_max_cooling,color='tab:blue',label='Cooling',marker='o')
        ax.set_ylabel('Maximal power needs (Wh)')
        # ax.set_ylim(bottom=0.)
        ax.legend()
        ax.set_xlabel('Temporal resolution time step (s)')
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('resolution_effect_max_power_{}_{}'.format(city,period[0]))),bbox_inches='tight')
        plt.show()
    
    #%% Premier test pour le poster de SGR
    if False:
        
        # Définition de la typologie
        # typo_name = 'FR.N.SFH.01.Test'
        typo_name = 'FR.N.SFH.01.Gen'
        typo = Typology(typo_name)
        
        # typo.w0_structure_thickness = 0.3
        # typo.w2_structure_thickness = 0.3
        typo.roof_U = 0.36
        
        no_insulation = False
        
        if no_insulation:
            typo.w0_insulation_thickness = 0
            typo.w1_insulation_thickness = 0
            typo.w2_insulation_thickness = 0
            typo.w3_insulation_thickness = 0
            
            typo.roof_U = 1.35
            
            typo.floor_insulation_thickness = 0
        
        # typo.heater_maximum_power = 0
        # typo.cooler_maximum_power = 0
        
        # typo.w0_orientation = 'E'
        # typo.update_orientation()
        
        # Génération du fichier météo
        city = 'Paris'
        period = [2020]*2
        principal_orientation = typo.w0_orientation
        
        weather_data = get_historical_weather_data(city,period,principal_orientation)
        weather_data = refine_resolution(weather_data, resolution='600s')
        
        # Définition des habitudes
        conventionnel = Behaviour('conventionnel_th-bce_2020')
        conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
        conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
        
        
        # Affichage des variables d'entrée
        if False:
            pass 
        
        # Simulation
        if True:
            year = period[0]
            
            simulation_data = SFH_test_model(typo, conventionnel, weather_data)
            simulation_data = aggregate_resolution(simulation_data, resolution='h')
            
            plot_timeserie(simulation_data[['temperature_2m','internal_temperature']],figsize=(15,5),
                           xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],ylabel='Temperature (°C)',
                           figs_folder=figs_folder, save_fig='thermal_model_temperature_{}_{}_{}'.format(city,year,typo_name))
            
            plot_timeserie(simulation_data[['heating_needs','cooling_needs']],figsize=(15,5),
                           xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],ylabel='Energy needs (Wh)',
                           figs_folder=figs_folder, save_fig='thermal_model_energy_needs_{}_{}_{}'.format(city,year,typo_name))
            
            
            annual_heating_consumption = simulation_data.heating_needs.sum()/1000
            surface_annual_heating_consumption = annual_heating_consumption/typo.surface
            
            annual_cooling_consumption = simulation_data.cooling_needs.sum()/1000
            surface_annual_cooling_consumption = annual_cooling_consumption/typo.surface
            
            print('Besoins annuels de chauffage à {} en {}: {:.0f} kWh/an'.format(city,year, annual_heating_consumption))
            print('Besoins annuels de chauffage à {} en {}: {:.0f} kWh/(m2.an)'.format(city, year, surface_annual_heating_consumption))
            print('Besoins annuels de refroidissement à {} en {}: {:.0f} kWh/an'.format(city, year, annual_cooling_consumption))
            print('Besoins annuels de refroidissement à {} en {}: {:.0f} kWh/(m2.an)'.format(city, year, surface_annual_cooling_consumption))
            
            monthly_data = aggregate_resolution(simulation_data, resolution='ME',agg_method='sum')
            monthly_data = monthly_data/1000
            
            plot_timeserie(monthly_data[['heating_needs','cooling_needs']],figsize=(5,5),
                           labels = ['{} ({:.1f} kWh/(m2.an))'.format(e,{'heating_needs':surface_annual_heating_consumption,'cooling_needs':surface_annual_cooling_consumption}.get(e)) for e in ['heating_needs','cooling_needs']],
                           ylabel='Energy needs (kWh/month) - {:.0f} m$^2$'.format(typo.surface),
                           xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],
                           figs_folder=figs_folder, save_fig='monthly_thermal_model_energy_needs_{}_{}_{}'.format(city,year,typo_name))
            
            # thermosensibilité
            if False:
                data_sensitivity = simulation_data[['temperature_2m','heating_needs','cooling_needs']].copy()
                data_sensitivity = data_sensitivity.sort_values(by='temperature_2m')
                data_sensitivity['energy_needs'] = (data_sensitivity.heating_needs + data_sensitivity.cooling_needs)
                # data_sensitivity = data_sensitivity[data_sensitivity.energy_needs>0.]
                
                
                plot_thermal_sensitivity(temperature=data_sensitivity.temperature_2m.to_list(), consumption=data_sensitivity.energy_needs.to_list(), 
                                         figs_folder=figs_folder, reg_code=city, reg_name='SFH test typology - {}'.format(city), year=year,C0_init=0,k_init=500,ylabel='Hourly energy needs (Wh)')
                
    #%% Graphes Préliminaires
    if False:
        
        # Étude de l'effet de l'épaisseur d'isolant sur la consommation annuelle
        if False:
            def get_annual_energy_needs(typology,weather_data,behaviour, by_surface=True):
                simulation_data = SFH_test_model(typology, conventionnel, weather_data)
                simulation_data = aggregate_resolution(simulation_data, resolution='h')
                
                yearly_data = aggregate_resolution(simulation_data, resolution='YE',agg_method='sum')
                yearly_data = yearly_data/1000
                surface_yearly_data = yearly_data/typology.surface
                
                if by_surface:
                    return surface_yearly_data[['heating_needs','cooling_needs']]
                else:
                    return yearly_data[['heating_needs','cooling_needs']]
                
            thickness_list = np.linspace(0,0.3,10)
            energy_needs_mean_list_1 = []
            energy_needs_std_list_1 = []
            energy_needs_mean_list_2 = []
            energy_needs_std_list_2 = []
            
            typo_name = 'FR.N.SFH.01.Test'
            typo = Typology(typo_name)
            typo.roof_U = 0.36
            typo.floor_insulation_thickness = 0
            
            # Génération du fichier météo
            city_1 = 'Paris'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_1 = get_historical_weather_data(city_1,period,principal_orientation)
            weather_data_1 = refine_resolution(weather_data_1, resolution='600s')
            
            city_2 = 'Marseille'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_2 = get_historical_weather_data(city_2,period,principal_orientation)
            weather_data_2 = refine_resolution(weather_data_2, resolution='600s')
            
            # Définition des habitudes
            conventionnel = Behaviour('conventionnel_th-bce_2020')
            conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
            conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
            
            for thickness in thickness_list:
                
                typo.w0_insulation_thickness = thickness
                typo.w1_insulation_thickness = thickness
                typo.w2_insulation_thickness = thickness
                typo.w3_insulation_thickness = thickness
                # typo.roof_U = 1.35
                
                energy_needs_1 = get_annual_energy_needs(typo, weather_data_1, conventionnel)
                energy_needs_mean_list_1.append(energy_needs_1.mean(axis=0))
                energy_needs_std_list_1.append(energy_needs_1.std(axis=0))
                
                energy_needs_2 = get_annual_energy_needs(typo, weather_data_2, conventionnel)
                energy_needs_mean_list_2.append(energy_needs_2.mean(axis=0))
                energy_needs_std_list_2.append(energy_needs_2.std(axis=0))
            
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.errorbar(thickness_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_2],
                        color='tab:red',label='Heating needs ({})'.format(city_2),ls='--',capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
                        color='tab:blue',label='Cooling needs ({})'.format(city_2),ls='--',capsize=3)
            ax.set_ylabel('Annual energy needs over {}-{} '.format(period[0],period[1])+'(kWh.m$^{-2}$.yr$^{-1}$)')
            ax.legend()
            ax.set_xlabel('Walls insulation thickness (m)')
            ax.set_ylim(bottom=0.,top=200)
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('effect_wall_insulation_thickness_{}_{}_{}-{}'.format(city_1,city_2,period[0],period[1]))),bbox_inches='tight')
            
            plt.show()
            
            
        # Étude de l'effet de la valeur U des fenêtres
        if False:
            def get_annual_energy_needs(typology,weather_data,behaviour, by_surface=True):
                simulation_data = SFH_test_model(typology, conventionnel, weather_data)
                simulation_data = aggregate_resolution(simulation_data, resolution='h')
                
                yearly_data = aggregate_resolution(simulation_data, resolution='YE',agg_method='sum')
                yearly_data = yearly_data/1000
                surface_yearly_data = yearly_data/typology.surface
                
                if by_surface:
                    return surface_yearly_data[['heating_needs','cooling_needs']]
                else:
                    return yearly_data[['heating_needs','cooling_needs']]
                
            U_value_windows_list = np.linspace(0.5,5,10)
            energy_needs_mean_list_1 = []
            energy_needs_std_list_1 = []
            energy_needs_mean_list_2 = []
            energy_needs_std_list_2 = []
            
            typo_name = 'FR.N.SFH.01.Test'
            typo = Typology(typo_name)
            typo.roof_U = 0.36
            typo.floor_insulation_thickness = 0.1
            typo.w0_insulation_thickness = 0.1
            typo.w1_insulation_thickness = 0.1
            typo.w2_insulation_thickness = 0.1
            typo.w3_insulation_thickness = 0.1
            
            # Génération du fichier météo
            city_1 = 'Paris'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_1 = get_historical_weather_data(city_1,period,principal_orientation)
            weather_data_1 = refine_resolution(weather_data_1, resolution='600s')
            
            city_2 = 'Marseille'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_2 = get_historical_weather_data(city_2,period,principal_orientation)
            weather_data_2 = refine_resolution(weather_data_2, resolution='600s')
            
            # Définition des habitudes
            conventionnel = Behaviour('conventionnel_th-bce_2020')
            conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
            conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
            
            for uw in tqdm.tqdm(U_value_windows_list):
                
                typo.windows_U = uw
                
                energy_needs_1 = get_annual_energy_needs(typo, weather_data_1, conventionnel)
                energy_needs_mean_list_1.append(energy_needs_1.mean(axis=0))
                energy_needs_std_list_1.append(energy_needs_1.std(axis=0))
                
                energy_needs_2 = get_annual_energy_needs(typo, weather_data_2, conventionnel)
                energy_needs_mean_list_2.append(energy_needs_2.mean(axis=0))
                energy_needs_std_list_2.append(energy_needs_2.std(axis=0))
            
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.errorbar(U_value_windows_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3)
            ax.errorbar(U_value_windows_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3)
            ax.errorbar(U_value_windows_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_2],
                        color='tab:red',label='Heating needs ({})'.format(city_2),ls='--',capsize=3)
            ax.errorbar(U_value_windows_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
                        color='tab:blue',label='Cooling needs ({})'.format(city_2),ls='--',capsize=3)
            ax.set_ylabel('Annual energy needs over {}-{} '.format(period[0],period[1])+'(kWh.m$^{-2}$.yr$^{-1}$)')
            ax.legend()
            ax.set_xlabel('Windows U-value (W.m$^{-2}$.K$^{-1}$)')
            ax.set_ylim(bottom=0.,top=100)
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('effect_windows_u_value_{}_{}_{}-{}'.format(city_1,city_2,period[0],period[1]))),bbox_inches='tight')
            
            plt.show()
        
        # Effets de l'orientation
        if False:
            def get_annual_energy_needs(typology,weather_data,behaviour, by_surface=True):
                simulation_data = SFH_test_model(typology, conventionnel, weather_data)
                simulation_data = aggregate_resolution(simulation_data, resolution='h')
                
                yearly_data = aggregate_resolution(simulation_data, resolution='YE',agg_method='sum')
                yearly_data = yearly_data/1000
                surface_yearly_data = yearly_data/typology.surface
                
                if by_surface:
                    return surface_yearly_data[['heating_needs','cooling_needs']]
                else:
                    return yearly_data[['heating_needs','cooling_needs']]
                
            orientation_list = ['N','NE','E','SE','S','SW','W','NW']
            energy_needs_mean_list_1 = []
            energy_needs_std_list_1 = []
            energy_needs_mean_list_2 = []
            energy_needs_std_list_2 = []
            
            typo_name = 'FR.N.SFH.01.Test'
            typo = Typology(typo_name)
            typo.roof_U = 0.36
            typo.floor_insulation_thickness = 0.1
            typo.w0_insulation_thickness = 0.1
            typo.w1_insulation_thickness = 0.1
            typo.w2_insulation_thickness = 0.1
            typo.w3_insulation_thickness = 0.1
            
            # Définition des habitudes
            conventionnel = Behaviour('conventionnel_th-bce_2020')
            conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
            conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
            
            for ori in tqdm.tqdm(orientation_list):
                typo.w0_orientation = ori
                typo.update_orientation()
                
                # Génération du fichier météo
                city_1 = 'Paris'
                period = [2015,2020]
                principal_orientation = typo.w0_orientation
                weather_data_1 = get_historical_weather_data(city_1,period,ori)
                weather_data_1 = refine_resolution(weather_data_1, resolution='600s')
                
                city_2 = 'Marseille'
                period = [2015,2020]
                principal_orientation = typo.w0_orientation
                weather_data_2 = get_historical_weather_data(city_2,period,ori)
                weather_data_2 = refine_resolution(weather_data_2, resolution='600s')
                
                
                energy_needs_1 = get_annual_energy_needs(typo, weather_data_1, conventionnel)
                energy_needs_mean_list_1.append(energy_needs_1.mean(axis=0))
                energy_needs_std_list_1.append(energy_needs_1.std(axis=0))
                
                energy_needs_2 = get_annual_energy_needs(typo, weather_data_2, conventionnel)
                energy_needs_mean_list_2.append(energy_needs_2.mean(axis=0))
                energy_needs_std_list_2.append(energy_needs_2.std(axis=0))
            
            now_ts = datetime.now().strftime("%Y-%m-%dT%H")
            pickle_file_name = "annual_needs_orientation" + now_ts + ".pickle"
            pickle.dump((energy_needs_mean_list_1, 
                         energy_needs_std_list_1,
                         energy_needs_mean_list_2,
                         energy_needs_std_list_2), open(pickle_file_name, "wb"))
            
            
            energy_needs_mean_list_1, energy_needs_std_list_1,energy_needs_mean_list_2,energy_needs_std_list_2 = pickle.load(open('annual_needs_orientation2024-10-24T17.pickle', 'rb'))
            
            # Plot cartésien (pas fun)
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.errorbar(orientation_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3,ls='',marker='o')
            ax.errorbar(orientation_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3,ls='',marker='o')
            ax.errorbar(orientation_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_2],
                        color='tab:red',label='Heating needs ({})'.format(city_2),ls='',capsize=3,marker='o',mfc='w')
            ax.errorbar(orientation_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
                        color='tab:blue',label='Cooling needs ({})'.format(city_2),ls='',capsize=3,marker='o',mfc='w')
            ax.set_ylabel('Annual energy needs over {}-{} '.format(period[0],period[1])+'(kWh.m$^{-2}$.yr$^{-1}$)')
            ax.legend()
            ax.set_xlabel('Main façade orientation')
            ax.set_ylim(bottom=0.,top=100)
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('effect_orientation_{}_{}_{}-{}'.format(city_1,city_2,period[0],period[1]))),bbox_inches='tight')
            plt.show()
            
            
            # Plot polaire, bcp plus marrant mais illisible : annueler
            # fig,ax = plt.subplots(dpi=300,figsize=(5,5),subplot_kw={'projection': 'polar'})

            # ax.set_theta_zero_location("N")
            # ax.set_theta_direction(-1)
            # ax.set_rlabel_position(0)
            # # ax.set_rlim(bottom=90, top=0)
            # # ax.set_rticks(list(range(0,91,20)))
            # # ax.set_yticklabels(['{}°'.format(e) if e!=0 else '' for e in ax.get_yticks()])
            # ax.set_xticks(np.pi/180. * np.linspace(0,  360, 12, endpoint=False))
            # ax.grid(True)
            # ax.set_xticklabels([{0:'N',90:'E',180:'S',270:'W'}.get(e,'{:.0f}°'.format(e)) for e in np.linspace(0,  360, 12, endpoint=False)])
            
            # ax.errorbar([2*np.pi/360*dict_orientation_angle.get(e) for e in orientation_list], 
            #             [e.loc['heating_needs'] for e in energy_needs_mean_list_1], 
            #             [e.loc['heating_needs'] for e in energy_needs_std_list_1],
            #             color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3,ls='',marker='o')
            # ax.errorbar([2*np.pi/360*dict_orientation_angle.get(e) for e in orientation_list], 
            #             [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
            #             [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
            #             color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3,ls='',marker='o')
            # ax.errorbar([2*np.pi/360*dict_orientation_angle.get(e) for e in orientation_list], 
            #             [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
            #             [e.loc['heating_needs'] for e in energy_needs_std_list_2],
            #             color='tab:red',label='Heating needs ({})'.format(city_2),ls='',capsize=3,marker='o',mfc='w')
            # ax.errorbar([2*np.pi/360*dict_orientation_angle.get(e) for e in orientation_list], 
            #             [e.loc['cooling_needs'] for e in energy_needs_mean_list_2], 
            #             [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
            #             color='tab:blue',label='Cooling needs ({})'.format(city_2),ls='',capsize=3,marker='o',mfc='w')
            # ax.set_title('Annual energy needs over {}-{} '.format(period[0],period[1])+'(kWh.m$^{-2}$.yr$^{-1}$)')
            # plt.savefig(os.path.join(figs_folder,'{}.png'.format('effect_orientation_polar_plot_{}_{}_{}-{}'.format(city_1,city_2,period[0],period[1]))),bbox_inches='tight')
            # plt.show()
            
        # Étude de l'effet de l'épaisseur d'isolant de plancher sur la consommation annuelle
        if True:
            def get_annual_energy_needs(typology,weather_data,behaviour, by_surface=True):
                simulation_data = SFH_test_model(typology, conventionnel, weather_data,progressbar=False)
                simulation_data = aggregate_resolution(simulation_data, resolution='h')
                
                yearly_data = aggregate_resolution(simulation_data, resolution='YE',agg_method='sum')
                yearly_data = yearly_data/1000
                surface_yearly_data = yearly_data/typology.surface
                
                if by_surface:
                    return surface_yearly_data[['heating_needs','cooling_needs']]
                else:
                    return yearly_data[['heating_needs','cooling_needs']]
                
            thickness_list = np.linspace(0,0.2,10)
            energy_needs_mean_list_1 = []
            energy_needs_std_list_1 = []
            energy_needs_mean_list_2 = []
            energy_needs_std_list_2 = []
            
            typo_name = 'FR.N.SFH.01.Test'
            typo = Typology(typo_name)
            typo.roof_U = 0.36
            typo.w0_insulation_thickness = 0.1
            typo.w1_insulation_thickness = 0.1
            typo.w2_insulation_thickness = 0.1
            typo.w3_insulation_thickness = 0.1
            
            
            # Génération du fichier météo
            city_1 = 'Paris'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_1 = get_historical_weather_data(city_1,period,principal_orientation)
            weather_data_1 = refine_resolution(weather_data_1, resolution='600s')
            
            city_2 = 'Marseille'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_2 = get_historical_weather_data(city_2,period,principal_orientation)
            weather_data_2 = refine_resolution(weather_data_2, resolution='600s')
            
            # Définition des habitudes
            conventionnel = Behaviour('conventionnel_th-bce_2020')
            conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
            conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
            
            for thickness in tqdm.tqdm(thickness_list):
                
                typo.floor_insulation_thickness = thickness
                
                energy_needs_1 = get_annual_energy_needs(typo, weather_data_1, conventionnel)
                energy_needs_mean_list_1.append(energy_needs_1.mean(axis=0))
                energy_needs_std_list_1.append(energy_needs_1.std(axis=0))
                
                energy_needs_2 = get_annual_energy_needs(typo, weather_data_2, conventionnel)
                energy_needs_mean_list_2.append(energy_needs_2.mean(axis=0))
                energy_needs_std_list_2.append(energy_needs_2.std(axis=0))
            
            
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.errorbar(thickness_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_2],
                        color='tab:red',label='Heating needs ({})'.format(city_2),ls='--',capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
                        color='tab:blue',label='Cooling needs ({})'.format(city_2),ls='--',capsize=3)
            ax.set_ylabel('Annual energy needs over {}-{} '.format(period[0],period[1])+'(kWh.m$^{-2}$.yr$^{-1}$)')
            # ax.legend()
            ax.set_xlabel('Floor insulation thickness (m)')
            ax.set_ylim(bottom=0.,top=150)
            # ax.set_ylim()
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('effect_floor_insulation_thickness_{}_{}_{}-{}'.format(city_1,city_2,period[0],period[1]))),bbox_inches='tight')
            
            plt.show()
            
        
        # Étude de l'effet de la surface de vitrage 
        if False:
            def get_annual_energy_needs(typology,weather_data,behaviour, by_surface=True):
                simulation_data = SFH_test_model(typology, conventionnel, weather_data,progressbar=False)
                simulation_data = aggregate_resolution(simulation_data, resolution='h')
                
                yearly_data = aggregate_resolution(simulation_data, resolution='YE',agg_method='sum')
                yearly_data = yearly_data/1000
                surface_yearly_data = yearly_data/typology.surface
                
                if by_surface:
                    return surface_yearly_data[['heating_needs','cooling_needs']]
                else:
                    return yearly_data[['heating_needs','cooling_needs']]
                
            thickness_list = np.linspace(0.5,2,15)
            energy_needs_mean_list_1 = []
            energy_needs_std_list_1 = []
            energy_needs_mean_list_2 = []
            energy_needs_std_list_2 = []
            
            typo_name = 'FR.N.SFH.01.Test'
            typo = Typology(typo_name)
            typo.roof_U = 0.36
            typo.w0_insulation_thickness = 0.1
            typo.w1_insulation_thickness = 0.1
            typo.w2_insulation_thickness = 0.1
            typo.w3_insulation_thickness = 0.1
            typo.floor_insulation_thickness = 0
            
            w0_gs = typo.w0_windows_surface
            w1_gs = typo.w1_windows_surface
            w2_gs = typo.w2_windows_surface
            w3_gs = typo.w3_windows_surface
            
            # Génération du fichier météo
            city_1 = 'Paris'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_1 = get_historical_weather_data(city_1,period,principal_orientation)
            weather_data_1 = refine_resolution(weather_data_1, resolution='600s')
            
            city_2 = 'Marseille'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_2 = get_historical_weather_data(city_2,period,principal_orientation)
            weather_data_2 = refine_resolution(weather_data_2, resolution='600s')
            
            # Définition des habitudes
            conventionnel = Behaviour('conventionnel_th-bce_2020')
            conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
            conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
            
            for surface_multi in tqdm.tqdm(thickness_list):
                
                typo.w0_windows_surface = w0_gs*surface_multi
                typo.w1_windows_surface = w1_gs*surface_multi
                typo.w2_windows_surface = w2_gs*surface_multi
                typo.w3_windows_surface = w3_gs*surface_multi
                
                energy_needs_1 = get_annual_energy_needs(typo, weather_data_1, conventionnel)
                energy_needs_mean_list_1.append(energy_needs_1.mean(axis=0))
                energy_needs_std_list_1.append(energy_needs_1.std(axis=0))
                
                energy_needs_2 = get_annual_energy_needs(typo, weather_data_2, conventionnel)
                energy_needs_mean_list_2.append(energy_needs_2.mean(axis=0))
                energy_needs_std_list_2.append(energy_needs_2.std(axis=0))
            
            
            glazing_surface_total = w0_gs + w1_gs + w2_gs + w3_gs
            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.errorbar(thickness_list*glazing_surface_total, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list*glazing_surface_total, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list*glazing_surface_total, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_2],
                        color='tab:red',label='Heating needs ({})'.format(city_2),ls='--',capsize=3)
            ax.errorbar(thickness_list*glazing_surface_total, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
                        color='tab:blue',label='Cooling needs ({})'.format(city_2),ls='--',capsize=3)
            ax.set_ylabel('Annual energy needs over {}-{} '.format(period[0],period[1])+'(kWh.m$^{-2}$.yr$^{-1}$)')
            # ax.legend()
            ax.set_xlabel('Total glazing surface (m2)')
            ax.set_ylim(bottom=0.,top=150)
            # ax.set_ylim()
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('effect_glazing_surface_{}_{}_{}-{}'.format(city_1,city_2,period[0],period[1]))),bbox_inches='tight')
            
            plt.show()
            
            
        # Étude de l'effet de la valeur Uw des vitrages
        if False:
            def get_annual_energy_needs(typology,weather_data,behaviour, by_surface=True):
                simulation_data = SFH_test_model(typology, conventionnel, weather_data,progressbar=False)
                simulation_data = aggregate_resolution(simulation_data, resolution='h')
                
                yearly_data = aggregate_resolution(simulation_data, resolution='YE',agg_method='sum')
                yearly_data = yearly_data/1000
                surface_yearly_data = yearly_data/typology.surface
                
                if by_surface:
                    return surface_yearly_data[['heating_needs','cooling_needs']]
                else:
                    return yearly_data[['heating_needs','cooling_needs']]
                
            Uw_list = np.linspace(0.5,5,15)
            energy_needs_mean_list_1 = []
            energy_needs_std_list_1 = []
            energy_needs_mean_list_2 = []
            energy_needs_std_list_2 = []
            
            typo_name = 'FR.N.SFH.01.Test'
            typo = Typology(typo_name)
            typo.roof_U = 0.36
            typo.w0_insulation_thickness = 0.1
            typo.w1_insulation_thickness = 0.1
            typo.w2_insulation_thickness = 0.1
            typo.w3_insulation_thickness = 0.1
            typo.floor_insulation_thickness = 0
            
            # Génération du fichier météo
            city_1 = 'Paris'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_1 = get_historical_weather_data(city_1,period,principal_orientation)
            weather_data_1 = refine_resolution(weather_data_1, resolution='600s')
            
            city_2 = 'Marseille'
            period = [2015,2020]
            principal_orientation = typo.w0_orientation
            weather_data_2 = get_historical_weather_data(city_2,period,principal_orientation)
            weather_data_2 = refine_resolution(weather_data_2, resolution='600s')
            
            # Définition des habitudes
            conventionnel = Behaviour('conventionnel_th-bce_2020')
            conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
            conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
            
            for u in tqdm.tqdm(Uw_list):
                
                typo.windows_U = u
                
                energy_needs_1 = get_annual_energy_needs(typo, weather_data_1, conventionnel)
                energy_needs_mean_list_1.append(energy_needs_1.mean(axis=0))
                energy_needs_std_list_1.append(energy_needs_1.std(axis=0))
                
                energy_needs_2 = get_annual_energy_needs(typo, weather_data_2, conventionnel)
                energy_needs_mean_list_2.append(energy_needs_2.mean(axis=0))
                energy_needs_std_list_2.append(energy_needs_2.std(axis=0))
            
                if u > 4.9:
                    simulation_data = SFH_test_model(typo, conventionnel, weather_data_1,progressbar=False)
                    plot_timeserie(simulation_data[['temperature_2m','internal_temperature']],figsize=(15,5),
                                   xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],ylabel='Temperature (°C)',)
                                   # figs_folder=figs_folder, save_fig='thermal_model_temperature_{}_{}_{}'.format(city,year,typo_name))

            fig,ax = plt.subplots(figsize=(5,5),dpi=300)
            ax.errorbar(Uw_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3)
            ax.errorbar(Uw_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3)
            ax.errorbar(Uw_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['heating_needs'] for e in energy_needs_std_list_2],
                        color='tab:red',label='Heating needs ({})'.format(city_2),ls='--',capsize=3)
            ax.errorbar(Uw_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
                        color='tab:blue',label='Cooling needs ({})'.format(city_2),ls='--',capsize=3)
            ax.set_ylabel('Annual energy needs over {}-{} '.format(period[0],period[1])+'(kWh.m$^{-2}$.yr$^{-1}$)')
            # ax.legend()
            ax.set_xlabel('Windows U-value (W.m$^{-2}$.K$^{-1}$)')
            ax.set_ylim(bottom=0.,top=150)
            # ax.set_ylim()
            
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('effect_windows_U_{}_{}_{}-{}'.format(city_1,city_2,period[0],period[1]))),bbox_inches='tight')
            
            plt.show()
            
            
        # Étude de l'effet des scénarios de chgmt climatique
        if False:
            
            # Premier test :
            if True:
                # Définition de la typologie
                typo_name = 'FR.N.SFH.01.Test'
                typo = Typology(typo_name)
                typo.roof_U = 0.36
                typo.w0_insulation_thickness = 0.1
                typo.w1_insulation_thickness = 0.1
                typo.w2_insulation_thickness = 0.1
                typo.w3_insulation_thickness = 0.1
                typo.floor_insulation_thickness = 0
                
                # Génération du fichier météo
                city = 'Marseille'
                climat = Climat(Departement(13).climat)
                
                period = [2085,2090]
                
                rcp = 85
                nmod = 0
                
                principal_orientation = typo.w0_orientation
                
                weather_data = get_projected_weather_data(city=city,
                                                          zcl_codint=climat.codint,
                                                          nmod=nmod,
                                                          rcp=rcp,
                                                          future_period=period,
                                                          principal_orientation=principal_orientation)
                weather_data = refine_resolution(weather_data, resolution='600s')
                
                # Définition des habitudes
                conventionnel = Behaviour('conventionnel_th-bce_2020')
                conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
                conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
                
                year = 2020
                
                simulation_data = SFH_test_model(typo, conventionnel, weather_data,progressbar=False)
                simulation_data = aggregate_resolution(simulation_data, resolution='h')
                
                plot_timeserie(simulation_data[['temperature_2m','internal_temperature']],figsize=(15,5),
                               xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],ylabel='Temperature (°C)',
                               figs_folder=figs_folder, save_fig='thermal_model_temperature_{}_{}_{}'.format(city,period[0],typo_name))
                
                plot_timeserie(simulation_data[['heating_needs','cooling_needs']],figsize=(15,5),
                               xlim=[pd.to_datetime('{}-01-01'.format(year)), pd.to_datetime('{}-12-31'.format(year))],ylabel='Energy needs (Wh)',
                               figs_folder=figs_folder, save_fig='thermal_model_energy_needs_{}_{}_{}'.format(city,period[0],typo_name))
                
                
                simulation_data = aggregate_resolution(simulation_data, resolution='YE',agg_method='sum')
                annual_heating_consumption = simulation_data.heating_needs.mean()/1000
                surface_annual_heating_consumption = annual_heating_consumption/typo.surface
                
                annual_cooling_consumption = simulation_data.cooling_needs.mean()/1000
                surface_annual_cooling_consumption = annual_cooling_consumption/typo.surface
                
                now_ts = datetime.now().strftime("%Y-%m-%dT%H")
                pickle_file_name = "annual_needs_" + now_ts + ".pickle"
                pickle.dump((annual_heating_consumption, annual_cooling_consumption), open(pickle_file_name, "wb"))
                
                print('Besoins annuels de chauffage à {} en {}: {:.0f} kWh/an'.format(city,year, annual_heating_consumption))
                print('Besoins annuels de chauffage à {} en {}: {:.0f} kWh/(m2.an)'.format(city, year, surface_annual_heating_consumption))
                print('Besoins annuels de refroidissement à {} en {}: {:.0f} kWh/an'.format(city, year, annual_cooling_consumption))
                print('Besoins annuels de refroidissement à {} en {}: {:.0f} kWh/(m2.an)'.format(city, year, surface_annual_cooling_consumption))
            
            # Graphe 3 sur les effets du changement climatique
            # attention, pour l'instant la météo future est en carton (un peu)
            if True:
                period_list = [2020,2040,2060,2080,2100]
                # period_list = [2099]
                mod_list = list(range(9))
                rcp_list = [45,85]
                    
                energy_needs_dict_1 = dict()
                energy_needs_dict_2 = dict()
                
                # Définition de la typologie
                typo_name = 'FR.N.SFH.01.Test'
                typo = Typology(typo_name)
                typo.roof_U = 0.36
                typo.w0_insulation_thickness = 0.1
                typo.w1_insulation_thickness = 0.1
                typo.w2_insulation_thickness = 0.1
                typo.w3_insulation_thickness = 0.1
                typo.floor_insulation_thickness = 0
                
                # Définition des habitudes
                conventionnel = Behaviour('conventionnel_th-bce_2020')
                conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
                conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
                
                for y in tqdm.tqdm(period_list):
                    for nmod in mod_list:
                        for rcp in rcp_list:
                            period = [y-10,y]
                            principal_orientation = typo.w0_orientation
                            
                            # Génération du fichier météo
                            city_1 = 'Paris'
                            climat_1 = Climat(Departement(75).climat)
                            
                            weather_data_1 = get_projected_weather_data(city=city_1,
                                                                        zcl_codint=climat_1.codint,
                                                                        nmod=nmod,
                                                                        rcp=rcp,
                                                                        future_period=period,
                                                                        principal_orientation=principal_orientation)
                            weather_data_1 = refine_resolution(weather_data_1, resolution='600s')
                            
                            city_2 = 'Marseille'
                            climat_2 = Climat(Departement(13).climat)
                            weather_data_2 = get_projected_weather_data(city=city_2,
                                                                        zcl_codint=climat_2.codint,
                                                                        nmod=nmod,
                                                                        rcp=rcp,
                                                                        future_period=period,
                                                                        principal_orientation=principal_orientation)
                            weather_data_2 = refine_resolution(weather_data_2, resolution='600s')
                            
                            simulation_data_1 = SFH_test_model(typo, conventionnel, weather_data_1,progressbar=False)
                            simulation_data_1 = aggregate_resolution(simulation_data_1, resolution='h')
                            simulation_data_1 = aggregate_resolution(simulation_data_1, resolution='YE',agg_method='sum')
                            
                            annual_heating_consumption_1 = simulation_data_1.heating_needs.mean()/1000
                            surface_annual_heating_consumption_1 = annual_heating_consumption_1/typo.surface
                            annual_cooling_consumption_1 = simulation_data_1.cooling_needs.mean()/1000
                            surface_annual_cooling_consumption_1 = annual_cooling_consumption_1/typo.surface
                            
                            simulation_data_2 = SFH_test_model(typo, conventionnel, weather_data_2,progressbar=False)
                            simulation_data_2 = aggregate_resolution(simulation_data_2, resolution='h')
                            simulation_data_2 = aggregate_resolution(simulation_data_2, resolution='YE',agg_method='sum')
                            
                            annual_heating_consumption_2 = simulation_data_2.heating_needs.mean()/1000
                            surface_annual_heating_consumption_2 = annual_heating_consumption_2/typo.surface
                            annual_cooling_consumption_2 = simulation_data_2.cooling_needs.mean()/1000
                            surface_annual_cooling_consumption_2 = annual_cooling_consumption_2/typo.surface
                            
                            energy_needs_dict_1[(y,rcp,nmod)] = surface_annual_heating_consumption_1,surface_annual_cooling_consumption_1
                            energy_needs_dict_2[(y,rcp,nmod)] = surface_annual_heating_consumption_2,surface_annual_cooling_consumption_2
                        
                
                # save data as pickles
                now_ts = datetime.now().strftime("%Y-%m-%dT%H")
                pickle_file_name = "energy_needs_" + now_ts + ".pickle"
                pickle.dump((energy_needs_dict_1, energy_needs_dict_2), open(pickle_file_name, "wb"))
                
                energy_needs_dict_1, energy_needs_dict_2 = pickle.load(open('energy_needs_2024-10-17T04.pickle', 'rb'))
                mod_list = list(range(9))
                rcp_list = [45,85]
                
                # from saver_energy_needs_climate_change import energy_needs_dict_1, energy_needs_dict_2
                period_list = [2020,2040,2060,2080,2100]
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                
                color_dict = {'cooling_rcp45':'cornflowerblue',
                              'cooling_rcp85':'darkblue',
                              'heating_rcp45':'lightcoral',
                              'heating_rcp85':'tab:red',}
                
                ls_dict = {'city1':'-',
                           'city2':'--',}
                
                cooling_rcp45_city1_mean = []
                cooling_rcp45_city1_std = []
                heating_rcp45_city1_mean = []
                heating_rcp45_city1_std = []
                
                cooling_rcp85_city1_mean = []
                cooling_rcp85_city1_std = []
                heating_rcp85_city1_mean = []
                heating_rcp85_city1_std = []
                
                # cooling_rcp45_city2_mean = []
                # cooling_rcp45_city2_std = []
                # heating_rcp45_city2_mean = []
                # heating_rcp45_city2_std = []
                
                # cooling_rcp85_city2_mean = []
                # cooling_rcp85_city2_std = []
                # heating_rcp85_city2_mean = []
                # heating_rcp85_city2_std = []
                
                for y in period_list:
                    cooling_rcp45_city1 = np.asarray([energy_needs_dict_1.get((y,45,nm))[1] for nm in mod_list])
                    cooling_rcp45_city1_mean.append(np.mean(cooling_rcp45_city1))
                    cooling_rcp45_city1_std.append(np.std(cooling_rcp45_city1))
                    heating_rcp45_city1 = np.asarray([energy_needs_dict_1.get((y,45,nm))[0] for nm in mod_list])
                    heating_rcp45_city1_mean.append(np.mean(heating_rcp45_city1))
                    heating_rcp45_city1_std.append(np.std(heating_rcp45_city1))
                    
                    cooling_rcp85_city1 = np.asarray([energy_needs_dict_1.get((y,85,nm))[1] for nm in mod_list])
                    cooling_rcp85_city1_mean.append(np.mean(cooling_rcp85_city1))
                    cooling_rcp85_city1_std.append(np.std(cooling_rcp85_city1))
                    heating_rcp85_city1 = np.asarray([energy_needs_dict_1.get((y,85,nm))[0] for nm in mod_list])
                    heating_rcp85_city1_mean.append(np.mean(heating_rcp85_city1))
                    heating_rcp85_city1_std.append(np.std(heating_rcp85_city1))
                    
                    # cooling_rcp45_city2 = np.asarray([energy_needs_dict_2.get((y,45,nm))[1] for nm in mod_list])
                    # cooling_rcp45_city2_mean.append(np.mean(cooling_rcp45_city2))
                    # cooling_rcp45_city2_std.append(np.std(cooling_rcp45_city2))
                    # heating_rcp45_city2 = np.asarray([energy_needs_dict_2.get((y,45,nm))[0] for nm in mod_list])
                    # heating_rcp45_city2_mean.append(np.mean(heating_rcp45_city2))
                    # heating_rcp45_city2_std.append(np.std(heating_rcp45_city2))
                    
                    # cooling_rcp85_city2 = np.asarray([energy_needs_dict_2.get((y,85,nm))[1] for nm in mod_list])
                    # cooling_rcp85_city2_mean.append(np.mean(cooling_rcp85_city2))
                    # cooling_rcp85_city2_std.append(np.std(cooling_rcp85_city2))
                    # heating_rcp85_city2 = np.asarray([energy_needs_dict_2.get((y,85,nm))[0] for nm in mod_list])
                    # heating_rcp85_city2_mean.append(np.mean(heating_rcp85_city2))
                    # heating_rcp85_city2_std.append(np.std(heating_rcp85_city2))
                    
                ax.errorbar(period_list, 
                            cooling_rcp45_city1_mean, 
                            cooling_rcp45_city1_std,
                            color=color_dict.get('cooling_rcp45'),
                            ls=ls_dict.get('city1'),capsize=3)
                
                ax.errorbar(period_list, 
                            heating_rcp45_city1_mean, 
                            heating_rcp45_city1_std,
                            color=color_dict.get('heating_rcp45'),
                            ls=ls_dict.get('city1'),capsize=3)
                
                ax.errorbar(period_list, 
                            cooling_rcp85_city1_mean, 
                            cooling_rcp85_city1_std,
                            color=color_dict.get('cooling_rcp85'),
                            ls=ls_dict.get('city1'),capsize=3)
                
                ax.errorbar(period_list, 
                            heating_rcp85_city1_mean, 
                            heating_rcp85_city1_std,
                            color=color_dict.get('heating_rcp85'),
                            ls=ls_dict.get('city1'),capsize=3)
                
                # ax.errorbar(period_list, 
                #             cooling_rcp45_city2_mean, 
                #             cooling_rcp45_city2_std,
                #             color=color_dict.get('cooling_rcp45'),
                #             ls=ls_dict.get('city2'),capsize=3)
                
                # ax.errorbar(period_list, 
                #             heating_rcp45_city2_mean, 
                #             heating_rcp45_city2_std,
                #             color=color_dict.get('heating_rcp45'),
                #             ls=ls_dict.get('city2'),capsize=3)
                
                # ax.errorbar(period_list, 
                #             cooling_rcp85_city2_mean, 
                #             cooling_rcp85_city2_std,
                #             color=color_dict.get('cooling_rcp85'),
                #             ls=ls_dict.get('city2'),capsize=3)
                
                # ax.errorbar(period_list, 
                #             heating_rcp85_city2_mean, 
                #             heating_rcp85_city2_std,
                #             color=color_dict.get('heating_rcp85'),
                #             ls=ls_dict.get('city2'),capsize=3)
                
                ax.plot([2020],[0],label='RCP4.5',color='silver')
                ax.plot([2020],[0],label='RCP8.5',color='grey')
                
                ax.set_ylabel('Annual energy needs (kWh.m$^{-2}$.yr$^{-1}$)')
                ax.legend()
                # ax.set_xlabel('Floor insulation thickness (m)')
                ax.set_ylim(bottom=0.,top=150)
                # ax.set_ylim()
                
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('effect_climate_change_{}'.format(city_1))),bbox_inches='tight')
                
                plt.show()
                
                
                period_list = [2020,2040,2060,2080,2100]
                
                fig,ax = plt.subplots(figsize=(5,5),dpi=300)
                
                color_dict = {'cooling_rcp45':'cornflowerblue',
                              'cooling_rcp85':'darkblue',
                              'heating_rcp45':'lightcoral',
                              'heating_rcp85':'tab:red',}
                
                ls_dict = {'city1':'-',
                           'city2':'--',}
                
                # cooling_rcp45_city1_mean = []
                # cooling_rcp45_city1_std = []
                # heating_rcp45_city1_mean = []
                # heating_rcp45_city1_std = []
                
                # cooling_rcp85_city1_mean = []
                # cooling_rcp85_city1_std = []
                # heating_rcp85_city1_mean = []
                # heating_rcp85_city1_std = []
                
                cooling_rcp45_city2_mean = []
                cooling_rcp45_city2_std = []
                heating_rcp45_city2_mean = []
                heating_rcp45_city2_std = []
                
                cooling_rcp85_city2_mean = []
                cooling_rcp85_city2_std = []
                heating_rcp85_city2_mean = []
                heating_rcp85_city2_std = []
                
                for y in period_list:
                    # cooling_rcp45_city1 = np.asarray([energy_needs_dict_1.get((y,45,nm))[1] for nm in mod_list])
                    # cooling_rcp45_city1_mean.append(np.mean(cooling_rcp45_city1))
                    # cooling_rcp45_city1_std.append(np.std(cooling_rcp45_city1))
                    # heating_rcp45_city1 = np.asarray([energy_needs_dict_1.get((y,45,nm))[0] for nm in mod_list])
                    # heating_rcp45_city1_mean.append(np.mean(heating_rcp45_city1))
                    # heating_rcp45_city1_std.append(np.std(heating_rcp45_city1))
                    
                    # cooling_rcp85_city1 = np.asarray([energy_needs_dict_1.get((y,85,nm))[1] for nm in mod_list])
                    # cooling_rcp85_city1_mean.append(np.mean(cooling_rcp85_city1))
                    # cooling_rcp85_city1_std.append(np.std(cooling_rcp85_city1))
                    # heating_rcp85_city1 = np.asarray([energy_needs_dict_1.get((y,85,nm))[0] for nm in mod_list])
                    # heating_rcp85_city1_mean.append(np.mean(heating_rcp85_city1))
                    # heating_rcp85_city1_std.append(np.std(heating_rcp85_city1))
                    
                    cooling_rcp45_city2 = np.asarray([energy_needs_dict_2.get((y,45,nm))[1] for nm in mod_list])
                    cooling_rcp45_city2_mean.append(np.mean(cooling_rcp45_city2))
                    cooling_rcp45_city2_std.append(np.std(cooling_rcp45_city2))
                    heating_rcp45_city2 = np.asarray([energy_needs_dict_2.get((y,45,nm))[0] for nm in mod_list])
                    heating_rcp45_city2_mean.append(np.mean(heating_rcp45_city2))
                    heating_rcp45_city2_std.append(np.std(heating_rcp45_city2))
                    
                    cooling_rcp85_city2 = np.asarray([energy_needs_dict_2.get((y,85,nm))[1] for nm in mod_list])
                    cooling_rcp85_city2_mean.append(np.mean(cooling_rcp85_city2))
                    cooling_rcp85_city2_std.append(np.std(cooling_rcp85_city2))
                    heating_rcp85_city2 = np.asarray([energy_needs_dict_2.get((y,85,nm))[0] for nm in mod_list])
                    heating_rcp85_city2_mean.append(np.mean(heating_rcp85_city2))
                    heating_rcp85_city2_std.append(np.std(heating_rcp85_city2))
                    
                # ax.errorbar(period_list, 
                #             cooling_rcp45_city1_mean, 
                #             cooling_rcp45_city1_std,
                #             color=color_dict.get('cooling_rcp45'),
                #             ls=ls_dict.get('city1'),capsize=3)
                
                # ax.errorbar(period_list, 
                #             heating_rcp45_city1_mean, 
                #             heating_rcp45_city1_std,
                #             color=color_dict.get('heating_rcp45'),
                #             ls=ls_dict.get('city1'),capsize=3)
                
                # ax.errorbar(period_list, 
                #             cooling_rcp85_city1_mean, 
                #             cooling_rcp85_city1_std,
                #             color=color_dict.get('cooling_rcp85'),
                #             ls=ls_dict.get('city1'),capsize=3)
                
                # ax.errorbar(period_list, 
                #             heating_rcp85_city1_mean, 
                #             heating_rcp85_city1_std,
                #             color=color_dict.get('heating_rcp85'),
                #             ls=ls_dict.get('city1'),capsize=3)
                
                ax.errorbar(period_list, 
                            cooling_rcp45_city2_mean, 
                            cooling_rcp45_city2_std,
                            color=color_dict.get('cooling_rcp45'),
                            ls=ls_dict.get('city2'),capsize=3)
                
                ax.errorbar(period_list, 
                            heating_rcp45_city2_mean, 
                            heating_rcp45_city2_std,
                            color=color_dict.get('heating_rcp45'),
                            ls=ls_dict.get('city2'),capsize=3)
                
                ax.errorbar(period_list, 
                            cooling_rcp85_city2_mean, 
                            cooling_rcp85_city2_std,
                            color=color_dict.get('cooling_rcp85'),
                            ls=ls_dict.get('city2'),capsize=3)
                
                ax.errorbar(period_list, 
                            heating_rcp85_city2_mean, 
                            heating_rcp85_city2_std,
                            color=color_dict.get('heating_rcp85'),
                            ls=ls_dict.get('city2'),capsize=3)
                
                ax.plot([2020],[0],label='RCP4.5',color='silver')
                ax.plot([2020],[0],label='RCP8.5',color='grey')
                
                ax.set_ylabel('Annual energy needs (kWh.m$^{-2}$.yr$^{-1}$)')
                ax.legend()
                # ax.set_xlabel('Floor insulation thickness (m)')
                ax.set_ylim(bottom=0.,top=150)
                # ax.set_ylim()
                
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('effect_climate_change_{}'.format(city_2))),bbox_inches='tight')
                
                plt.show()
    
    #%% Comparaisons des typologies TABULA
    if True:
        # Génération du fichier météo
        city = 'Beauvais'
        # period = [2010,2020]
        # period = [1990,2000]
        period = [2010,2020]
        
        # Checkpoint weather data
        weather_data_checkfile = ".weather_data_{}_{}_{}_".format(city,period[0],period[1]) + today + ".pickle"
        if weather_data_checkfile not in os.listdir():
            weather_data = get_historical_weather_data(city,period)
            weather_data = refine_resolution(weather_data, resolution='600s')
            pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
        else:
            weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
        
        # Carte de France avec la localisation de la ville
        if False:
            draw_climat_map({Climat(e):None for e in France().climats},zcl_label=False, 
                            figs_folder=figs_folder, save='map_France_{}'.format(city),
                            add_city_points=[city])
        
        # Affichage des données météo 
        if False:
            plot_weather_data = aggregate_resolution(weather_data.copy(), resolution='d')
            plot_timeserie(plot_weather_data[['temperature_2m']],figsize=(15,5),
                           labels=[city], ylabel='Mean air temperature (°C)',
                           xlim=[pd.to_datetime('{}-01-01'.format(period[0])), pd.to_datetime('{}-12-31'.format(period[1]))],
                           figs_folder=figs_folder, save_fig='temperature_{}_{}_{}'.format(city,period[0],period[1]))
        
        # Définition des habitudes
        conventionnel = Behaviour('conventionnel_th-bce_2020')
        conventionnel.heating_rules = {i:[19]*24 for i in range(1,8)}
        conventionnel.cooling_rules = {i:[26]*24 for i in range(1,8)}
        # changer les règles de présence ? 
    
        
        # Premier test du modèle général
        if False:
            # Définition de la typologie
            typo_name = 'FR.N.SFH.01.Gen'
            typo = Typology(typo_name)
            
            t1 = time.time()
            simulation = run_thermal_model(typo, conventionnel, weather_data)
            simulation = aggregate_resolution(simulation, resolution='h')
            t2 = time.time()
            print('{} ans de simulation : {:.2f}s.'.format(len(list(range(*period))),t2-t1))
            
            # print(simulation_data.columns)
            heating_cooling_modelling = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
            heating_cooling_modelling = heating_cooling_modelling/1000
            heating_cooling_modelling = heating_cooling_modelling/typo.surface
            heating_cooling_modelling.index = heating_cooling_modelling.index.year
            
            print(heating_cooling_modelling)
            
        
        # Test de vitesse de calcul en fonction de la durée de la période
        if False:
            period_len_list = list(range(1,31))
            speed_list = []
            
            tmp_checkfile = ".speed_test_{}_".format(city) + today + ".pickle"
            if tmp_checkfile not in os.listdir():
                for y in tqdm.tqdm(period_len_list):
                    period_test = [1990,1990+y-1]
                    
                    # Checkpoint weather data
                    weather_data_checkfile = ".weather_data_{}_{}_{}_".format(city,period_test[0],period_test[1]) + today + ".pickle"
                    if weather_data_checkfile not in os.listdir():
                        weather_data = get_historical_weather_data(city,period_test)
                        weather_data = refine_resolution(weather_data, resolution='600s')
                        pickle.dump(weather_data, open(weather_data_checkfile, "wb"))
                    else:
                        weather_data = pickle.load(open(weather_data_checkfile, 'rb'))
                
                    # Définition de la typologie
                    typo_name = 'FR.N.SFH.01.Gen'
                    typo = Typology(typo_name)
                    
                    t1 = time.time()
                    simulation = run_thermal_model(typo, conventionnel, weather_data)
                    simulation = aggregate_resolution(simulation, resolution='h')
                    t2 = time.time()
                    speed_list.append(t2-t1)
                
            if tmp_checkfile not in os.listdir():
                pickle.dump(speed_list, open(tmp_checkfile, "wb"))
            
            speed_list = pickle.load(open(tmp_checkfile, 'rb'))
            
            fig,ax= plt.subplots(figsize=(5,5),dpi=300)
            ax.plot(period_len_list,speed_list,marker='o',ls='',label='data')
            ax.set_ylim(bottom=0.)
            ax.set_xlim(left=0.)
            ax.set_xlabel('Modelling period (yr)')
            ax.set_ylabel('Computation time (s)')
            X = np.asarray(period_len_list)
            Y = np.asarray(speed_list)
            a,b = np.polyfit(X,Y,deg=1)
            Y_hat = a*X+b
            r2 = r2_score(Y, Y_hat)
            ax.plot(X,Y_hat,color='k',label='linear fit (R$^2$={:.2f}, {:.2f} s'.format(r2,a)+'.yr$^{-1}$)')
            ax.legend()
            plt.savefig(os.path.join(figs_folder,'{}.png'.format('computation_speed')),bbox_inches='tight')
            
            plt.show()
            
        
        # Comparaison entre typologies
        if True:
            
            # for building_type in ['SFH','TH','MFH','AB']:
            for building_type in ['TH']:
            
                heating_needs_TABULA = {}
                heating_needs_GENMOD = {}
                for i in tqdm.tqdm(range(1,11),desc=building_type):
                    code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
    
                    for level in ['initial','standard','advanced']:
                        typo = Typology(code,level)
                        
                        heating_needs_TABULA[(code,level)] = typo.tabula_heating_needs
                        
                        tmp_checkfile = ".heating_needs_{}_{}_{}_{}_".format(city,period[0],period[1],building_type) + today + ".pickle"
                        if tmp_checkfile not in os.listdir():
                            simulation = run_thermal_model(typo, conventionnel, weather_data)
                            simulation = aggregate_resolution(simulation, resolution='h')
                            
                            heating_cooling_modelling = aggregate_resolution(simulation[['heating_needs','cooling_needs']], resolution='YE',agg_method='sum')
                            heating_cooling_modelling = heating_cooling_modelling/1000
                            heating_cooling_modelling = heating_cooling_modelling/typo.surface
                            heating_cooling_modelling.index = heating_cooling_modelling.index.year
                            
                            heating_needs_GENMOD[(code,level)] = heating_cooling_modelling.heating_needs.values
                        
                
                if tmp_checkfile not in os.listdir():
                    pickle.dump(heating_needs_GENMOD, open(tmp_checkfile, "wb"))
                
                heating_needs_GENMOD = pickle.load(open(tmp_checkfile, 'rb'))
                
                fig,ax = plt.subplots(figsize=(15,5),dpi=300)
                for i in range(1,11):
                    code = 'FR.N.{}.{:02d}.Gen'.format(building_type,i)
                    
                    j = i*7
                    X = [j,j+2,j+4]
                    Y = [heating_needs_TABULA.get(('FR.N.{}.{:02d}.Gen'.format(building_type,i),e)) for e in ['initial','standard','advanced']]
                    
                    if i == 1:
                        ax.plot(X,Y,color='tab:blue',ls=':',marker='o',label='TABULA')
                    else:
                        ax.plot(X,Y,color='tab:blue',ls=':',marker='o')
                        
                    for k,level in enumerate(['initial','standard','advanced']):
                        if i==1 and k==1:
                            ax.boxplot(heating_needs_GENMOD[(code,level)],positions=[X[k]],widths=1, label='Model')
                        else:
                            ax.boxplot(heating_needs_GENMOD[(code,level)],positions=[X[k]],widths=1)
                        
                        
                ax.set_ylim(bottom=0.)
                ax.set_ylabel('Heating needs (kWh.m$^{-2}$.yr$^{-1}$)')
                ax.legend()
                ax.set_xticks([(i*7)+2 for i in range(1,11)],['{}.{:02d}'.format(building_type,i) for i in range(1,11)])
                
                plt.savefig(os.path.join(figs_folder,'{}.png'.format('{}_TABULA_consumption'.format(building_type))),bbox_inches='tight')
            
            
        
    #%% Résolution de matrice singulière 
    if False:
        print()
        
        A = pickle.load(open('.A_SFH.pickle', "rb"))
        # print(A)
        for i in range(len(A)):
            L = []
            for j in range(len(A[0])):
                L.append('{:.1E}'.format(A[i][j]))
            print(L)
            
        A = np.asarray(A)
        print(np.linalg.det(A))
        
        print()
        
        A = pickle.load(open('.A_GENMOD.pickle', "rb"))
        # print(A)
        for i in range(len(A)):
            L = []
            for j in range(len(A[0])):
                L.append('{:.1E}'.format(A[i][j]))
            print(L)
            
        A = np.asarray(A)
        print(np.linalg.det(A))
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()