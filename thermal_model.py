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
from typologies import Typology
from behaviour import Behaviour
from thermal_sensitivity import plot_thermal_sensitivity
from future_meteorology import get_projected_weather_data
from administrative import Departement, Climat


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


def get_external_convection_heat_transfer(wind_speed=5,method='th-bat',plot=False,figs_folder=None):
    """
    Supposition d'un vent moyen de 10m/s, à corriger avec des données météo ?

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
    Supposition d'un vent moyen de 10m/s, à corriger avec des données météo ?

    """
    if method=='cste':
        # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf p15
        # pour une température moyenne de 10°C
        h = get_external_radiation_heat_transfer(Tm=20, method='th-bat')
    
    elif method=='th-bat':
        # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf p15
        epsilon = 0.9
        sigma_SB = 5.67e-8 # W/(m2.K4)
        hro = 4 * sigma_SB  * (Tm+273.15)**3
        h = epsilon * hro # W/(m2.K)
    
    return h


def compute_R_ueh(typology):
    h_ext_conv = get_external_convection_heat_transfer()
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
        # TODO penser à rajouter les ponts thermiques
        R_w0iso_in = typology.w0_insulation_thickness/(typology.w0_insulation_material.thermal_conductivity * typology.w0_surface)
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
        # TODO penser à rajouter les ponts thermiques
        R_w1iso_in = typology.w1_insulation_thickness/(typology.w1_insulation_material.thermal_conductivity * typology.w1_surface)
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
        # TODO penser à rajouter les ponts thermiques
        R_w2iso_in = typology.w2_insulation_thickness/(typology.w2_insulation_material.thermal_conductivity * typology.w2_surface)
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
        # TODO penser à rajouter les ponts thermiques
        R_w3iso_in = typology.w3_insulation_thickness/(typology.w3_insulation_material.thermal_conductivity * typology.w3_surface)
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

def compute_Rdi(typology):
    R_df = typology.floor_structure_thickness/(typology.floor_structure_material.thermal_conductivity * typology.ground_surface)
    
    # TODO modifier le modèle pour pouvoir avoir différents type d'isolation du sol
    # TODO ajouter les ponts thermiques
    R_diso_in = typology.floor_insulation_thickness/(typology.floor_insulation_material.thermal_conductivity * typology.ground_surface)
    
    # https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/4-fascicule_parois_opaques_methodes.pdf p16
    hi = 2.9 # W/(m2.K)
    R_dih = 1/(hi * typology.w0_surface) # K/W
    
    R_di = R_df/2 + R_diso_in + R_dih
    return R_di

def compute_R_df(typology):
    R_df = typology.floor_structure_thickness/(typology.floor_structure_material.thermal_conductivity * typology.ground_surface)
    return R_df

def compute_C_f(typology):
    volume_f = typology.floor_structure_thickness * typology.ground_surface # m3
    mass_f = volume_f * typology.floor_structure_material.density # kg
    C_f = typology.floor_structure_material.thermal_capacity * mass_f # J/K
    return C_f

def compute_C_i(typology):
    mass_air = AIR_DENSITY * typology.volume
    C_air = AIR_THERMAL_CAPACITY * mass_air
    
    # estimations au doigt mouillé : cf Antonopoulos and Koronaki (1999) #TODO à rafiner
    C_internal_partitions = 10*C_air
    C_mobilier = 0.2*C_internal_partitions 
    
    C_i = C_air + C_mobilier + C_internal_partitions
    return C_i

def compute_R_g(typology):
    R_g = typology.floor_ground_distance/(GROUND_THERMAL_CONDUCTIVITY * typology.ground_section)
    return R_g

def compute_C_g(typology):
    C_g = GROUND_THERMAL_CAPACITY * GROUND_DENSITY * typology.ground_volume
    return C_g


def get_solar_absorption_coefficient(typology):
    dict_color_absorption = {'light':0.4,
                             'medium':0.6,
                             'dark':0.8,
                             'black':1.}
    
    absorption_coefficient = dict_color_absorption.get(typology.roof_color)
    return absorption_coefficient


def compute_external_Phi(typology, weather_data, wall):
    # coefficient d'absorption du flux solaire
    alpha = get_solar_absorption_coefficient(typology)
    
    # orientation de la paroi
    if wall == 'roof':
        orientation = 'H'
        surface = typology.roof_surface
    else:
        orientation = {0:typology.w0_orientation,
                       1:typology.w1_orientation,
                       2:typology.w2_orientation,
                       3:typology.w3_orientation,}.get(wall)
        surface = {0:typology.w0_surface,
                   1:typology.w1_surface,
                   2:typology.w2_surface,
                   3:typology.w3_surface,}.get(wall)
        
    sun_radiation = weather_data['direct_sun_radiation_{}'.format(orientation)].values + weather_data['diffuse_sun_radiation_{}'.format(orientation)].values
    Phi_se = sun_radiation * surface * alpha
    return Phi_se


def get_solar_transmission_factor(typology,weather_data):
    # Dans les règles Th-bat : voir norme NF P50 777, puis norme NF EN 410 
    # TODO à raffiner selon le nombre de couches principalement (et peut-être l'angle d'incidence ?)
    solar_factor = 0.5 # g (ratio)
    return solar_factor

def compute_internal_Phi(typology, weather_data, wall):
    # coefficient d'absorption du flux solaire
    g = get_solar_transmission_factor(typology,weather_data)
    
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
    
    sun_radiation = weather_data['direct_sun_radiation_{}'.format(orientation)].values + weather_data['diffuse_sun_radiation_{}'.format(orientation)].values
    Phi_si = sun_radiation * surface * g
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
    R_di = compute_Rdi(typology)
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
        
        U[i,11] = P_heater + P_cooler # i-1 ou i #TODO : à vérifier Rouchier, Madsen : peu de différence en tout cas
        
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
            
            simulation_data = SFH_test_model(typo, conventionnel, weather_data_fine_res,progressbar=False)
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
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.semilogy(resolution_list, time_compute,color='k',marker='o')
        ax.set_ylabel('Computation time for one year simulation (s)')
        # ax.set_ylim(bottom=0.)
        ax.set_xlabel('Temporal resolution time step (s)')
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('resolution_effect_time_computation_{}_{}'.format(city,period[0]))),bbox_inches='tight')
        plt.show()
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(resolution_list, heating_needs_list,color='tab:red',label='Heating',marker='o')
        # ax.plot(resolution_list, cooling_needs_list,color='tab:blue',label='Cooling')
        ax.set_ylabel('Annual energy needs (kWh.m$^{-2}$.yr$^{-1}$)')
        # ax.set_ylim(bottom=0.)
        ax.legend()
        ax.set_xlabel('Temporal resolution time step (s)')
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('resolution_effect_energy_needs_heating_{}_{}'.format(city,period[0]))),bbox_inches='tight')
        plt.show()
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        ax.plot(resolution_list, power_max_heating,color='tab:red',label='Heating',marker='o')
        # ax.plot(resolution_list, power_max_cooling,color='tab:blue',label='Cooling')
        ax.set_ylabel('Maximal power needs (Wh)')
        # ax.set_ylim(bottom=0.)
        ax.legend()
        ax.set_xlabel('Temporal resolution time step (s)')
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('resolution_effect_max_power_heating_{}_{}'.format(city,period[0]))),bbox_inches='tight')
        plt.show()
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        # ax.plot(resolution_list, heating_needs_list,color='tab:red',label='Heating')
        ax.plot(resolution_list, cooling_needs_list,color='tab:blue',label='Cooling',marker='o')
        ax.set_ylabel('Annual energy needs (kWh.m$^{-2}$.yr$^{-1}$)')
        # ax.set_ylim(bottom=0.)
        ax.legend()
        ax.set_xlabel('Temporal resolution time step (s)')
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('resolution_effect_energy_needs_cooling_{}_{}'.format(city,period[0]))),bbox_inches='tight')
        plt.show()
        
        fig,ax = plt.subplots(dpi=300,figsize=(5,5))
        # ax.plot(resolution_list, power_max_heating,color='tab:red',label='Heating')
        ax.plot(resolution_list, power_max_cooling,color='tab:blue',label='Cooling',marker='o')
        ax.set_ylabel('Maximal power needs (Wh)')
        # ax.set_ylim(bottom=0.)
        ax.legend()
        ax.set_xlabel('Temporal resolution time step (s)')
        plt.savefig(os.path.join(figs_folder,'{}.png'.format('resolution_effect_max_power_cooling_{}_{}'.format(city,period[0]))),bbox_inches='tight')
        plt.show()
        
    #%% Premier test pour le poster de SGR
    if False:
        
        # Définition de la typologie
        typo_name = 'FR.N.SFH.01.Test'
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
            
            monthly_data = aggregate_resolution(simulation_data, resolution='M',agg_method='sum')
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
                
    #%% Graphes Poster SGR 
    if True:
        
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
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
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
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3)
            ax.errorbar(U_value_windows_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3)
            ax.errorbar(U_value_windows_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
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
            
        # Étude de l'effet de l'épaisseur d'isolant de plancher sur la consommation annuelle
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
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
                        color='tab:red',label='Heating needs ({})'.format(city_2),ls='--',capsize=3)
            ax.errorbar(thickness_list, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
                        color='tab:blue',label='Cooling needs ({})'.format(city_2),ls='--',capsize=3)
            ax.set_ylabel('Annual energy needs over {}-{} '.format(period[0],period[1])+'(kWh.m$^{-2}$.yr$^{-1}$)')
            # ax.legend()
            ax.set_xlabel('Floor insulation thickness (m)')
            ax.set_ylim(bottom=0.,top=100)
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
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:red',label='Heating needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list*glazing_surface_total, 
                        [e.loc['cooling_needs'] for e in energy_needs_mean_list_1], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_1],
                        color='tab:blue',label='Cooling needs ({})'.format(city_1),capsize=3)
            ax.errorbar(thickness_list*glazing_surface_total, 
                        [e.loc['heating_needs'] for e in energy_needs_mean_list_2], 
                        [e.loc['cooling_needs'] for e in energy_needs_std_list_2],
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
            # TODO : attention, pour l'instant la météo future est en carton (un peu)
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
                
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()