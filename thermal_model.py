#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:28:45 2024

@author: amounier
"""

import time
from datetime import date
import os
import pandas as pd
import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from utils import plot_timeserie
from meteorology import get_init_ground_temperature, get_historical_weather_data
from energy_consumption_RC import get_P_cooler, get_P_heater
from typologies import Typology
from behaviour import Behaviour

air_thermal_capacity = 1000 # J/(kg.K)
air_density = 1.2 # kg/m3

ground_thermal_capacity = 1000 #J /(kg.K)
ground_density = 2500 # kg/m3
ground_thermal_conductivity = 1.5 # W/(m.K)

def dot3(A,B,C):
    return np.dot(A,np.dot(B,C))


def get_ventilation_minimum_air_flow(typology, plot=False, figs_folder=None):
    """
    https://www.legifrance.gouv.fr/loda/id/JORFTEXT000000862344/2021-01-08/
    Et hypothèse de 20m2 par pièce

    """
    ventilation_minimum_air_flow = typology.surface * 0.8 + 24.3 # m3/h
    ventilation_minimum_air_flow = air_density*ventilation_minimum_air_flow/3600 # kg/s
    
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
    air_infiltration = air_density*air_infiltration/3600 # kg/s
    
    return air_infiltration


def compute_R_air(typology):
    # total_air_flow = get_infiltration_air_flow(typology) + get_ventilation_minimum_air_flow(typology)
    R_air = get_infiltration_air_flow(typology) * air_thermal_capacity + get_ventilation_minimum_air_flow(typology) * air_thermal_capacity * (1-typology.ventilation_efficiency)
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


def compute_R_ueh(typology):
    h_ext = get_external_convection_heat_transfer()
    R_ueh = 1/(h_ext * typology.roof_surface) # K/W
    return R_ueh

def compute_R_w0eh(typology):
    h_ext = get_external_convection_heat_transfer()
    R_w0eh = 1/(h_ext * typology.w0_surface) # K/W
    return R_w0eh

def compute_R_w1eh(typology):
    h_ext = get_external_convection_heat_transfer()
    R_w1eh = 1/(h_ext * typology.w1_surface) # K/W
    return R_w1eh

def compute_R_w2eh(typology):
    h_ext = get_external_convection_heat_transfer()
    R_w2eh = 1/(h_ext * typology.w2_surface) # K/W
    return R_w2eh

def compute_R_w3eh(typology):
    h_ext = get_external_convection_heat_transfer()
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
    mass_air = air_density * typology.volume
    C_air = air_thermal_capacity * mass_air
    
    # estimations au doigt mouillé : cf Antonopoulos and Koronaki (1999)
    C_mobilier = C_air 
    C_internal_partitions = 1.5*C_air
    
    C_i = C_air + C_mobilier + C_internal_partitions
    return C_i

def compute_R_g(typology):
    R_g = typology.floor_ground_distance/(ground_thermal_conductivity * typology.ground_section)
    return R_g

def compute_C_g(typology):
    C_g = ground_thermal_capacity * ground_density * typology.ground_volume
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
    solar_factor = 0.6 # g (ratio)
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
    
    sun_radiation = sun_radiation = weather_data['direct_sun_radiation_{}'.format(orientation)].values + weather_data['diffuse_sun_radiation_{}'.format(orientation)].values
    Phi_si = sun_radiation * surface * g
    return Phi_si



def SFH_test_model(typology, behaviour, weather_data):
    """
    Maison individuelle détachée (SFH), sans cave et avec des combles aménagées
    Une seule zone thermique.
    
    La ventilation va devoir être constante dans ce premier modèle.
    Idem pour les coefficient de transfert surfaciques :
        (rayonnement (absent pour l'instant) et convection)

    """
    #  remplacer les None par des valeurs issues de typology
    
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
    
    # TODO : vérifier les signes 
    
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
    
    A[1,1] = 1/C_w0 * (-1/R_w0e -1/R_w0i + R_w0eh/(R_w0eh+R_w0e))
    A[2,2] = 1/C_w1 * (-1/R_w1e -1/R_w1i + R_w1eh/(R_w1eh+R_w1e))
    A[3,3] = 1/C_w2 * (-1/R_w2e -1/R_w2i + R_w2eh/(R_w2eh+R_w2e))
    A[4,4] = 1/C_w3 * (-1/R_w3e -1/R_w3i + R_w3eh/(R_w3eh+R_w3e))
    
    A[5,0] = 1/C_f * 1/R_di
    A[5,5] = 1/C_f * (-1/(R_df/2) - 1/R_di)
    A[5,6] = 1/C_f * 1/(R_df/2)
    
    A[6,5] = 1/C_g * 1/(R_df/2)
    A[6,6] = 1/C_g * (-1/R_g - 1/(R_df/2))
    
    print(A)
    
    # Définition de la matrice B
    B = np.zeros((7,13))
    B[0,0] = 1/C_i * (1/R_air 
                      # +1/R_uw
                      +1/(R_ui+R_ueh)
                      +1/R_w0w
                      +1/R_w1w
                      +1/R_w2w
                      +1/R_w3w)
    
    B[0,0] = 1/C_i * R_ueh/(R_ui+R_ueh)
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
    U[:,1] = [0]*len(weather_data) # compute_external_Phi(typology, weather_data, wall='roof') # Phi_sue
    U[:,2] = [0]*len(weather_data) # Phi_sui
    U[:,3] = compute_external_Phi(typology, weather_data, wall=0) # Phi_sw0e
    U[:,4] = compute_internal_Phi(typology, weather_data, wall=0) # Phi_sw0i
    U[:,5] = compute_external_Phi(typology, weather_data, wall=1) # Phi_sw1e
    U[:,6] = compute_internal_Phi(typology, weather_data, wall=1) # Phi_sw1i
    U[:,7] = compute_external_Phi(typology, weather_data, wall=2) # Phi_sw2e
    U[:,8] = compute_internal_Phi(typology, weather_data, wall=2) # Phi_sw2i
    U[:,9] = compute_external_Phi(typology, weather_data, wall=3) # Phi_sw3e
    U[:,10] = compute_internal_Phi(typology, weather_data, wall=3) # Phi_sw3i
    U[:,12] = internal_thermal_gains 
    
    X[0,0] = Ti_setpoint_winter[0]
    X[0,1] = 1/(R_w0e+R_w0eh+R_w0i) * (R_w0i * U[0,0] + (R_w0e+R_w0eh) * X[0,0])
    X[0,2] = 1/(R_w1e+R_w1eh+R_w1i) * (R_w1i * U[0,0] + (R_w1e+R_w1eh) * X[0,0])
    X[0,3] = 1/(R_w2e+R_w2eh+R_w2i) * (R_w2i * U[0,0] + (R_w2e+R_w2eh) * X[0,0])
    X[0,4] = 1/(R_w3e+R_w3eh+R_w3i) * (R_w3i * U[0,0] + (R_w3e+R_w3eh) * X[0,0])
    X[0,6] = get_init_ground_temperature(foundation_depth, weather_data)
    X[0,5] = 1/(R_df/2+R_di) * (R_di * X[0,6] + (R_df/2) * X[0,0])
    
    # Simulation
    for i in tqdm.tqdm(range(1,len(time_)), total=len(time_)-1):
        # Te = U[i,0]
        Ti = X[i-1,0]
        
        Ts_heater = Ti_setpoint_winter[i]
        Ts_cooler = Ti_setpoint_summer[i]
        
        # Te = X[i-1,0]
        P_heater = get_P_heater(Ti, Ti_min=Ts_heater, Pmax=P_max_heater, method='all_or_nothing')
        P_cooler = get_P_cooler(Ti, Ti_max=Ts_cooler, Pmax=P_max_cooler, method='all_or_nothing')
        
        U[i-1,11] = P_heater + P_cooler
        
        X[i] = np.dot(F,X[i-1]) + np.dot(G, U[i-1].T)
    
    print(X.shape)
    weather_data['internal_temperature'] = X[:,0]
    weather_data['ground_temperature'] = X[:,6]
    weather_data['heater_cooler_power'] = U[:,11]
    
    # print(weather_data[['temperature_2m','internal_temperature']])
    # plot_timeserie(weather_data[['temperature_2m','internal_temperature','ground_temperature']],
    #                xlim=[pd.to_datetime('{}-02-01'.format(2020)), pd.to_datetime('{}-02-08'.format(2020))])
    # plot_timeserie(weather_data[['temperature_2m','internal_temperature','ground_temperature']],
    #                xlim=[pd.to_datetime('{}-07-01'.format(2020)), pd.to_datetime('{}-07-08'.format(2020))])
    
    for i in range(7):
        plt.plot(U[:,0])
        plt.plot(X[:,i])
        plt.show()
        
    for i in range(1,13):
        plt.plot(U[:,0])
        plt.plot(U[:,i])
        plt.show()
    
    # plot_timeserie(weather_data[['heater_cooler_power']],
    #                xlim=[pd.to_datetime('{}-07-01'.format(2020)), pd.to_datetime('{}-07-31'.format(2020))])
    return X,U
    
    
def refine_resolution(data, resolution):
    data_high_res = pd.DataFrame(index=pd.date_range(data.index[0],data.index[-1],freq=resolution))
    data_high_res = data_high_res.join(data,how='left')
    data_high_res = data_high_res.interpolate()
    return data_high_res


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
        
    #%% Premier test pour le poster de SGR
    if True:
        # Définition de la typologie
        typo = Typology('FR.N.SFH.01.Test')
        # typo.w0_insulation_thickness = 0.6
        # typo.ventilation_efficiency = 1
        typo.heater_maximum_power = 0
        typo.cooler_maximum_power = 0
        
        # Génération du fichier météo
        city = 'Marseille'
        year = 2020
        principal_orientation = typo.w0_orientation
        
        weather_data = get_historical_weather_data(city,year,principal_orientation)
        weather_data = refine_resolution(weather_data, resolution='30s')
        
        # Définition des habitudes (simplifiées pour l'instant)
        conventionnel = Behaviour('conventionnel_th-bce_2020')
        
        # Affichage des variables d'entrée
        if False:
            pass 
        
        # Simulation
        if True:
            X,U = SFH_test_model(typo, conventionnel, weather_data)
        
        
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()