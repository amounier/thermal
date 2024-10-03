#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:00:06 2024

@author: amounier
"""

import time 
import os
from datetime import date
import pandas as pd





#%% ===========================================================================
# script principal
# =============================================================================
def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_future_meteorology'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
        
    #%% Téléchargement des données CORDEX sur CDS (ne marche pas)
    if False:
        import cdsapi
        
        dataset = "projections-cordex-domains-single-levels"
        request = {
            "domain": "europe",
            "experiment": "rcp_8_5",
            "horizontal_resolution": "0_11_degree_x_0_11_degree",
            "temporal_resolution": "3_hours",
            "variable": [
                "2m_air_temperature",
                "surface_solar_radiation_downwards",
                "surface_thermal_radiation_downward"
            ],
            "gcm_model": "ipsl_cm5a_mr",
            "rcm_model": "knmi_racmo22e",
            "ensemble_member": "r1i1p1",
            "start_year": ["2078"],
            "end_year": ["2079"]
        }
        
        client = cdsapi.Client()
        client.retrieve(dataset, request).download()
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()

