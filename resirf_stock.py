#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:52:47 2025

@author: amounier
"""

import time 
import pandas as pd
import os 
from datetime import date


def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')
    
    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_resirf'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
        
    #%% Premier essai sur le stock initial
    if True:
        stock = pd.read_csv(os.path.join('data','Res-IRF','buildingstock_sdes2018_update_hpdiff_ac_reduced.csv'))
        
    
    
    
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__=='__main__':
    main()