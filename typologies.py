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


dict_angle_orientation = {i*45:o for i,o in enumerate(['N','NE','E','SE','S','SW','W','NW'])}
dict_orientation_angle = {v:k for k,v in dict_angle_orientation.items()}


def open_tabula_typologies():
    path = os.path.join('data','TABULA','TABULA_typologies.csv')
    data = pd.read_csv(path)
    return data


def main():
    tic = time.time()
    
    # # Défintion de la date du jour
    # today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # # Défintion des dossiers de sortie 
    # output = 'output'
    # folder = '{}_typologies'.format(today)
    # figs_folder = os.path.join(output, folder, 'figs')
    
    # # Création des dossiers de sortie 
    # if folder not in os.listdir(output):
    #     os.mkdir(os.path.join(output,folder))
    # if 'figs' not in os.listdir(os.path.join(output, folder)):
    #     os.mkdir(figs_folder)
        
    # -------------------------------------------------------------------------
    
    typologies = open_tabula_typologies()
    
    
    
    
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__=='__main__':
    main()