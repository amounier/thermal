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


def dot3(A,B,C):
    return np.dot(A,np.dot(B,C))


def SFH_test_model():
    """
    Maison individuelle détachée (SFH), sans cave et avec des combles aménagées
    Une seule zone thermique.
    
    """
    # TODO aller 
    
    
    
    



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
        
    tac = time.time()
    print('Done in {:.2f}s.'.format(tac-tic))
    
if __name__ == '__main__':
    main()