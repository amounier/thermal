#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:44:11 2024

@author: amounier
"""

import time 
import os
import pandas as pd
import geopandas as gpd


# ouverture des fichiers administratifs (0.2s)
adm = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','communes-departement-region.csv'))
adm = adm.dropna(subset=['code_departement'])
adm['code_departement'] = ['0{}'.format(c) if len(c) == 1 else c for c in adm.code_departement]

geo = gpd.read_file(os.path.join('data','INSEE','decoupage_administratif','departements.geojson'))

zcl = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','zones_climatiques.csv'))
zcl['code_departement'] = ['0{}'.format(c) if len(c) == 1 else c for c in zcl.code_departement]

dict_code_dep_name_dep = {c:n for c,n in zip(adm.code_departement,adm.nom_departement)}
dict_code_dep_name_reg = {d:r for d,r in zip(adm.code_departement,adm.nom_region)}
dict_code_dep_geom_dep = {d:g for d,g in zip(geo.code,geo.geometry)}
dict_code_dep_code_zcl = {d:c for d,c in zip(zcl.code_departement,zcl.zone_climatique)}

# list_dep = list(dict_code_dep_geom_dep.keys())
# list_reg = list(set([dict_code_dep_name_reg.get(cd) for cd in list_dep]))


class Departement:
    def __init__(self,dep_code):
        if type(dep_code) == int:
            self.code = "{:02d}".format(dep_code)
        elif type(dep_code) == str and len(dep_code) == 1:
            self.code = "{:02d}".format(int(dep_code))
        else:
            self.code = dep_code
            
        self.name = dict_code_dep_name_dep.get(self.code)
        self.codint = int(self.code.replace('A','01').replace('B','02'))
        self.region = dict_code_dep_name_reg.get(self.code)
        self.geometry = dict_code_dep_geom_dep.get(self.code)
        self.climat = dict_code_dep_code_zcl.get(self.code)
        
    def __str__(self):
        return '{} ({})'.format(self.name, self.code)
    

# =============================================================================
# Script principal
# =============================================================================

def main():
    tic = time.time()
    
    # dep = Departement('13')
    # print(dep)
    
    
    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
    
if __name__ == '__main__':
    main()