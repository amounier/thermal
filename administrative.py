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
import matplotlib.pyplot as plt
from datetime import date
import matplotlib
import cartopy.crs as ccrs
from shapely.ops import unary_union

from utils import blank_national_map

# ouverture des fichiers administratifs (0.2s)
adm = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','communes-departement-region.csv'))
adm = adm.dropna(subset=['code_departement'])
adm['code_departement'] = ['0{}'.format(c) if len(c) == 1 else c for c in adm.code_departement]

geo = gpd.read_file(os.path.join('data','INSEE','decoupage_administratif','departements.geojson'))

zcl = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','zones_climatiques.csv'))
zcl['code_departement'] = ['0{}'.format(c) if len(c) == 1 else c for c in zcl.code_departement]

dict_code_dep_name_dep = {c:n for c,n in zip(adm.code_departement,adm.nom_departement)}
dict_name_dep_code_dep = {n:c for c,n in dict_code_dep_name_dep.items()}
dict_code_dep_name_reg = {d:r for d,r in zip(adm.code_departement,adm.nom_region)}
dict_code_dep_geom_dep = {d:g for d,g in zip(geo.code,geo.geometry)}
dict_code_dep_code_zcl = {d:c for d,c in zip(zcl.code_departement,zcl.zone_climatique)}

prf = pd.read_csv(os.path.join('data','INSEE','decoupage_administratif','prefectures.csv'))
dict_name_dep_name_prf = {n:p for n,p in zip(prf.Département,prf.Préfecture)}

list_dep_code = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', 
                 '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', 
                 '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31', '32', 
                 '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', 
                 '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', 
                 '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', 
                 '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', 
                 '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', 
                 '88', '89', '90', '91', '92', '93', '94', '95']

# list_reg = list(set([dict_code_dep_name_reg.get(cd) for cd in list_dep]))

# TODO : rajouter une classe France pour les stats nationale par exemple
# TODO : rajouter une classe city pour faire le lien avec les données météo

class Departement:
    def __init__(self,dep_code):
        if type(dep_code) == int:
            self.code = "{:02d}".format(dep_code)
        elif type(dep_code) == str and len(dep_code) == 1:
            self.code = "{:02d}".format(int(dep_code))
        elif type(dep_code) == str and len(dep_code) == 2:
            if dep_code in ['2A','2B']:
                self.code = dep_code
            else:
                self.code = "{:02d}".format(int(dep_code))
        else:
            #TODO à renforcer
            self.code = dict_name_dep_code_dep.get(dep_code)
            
        self.name = dict_code_dep_name_dep.get(self.code)
        self.codint = int(self.code.replace('A','01').replace('B','02'))
        self.region = dict_code_dep_name_reg.get(self.code)
        self.geometry = dict_code_dep_geom_dep.get(self.code)
        self.climat = dict_code_dep_code_zcl.get(self.code)
        
        self.prefecture = dict_name_dep_name_prf.get(self.name)
        
        # statistiques réalisées dans typologies.py
        self.typologies_batiments_groupe = None
        self.typologies_households_number = None
        
    def __str__(self):
        return '{} ({})'.format(self.name, self.code)
    

class Climat:
    def __init__(self,code):
        self.code = code
        self.departements = [Departement(k) for k,e in dict_code_dep_code_zcl.items() if e == self.code]
        self.geometry = unary_union([d.geometry for d in self.departements])
        self.codint = int(''.join([str({chr(ord('@')+n).lower():n for n in range(1,27)}.get(l.lower(),l)) for l in self.code]))
        # gestion du climat H3
        if self.codint<100:
            self.codint *= 10
        
    def __str__(self):
        return self.name
    
    
    
class France:
    def __init__(self):
        self.departements = [Departement(e) for e in list_dep_code]
        


def draw_departement_map(dict_dep,figs_folder,cbar_min=0,cbar_max=1.,
                         automatic_cbar_values=False, cbar_label=None, 
                         map_title=None,save=None):
    fig,ax = blank_national_map()
    
    cmap = matplotlib.colormaps.get_cmap('viridis')
    
    plotter = pd.DataFrame().from_dict({'departements':dict_dep.keys(),'vals':dict_dep.values()})
    plotter['geometry'] = [d.geometry for d in plotter.departements]
    plotter = gpd.GeoDataFrame(plotter, geometry=plotter.geometry)
    
    if automatic_cbar_values:
        cbar_max = plotter.values.quantile(0.99)
        cbar_min = plotter.values.quantile(0.01)
    
    plotter['color'] = (plotter.vals-cbar_min)/(cbar_max-cbar_min)
    plotter['color'] = plotter['color'].apply(cmap)
    
    plotter.plot(color=plotter.color, ax=ax, transform=ccrs.PlateCarree(),)
    plotter.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k',lw=0.5)
    
    cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
    posn = ax.get_position()
    cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
    norm = matplotlib.colors.Normalize(vmin=cbar_min, vmax=cbar_max)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    cbar_label_var = cbar_label
    _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend='neither', extendfrac=0.02)
    
    ax.set_title(map_title)
    if save is not None:
        plt.savefig(os.path.join(figs_folder,'{}.png'.format(save)),bbox_inches='tight')
    return fig,ax

#%% ===========================================================================
# Script principal
# =============================================================================

def main():
    tic = time.time()
    
    # Défintion de la date du jour
    today = pd.Timestamp(date.today()).strftime('%Y%m%d')

    # Défintion des dossiers de sortie 
    output = 'output'
    folder = '{}_administrative'.format(today)
    figs_folder = os.path.join(output, folder, 'figs')
    
    # Création des dossiers de sortie 
    if folder not in os.listdir(output):
        os.mkdir(os.path.join(output,folder))
    if 'figs' not in os.listdir(os.path.join(output, folder)):
        os.mkdir(figs_folder)
    
    #%% Test de cartographie d'un département, d'une région climatique
    if False:
        dep = Departement('13')
        # print(dep)
        draw_departement_map({dep:0.5}, figs_folder=figs_folder, save='test_{}'.format(dep.code))
    
    if True:
        zcl = Climat('H1a')
        print(zcl.code)
        print(zcl.codint)
        # [print(d) for d in zcl.departements]
        
    #%% Téléchargement des préfectures pour intégratiuon à département
    if False:
        prefectures = pd.read_html('https://fr.wikipedia.org/wiki/Liste_des_pr%C3%A9fectures_de_France')[0]
        prefectures = prefectures[['No Insee', 'Département', 'Préfecture']]
        prefectures.to_csv('data/prefectures.csv',index=False)
        
    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
    
if __name__ == '__main__':
    main()