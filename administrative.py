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
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
from scipy.spatial.distance import euclidean
import numpy as np

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
dict_code_zcl_code_zcw = {e:e[:2] for e in ['H1a', 'H1b', 'H1c', 'H2a', 'H2b', 'H2c', 'H2d', 'H3']}
dict_code_zcl_code_zcs = {e:'d' if e[-1]=='3' else e[-1] for e in ['H1a', 'H1b', 'H1c', 'H2a', 'H2b', 'H2c', 'H2d', 'H3']}

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

# rajouter une classe France pour les stats nationale par exemple
# rajouter une classe city pour faire le lien avec les données météo
    
def get_coordinates(city,max_attempt=10):
    """
    Récupération des coordonnées d'une ville via l'API OSM. 

    Parameters
    ----------
    city : str
        DESCRIPTION.

    Returns
    -------
    longitude : float
        DESCRIPTION.
    latitude : float
        DESCRIPTION.

    """

    coordinates_dict = {"Bourg-en-Bresse":(5.225032, 46.205119),
                        "Laon":(3.620686, 49.564665),
                        "Moulins":(3.33317, 46.566053),
                        "Digne-les-Bains":(6.235143, 44.091814),
                        "Gap":(6.082064, 44.561203),
                        "Nice":(7.268391, 43.700936),
                        "Privas":(4.598673, 44.735271),
                        "Charleville-Mézières":(4.720694, 49.773571),
                        "Foix":(1.605381, 42.9639),
                        "Troyes":(4.074626, 48.297163),
                        "Carcassonne":(2.349107, 43.213036),
                        "Rodez":(2.572849, 44.351141),
                        "Marseille":(5.369953, 43.296174),
                        "Caen":(-0.363561, 49.18134),
                        "Aurillac":(2.44331, 44.928544),
                        "Angoulême":(0.156195, 45.648451),
                        "La Rochelle":(-1.151595, 46.159732),
                        "Bourges":(2.399125, 47.081166),
                        "Tulle":(1.77068, 45.267835),
                        "Dijon":(5.04147, 47.321581),
                        "Saint-Brieuc":(-2.760328, 48.514113),
                        "Guéret":(1.871576, 46.168952),
                        "Périgueux":(0.718441, 45.190936),
                        "Besançon":(6.024362, 47.238022),
                        "Valence":(-0.376335, 39.469707),
                        "Évreux":(1.151016, 49.02689),
                        "Chartres":(1.488143, 48.44386),
                        "Quimper":(-4.102478, 47.996032),
                        "Ajaccio":(8.737603, 41.926399),
                        "Bastia":(9.450919, 42.699398),
                        "Nîmes":(4.360069, 43.837425),
                        "Toulouse":(1.444247, 43.604462),
                        "Auch":(0.585051, 43.646356),
                        "Bordeaux":(-0.580036, 44.841225),
                        "Montpellier":(3.876734, 43.611242),
                        "Rennes":(-1.68002, 48.111339),
                        "Châteauroux":(1.677096, 46.820378),
                        "Tours":(0.688927, 47.390047),
                        "Grenoble":(5.735782, 45.18756),
                        "Lons-le-Saunier":(5.558997, 46.672704),
                        "Mont-de-Marsan":(-0.500972, 43.891132),
                        "Blois":(1.333764, 47.587686),
                        "Saint-Étienne":(4.387306, 45.440147),
                        "Le Puy-en-Velay":(3.885554, 45.045974),
                        "Nantes":(-1.554136, 47.218637),
                        "Orléans":(1.908607, 47.902734),
                        "Cahors":(1.4365, 44.4495),
                        "Agen":(0.617611, 44.201583),
                        "Mende":(3.499106, 44.518023),
                        "Angers":(-0.551559, 47.473988),
                        "Saint-Lô":(-1.090664, 49.1157),
                        "Châlons-en-Champagne":(4.362885, 48.956622),
                        "Chaumont":(5.139585, 48.111132),
                        "Laval":(-0.773402, 48.070669),
                        "Nancy":(6.18341, 48.693722),
                        "Bar-le-Duc":(5.162381, 48.771267),
                        "Vannes":(-2.759908, 47.658677),
                        "Metz":(6.176355, 49.119696),
                        "Nevers":(3.15772, 46.98766),
                        "Lille":(3.063528, 50.636565),
                        "Beauvais":(2.082336, 49.4301),
                        "Alençon":(0.091137, 48.431206),
                        "Arras":(2.777221, 50.291048),
                        "Clermont-Ferrand":(3.081943, 45.777455),
                        "Pau":(-0.368567, 43.295755),
                        "Tarbes":(0.078102, 43.232858),
                        "Perpignan":(2.895312, 42.69853),
                        "Strasbourg":(7.750713, 48.584614),
                        "Colmar":(7.357964, 48.077752),
                        "Lyon":(4.832011, 45.757814),
                        "Vesoul":(6.154469, 47.61974),
                        "Mâcon":(4.832227, 46.303668),
                        "Le Mans":(0.196785, 48.007385),
                        "Chambéry":(5.920364, 45.566267),
                        "Annecy":(6.128885, 45.899235),
                        "Paris":(2.320041, 48.85889),
                        "Rouen":(1.093966, 49.440459),
                        "Melun":(2.660817, 48.539927),
                        "Versailles":(2.126689, 48.80354),
                        "Niort":(-0.464606, 46.323923),
                        "Amiens":(2.295695, 49.894171),
                        "Albi":(2.147899, 43.927755),
                        "Montauban":(1.354999, 44.017584),
                        "Toulon":(5.930492, 43.125731),
                        "Avignon":(4.805901, 43.949249),
                        "La Roche-sur-Yon":(-1.42697, 46.670543),
                        "Poitiers":(0.340196, 46.58026),
                        "Limoges":(1.264485, 45.835424),
                        "Épinal":(6.450364, 48.174768),
                        "Auxerre":(3.570579, 47.796129),
                        "Belfort":(6.862894, 47.63796),
                        "Évry-Courcouronnes":(2.438182, 48.629966),
                        "Nanterre":(2.207127, 48.892427),
                        "Bobigny":(2.445223, 48.906387),
                        "Créteil":(2.453073, 48.777149),
                        "Cergy":(2.038874, 49.052753),}
        
    if city in coordinates_dict.keys():
        longitude, latitude = coordinates_dict[city]
        return longitude, latitude
    else:
        try:
            # initialisation de l'instance Nominatim (API OSM), changer l'agent si besoin
            geolocator = Nominatim(user_agent="amounier")
            location = geolocator.geocode(city)
            longitude, latitude = round(location.longitude,ndigits=6), round(location.latitude, ndigits=6)
        except GeocoderUnavailable:
            if max_attempt>0:
                get_coordinates(city,max_attempt=max_attempt-1)
            raise KeyError('No internet connexion, offline availables cities are : {}'.format(', '.join(list(coordinates_dict.keys()))))
    return longitude, latitude


class City:
    def __init__(self,name):
        self.name = name
        self.coordinates = get_coordinates(self.name)
        
    def __str__(self):
        return self.name
    

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
            
        # TODO : ville centrale de la zone
        # (prefecture du departement dont le centroid est le plus proche du centroid du climat)
        self.center_departement = self.get_center_departement()
        self.center_prefecture = self.get_center_prefecture()
        
    def __str__(self):
        return self.code
    
    def get_center_departement(self):
        distance_list = [np.nan]*len(self.departements)
        for i,dep in enumerate(self.departements):
            distance_list[i] = euclidean(self.geometry.centroid.coords[0], dep.geometry.centroid.coords[0],)
            
        center_departement = self.departements[np.argmin(distance_list)]
        return center_departement
    
    def get_center_prefecture(self):
        distance_list = [np.nan]*len(self.departements)
        for i,dep in enumerate(self.departements):
            prefecture = dep.prefecture
            coords_prefecture = get_coordinates(prefecture)
            distance_list[i] = euclidean(self.geometry.centroid.coords[0], coords_prefecture)
            
        center_prefecture = self.departements[np.argmin(distance_list)].prefecture
        return center_prefecture
    
    
class Climat_winter:
    def __init__(self,code):
        self.code = code
        self.climats = [Climat(k) for k,e in dict_code_zcl_code_zcw.items() if e == self.code]
        self.geometry = unary_union([d.geometry for d in self.climats])
        self.codint = int(''.join([str({chr(ord('@')+n).lower():n for n in range(1,27)}.get(l.lower(),l)) for l in self.code]))
        # gestion du climat H3
        if self.codint<100:
            self.codint *= 10
        
    def __str__(self):
        return self.code
    

class Climat_summer:
    def __init__(self,code):
        self.code = code
        self.climats = [Climat(k) for k,e in dict_code_zcl_code_zcs.items() if e == self.code]
        self.geometry = unary_union([d.geometry for d in self.climats])
        self.codint = int(''.join([str({chr(ord('@')+n).lower():n for n in range(1,27)}.get(l.lower(),l)) for l in self.code]))
        
    def __str__(self):
        return self.code
    
    
    
class France:
    def __init__(self):
        self.departements = [Departement(e) for e in list_dep_code]
        self.climats = sorted(list(set([e.climat for e in self.departements])))
        self.climats_winter = ['H1','H2','H3']
        self.climats_summer = ['a','b','c','d']
        


def draw_departement_map(dict_dep,figs_folder,cbar_min=0,cbar_max=1.,
                         automatic_cbar_values=False, cbar_label=None, 
                         map_title=None,save=None):
    fig,ax = blank_national_map()
    
    cmap = matplotlib.colormaps.get_cmap('viridis')
    
    plotter = pd.DataFrame().from_dict({'departements':dict_dep.keys(),'vals':dict_dep.values()})
    plotter['geometry'] = [d.geometry for d in plotter.departements]
    plotter = gpd.GeoDataFrame(plotter, geometry=plotter.geometry)
    
    if automatic_cbar_values:
        cbar_max = plotter.vals.quantile(0.99)
        cbar_min = plotter.vals.quantile(0.01)
        cbar_extend = 'both'
    else:
        cbar_extend = 'neither'
    
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
    _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend=cbar_extend, extendfrac=0.02)
    
    ax.set_title(map_title)
    if save is not None:
        plt.savefig(os.path.join(figs_folder,'{}.png'.format(save)),bbox_inches='tight')
    return fig,ax


def draw_climat_map(dict_dep,figs_folder,cbar_min=0,cbar_max=1.,
                    automatic_cbar_values=False, cbar_label=None, 
                    map_title=None,save=None, cmap=None,zcl_label=False,
                    add_city_points=None, add_legend=True,lw=None):
    
    fig,ax = blank_national_map()
    
    if cmap is None:
        cmap = matplotlib.colormaps.get_cmap('viridis')
    
    plotter = pd.DataFrame().from_dict({'climats':dict_dep.keys(),'vals':dict_dep.values()})
    plotter['geometry'] = [d.geometry for d in plotter.climats]
    plotter = gpd.GeoDataFrame(plotter, geometry=plotter.geometry)
    
    if automatic_cbar_values:
        cbar_max = plotter.vals.quantile(0.99)
        cbar_min = plotter.vals.quantile(0.01)
        cbar_extend = 'both'
    else:
        cbar_extend = 'neither'
    
    plotter['color'] = (plotter.vals-cbar_min)/(cbar_max-cbar_min)
    plotter['color'] = plotter['color'].apply(cmap)
    
    # print(plotter.color)
    
    plotter.plot(color=plotter.color, ax=ax, transform=ccrs.PlateCarree(),)
    plotter.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color='k',lw=lw)
    
    if zcl_label:
        for zcl in dict_dep.keys():
            ax.text(zcl.geometry.centroid.x, zcl.geometry.centroid.y, '{}'.format(zcl.code), 
                    horizontalalignment='center', transform=ccrs.Geodetic(), zorder=20, color='w',
                    bbox=dict(facecolor='k', alpha=0.5))
            
    if add_city_points is not None:
        if len(add_city_points) == len(dict_dep.keys()):
            for i,city in enumerate(add_city_points):
                if len(add_city_points)>1:
                    color = None
                else:
                    color = cmap(0.5)
                city = City(city)
                zcl = list(dict_dep.keys())[i]
                ax.plot(city.coordinates[0],city.coordinates[1], 
                        transform=ccrs.PlateCarree(), color=color,ls='',
                        marker='o',label='{} - {}'.format(city.name,zcl.code),mec='k',zorder=5)
        else:
            for i,city in enumerate(add_city_points):
                if len(add_city_points)>1:
                    color = None
                else:
                    color = cmap(0.5)
                city = City(city)
                ax.plot(city.coordinates[0],city.coordinates[1], 
                        transform=ccrs.PlateCarree(), color=color,ls='',
                        marker='o',label=city.name,mec='k',zorder=5)
        
    
    if not all(plotter.color==(0.0, 0.0, 0.0, 0.0)):
        cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0+posn.width+0.02, posn.y0, 0.04, posn.height])
        norm = matplotlib.colors.Normalize(vmin=cbar_min, vmax=cbar_max)
        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        cbar_label_var = cbar_label
        _ = plt.colorbar(mappable, cax=cbar_ax, label=cbar_label_var, extend=cbar_extend, extendfrac=0.02)
    
    if add_legend:
        ax.legend(ncol=3,loc='lower center')
        
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
        zcl = Climat('H1a')
        dep = zcl.center_departement
        # print(dep)
        draw_departement_map({dep:None}, figs_folder=figs_folder, save='dep_{}'.format(dep.code))
    
    # zone climatique 8
    if True:
        zcl = Climat('H3')
        # print(zcl.code)
        # print(zcl.codint)
        
        france = France()
        
        draw_climat_map({Climat(e):None for e in france.climats},zcl_label=False, 
                        figs_folder=figs_folder, save='zcl',
                        add_city_points=[Climat(c).center_prefecture for c in france.climats],lw=0.7)
        
        # [print(d) for d in zcl.departements]
        
    # zones climatiques d'été et d'hiver
    if True:
       france = France()
       
       climats_winter = [Climat_winter(e) for e in france.climats_winter]
       draw_climat_map({c:None for c in climats_winter},zcl_label=True, 
                       figs_folder=figs_folder, save='zcl_winter',
                       add_legend=False,lw=0.7)
       
       climats_summer = [Climat_summer(e) for e in france.climats_summer]
       draw_climat_map({c:None for c in climats_summer},zcl_label=True, 
                       figs_folder=figs_folder, save='zcl_summer',
                       add_legend=False,lw=0.7)
       
       # climats = [Climat(e) for e in france.climats]
       # draw_climat_map({c:None for c in climats},zcl_label=True, 
       #                 figs_folder=figs_folder, save='zcl',
       #                 add_legend=False,lw=0.7)
        
    #%% Téléchargement des préfectures pour intégratiuon à département
    if False:
        prefectures = pd.read_html('https://fr.wikipedia.org/wiki/Liste_des_pr%C3%A9fectures_de_France')[0]
        prefectures = prefectures[['No Insee', 'Département', 'Préfecture']]
        prefectures.to_csv('data/prefectures.csv',index=False)
        
    tac = time.time()
    print("Done in {:.2f}s.".format(tac-tic))
    
    
if __name__ == '__main__':
    main()