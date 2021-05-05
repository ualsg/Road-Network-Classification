import warnings
warnings.filterwarnings("ignore")
import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import img_to_array
from crhd_generator import PlotCRHD_grid

from Build_model import Build_model
from config import config


def get_index(grid_path, road_path, building_path, landuse_path):
    
    # load grids
    grids = gpd.read_file(grid_path)
    grids.set_index('id',drop=False,inplace=True)
    grids_for_match = grids[['id', 'geometry']]
    print('Grids loaded!')

    # Road Density & Intersection
    roads = gpd.read_file(road_path)
    roads = roads[roads.geometry.notnull()]
    intersection = gpd.sjoin(grids_for_match, roads, how="right")
    for _id in grids_for_match.index:
        intersection[intersection.id == _id] = gpd.clip(intersection[intersection.id == _id],
                                                        grids_for_match.loc[[_id]])
    intersection = intersection[intersection.geometry.notnull()]
    intersection['length'] = intersection.to_crs(epsg=3857).geometry.length
    grids['RD'] = intersection.groupby('id')['length'].sum()

    # for i, row in grids.iterrows():
    #     try:
    #         centroid = row.geometry.centroid
    #         G = ox.graph_from_point(center_point=(centroid.y, centroid.x), network_type='all', dist=500)
    #         intersections = ox.graph_to_gdfs(G, nodes=True, edges=True)[0]
    #         grids.loc[i,'ID'] = intersections.shape[0]
    #     except:
    #         grids.loc[i,'ID'] = 0
    print('Road completed!')

    # building density & average building area
    buildings = gpd.read_file(building_path)
    buildings = buildings[buildings.geometry.notnull()]
    buildings = buildings.to_crs(epsg=3857)
    buildings['area'] = buildings.geometry.area
    buildings = buildings.to_crs(epsg=4326)
    intersection = gpd.sjoin(grids_for_match, buildings, how="right")
    grids['BuD'] = intersection.groupby('id')['area'].sum()
    grids['ABFA'] = intersection.groupby('id')['area'].mean()
    print('Building completed!')

    # block density & average block area & entropy

    def cross_entropy(row):
        p = np.array(row)
        log_p = np.log(p)
        log_p[log_p==-inf] = 0
        return -np.sum(p*log_p)

    landuse = gpd.read_file(landuse_path)
    landuse = landuse[landuse.geometry.notnull()]

    intersection = gpd.sjoin(grids_for_match, landuse, how="right")
    intersection['geometry'] = intersection.buffer(0)
    for _id in grids_for_match.index:
        intersection[intersection.id == _id] = gpd.clip(intersection[intersection.id == _id],
                                                        grids_for_match.loc[[_id]])
    intersection = intersection[intersection.geometry.notnull()]
    intersection['area'] = intersection.to_crs(epsg=3857).geometry.area
    grids['BID'] = intersection.groupby('id')['area'].count()
    grids['ABA'] = intersection.groupby('id')['area'].mean()
    land_use_areas = intersection.groupby(['id', 'fclass'])['area'].sum().unstack(1).fillna(0)
    land_use_areas['_sum'] = land_use_areas.sum(axis=1)
    land_use_areas = land_use_areas.apply(lambda x: x / x._sum, axis=1)
    grids['LUM'] = land_use_areas.apply(cross_entropy, axis=1)
    print('Block completed!')

    # Save
    #grids.drop(columns=['id'], inplace=True)
    #grids.to_file(save_path if save_path else grid_path,
    #              driver='ESRI Shapefile',
    #              encoding='utf-8')

    return grids

def getIntersection(grids):
    grids.set_index('id', inplace=True)
    print('Grids loaded!')

    i_cnt=[]
    cnt=0
    for i, row in grids.iterrows():
        try:
            centroid = row.geometry.centroid
            G = ox.graph_from_point(center_point=(centroid.y, centroid.x), network_type='all', dist=500)
            intersections = ox.graph_to_gdfs(G, nodes=True, edges=True)[0]
            i_cnt.append(intersections.shape[0])
        except:
            i_cnt.append(0)
        cnt += 1
        if cnt%50==0:
            print(f'current id: {i}')

    grids['ID'] = i_cnt
    #grids.to_file(save_path if save_path else grid_path,
    #              driver = 'ESRI Shapefile',
    #              encoding='utf-8')
    print('Completed!')
    return grids

def dropNonbuiltGrid(grids):
    grids.dropna(subset=['BuD','BID'], how='any', inplace=True)
    #grids.reset_index(drop=True, inplace=True)
    #grids.id = grids.index.map(lambda x: f'{cityName}_{str(x)}')
    #grids.set_index('id', inplace=True)
    #grids.to_file(save_path if save_path else grid_path,
    #              driver = 'ESRI Shapefile',
    #              encoding='utf-8')
    return grids

def get_MorphoIndex(grid_path, road_path, building_path, landuse_path, save_path=None, get_intersection=False, drop_nonbuilt=True):
    '''
    :param grid_path: filepath of grid Shapefile.
    :param road_path: filepath of road Shapefile.
    :param building_path: filepath of building Shapefile.
    :param landuse_path: filepath of land use Shapefile (should include a column named 'fclass' denoting the land use type).
    :param get_intersection: Boolean, whether to calculate number of road intersections.
        *Notice: This calculation needs internet and is very time consuming.
    :return: grids GeoDataFrame
    '''
    print('Start getting mophoindex')
    print('-' * 50)
    grids = get_index(grid_path, road_path, building_path, landuse_path)
    print('-'*50)
    if drop_nonbuilt:
        print('Dropping non-built grids')
        grids = dropNonbuiltGrid(grids)
        print('-'*50)
    if get_intersection:
        print('Getting intersections')
        grids = getIntersection(grids)

    # Save
    grids.drop(columns=['id'], inplace=True)
    grids.to_file(save_path if save_path else grid_path,
                  driver='ESRI Shapefile',
                  encoding='utf-8')
    return grids


class prob_calculator():
    def __init__(self, channeles=3):
        self.model = None
        self.channeles = channeles

    def load_model(self, model_path):
        modelPath = model_path
        if self.channeles == 3:
            self.model = Build_model(config).build_model()
        elif self.channeles == 6:
            self.model = Build_model(config).build_mymodel()
        self.model.load_weights(modelPath)

    def deal_image(self, img_path, channel=3):
        image = cv2.imread(img_path)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if channel == 1:
            return gray_img
        thre_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_TRUNC)[1]
        mask = np.where(thre_img > 190)
        image[mask] = 255
        return image

    def deal_divide_image(self, img_path, channel=3):
        image1 = cv2.imread(img_path)
        image2 = image1.copy()
        gray_img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if channel == 1:
            return gray_img
        thre_img = cv2.threshold(gray_img, 200, 255, cv2.THRESH_TRUNC)[1]
        mask1 = np.where(np.logical_or(thre_img < 130, thre_img > 190))
        mask2 = np.where(thre_img > 180)
        image1[mask1], image2[mask2] = 255, 255
        return image1, image2

    def getProb(self, grid_path, image_path, save_path=None):
        if not self.model:
            raise Exception('No model is loaded.')
        grids = gpd.read_file(grid_path)
        grids.set_index('id', inplace=True)
        grids[['Prob_O', 'Prob_N', 'Prob_G', 'Prob_R']] = None

        imgNameList = grids.index.to_list()
        # load images

        for i, imgName in enumerate(imgNameList):
            imgPath = imgName + '.png'
            file = os.path.join(image_path, imgPath)

            if not os.path.exists(file):  # if image not generated yet, plot it now
                print(f'trying to plot grid {imgName}')
                centroid = grids.loc[imgName].geometry.centroid
                point = (centroid.y, centroid.x)
                try:
                    PlotCRHD_grid(imgName, point, 1000, save_path)
                except:
                    grids.loc[imgName, ['Prob_O', 'Prob_N', 'Prob_G', 'Prob_R']] = None
                    grids['cls'] = 'Nopattern'
                    continue
            if self.channeles==3:
                img = self.deal_image(file)
                img = cv2.resize(img, (config.normal_size, config.normal_size))
                img = img_to_array(img) / 255
                img = img[np.newaxis, :]
                pred = self.model.predict(img)[0]
            elif self.channeles==6:
                img1, img2 = self.deal_divide_image(file)
                img1 = cv2.resize(img1, (config.normal_size, config.normal_size))
                img1 = img_to_array(img1) / 255
                img1 = img1[np.newaxis, :]
                img2 = cv2.resize(img2, (config.normal_size, config.normal_size))
                img2 = img_to_array(img2) / 255
                img2 = img2[np.newaxis, :]
                pred = self.model.predict([img2, img1])[0]
            grids.loc[imgName, ['Prob_O', 'Prob_N', 'Prob_G', 'Prob_R']] = pred
            if i % 100 == 0:
                print(f'Current grid: {imgName}')

        grids[['Prob_O', 'Prob_N', 'Prob_G', 'Prob_R']] = grids[['Prob_O', 'Prob_N', 'Prob_G', 'Prob_R']].astype(float)
        class_idx = {0: 'Organic', 1: 'Nopattern', 2: 'Gridiron', 3: 'Radial'}
        cls = grids[grids.Prob_O.notna()].apply(lambda row: np.argmax([row.Prob_O, row.Prob_N, row.Prob_G, row.Prob_R]), axis=1)
        grids['cls'][grids.Prob_O.notna()] = cls.map(class_idx)
        grids.to_file(save_path if save_path else grid_path,
                      driver='ESRI Shapefile',
                      encoding='utf-8')
        print('Completed!')
        return grids