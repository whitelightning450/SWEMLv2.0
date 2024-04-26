# Import packages
# Dataframe Packages
import numpy as np
from numpy import gradient, rad2deg, arctan2
import xarray as xr
import pandas as pd

# Vector Packages
import geopandas as gpd
import shapely
from shapely import wkt
from shapely.geometry import Point, Polygon
from pyproj import CRS, Transformer

# Raster Packages
import rioxarray as rxr
import rasterio
from rasterio.mask import mask
from rioxarray.merge import merge_arrays
import rasterstats as rs
from osgeo import gdal
from osgeo import gdalconst

# Data Access Packages
import earthaccess as ea
import h5py
import pickle
import pystac_client
import richdem as rd
import planetary_computer
from planetary_computer import sign

# General Packages
import os
import re
import shutil
import math
from datetime import datetime
import glob
from pprint import pprint
from typing import Union
from pathlib import Path
from tqdm import tqdm
import time
import requests
import concurrent.futures as cf
import dask
import dask.dataframe as dd
from dask.distributed import progress
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from retrying import retry
import fiona
import re
import s3fs

#need to mamba install gdal, earthaccess 
#pip install pystac_client, richdem, planetary_computer, dask, distributed, retrying

#connecting to AWS
import warnings; warnings.filterwarnings("ignore")
import boto3
import pickle as pkl
'''
To create .netrc file:
import earthaccess
earthaccess.login(persist=True)
'''

#load access key
HOME = os.path.expanduser('~')
KEYPATH = "SWEML/AWSaccessKeys.csv"
ACCESS = pd.read_csv(f"{HOME}/{KEYPATH}")

#start session
SESSION = boto3.Session(
    aws_access_key_id=ACCESS['Access key ID'][0],
    aws_secret_access_key=ACCESS['Secret access key'][0],
)
S3 = SESSION.resource('s3')
#AWS BUCKET information
BUCKET_NAME = 'national-snow-model'
BUCKET = S3.Bucket(BUCKET_NAME)


# Calculating nearest SNOTEL sites
def calculate_nearest_snotel(region, aso_gdf, snotel_gdf, n=6, distance_cache=None):

    nearest_snotel_dict_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}"
    #check to see if regional TrainingDF path exists, if not, make one
    if not os.path.exists(nearest_snotel_dict_path):
        os.makedirs(nearest_snotel_dict_path, exist_ok=True)

    if distance_cache is None:
        distance_cache = {}

    nearest_snotel = {}
    print(f"Calculating haversine distance for {len(aso_gdf)} locations to in situ OBS, and saving cell-obs relationships in dictionary")
    for idx, aso_row in tqdm(aso_gdf.iterrows()):
        cell_id = idx
        # Check if distances for this cell_id are already calculated and cached
        if cell_id in distance_cache:
            nearest_snotel[idx] = distance_cache[cell_id]
        else:
            # Calculate Haversine distances between the grid cell and all SNOTEL locations
            distances = haversine_vectorized(
                aso_row.geometry.y, aso_row.geometry.x,
                snotel_gdf.geometry.y.values, snotel_gdf.geometry.x.values)

            # Store the nearest stations in the cache
            nearest_snotel[idx] = list(snotel_gdf['station_id'].iloc[distances.argsort()[:n]])
            distance_cache[cell_id] = nearest_snotel[idx]
    #saving nearest snotel file
    print(f"Saving nearest SNOTEL in {region} for each cell id in a pkl file")        
    with open(f"{nearest_snotel_dict_path}/nearest_SNOTEL.pkl", 'wb') as handle:
        pkl.dump(nearest_snotel, handle, protocol=pkl.HIGHEST_PROTOCOL)
    #return nearest_snotel

def haversine_vectorized(lat1, lon1, lat2, lon2):
    
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371.0
    # Distance calculation
    distances = r * c

    return distances

def calculate_distances_for_cell(aso_row, snotel_gdf, n=6):
   
    distances = haversine_vectorized(
        aso_row.geometry.y, aso_row.geometry.x,
        snotel_gdf.geometry.y.values, snotel_gdf.geometry.x.values)
    
    nearest_sites = list(snotel_gdf['station_id'].iloc[distances.argsort()[:n]])
    
    return nearest_sites

def create_polygon(row):
        return Polygon([(row['BL_Coord_Long'], row['BL_Coord_Lat']),
                        (row['BR_Coord_Long'], row['BR_Coord_Lat']),
                        (row['UR_Coord_Long'], row['UR_Coord_Lat']),
                        (row['UL_Coord_Long'], row['UL_Coord_Lat'])])

def fetch_snotel_sites_for_cellids(region):  
    #relative file paths
    aso_swe_files_folder_path = f"{HOME}/SWEMLv2.0/data/Processed_SWE/{region}"
    snotel_path = f"{HOME}/SWEMLv2.0/data/SNOTEL_Data/"
    Snotelmeta_path = f"{snotel_path}ground_measures_metadata.csv"
    
    try:
        snotel_file = pd.read_csv(Snotelmeta_path)
    except:
        print("Snotel meta not found, retreiving from AWS S3")
        if not os.path.exists(snotel_path):
            os.makedirs(snotel_path, exist_ok=True)
        key = "NSMv2.0"+Snotelmeta_path.split("SWEMLv2.0",1)[1]       
        S3.meta.client.download_file(BUCKET_NAME, key,Snotelmeta_path)
        snotel_file = pd.read_csv(Snotelmeta_path)

    ASO_meta_loc_DF = pd.DataFrame()

    #add new prediction location here at this step - 
    #will need to make 100m grid for RegionVal.pkl. currenlty 1.8 million sites with just ASO
    #build in method for adding to existing dictionary rather than rerunning for entire region...
    print('Loading all Geospatial prediction/observation files and concatenating into one dataframe')
    for aso_swe_file in tqdm(os.listdir(aso_swe_files_folder_path)):
        aso_file = pd.read_csv(os.path.join(aso_swe_files_folder_path, aso_swe_file))
        ASO_meta_loc_DF = pd.concat([ASO_meta_loc_DF, aso_file])
    
    #removing bad ASO sites
    print(f"Removing nan ASO sites and saving geodataframe to TrainingDFs {region} folder.")
    ASO_meta_loc_DF = ASO_meta_loc_DF[ASO_meta_loc_DF['swe']>=0]
    
    print('Identifying unique sites to create geophysical information dataframe') 
    ASO_meta_loc_DF.drop_duplicates(subset=['cell_id'], inplace=True)
    ASO_meta_loc_DF.set_index('cell_id', inplace=True)
    ASO_meta_loc_DF.to_csv(f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/ASO_meta.parquet")


    print('converting to geodataframe')
    aso_geometry = [Point(xy) for xy in zip(ASO_meta_loc_DF['x'], ASO_meta_loc_DF['y'])]
    aso_gdf = gpd.GeoDataFrame(ASO_meta_loc_DF, geometry=aso_geometry)

    print('Loading SNOTEL metadata and processing snotel geometry')
    snotel_path = f"{HOME}/SWEMLv2.0/data/SNOTEL_Data/"
    Snotelmeta_path = f"{snotel_path}ground_measures_metadata.csv"
    snotel_file = pd.read_csv(Snotelmeta_path)

    snotel_geometry = [Point(xy) for xy in zip(snotel_file['longitude'], snotel_file['latitude'])]
    snotel_gdf = gpd.GeoDataFrame(snotel_file, geometry=snotel_geometry)

    print('Processing snotel geometry')
    snotel_geometry = [Point(xy) for xy in zip(snotel_file['longitude'], snotel_file['latitude'])]
    snotel_gdf = gpd.GeoDataFrame(snotel_file, geometry=snotel_geometry)

    # Calculating nearest SNOTEL sites
    calculate_nearest_snotel(region,aso_gdf, snotel_gdf, n=6)


def GeoSpatial(region):
    print(f"Loading geospatial data for {region}")
    ASO_meta_loc_DF = pd.read_csv(f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/ASO_meta.parquet")

    cols = ['cell_id', 'x', 'y']
    ASO_meta_loc_DF = ASO_meta_loc_DF[cols]

    print(f"Converting to geodataframe")
    aso_geometry = [Point(xy) for xy in zip(ASO_meta_loc_DF['x'], ASO_meta_loc_DF['y'])]
    aso_gdf = gpd.GeoDataFrame(ASO_meta_loc_DF, geometry=aso_geometry)
    
    #renaming columns x/y to lat/long
    aso_gdf.rename(columns = {'y':'cen_lat', 'x':'cen_lon'}, inplace = True)

    return aso_gdf


#Processing using gdal
def process_single_location(args):
    #lat, lon, index_id, tiles = args
    cell_id, lat, lon, DEMs, tiles = args
    #print(lat, lon, index_id, tiles)
    
    #maybe thorugh a try/except here, look up how to find copernicus data
    tile_id = f"Copernicus_DSM_COG_30_N{str(math.floor(lat))}_00_W{str(math.ceil(abs(lon)))}_00_DEM"
    # print(tile_id)
    # display(DEMs)
    index_id = DEMs.loc[tile_id]['sliceID']

    signed_asset = planetary_computer.sign(tiles[int(index_id)].assets["data"])
    #signed_asset = planetary_computer.sign(tiles)
    elevation = rxr.open_rasterio(signed_asset.href)
    
    slope = elevation.copy()
    aspect = elevation.copy()

    transformer = Transformer.from_crs("EPSG:4326", elevation.rio.crs, always_xy=True)
    xx, yy = transformer.transform(lon, lat)

    tilearray = np.around(elevation.values[0]).astype(int)
    geo = (math.floor(float(lon)), 90, 0.0, math.ceil(float(lat)), 0.0, -90)

    driver = gdal.GetDriverByName('MEM')
    temp_ds = driver.Create('', tilearray.shape[1], tilearray.shape[0], 1, gdalconst.GDT_Float32)

    temp_ds.GetRasterBand(1).WriteArray(tilearray)

    tilearray_np = temp_ds.GetRasterBand(1).ReadAsArray()
    grad_y, grad_x = gradient(tilearray_np)

    # Calculate slope and aspect
    slope_arr = np.sqrt(grad_x**2 + grad_y**2)
    aspect_arr = rad2deg(arctan2(-grad_y, grad_x)) % 360 
    
    slope.values[0] = slope_arr
    aspect.values[0] = aspect_arr

    elev = round(elevation.sel(x=xx, y=yy, method="nearest").values[0])
    slop = round(slope.sel(x=xx, y=yy, method="nearest").values[0])
    asp = round(aspect.sel(x=xx, y=yy, method="nearest").values[0])

    return cell_id, elev, slop, asp

def extract_terrain_data_threaded(metadata_df, region):
    global elevation_cache 
    elevation_cache = {} 

    print('Calculating dataframe bounding box')
    bounding_box = metadata_df.geometry.total_bounds
    min_x, min_y, max_x, max_y = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
    
    client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            ignore_conformance=True,
        )

    search = client.search(
                    collections=["cop-dem-glo-90"],
                    intersects = {
                            "type": "Polygon",
                            "coordinates": [[
                            [min_x, min_y],
                            [max_x, min_y],
                            [max_x, max_y],
                            [min_x, max_y],
                            [min_x, min_y]  
                        ]]})

    tiles = list(search.items())

    DEMs = []

    print("Retrieving Copernicus 90m DEM tiles")
    for i in tqdm(range(0, len(tiles))):
        row = [i, tiles[i].id]
        DEMs.append(row)
    DEMs = pd.DataFrame(columns = ['sliceID', 'tileID'], data = DEMs)
    DEMs = DEMs.set_index(DEMs['tileID'])
    del DEMs['tileID']
    print(f"There are {len(DEMs)} tiles in the region")

    print("Determining Grid Cell Spatial Features")

    
    results = []
    with cf.ThreadPoolExecutor(max_workers=None) as executor:
        jobs = {executor.submit(process_single_location, (metadata_df.iloc[i]['cell_id'], metadata_df.iloc[i]['cen_lat'], metadata_df.iloc[i]['cen_lon'], DEMs, tiles)): 

                i for i in tqdm(range(len(metadata_df)))}
        
        print(f"Job complete for getting geospatial metadata, processing dataframe")
        for job in tqdm(cf.as_completed(jobs)):
            results.append(job.result())

    meta = pd.DataFrame(results, columns=['cell_id', 'Elevation_m', 'Slope_Deg', 'Aspect_Deg'])
    meta.set_index('cell_id', inplace=True)
    metadata_df.set_index('cell_id', inplace=True)
    metadata_df = pd.concat([metadata_df, meta], axis = 1)

    #save regional dataframe
    dfpath = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}"
    print(f"Saving {region} dataframe in {dfpath}")
    metadata_df.to_csv(f"{dfpath}/{region}_metadata.parquet")
        
    return metadata_df


def add_geospatial_threaded(region, output_res):
    # Processed ASO observations folder with snotel measurements
    TrainingDFpath = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}"
    GeoObsdfs = f"{TrainingDFpath}/GeoObsDFs"

    #Make directory
    if not os.path.exists(GeoObsdfs):
        os.makedirs(GeoObsdfs, exist_ok=True)

    #Get Geospatial meta data
    print(f"Loading goeospatial meta data for grids in {region}")
    aso_gdf = pd.read_csv(f"{TrainingDFpath}/{region}_metadata.parquet")

    #create dataframe
    print(f"Loading all available processed ASO observations for the {region} at {output_res}M resolution")
    aso_swe_files = []
    for aso_swe_file in tqdm(os.listdir(f"{TrainingDFpath}/Obsdf")):  #add file names to aso_swe_files
        aso_swe_files.append(aso_swe_file)
    print(f"Concatenating {len(aso_swe_files)} with geospatial data...")
    with cf.ProcessPoolExecutor(max_workers=None) as executor: 
        # Start the load operations and mark each future with its process function
        [executor.submit(add_geospatial_single, (f"{TrainingDFpath}/Obsdf", aso_swe_files[i], aso_gdf,GeoObsdfs)) for i in tqdm(range(len(aso_swe_files)))]
        
        print(f"Job complete for connecting obs with geospatial data, the files can be found in {GeoObsdfs}")
    


def add_geospatial_single(args):

    aso_swe_path, aso_swe_file, aso_gdf, GeoObsdfs = args

    ObsDF = pd.read_csv(f"{aso_swe_path}/{aso_swe_file}")

 
    #combine df with geospatial meta data
    final_df = pd.merge(ObsDF, aso_gdf, on = 'cell_id', how = 'left')
    cols = [
        'cell_id', 'Date',  'cen_lat', 'cen_lon', 'geometry', 'Elevation_m', 'Slope_Deg',
        'Aspect_Deg', 'swe', 'nearest_site_1', 'nearest_site_2', 'nearest_site_3', 'nearest_site_4', 
        'nearest_site_5', 'nearest_site_6'
        ]
    final_df = final_df[cols]

    #display(final_df)

    final_df.to_csv(f"{GeoObsdfs}/GeoObsdfs_{aso_swe_file[:8]}.parquet")

