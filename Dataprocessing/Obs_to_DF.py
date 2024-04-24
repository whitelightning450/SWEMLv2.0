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






#This likely needs to be a new .py file...
def Nearest_Snotel_2_obs(region, output_res, dropna = True):    
    print('Connecting site observations with nearest monitoring network obs')

    #get Snotel observations
    snotel_path = f"{HOME}/SWEMLv2.0/data/SNOTEL_Data/"
    Snotelobs_path = f"{snotel_path}ground_measures_train_featuresALLDATES.parquet"
    #ASO observations
    aso_swe_files_folder_path = f"{HOME}/SWEMLv2.0/data/Processed_SWE/{region}"
    #nearest snotel path
    nearest_snotel_dict_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}"

    #Get sites/snotel observations from 2013-2019
    print('Loading observations from 2013-2019')
    try:
        snotel_data = pd.read_csv(Snotelobs_path)
    except:
        print("Snotel obs not found, retreiving from AWS S3")
        if not os.path.exists(snotel_path):
            os.makedirs(snotel_path, exist_ok=True)
        key = "NSMv2.0"+Snotelobs_path.split("SWEMLv2.0",1)[1]        
        S3.meta.client.download_file(BUCKET_NAME, key,Snotelobs_path)
        snotel_data = pd.read_csv(Snotelobs_path)

    #Load dictionary of nearest sites
    print(f"Loading {output_res}M resolution grids for {region} region")
    with open(f"{nearest_snotel_dict_path}/nearest_SNOTEL.pkl", 'rb') as handle:
        nearest_snotel = pickle.load(handle)

    #Processing SNOTEL Obs to correct date/time
    print('Processing datetime component of SNOTEL observation dataframe')
    date_columns = snotel_data.columns[1:]
    new_column_names = {col: pd.to_datetime(col, format='%Y-%m-%d').strftime('%Y%m%d') for col in date_columns}
    snotel_data_f = snotel_data.rename(columns=new_column_names)

    #create data 
    final_df = pd.DataFrame()
    #aso_gdf = pd.DataFrame()

    print(f"Loading all available processed ASO observations for the {region} at {output_res}M resolution")
    for aso_swe_file in tqdm(os.listdir(aso_swe_files_folder_path)):      
        timestamp = aso_swe_file.split('_')[-1].split('.')[0]

        #load in SWE data from ASO
        aso_swe_data = pd.read_csv(os.path.join(aso_swe_files_folder_path, aso_swe_file))
        #aso_gdf = load_aso_snotel_geometry(aso_swe_file, aso_swe_files_folder_path)
        if dropna == True:
            aso_swe_data.dropna(inplace=True)
            aso_swe_data = aso_swe_data[aso_swe_data['swe'] >= 0]
            aso_swe_data.reset_index(inplace=True)
        transposed_data = {}

        if timestamp in new_column_names.values():
            print(f"Connecting ASO observations and Snotel observations for {timestamp}")
            for row in tqdm(np.arange(0, len(aso_swe_data),1)):
                cell_id = aso_swe_data.loc[0]['cell_id']
                station_ids = nearest_snotel[cell_id]
                selected_snotel_data = snotel_data_f[['station_id', timestamp]].loc[snotel_data_f['station_id'].isin(station_ids)]
                station_mapping = {old_id: f"nearest site {i+1}" for i, old_id in enumerate(station_ids)}
                
                # Rename the station IDs in the selected SNOTEL data
                selected_snotel_data['station_id'] = selected_snotel_data['station_id'].map(station_mapping)

                # Transpose and set the index correctly
                transposed_data[cell_id] = selected_snotel_data.set_index('station_id').T
            
            #Convert dictionary of sites to dataframe
            transposed_df = pd.concat(transposed_data, axis=0)

            # Reset index and rename columns
            transposed_df.reset_index(inplace = True)
            transposed_df.rename(columns={'level_0': 'cell_id', 'level_1': 'Date'}, inplace = True)
            transposed_df['Date'] = pd.to_datetime(transposed_df['Date'])

            aso_swe_data['Date'] = pd.to_datetime(timestamp)
            aso_swe_data = aso_swe_data[['cell_id', 'Date', 'swe']]
            merged_df = pd.merge(aso_swe_data, transposed_df, how='left', on=['cell_id', 'Date'])

            final_df = pd.concat([final_df, merged_df], ignore_index=True)

        else:
            aso_swe_data['Date'] = pd.to_datetime(timestamp)
            aso_swe_data = aso_swe_data[['cell_id', 'Date', 'swe']]

            # No need to merge in this case, directly concatenate
            final_df = pd.concat([final_df, aso_swe_data], ignore_index=True)

    final_df.to_csv(f"{nearest_snotel_dict_path}/ASO_Obs_DF.parquet")
    return final_df