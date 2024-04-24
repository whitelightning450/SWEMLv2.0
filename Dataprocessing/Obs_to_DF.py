# Import packages
# Dataframe Packages
import numpy as np
import xarray as xr
import pandas as pd

# Vector Packages
import geopandas as gpd
import shapely
from shapely import wkt
from shapely.geometry import Point, Polygon

# General Packages
import os
import re
from datetime import datetime
import glob
from pathlib import Path
from tqdm import tqdm
import time
import requests
import concurrent.futures as cf
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


#function for processing a single timestamp
def process_single_timestamp(args):
    #get key variable from args
    aso_swe_file, new_column_names, snotel_data_f, region, nearest_snotel , Obsdf = args
    #ASO observations
    aso_swe_files_folder_path = f"{HOME}/SWEMLv2.0/data/Processed_SWE/{region}"
        
    timestamp = aso_swe_file.split('_')[-1].split('.')[0]

    #load in SWE data from ASO
    aso_swe_data = pd.read_csv(os.path.join(aso_swe_files_folder_path, aso_swe_file))
    aso_swe_data.dropna(inplace=True)
    aso_swe_data = aso_swe_data[aso_swe_data['swe'] >= 0]
    aso_swe_data.reset_index(inplace=True)
    transposed_data = {}

    if timestamp in new_column_names.values():
        #print(f"Connecting ASO observations and Snotel observations for {timestamp}")
        for row in tqdm(np.arange(0, len(aso_swe_data),1)):
            cell_id = aso_swe_data.loc[row]['cell_id']
            station_ids = nearest_snotel[cell_id]
            selected_snotel_data = snotel_data_f[['station_id', timestamp]].loc[snotel_data_f['station_id'].isin(station_ids)]
            station_mapping = {old_id: f"nearest_site_{i+1}" for i, old_id in enumerate(station_ids)}
            
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

        Obsdf = pd.concat([Obsdf, merged_df], ignore_index=True)

    else:
        aso_swe_data['Date'] = pd.to_datetime(timestamp)
        aso_swe_data = aso_swe_data[['cell_id', 'Date', 'swe']]

        # No need to merge in this case, directly concatenate
        Obsdf = pd.concat([Obsdf, aso_swe_data], ignore_index=True)

    #save each timesteps df in case of error in data size for all...
    obsdf_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/Obsdf"
    if not os.path.exists(obsdf_path):
        os.makedirs(obsdf_path, exist_ok=True)

    Obsdf.to_csv(f"{obsdf_path}/{timestamp}_ObsDF.parquet")
    return Obsdf

def Nearest_Snotel_2_obs_MultiProcess(region, output_res,max_workers = None):    

    print('Connecting site observations with nearest monitoring network obs')
    #get Snotel observations
    snotel_path = f"{HOME}/SWEMLv2.0/data/SNOTEL_Data/"
    Snotelobs_path = f"{snotel_path}ground_measures_train_featuresALLDATES.parquet"
    #nearest snotel path
    nearest_snotel_dict_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}"
    #ASO observations
    aso_swe_files_folder_path = f"{HOME}/SWEMLv2.0/data/Processed_SWE/{region}"

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

    #Get Geospatial meta data
    print(f"Loading goeospatial meta data for grids in {region}")
    geodf_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}"
    try:
        aso_gdf = pd.read_csv(f"{geodf_path}/{region}_metadata.parquet")
    except:
        print("Snotel obs not found, retreiving from AWS S3")
        if not os.path.exists(snotel_path):
            os.makedirs(snotel_path, exist_ok=True)
        key = "NSMv2.0"+geodf_path.split("SWEMLv2.0",1)[1]        
        S3.meta.client.download_file(BUCKET_NAME, key,f"{geodf_path}/{region}_metadata.parquet")
        aso_gdf = pd.read_csv(Snotelobs_path)

    #Load dictionary of nearest sites
    print(f"Loading {output_res}M resolution grids for {region} region")
    with open(f"{nearest_snotel_dict_path}/nearest_SNOTEL.pkl", 'rb') as handle:
        nearest_snotel = pkl.load(handle)

    #Processing SNOTEL Obs to correct date/time
    print('Processing datetime component of SNOTEL observation dataframe')
    date_columns = snotel_data.columns[1:]
    new_column_names = {col: pd.to_datetime(col, format='%Y-%m-%d').strftime('%Y%m%d') for col in date_columns}
    snotel_data_f = snotel_data.rename(columns=new_column_names)
    
    #create dataframe
    print(f"Loading all available processed ASO observations for the {region} at {output_res}M resolution")
    aso_swe_files = [
        f"ASO_100M_SWE_20130403.csv",
         f"ASO_100M_SWE_20130429.csv",
        # f"ASO_100M_SWE_20130503.csv"
    ]
    # aso_swe_files = []
    # for aso_swe_file in tqdm(os.listdir(aso_swe_files_folder_path)):  #add file names to aso_swe_files
    #     aso_swe_files.append(aso_swe_file)

    print(f"Connecting {len(aso_swe_files)} timesteps of observations for {region}")
    Obsdf = pd.DataFrame()
    #using ProcessPool here because of the python function used (e.g., not getting data but processing it)
    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor: 
        # Start the load operations and mark each future with its process function
        jobs = [executor.submit(process_single_timestamp, (aso_swe_files[i],new_column_names, snotel_data_f, region, nearest_snotel, Obsdf)) for i in tqdm(range(len(aso_swe_files)))]
        
        print(f"Job complete for connecting SNOTEL obs to sites/dates, processing into dataframe")
        for job in tqdm(cf.as_completed(jobs)):
            Obsdf = pd.concat([Obsdf,job.result()])

    print(f"Connecting dataframe with geospatial features...")
    #combine df with geospatial meta data
    final_df = pd.merge(Obsdf, aso_gdf, on = 'cell_id', how = 'left')
    cols = [
        'cell_id', 'Date',  'cen_lat', 'cen_lon', 'geometry', 'Elevation_m', 'Slope_Deg',
       'Aspect_Deg', 'swe', 'nearest_site_1', 'nearest_site_2', 'nearest_site_3', 'nearest_site_4', 
       'nearest_site_5', 'nearest_site_6'
      ]
    final_df = final_df[cols]
    
    final_df.to_csv(f"{nearest_snotel_dict_path}/{region}_Training_DF.parquet")
    return final_df



#same as above multiprocessing function but without multiprocessing
def Nearest_Snotel_2_obs(region, output_res):    
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

    #Get Geospatial meta data
    print(f"Loading goeospatial meta data for grids in {region}")
    geodf_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}"
    try:
        aso_gdf = pd.read_csv(f"{geodf_path}/{region}_metadata.parquet")
    except:
        print("Snotel obs not found, retreiving from AWS S3")
        if not os.path.exists(snotel_path):
            os.makedirs(snotel_path, exist_ok=True)
        key = "NSMv2.0"+geodf_path.split("SWEMLv2.0",1)[1]        
        S3.meta.client.download_file(BUCKET_NAME, key,f"{geodf_path}/{region}_metadata.parquet")
        aso_gdf = pd.read_csv(Snotelobs_path)

    #Load dictionary of nearest sites
    print(f"Loading {output_res}M resolution grids for {region} region")
    with open(f"{nearest_snotel_dict_path}/nearest_SNOTEL.pkl", 'rb') as handle:
        nearest_snotel = pkl.load(handle)

    #Processing SNOTEL Obs to correct date/time
    print('Processing datetime component of SNOTEL observation dataframe')
    date_columns = snotel_data.columns[1:]
    new_column_names = {col: pd.to_datetime(col, format='%Y-%m-%d').strftime('%Y%m%d') for col in date_columns}
    snotel_data_f = snotel_data.rename(columns=new_column_names)
    
    #create dataframe
    print(f"Loading all available processed ASO observations for the {region} at {output_res}M resolution")
    Obsdf = pd.DataFrame()

    for aso_swe_file in tqdm(os.listdir(aso_swe_files_folder_path)):  #multithread this line!!!
          
        timestamp = aso_swe_file.split('_')[-1].split('.')[0]

        #load in SWE data from ASO
        aso_swe_data = pd.read_csv(os.path.join(aso_swe_files_folder_path, aso_swe_file))
        aso_swe_data.dropna(inplace=True)
        aso_swe_data = aso_swe_data[aso_swe_data['swe'] >= 0]
        aso_swe_data.reset_index(inplace=True)
        transposed_data = {}

        if timestamp in new_column_names.values():
            print(f"Connecting ASO observations and Snotel observations for {timestamp}")
            for row in tqdm(np.arange(0, len(aso_swe_data),1)):
                cell_id = aso_swe_data.loc[row]['cell_id']
                station_ids = nearest_snotel[cell_id]
                selected_snotel_data = snotel_data_f[['station_id', timestamp]].loc[snotel_data_f['station_id'].isin(station_ids)]
                station_mapping = {old_id: f"nearest_site_{i+1}" for i, old_id in enumerate(station_ids)}
                
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

            Obsdf = pd.concat([Obsdf, merged_df], ignore_index=True)

        else:
            aso_swe_data['Date'] = pd.to_datetime(timestamp)
            aso_swe_data = aso_swe_data[['cell_id', 'Date', 'swe']]

            # No need to merge in this case, directly concatenate
            Obsdf = pd.concat([Obsdf, aso_swe_data], ignore_index=True)

    print(f"Connecting dataframe with geospatial features...")
    #combine df with geospatial meta data
    final_df = pd.merge(Obsdf, aso_gdf, on = 'cell_id', how = 'left')
    cols = [
        'cell_id', 'Date',  'cen_lat', 'cen_lon', 'geometry', 'Elevation_m', 'Slope_Deg',
       'Aspect_Deg', 'swe', 'nearest_site_1', 'nearest_site_2', 'nearest_site_3', 'nearest_site_4', 
       'nearest_site_5', 'nearest_site_6'
      ]
    final_df = final_df[cols]
    
    final_df.to_csv(f"{nearest_snotel_dict_path}/{region}_Training_DF.parquet")
    return final_df