# Import packages
# Dataframe Packages
import numpy as np
import xarray as xr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import concurrent.futures as cf

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
# import h5py
import pickle
from pystac_client import Client
#import richdem as rd
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
from tqdm._tqdm_notebook import tqdm_notebook
import requests
# import dask
# import dask.dataframe as dd
# from dask.distributed import progress
# from dask.distributed import Client
# from dask.diagnostics import ProgressBar
#from retrying import retry
import fiona
import re
import s3fs

#need to mamba install gdal, earthaccess 
#pip install pystac_client, richdem, planetary_computer, dask, distributed, retrying

#connecting to AWS
import warnings; warnings.filterwarnings("ignore")
import boto3
from botocore import UNSIGNED
from botocore.client import Config

import utils.NSIDC_Data
import netrc
import base64
import getpass

'''
To create .netrc file:  https://earthaccess.readthedocs.io/en/latest/howto/authenticate/
import earthaccess
earthaccess.login(persist=True)
'''

#load access key
HOME = os.getcwd()
KEYPATH = "utils/AWSaccessKeys.csv"

if os.path.isfile(f"{HOME}/{KEYPATH}") == True:
    ACCESS = pd.read_csv(f"{HOME}/{KEYPATH}")

    #start session
    SESSION = boto3.Session(
        aws_access_key_id=ACCESS['Access key ID'][0],
        aws_secret_access_key=ACCESS['Secret access key'][0],
    )
    S3 = SESSION.resource('s3')
    #AWS BUCKET information
    BUCKET_NAME = 'national-snow-model'
    #S3 = boto3.resource('S3', config=Config(signature_version=UNSIGNED))
    BUCKET = S3.Bucket(BUCKET_NAME)
    print('AWS access keys loaded')
else:
    print(f"no AWS credentials present, skipping, {HOME}/{KEYPATH}")


class ASODataTool:
    def __init__(self, short_name, version, polygon='', filename_filter=''):
        self.short_name = short_name
        self.version = version
        self.polygon = polygon
        self.filename_filter = filename_filter
        self.url_list = []
        self.CMR_URL = 'https://cmr.earthdata.nasa.gov'
        self.CMR_PAGE_SIZE = 2000
        self.CMR_FILE_URL = ('{0}/search/granules.json?provider=NSIDC_ECS'
                             '&sort_key[]=start_date&sort_key[]=producer_granule_id'
                             '&scroll=true&page_size={1}'.format(self.CMR_URL, self.CMR_PAGE_SIZE))

    def cmr_search(self, time_start, time_end, region, bounding_box):
        print(f"Fetching file URLs in progress for {region} from {time_start} to {time_end}")
        try:
            if not self.url_list:
                self.url_list = NSIDC_Data.cmr_search(
                    self.short_name, self.version, time_start, time_end,
                    bounding_box=self.bounding_box, polygon=self.polygon,
                    filename_filter=self.filename_filter, quiet=False)
            return self.url_list
        except KeyboardInterrupt:
            quit()

    def get_credentials(self):
        """
        Get credentials from .netrc file
        """
        print('getting credentials NSIDC')

        try:
            info = netrc.netrc()
            username, account, password = info.authenticators("urs.earthdata.nasa.gov")
            credentials = f'{username}:{password}'
            credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')
        except Exception:
            username = input("Earthdata Login Username: ")
            password = getpass.getpass("Earthdata Login Password: ")
            credentials = f'{username}:{password}'
            credentials = base64.b64encode(credentials.encode('ascii')).decode('ascii')
        return credentials

    def cmr_download(self, directory, region):
        #dpath = f"{HOME}/SWEMLv2.0/data/ASO/{region}/{directory}"
        dpath = f"{HOME}/data/ASO/{region}/{directory}"
        if not os.path.exists(dpath):
            os.makedirs(dpath, exist_ok=True)
        
        #Get credential
        credentials = self.get_credentials()

        with cf.ThreadPoolExecutor(max_workers=5) as executor: #setting max workers to none leads to 503 error, NASA does not like us pinging a lot
            {executor.submit(NSIDC_Data.cmr_download, (self.url_list[i]),credentials, dpath,region, False):i for i in tqdm_notebook(range(len(self.url_list)))}

        print(f"All NASA ASO data collected for given date range and can be found in {dpath}...")
        print("Files with .xml extension moved to the destination folder.")

    @staticmethod
    def get_bounding_box(region):
        #dpath = f"{HOME}/SWEMLv2.0/data/PreProcessed"
        dpath = f"{HOME}/data/PreProcessed"

        regions = pd.read_pickle(f"{dpath}/SWEMLV2Regions.pkl")
        # except:
        #     print('File not local, getting from AWS S3.')
        #     if not os.path.exists(dpath):
        #         os.makedirs(dpath, exist_ok=True)
        #     key = f"data/PreProcessed/RegionVal.pkl"            
        #     S3.meta.client.download_file(BUCKET_NAME, key,f"{dpath}/RegionVal.pkl")
        #     regions = pd.read_pickle(f"{dpath}/RegionVal.pkl")


        
        superset = []

        superset.append(regions[region])
        superset = pd.concat(superset)
        superset = gpd.GeoDataFrame(superset, geometry=gpd.points_from_xy(superset.Long, superset.Lat, crs="EPSG:4326"))
        bounding_box = list(superset.total_bounds)

        return f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}"

class ASODownload(ASODataTool):
    def __init__(self, short_name, version, polygon='', filename_filter=''):
        super().__init__(short_name, version, polygon, filename_filter)
        self.region_list =    [ 'N_Sierras',
                                'S_Sierras',
                                'Greater_Yellowstone',
                                'N_Co_Rockies',
                                'SW_Mont',
                                'SW_Co_Rockies',
                                'GBasin',
                                'N_Wasatch',
                                'N_Cascade',
                                'S_Wasatch',
                                'SW_Mtns',
                                'E_WA_N_Id_W_Mont',
                                'S_Wyoming',
                                'SE_Co_Rockies',
                                'Sawtooth',
                                'Ca_Coast',
                                'E_Or',
                                'N_Yellowstone',
                                'S_Cascade',
                                'Wa_Coast',
                                'Greater_Glacier',
                                'Or_Coast'  ]

    def BoundingBox(self, region):
        try:
            self.bounding_box = self.get_bounding_box(region)
            print(f"Bounding Box collected for {region}: {self.bounding_box}")
            return self.bounding_box
        except ValueError:
            print("Invalid input. Please enter a valid index.")
    


class ASODataProcessing:
    
    @staticmethod
    def processing_tiff(region, input_file, output_path, output_res):
        try:
            date = os.path.splitext(input_file)[0].split("_")[-1]
            
            # Define the output file path
            output_folder = os.path.join(output_path, f"{region}/ASO_Processed_{output_res}M_tif")
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"ASO_{output_res}M_{date}.tif")
    
            ds = gdal.Open(input_file)
            if ds is None:
                print(f"Failed to open '{input_file}'. Make sure the file is a valid GeoTIFF file.")
                return None
            
            # Reproject and resample, the Res # needs to be in degrees, ~111,111m to 1 degree
            Res = output_res/111111
            gdal.Warp(output_file, ds, dstSRS="EPSG:4326", xRes=Res, yRes=-Res, resampleAlg="bilinear")
            print('gdal done')
            # Read the processed TIFF file using rasterio
            rds = rxr.open_rasterio(output_file)
            print('rds made')
            rds = rds.squeeze().drop("spatial_ref").drop("band")
            rds.name = "data"
            df = rds.to_dataframe().reset_index()
            return df
    
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
        
        
    def processing_tiff(self, input_file, output_path, output_res, region):
        try:
            date = os.path.splitext(input_file)[0].split("_")[-1]
            
            # Define the output file path
            output_folder = os.path.join(output_path, f"{region}/Processed_{output_res}M_SWE")
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"ASO_{output_res}M_{date}.tif")

            ds = gdal.Open(input_file)
            if ds is None:
                print(f"Failed to open '{input_file}'. Make sure the file is a valid GeoTIFF file.")
                return None
            
            # Reproject and resample, the Res # needs to be in degrees 0.00009 is equivalent to ~100 m
            Res = output_res/111111
            
            gdal.Warp(output_file, ds, dstSRS="EPSG:4326", xRes=Res, yRes=-Res, resampleAlg="bilinear")

            # Read the processed TIFF file using rasterio
            rds = rxr.open_rasterio(output_file)
            rds = rds.squeeze().drop("spatial_ref").drop("band")
            rds.name = "data"
            df = rds.to_dataframe().reset_index()
            return df
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
        
    def region_sort(self,input_file, region):
        try:
            dir = f"{HOME}/data/ASO/"
            output_folder = os.path.join(dir, f"{region}/Raw_ASO_Data")
            os.makedirs(output_folder, exist_ok=True)
            # print(output_folder)
            ## clean up file name; retrieve & reformat date
            date = next(element for element in os.path.splitext(input_file)[0].split("_") if element.startswith('20'))
            if type(date[4]) == str:
                date_singleday = os.path.splitext(date)[0].split("-")[0]
                datetime_object = datetime.strptime(date_singleday, "%Y%b%d")
                date = datetime_object.strftime('%Y%m%d')
            basin = os.path.splitext(input_file)[0].split("_")[1]
            dst = f"{output_folder}/ASO_50M_SWE_{basin}_{date}.tif"
            BBox = ASODataTool.get_bounding_box_list(region)

            #open file, reproject to WGS84, retrieve file extent
            rds = rxr.open_rasterio(input_file)
            rds_4326 = rds.rio.reproject("EPSG:4326")
            xmin = np.float64(np.min(rds_4326['x']))
            xmax = np.float64(np.max(rds_4326['x']))
            ymin = np.float64(np.min(rds_4326['y']))
            ymax = np.float64(np.max(rds_4326['y']))
            rds_bbox = [xmin,ymin,xmax,ymax]

            #check if corners of file extent are within region bbox:
            if ((xmin > BBox[0] and xmin < BBox[2]) or (xmax > BBox[0] and xmax < BBox[2])):
                if ((ymin > BBox[1] and ymin < BBox[3]) or (ymax > BBox[1] and ymax < BBox[3])):
                    print(region)
                    #save file w/ original projection in Raw_ASO_Data directory in correct region
                    shutil.copy(input_file, dst)
            # else:
                # print('no match')         
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def average_duplicates(self, cell_id, aso_file, siteave_dic):
        sitex = aso_file[aso_file['cell_id'] == cell_id]
        mean_lat = np.round(np.mean(sitex['cen_lat']),3)
        mean_lon = np.round(np.mean(sitex['cen_lon']),3)
        mean_swe = np.round(np.mean(sitex['swe_m']),2)

        tempdic = {'cell_id': cell_id,
                'cen_lat': mean_lat,
                'cen_lon': mean_lon,
                'swe_m': mean_swe
        }

        sitedf = pd.DataFrame(tempdic, index = [cell_id])
        siteave_dic[cell_id] = sitedf
            

    def process_single_ASO_file(self, args):
            
        folder_path, tiff_filename, output_res, region, dir = args
        # Open the TIFF file
        tiff_filepath = os.path.join(folder_path, tiff_filename)
        df = self.processing_tiff(tiff_filepath, dir, output_res, region)

        date = os.path.splitext(tiff_filename)[0].split("_")[-1]

        #process file for more efficient saving and fix column headers
        df.rename(columns = {'x': 'cen_lon', 'y':'cen_lat', 'data':'swe_m'}, inplace = True)
        df = df[df['swe_m'] >=0]
        #make cell_id for each site
        df['cell_id'] = df.apply(lambda row: self.make_cell_id(region, output_res, row['cen_lat'], row['cen_lon']), axis=1)

        #get unique cell ids
        cell_ids = df.cell_id.unique()

        if len(df) > len(cell_ids):
            siteave_dic = {}
            print(f"Taking {len(df)} observations down to {len(cell_ids)} unique cell ids and taking the spatial average to get {output_res} m resolution for timestep {date}")
            [self.average_duplicates(cell_id, df, siteave_dic) for cell_id in cell_ids]
            df = pd.concat(siteave_dic)

        if df is not None:
            # Define the parquet filename and folder
            parquet_filename = f"ASO_{output_res}M_SWE_{date}.parquet"
            parquet_folder = os.path.join(dir, f"{region}/{output_res}M_SWE_parquet")
            os.makedirs(parquet_folder, exist_ok=True)
            parquet_filepath = os.path.join(parquet_folder, parquet_filename)

            # Save the DataFrame as a parquet file
            #Convert DataFrame to Apache Arrow Table
            table = pa.Table.from_pandas(df)
            # Parquet with Brotli compression
            pq.write_table(table, parquet_filepath, compression='BROTLI')
        
            
    def convert_tiff_to_parquet_multiprocess(self, input_folder, output_res, region):

        print('Converting .tif to parquet')
        # dir = f"{HOME}/SWEMLv2.0/data/ASO/"
        dir = f"{HOME}/data/ASO/"
        folder_path = os.path.join(dir, input_folder)
        
        # Check if the folder exists and is not empty
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return
        
        if not os.listdir(folder_path):
            print(f"The folder '{input_folder}' is empty.")
            return

        tiff_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".tif")]
        print(f"Converting {len(tiff_files)} ASO tif files to parquet")
        
        # Create parquet files from TIFF files
        with cf.ProcessPoolExecutor(max_workers=None) as executor: 
        # Start the load operations and mark each future with its process function
            [executor.submit(self.process_single_ASO_file, (folder_path, tiff_files[i], output_res, region, dir)) for i in tqdm_notebook(range(len(tiff_files)))]

        print('Checking to make sure all files successfully converted...')
        parquet_folder = os.path.join(dir, f"{region}/{output_res}M_SWE_parquet")
        for parquet_file in tqdm_notebook(os.listdir(parquet_folder)):
            try:
                aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
            except:# add x number of attempts
                print(f"Bad file conversion for {parquet_file}, attempting to reprocess")
                tiff_file = f"ASO_50M_SWE_USCACE_{parquet_file[-16:-8]}.tif"
                print(tiff_file)
                # redo function to convert tiff to parquet
                args = folder_path, tiff_file, output_res, region, dir
                self.process_single_ASO_file(args)
                try:
                    print('Attempt 2')
                    # try to reloade again
                    aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
                    print(f"Bad file conversion for {tiff_file}, attempting to reprocess")
                except:
                    # redo function to convert tiff to parquet
                    self.process_single_ASO_file(args)
                    try:
                        print('Attempt 3')
                        # try to reloade again
                        aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
                        print(f"Bad file conversion for {tiff_file}, attempting to reprocess")
                    except:
                        # redo function to convert tiff to parquet
                        self.process_single_ASO_file(args)
                        try:
                            print('Attempt 4')
                            # try to reloade again
                            aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
                            print(f"Bad file conversion for {tiff_file}, attempting to reprocess")
                        except:
                            # redo function to convert tiff to parquet
                            self.process_single_ASO_file(args)
                            try:
                                print('Attempt 5')
                                # try to reloade again
                                aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
                                print(f"Bad file conversion for {tiff_file}, attempting to reprocess")
                            except:
                                # redo function to convert tiff to parquet
                                self.process_single_ASO_file(args)
                                
                                
class ASODataProcessing_v2: #Revised script to work with 2020-2024 data put into Water Years rather than regions
    
    @staticmethod
    def processing_tiff(WY, input_file, output_path, output_res):
        try:
            print('Using function file')
            #code block for converting input file string to date to match WY2013-2019 format
            #2020-2024 is formatted differently
            if WY in ['2020', '2021', '2022', '2023', '2024']:

                if input_file[-7:-4] == '50m':
                    date = os.path.splitext(input_file)[0].split("_")[-3]
                if input_file[-7:-4] == 'agg':
                    date = os.path.splitext(input_file)[0].split("_")[-4]

                if '-' in date:
                    date = date.split("-")[0]

                yr = date[:4]
                m = str(datetime.strptime(date[4:7], "%b").month)
                d = date[7:]

                if len(m) == 1:
                    m = f"0{m}"
                if len(d) == 1:
                    d = f"0{d}"

                date = f"{yr}{m}{d}"

                #get location
                loc = os.path.splitext(input_file)[0].split("_")[1]
            #formatting for <2020
            else:
                date = os.path.splitext(input_file)[0].split("_")[-1]
                loc = os.path.splitext(input_file)[0].split("_")[-2]
            # Define the output file path
            output_folder = os.path.join(output_path, f"{WY}/ASO_Processed_{output_res}M_tif")
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"ASO_{loc}_{output_res}M_{date}.tif")
    
            ds = gdal.Open(input_file)
            if ds is None:
                print(f"Failed to open '{input_file}'. Make sure the file is a valid GeoTIFF file.")
                return None
            
            # Reproject and resample, the Res # needs to be in degrees, ~111,111m to 1 degree
            Res = output_res/111111
            gdal.Warp(output_file, ds, dstSRS="EPSG:4326", xRes=Res, yRes=-Res, resampleAlg="bilinear")
            print('gdal done')
            # Read the processed TIFF file using rasterio
            rds = rxr.open_rasterio(output_file)
            print('rds made')
            rds = rds.squeeze().drop("spatial_ref").drop("band")
            rds.name = "data"
            df = rds.to_dataframe().reset_index()
            return df
    
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
        
        
    def processing_tiff(self, input_file, output_path, output_res, WY):
        try:
            #code block for converting input file string to date to match WY2013-2019 format
                   #2020-2024 is formatted differently
            if WY in ['2020', '2021', '2022', '2023', '2024']:

                if input_file[-7:-4] == '50m':
                    date = os.path.splitext(input_file)[0].split("_")[-3]
                if input_file[-7:-4] == 'agg':
                    date = os.path.splitext(input_file)[0].split("_")[-4]

                if '-' in date:
                    date = date.split("-")[0]

                yr = date[:4]
                m = str(datetime.strptime(date[4:7], "%b").month)
                d = date[7:]

                if len(m) == 1:
                    m = f"0{m}"
                if len(d) == 1:
                    d = f"0{d}"

                date = f"{yr}{m}{d}"

                #get location
                loc = os.path.splitext(input_file)[0].split("_")[1]
            #formatting for <2020
            else:
                date = os.path.splitext(input_file)[0].split("_")[-1]
                loc = os.path.splitext(input_file)[0].split("_")[-2]

            
            # Define the output file path
            output_folder = os.path.join(output_path, f"{WY}/Processed_{output_res}M_SWE")
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"ASO_{loc}_{output_res}M_{date}.tif")
            

            ds = gdal.Open(input_file)
            if ds is None:
                print(f"Failed to open '{input_file}'. Make sure the file is a valid GeoTIFF file.")
                return None
            
            # Reproject and resample, the Res # needs to be in degrees 0.00009 is equivalent to ~100 m
            Res = output_res/111111
            
            gdal.Warp(output_file, ds, dstSRS="EPSG:4326", xRes=Res, yRes=-Res, resampleAlg="bilinear")

            # Read the processed TIFF file using rasterio
            rds = rxr.open_rasterio(output_file)
            rds = rds.squeeze().drop("spatial_ref").drop("band")
            rds.name = "data"
            df = rds.to_dataframe().reset_index()
            return df
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
    
    def average_duplicates(self, cell_id, aso_file, siteave_dic):
        sitex = aso_file[aso_file['cell_id'] == cell_id]
        mean_lat = np.round(np.mean(sitex['cen_lat']),3)
        mean_lon = np.round(np.mean(sitex['cen_lon']),3)
        mean_swe = np.round(np.mean(sitex['swe_m']),2)

        tempdic = {'cell_id': cell_id,
                'cen_lat': mean_lat,
                'cen_lon': mean_lon,
                'swe_m': mean_swe
        }

        sitedf = pd.DataFrame(tempdic, index = [cell_id])
        siteave_dic[cell_id] = sitedf
            

    def process_single_ASO_file(self, args):
            
        folder_path, tiff_filename, output_res, WY, dir = args
        # Open the TIFF file
        tiff_filepath = os.path.join(folder_path, tiff_filename)
        df = self.processing_tiff(tiff_filepath, dir, output_res, WY)
        
        #2020-2024 is formatted differently
        if WY in ['2020', '2021', '2022', '2023', '2024']:

            if tiff_filename[-7:-4] == '50m':
                date = os.path.splitext(tiff_filename)[0].split("_")[-3]
            if tiff_filename[-7:-4] == 'agg':
                date = os.path.splitext(tiff_filename)[0].split("_")[-4]

            if '-' in date:
                date = date.split("-")[0]

            yr = date[:4]
            m = str(datetime.strptime(date[4:7], "%b").month)
            d = date[7:]

            if len(m) == 1:
                m = f"0{m}"
            if len(d) == 1:
                d = f"0{d}"

            date = f"{yr}{m}{d}"

            #get location
            loc = os.path.splitext(tiff_filename)[0].split("_")[1]
        #formatting for <2020
        else:
            date = os.path.splitext(tiff_filename)[0].split("_")[-1]
            loc = os.path.splitext(tiff_filename)[0].split("_")[-2]
                

        #process file for more efficient saving and fix column headers
        df.rename(columns = {'x': 'cen_lon', 'y':'cen_lat', 'data':'swe_m'}, inplace = True)
        df = df[df['swe_m'] >=0]
        #make cell_id for each site
        df['cell_id'] = df.apply(lambda row: self.make_cell_id(WY, output_res, row['cen_lat'], row['cen_lon']), axis=1)

        #get unique cell ids
        cell_ids = df.cell_id.unique()

        if len(df) > len(cell_ids):
            siteave_dic = {}
            print(f"Taking {len(df)} observations down to {len(cell_ids)} unique cell ids and taking the spatial average to get {output_res} m resolution for timestep {date}")
            [self.average_duplicates(cell_id, df, siteave_dic) for cell_id in cell_ids]
            df = pd.concat(siteave_dic)

        if df is not None:
            # Define the parquet filename and folder
            parquet_filename = f"ASO_{loc}_{output_res}M_SWE_{date}.parquet"
            parquet_folder = os.path.join(dir, f"{WY}/{output_res}M_SWE_parquet")
            os.makedirs(parquet_folder, exist_ok=True)
            parquet_filepath = os.path.join(parquet_folder, parquet_filename)

            # Save the DataFrame as a parquet file
            #Convert DataFrame to Apache Arrow Table
            table = pa.Table.from_pandas(df)
            # Parquet with Brotli compression
            pq.write_table(table, parquet_filepath, compression='BROTLI')
        
            
    def convert_tiff_to_parquet_multiprocess(self, input_folder, output_res, WY):

        print('Converting .tif to parquet')
        # dir = f"{HOME}/SWEMLv2.0/data/ASO/"
        dir = f"{HOME}/data/ASO/"
        folder_path = os.path.join(dir, input_folder)
        
        # Check if the folder exists and is not empty
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return
        
        if not os.listdir(folder_path):
            print(f"The folder '{input_folder}' is empty.")
            return

        tiff_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".tif")]
        print(f"Converting {len(tiff_files)} ASO tif files to parquet")
        
        # Create parquet files from TIFF files
        with cf.ProcessPoolExecutor(max_workers=None) as executor: 
        # Start the load operations and mark each future with its process function
            [executor.submit(self.process_single_ASO_file, (folder_path, tiff_files[i], output_res, WY, dir)) for i in tqdm_notebook(range(len(tiff_files)))]

        print('Checking to make sure all files successfully converted...')
        parquet_folder = os.path.join(dir, f"{WY}/{output_res}M_SWE_parquet")
        for parquet_file in tqdm_notebook(os.listdir(parquet_folder)):
            try:
                aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
            except:# add x number of attempts
                print(f"Bad file conversion for {parquet_file}, attempting to reprocess")
                tiff_file = f"ASO_50M_SWE_USCACE_{parquet_file[-16:-8]}.tif"
                print(tiff_file)
                # redo function to convert tiff to parquet
                args = folder_path, tiff_file, output_res, WY, dir
                self.process_single_ASO_file(args)
                try:
                    print('Attempt 2')
                    # try to reloade again
                    aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
                    print(f"Bad file conversion for {tiff_file}, attempting to reprocess")
                except:
                    # redo function to convert tiff to parquet
                    self.process_single_ASO_file(args)
                    try:
                        print('Attempt 3')
                        # try to reloade again
                        aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
                        print(f"Bad file conversion for {tiff_file}, attempting to reprocess")
                    except:
                        # redo function to convert tiff to parquet
                        self.process_single_ASO_file(args)
                        try:
                            print('Attempt 4')
                            # try to reloade again
                            aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
                            print(f"Bad file conversion for {tiff_file}, attempting to reprocess")
                        except:
                            # redo function to convert tiff to parquet
                            self.process_single_ASO_file(args)
                            try:
                                print('Attempt 5')
                                # try to reloade again
                                aso_file = pd.read_parquet(os.path.join(parquet_folder, parquet_file))
                                print(f"Bad file conversion for {tiff_file}, attempting to reprocess")
                            except:
                                # redo function to convert tiff to parquet
                                self.process_single_ASO_file(args)

                           
                
    def create_polygon(self, row):
        return Polygon([(row['BL_Coord_Long'], row['BL_Coord_Lat']),
                        (row['BR_Coord_Long'], row['BR_Coord_Lat']),
                        (row['UR_Coord_Long'], row['UR_Coord_Lat']),
                        (row['UL_Coord_Long'], row['UL_Coord_Lat'])])
    
    #make cell_id
    def make_cell_id(self,region, output_res, cen_lat, cen_lon):
        #round lat/long to similar sites, may redue file size...
        cen_lat = round(cen_lat,3) #rounding to 3 past the decimal, 100m =~0.001 degreest
        cen_lon = round(cen_lon,3)
        cell_id = f"{region}_{output_res}M_{cen_lat}_{cen_lon}"
        return cell_id
    
        #make cell_id
    def make_cell_id_v2(self,WY, output_res, cen_lat, cen_lon):
        #round lat/long to similar sites, may redue file size...
        cen_lat = round(cen_lat,3) #rounding to 3 past the decimal, 100m =~0.001 degreest
        cen_lon = round(cen_lon,3)
        cell_id = f"{WY}_{output_res}M_{cen_lat}_{cen_lon}"
        return cell_id

    