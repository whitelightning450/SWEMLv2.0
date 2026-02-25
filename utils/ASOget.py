# Import packages
# Dataframe Packages
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import concurrent.futures as cf

# Vector Packages
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS, Transformer

# Raster Packages
import rioxarray as rxr
import rasterio
from rioxarray.merge import merge_arrays
# import rasterstats as rs
from osgeo import gdal
from osgeo import gdalconst

# Data Access Packages
# import earthaccess as ea
# import h5py
# import pickle
# from pystac_client import Client
#import richdem as rd
# import planetary_computer
# from planetary_computer import sign

# General Packages
import os
import re
from datetime import datetime
import glob
# from pprint import pprint
# from typing import Union
# from pathlib import Path
from tqdm._tqdm_notebook import tqdm_notebook
from tqdm.auto import tqdm

# import requests
# import dask
# import dask.dataframe as dd
# from dask.distributed import progress
# from dask.distributed import Client
# from dask.diagnostics import ProgressBar
#from retrying import retry
# import fiona
# import re
# import s3fs

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
    
#set multiprocessing limits
CPUS = len(os.sched_getaffinity(0))
CPUS = int((CPUS/2)-2)


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
                                
                                
class ASODataProcessing_v2: 
    #Revised script to work with 2020-2024 data put into Water Years rather than regions
    """
    Converts raw ASO GeoTIFF files to Brotli-compressed Parquet.

    Internally reprojects to EPSG:5070 (NAD83 Conus Albers Equal Area) so
    resampling uses a truly square pixel grid in meters, then converts pixel
    centers back to WGS84 for downstream compatibility.

    Output parquet schema:
        cell_id : str    "{res}M_{cen_lat}_{cen_lon}"
        cen_lat : float  WGS84 latitude  (degrees, rounded to 3 dp)
        cen_lon : float  WGS84 longitude (degrees, rounded to 3 dp)
        swe_m   : float  snow water equivalent (meters)

    Handles both filename conventions:
        Pre-2020:  ASO_50M_SWE_{BASIN}_{YYYYMMDD}.tif
        2020+:     ASO_{Basin}[_Type]_{YYYY}{Mon}{D[-range]}_swe_50m.tif
                   (multi-day ranges use the first day)
    """

    @staticmethod
    def parse_filename(filename):
        """Returns (basin, date_str) where date_str is YYYYMMDD."""
        stem = os.path.splitext(filename)[0]
        parts = stem.split("_")

        # Pre-2020: ASO_50M_SWE_{BASIN}_{YYYYMMDD}
        if parts[1] == '50M':
            return parts[3], parts[4]

        # 2020+: find element starting with 4-digit year followed by letters
        basin = parts[1]
        date_part = next(p for p in parts if re.match(r'^20\d{2}[A-Za-z]', p))
        date_part = date_part.split('-')[0]  # first day of any multi-day range
        yr = date_part[:4]
        month_str = re.search(r'[A-Za-z]+', date_part[4:]).group()
        day_str   = re.search(r'\d+',       date_part[4:]).group()
        month_num = datetime.strptime(month_str[:3], "%b").month
        return basin, f"{yr}{month_num:02d}{int(day_str):02d}"
    
    @staticmethod
    def make_cell_id(output_res, cen_lat, cen_lon):
        return f"{output_res}M_{round(cen_lat, 6)}_{round(cen_lon, 6)}"
    
    def process_single_ASO_file(self, args):
        """Reproject one ASO tif to Albers, resample, convert back to WGS84, save as parquet."""
        folder_path, tiff_filename, output_res, WY, dir = args
        tiff_path = os.path.join(folder_path, tiff_filename)

        try:
            basin, date = self.parse_filename(tiff_filename)
        except Exception as e:
            print(f"Could not parse '{tiff_filename}': {e}")
            return

        try:
            # Reproject to Albers Equal Area — pixel size is true square meters
            rds = rxr.open_rasterio(tiff_path, masked=True)
            rds_albers = rds.rio.reproject(
                "EPSG:5070",
                resolution=output_res,
                resampling=rasterio.enums.Resampling.bilinear,
            ).squeeze(drop=True)
            rds_albers.name = "swe_m"

            data = rds_albers.values                  # 2D numpy array, NaN where nodata
            xs   = rds_albers.x.values                # 1D array of Albers x coords
            ys   = rds_albers.y.values                # 1D array of Albers y coords

            # Build pixel mask 
            xx, yy = np.meshgrid(xs, ys)
            mask = np.isfinite(data) & (data >= 0)
            if not mask.any():
                print(f"No valid SWE data in {tiff_filename}")
                return

            transformer = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(xx[mask], yy[mask])

            df = pd.DataFrame({
                'cen_lat': lat.round(6),
                'cen_lon': lon.round(6),
                'swe_m':   data[mask].round(4),
            })
            df['cell_id'] = df.apply(
                lambda row: self.make_cell_id(output_res, row['cen_lat'], row['cen_lon']), axis=1
            )
            df = df[['cell_id', 'cen_lat', 'cen_lon', 'swe_m']]

            parquet_folder = os.path.join(dir, f"{WY}/{output_res}M_SWE_parquet")
            os.makedirs(parquet_folder, exist_ok=True)
            pq.write_table(
                pa.Table.from_pandas(df),
                os.path.join(parquet_folder, f"ASO_{basin}_{output_res}M_SWE_{date}.parquet"),
                compression='BROTLI',
            )
        except Exception as e:
            print(f"Error processing {tiff_filename}: {e}")

    def _process_wrapper(self, args):
        self.process_single_ASO_file(args)
            
    def convert_tiff_to_parquet_multiprocess(self, input_folder, output_res, WY):
        aso_dir = f"{HOME}/data/ASO/"
        folder_path = os.path.join(aso_dir, input_folder)

        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}")
            return

        tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        if not tiff_files:
            print(f"No .tif files found in {folder_path}")
            return

        print(f"Converting {len(tiff_files)} files  WY{WY}  {output_res}m resolution")
        args_list = [
            (folder_path, f, output_res, WY, aso_dir)
            for f in tiff_files
        ]

        with cf.ProcessPoolExecutor(max_workers=CPUS) as executor:
            futures = [executor.submit(self._process_wrapper, args) for args in args_list]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"WY{WY}"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker error: {e}")


        parquet_folder = os.path.join(aso_dir, f"{WY}/{output_res}M_SWE_parquet")
        failed = []
        for pf in tqdm_notebook(os.listdir(parquet_folder), desc="Verifying"):
            try:
                pd.read_parquet(os.path.join(parquet_folder, pf))
            except Exception:
                failed.append(pf)

        if failed:
            print(f"{len(failed)} files failed verification: {failed}")
        else:
            print(f"All {len(os.listdir(parquet_folder))} parquet files verified.")                        
                
    def create_polygon(self, row):
        return Polygon([(row['BL_Coord_Long'], row['BL_Coord_Lat']),
                        (row['BR_Coord_Long'], row['BR_Coord_Lat']),
                        (row['UR_Coord_Long'], row['UR_Coord_Lat']),
                        (row['UL_Coord_Long'], row['UL_Coord_Lat'])])
    