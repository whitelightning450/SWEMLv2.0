#connecting to AWS
import warnings; warnings.filterwarnings("ignore")
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import zipfile
import pandas as pd
import numpy as np
import geopandas as gpd
import netrc
import concurrent.futures as cf
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
import xarray as xr
import shapely
from datetime import datetime, date, timedelta
import pyarrow as pa
import pyarrow.parquet as pq


# Raster Packages
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import rasterstats as rs
import earthaccess as ea
from nsidc_fetch import download, format_date, format_boundingbox


import glob
from pprint import pprint
from typing import Union
from pathlib import Path
import time

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


def get_VIIRS_from_AWS():

    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019] #note, likely will have to redo all of these! found error

    for year in years:
        SCA_directory = f"{HOME}/SWEMLv2.0/data/VIIRS/WY{year}"

        if os.path.exists(SCA_directory)== False:
            os.makedirs(SCA_directory)
            print('Getting VIIRS fSCA files')
            key = f"data/VIIRS/WY{year}.zip"            
            S3.meta.client.download_file(BUCKET_NAME, key,f"{HOME}/SWEMLv2.0/data/VIIRS/WY{year}.zip")
            with zipfile.ZipFile(f"{HOME}/SWEMLv2.0/data/VIIRS/WY{year}.zip", 'r') as Z:
                for elem in Z.namelist() :
                    Z.extract(elem, f"{SCA_directory}/")
            os.remove(f"{HOME}/SWEMLv2.0/data/VIIRS/WY{year}.zip")

def augment_SCA_mutliprocessing(region, output_res, threshold):
    """
        Augments the region's dataframe with SCA data.

        Parameters:
            region (str): The region to augment.

        Returns:
            adf (GeoDataFrame): The augmented dataframe.
    """
    # get list of dataframes dataframe
  
    GeoObsDF_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution/GeoObsDFs" #This may need to be the region
    ViirsGeoObsDF_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution/VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    VIIRSdata_path = f"{HOME}/SWEMLv2.0/data/VIIRS"

    #check and make directory for VIIRS DFs
    if not os.path.exists(ViirsGeoObsDF_path):
            os.makedirs(ViirsGeoObsDF_path, exist_ok=True)
 
    #Get list of GeoObsDF dataframes
    GeoObsDF_files = [filename for filename in os.listdir(GeoObsDF_path) if filename.endswith(".parquet")]
    GeoObsDF_files.sort()

    #GeoObsDF_files = GeoObsDF_files[0:1] #This is to develop code...

    #This will get multiprocessed
    print(f"Getting VIIRS fsca values for {len(GeoObsDF_files)} timesteps of observations for {region}")

    #using ProcessPool here because of the python function used (e.g., not getting data but processing it)
    with cf.ProcessPoolExecutor(max_workers=6) as executor:  #settting max works to 6 to not make NASA mad...
        # Start the load operations and mark each future with its process function
        [executor.submit(single_df_VIIRS, (GeoObsDF_files[i],GeoObsDF_path, VIIRSdata_path, ViirsGeoObsDF_path, threshold, output_res)) for i in range(len(GeoObsDF_files))]

    # for file in GeoObsDF_files:
    #     args = file,GeoObsDF_path, VIIRSdata_path, ViirsGeoObsDF_path, threshold, output_res
    #     geoRegionDF = single_df_VIIRS(args)
    print(f"Job complete for connecting VIIRS fsca to sites/dates, files can be found in {ViirsGeoObsDF_path}")

    
def single_df_VIIRS(args):
    GeoObsDF_file, GeoObsDF_path,  SCA_folder, ViirsGeoObsDF_path,threshold, output_res = args
   
    timestamp = GeoObsDF_file.split('_')[-1].split('.')[0]
    print(timestamp)

    region_df = pd.read_parquet(os.path.join(GeoObsDF_path, GeoObsDF_file),engine='fastparquet')
    #round lat/long, literally no need for any higher spatial resolution than 0.001 degrees, as this is ~100 m
    region_df['cen_lon'] =np.round(region_df['cen_lon'], 3)
    region_df['cen_lat'] =np.round(region_df['cen_lat'], 3)

    geoRegionDF = gpd.GeoDataFrame(region_df, geometry=gpd.points_from_xy(region_df.cen_lon, region_df.cen_lat,
                                                                            crs="EPSG:4326"))  # Convert to GeoDataFrame
    date = geoRegionDF.Date.unique().strftime('%Y-%m-%d')[0]

    try:
        # Fetch granules
        region_granules = fetchGranules(geoRegionDF.total_bounds, SCA_folder, date) 
         # Merge granules
        regional_raster = createMergedRxr(region_granules["filepath"]) 
    except:
        print(f"No granules found for {date}, requesting data from NSIDC...")
        #download data
        download_VIIRS(geoRegionDF.total_bounds, SCA_folder, date)
        # Fetch granules
        region_granules = fetchGranules(geoRegionDF.total_bounds, SCA_folder, date) 
         # Merge granules
        regional_raster = createMergedRxr(region_granules["filepath"]) 
    
    #set buffer around cell of interest,must be larger than 400 m
    if output_res >799:
        buffer = output_res/2
    else:
        buffer = 400

    adf = augmentGeoDF(geoRegionDF, regional_raster, buffer=buffer, threshold=threshold)  # Buffer by 500 meters -> 1km square

    adf.pop('geometry')

    # #save file as parquet
    table = pa.Table.from_pandas(adf)
    # Parquet with Brotli compression
    pq.write_table(table,f"{ViirsGeoObsDF_path}/VIIRS_GeoObsDF_{timestamp}.parquet", compression='BROTLI')

    print(f"dataprocessing VIIRS for {timestamp} complete...")


def fetchGranules(boundingBox: list[float, float, float, float],
                  dataFolder: Union[Path, str],
                  date: Union[datetime, str],
                  extentDF: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
            Fetches VIIRS granules from local storage.

            Parameters:
                boundingBox (list[float, float, float, float]): The bounding box of the region of interest. (west, south, east, north)
                date (datetime, str): The start date of the data to fetch.
                dataFolder (Path, str): The folder to save the data to, also used to check for existing data.
                extentDF (GeoDataFrame): A dataframe containing the horizontal and vertical tile numbers and their boundaries
                shouldDownload (bool): Whether to fetch the data from the API or not.

            Returns:
                df (GeoDataFrame): A dataframe of the granules that intersect with the bounding box
        """
    if extentDF is None:
        cells = calculateGranuleExtent(boundingBox, date)  # Fetch granules from API, no need to check bounding box
    else:
        # Find granules that intersect with the bounding box
        cells = extentDF.cx[boundingBox[0]:boundingBox[2],
                boundingBox[1]:boundingBox[3]]  # FIXME if there is only one point, this will fail

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if not isinstance(dataFolder, Path):
        dataFolder = Path(dataFolder)
    
    day = date.strftime("%Y-%m-%d")
    cells["date"] = date  # record the date
    cells["filepath"] = cells.apply(
        lambda x: granuleFilepath(createGranuleGlobpath(dataFolder, date, x['h'], x['v'])),
        axis=1
    )  # add filepath if it exists, otherwise add empty string
    
    #display(cells)
    return cells

def download_VIIRS(boundingBox: list[float, float, float, float],
                  dataFolder: Union[Path, str],
                  date: Union[datetime, str],
                  shouldDownload = True):

    cells = calculateGranuleExtent(boundingBox, date)  # Fetch granules from API, no need to check bounding box

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")
        year = date.year if date.month < 10 else date.year + 1  # Water years start on October 1st 

    #check to see if path exists, if not make one
    #dataFolder = f"{dataFolder}/WY{year}/{year-1}-{year}NASA"
    dataFolder = f"{dataFolder}/WY{year}"
    if os.path.exists(dataFolder)== False:
        os.makedirs(dataFolder)

    if not isinstance(dataFolder, Path):
        dataFolder = Path(dataFolder)
        dayOfYear = date.strftime("%Y%j") 

    
    day = date.strftime("%Y-%m-%d")
    cells["date"] = date  # record the date
    cells["filepath"] = cells.apply(
        lambda x: granuleFilepath(createGranuleGlobpath(dataFolder, date, x['h'], x['v'])),
        axis=1
                )  #
    # display(cells)
    #This line of code goes and gets the data if it is not in folder
    missingCells = cells[cells["filepath"] == ''][["h", "v"]].to_dict("records")
    attempts = 3  # how many times it will try and download the missing granules
    while shouldDownload and len(missingCells) > 0 and attempts > 0:
        temporal = format_date(date)  # Format date as YYYY-MM-DD
        bbox = format_boundingbox(boundingBox)  # Format bounding box as "W,S,E,N"
        version = "2" #if date > datetime(2018, 1, 1) else "1"  # Use version 1 if date is before 2018, looks like all products use 2 now, or this is backwards...
        cells["filepath"] = download("VNP10A1F", version, temporal, bbox, dataFolder, mode="async") #might need try/except here...
        # )  # add filepath if it exists, otherwise add empty string
        missingCells = cells[cells["filepath"] == ''][["h", "v"]].to_dict("records")
        if len(missingCells) > 0:
            attempts -= 1
            print(f"still missing {len(missingCells)} granules for {day}, retrying in 30 seconds, {attempts} tries left")
            time.sleep(30)
            print("retrying")
            cells["filepath"] = download("VNP10A1F", version, temporal, bbox, dataFolder, mode="async") 
            missingCells = cells[cells["filepath"] == ''][["h", "v"]].to_dict("records")  # before we try again, double check
    print('Processing data for dataframe now...')

def createGranuleGlobpath(dataRoot: str, date: datetime, h: int, v: int) -> str:
    """
        Creates a filepath for a VIIRS granule.

        Parameters:
            dataRoot (str): The root folder for the data.
            date (str): The date of the data.
            h (int): The horizontal tile number.
            v (int): The vertical tile number.

        Returns:
            filepath (str): The filepath of the granule.
    """
    dayOfYear = date.strftime("%Y%j")  # Format date as YearDayOfYear

    WY_split = datetime(date.year, 10, 1)  # Split water years on October 1st

    # if day is after water year, then we need to adjust the year
    if date.month > 10:
        year = date.year + 1
        next_year = date.year
    else:
        year = date.year
        next_year = date.year 
    
    #return os.path.join(dataRoot, f"WY{next_year}/{year}-{next_year}NASA/VNP10A1F_A{dayOfYear}_h{h}v{v}_*.tif")
    return os.path.join(dataRoot, f"WY{next_year}/VNP10A1F_A{dayOfYear}_h{h}v{v}_*.tif")

def granuleFilepath(filepath: str) -> str:
    """
        return matched filepath if it exists, otherwise return empty string
    """
    result = glob.glob(filepath)
    
    if result:
        return result[0]  # There should only be one match
    else:
        return ''

def augmentGeoDF(gdf: gpd.GeoDataFrame,
                 raster: xr.DataArray,
                 threshold: float = 20,  # TODO try 10
                 noData: int = 255,
                 buffer: float = None) -> gpd.GeoDataFrame:
    """
        Augments a GeoDataFrame with a raster's values.

        Parameters:
            gdf (GeoDataFrame): The GeoDataFrame to append the SCA to. Requires geometry to be an area, see buffer param
            raster (DataArray): The raster to augment the GeoDataFrame with.
            threshold (int): The threshold to use to determine if a pixel is snow or not.
            noData (int): The no data value of the raster.
            buffer (float): The buffer to use around the geometry. Set if the geometry is a point.

        Returns:
            gdf (GeoDataFrame): The augmented GeoDataFrame.
    """

    if buffer is not None:
        buffered = gdf.to_crs("3857").buffer(buffer,
                                             cap_style=3)  # Convert CRS to a projected CRS and buffer points into squares
        buffered = buffered.to_crs(raster.rio.crs)  # Convert the GeoDataFrame to the same CRS as the raster
    else:
        buffered = gdf.to_crs(raster.rio.crs)  # Convert the GeoDataFrame to the same CRS as the raster

    stats = rs.zonal_stats(buffered,  # pass the buffered geometry
                           raster.values[0],  # pass the raster values as a numpy array, TODO investigate passing GDAL
                           no_data=noData,
                           affine=raster.rio.transform(),  # required for passing numpy arrays
                           stats=['mean'],  # we only want the mean, others are available if needed
                           geojson_out=False,  # we will add the result back into a GeoDataFrame, so no need for GeoJSON
                           )

    gdf["VIIRS_SCA"] = [stat['mean'] for stat in stats]  # add the mean to the GeoDataFrame
    gdf["hasSnow"] = gdf["VIIRS_SCA"] > threshold

    return gdf



def createMergedRxr(files: list[str]) -> xr.DataArray:
    """
        Creates a merged (mosaic-ed) rasterio dataset from a list of files.

        Parameters:
            files (list[str]): A list of filepaths to open and merge.

        Returns:
            merged (DataArray): A merged DataArray.
    """

    # FIXME sometimes throws "CPLE_AppDefined The definition of geographic CRS EPSG:4035 got from GeoTIFF keys is not
    #   the same as the one from the EPSG registry, which may cause issues during reprojection operations. Set
    #   GTIFF_SRS_SOURCE configuration option to EPSG to use official parameters (overriding the ones from GeoTIFF
    #   keys), or to GEOKEYS to use custom values from GeoTIFF keys and drop the EPSG code."
    tifs = [rxr.open_rasterio(file) for file in files]  # Open all the files as Rioxarray DataArrays

    noLakes = [tif.where(tif != 237, other=0) for tif in tifs]  # replace all the lake values with 0
    noOceans = [tif.where(tif != 239, other=0) for tif in noLakes]  # replace all the ocean values with 0
    noErrors = [tif.where(tif <= 100, other=100) for tif in
                noOceans]  # replace all the other values with 100 (max Snow)
    return merge_arrays(noErrors, nodata=255)  # Merge the arrays

def calculateGranuleExtent(boundingBox: list[float, float, float, float],
                               day: Union[datetime, str] = datetime(2018, 7, 7)):

    if not isinstance(day, datetime):
        day = datetime.strptime(day, "%Y-%m-%d")

    # Get params situated
    datasetName = "VNP10A1F"  # NPP-SUOMI VIIRS, but JPSS1 VIIRS also exists
    version = "2" if day > datetime(2018, 1, 1) else "1"  # TODO v1 supports 2013-on, but v2 currently breaks <2018??? 
    #print('VIIRS version: ', version)
    query = (ea.granule_query()
                .short_name(datasetName)
                .version(version)
                .bounding_box(*boundingBox)
                .temporal(day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d"))
                # Grab one day's worth of data, we only care about spatial extent
                )
    results = query.get(100)  # The Western CONUS is usually 7, so this is plenty

    cells = []
    for result in results:
        geometry = shapely.geometry.Polygon(
            [(x["Longitude"], x["Latitude"]) for x in
                result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"][
                    "Points"]]
        )
        cell = {
            "h": result["umm"]["AdditionalAttributes"][1]["Values"][0],  # HORIZONTAL TILE NUMBER
            "v": result["umm"]["AdditionalAttributes"][2]["Values"][0],  # VERTICAL TILE NUMBER
            "geometry": geometry
        }
        cells.append(cell)

    geo = gpd.GeoDataFrame(cells, geometry="geometry", crs="EPSG:4326")
    return geo
