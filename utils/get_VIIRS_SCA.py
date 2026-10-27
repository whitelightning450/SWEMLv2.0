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
import concurrent.futures as cf
from tqdm.auto import tqdm
import xarray as xr
import shapely
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq


# Raster Packages
import h5py
import re
import rasterio
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import rasterstats as rs
import earthaccess as ea


import glob
from pprint import pprint
from typing import Union
from pathlib import Path

'''
To create .netrc file:
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
else:
    print(f"no AWS credentials present, {HOME}/{KEYPATH}")
    
#set multiprocessing limits
CPUS = len(os.sched_getaffinity(0))
CPUS = int((CPUS/2)-2)


def get_VIIRS_from_AWS():

    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019] #note, likely will have to redo all of these! found error

    for year in years:
        # SCA_directory = f"{HOME}/SWEMLv2.0/data/VIIRS/WY{year}"
        SCA_directory = f"{HOME}/data/VIIRS/WY{year}"

        if os.path.exists(SCA_directory)== False:
            os.makedirs(SCA_directory)
            print('Getting VIIRS fSCA files')
            key = f"data/VIIRS/WY{year}.zip"            
            #S3.meta.client.download_file(BUCKET_NAME, key,f"{HOME}/SWEMLv2.0/data/VIIRS/WY{year}.zip")
            S3.meta.client.download_file(BUCKET_NAME, key,f"{HOME}/data/VIIRS/WY{year}.zip")
            # with zipfile.ZipFile(f"{HOME}/SWEMLv2.0/data/VIIRS/WY{year}.zip", 'r') as Z:
            with zipfile.ZipFile(f"{HOME}/data/VIIRS/WY{year}.zip", 'r') as Z:
                for elem in Z.namelist() :
                    Z.extract(elem, f"{SCA_directory}/")
            #os.remove(f"{HOME}/SWEMLv2.0/data/VIIRS/WY{year}.zip")
            os.remove(f"{HOME}/data/VIIRS/WY{year}.zip")

def augment_SCA_multiprocessing(region, output_res, threshold):
    """
        Augments the region's dataframe with SCA data.

        Parameters:
            region (str): The region to augment.

        Returns:
            adf (GeoDataFrame): The augmented dataframe.
    """
    # get list of dataframes dataframe
  
    # GeoObsDF_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution/GeoObsDFs" #This may need to be the region
    # ViirsGeoObsDF_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution/VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    # VIIRSdata_path = f"{HOME}/SWEMLv2.0/data/VIIRS"
    GeoObsDF_path = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/GeoObsDFs" #This may need to be the region
    ViirsGeoObsDF_path = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    VIIRSdata_path = f"{HOME}/data/VIIRS"  #/WY{region}"
    
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
    with cf.ProcessPoolExecutor(max_workers=CPUS) as executor:  #settting max works to 6 to not make NASA mad...
        futures = {
            executor.submit(single_df_VIIRS, (f, GeoObsDF_path, VIIRSdata_path, ViirsGeoObsDF_path, threshold, output_res)): f
            for f in GeoObsDF_files
        }
        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Worker error ({futures[future]}): {e}")

    # for file in GeoObsDF_files:
    #     args = file,GeoObsDF_path, VIIRSdata_path, ViirsGeoObsDF_path, threshold, output_res
    #     geoRegionDF = single_df_VIIRS(args)
    print(f"Job complete for connecting VIIRS fsca to sites/dates, files can be found in {ViirsGeoObsDF_path}")

    
def single_df_VIIRS(args):
    GeoObsDF_file, GeoObsDF_path,  SCA_folder, ViirsGeoObsDF_path,threshold, output_res = args
   
    timestamp = GeoObsDF_file.split('_')[-1].split('.')[0]
    print(f"{timestamp}, ")

    region_df = pd.read_parquet(os.path.join(GeoObsDF_path, GeoObsDF_file))

    geoRegionDF = gpd.GeoDataFrame(region_df, geometry=gpd.points_from_xy(region_df.cen_lon, region_df.cen_lat,
                                                                            crs="EPSG:4326"))  # Convert to GeoDataFrame
    date = geoRegionDF.Date.unique().strftime('%Y-%m-%d')[0]
    
    #print(geoRegionDF.total_bounds, SCA_folder, date)

    try:
        region_granules = fetchGranules(geoRegionDF.total_bounds, SCA_folder, date)
        regional_raster = createMergedRxr(region_granules["filepath"])
    except Exception as e:
        print(f"Exception: {e}")
        print(f"No granules found for {date}, requesting data from NSIDC...")
        viirs_error = None
        try:
            download_VIIRS(geoRegionDF.total_bounds, SCA_folder, date)
            region_granules = fetchGranules(geoRegionDF.total_bounds, SCA_folder, date)
            regional_raster = createMergedRxr(region_granules["filepath"])
        except Exception as download_e:
            # store as string only — download_e may hold urllib3 SSL objects that
            # cannot be pickled across the ProcessPoolExecutor boundary; Python 3
            # deletes download_e from local scope on exit from this except block,
            # so raising after the block avoids carrying those references
            viirs_error = f"VIIRS data unavailable for {date}: {download_e}"
        if viirs_error is not None:
            raise RuntimeError(viirs_error)
    
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
    pq.write_table(table,f"{ViirsGeoObsDF_path}/VIIRS_{GeoObsDF_file.split('.')[0]}.parquet", compression='BROTLI')

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

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")
    year = date.year if date.month < 10 else date.year + 1

    dataFolder = f"{dataFolder}/WY{year}"
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)

    if shouldDownload:
        results = (ea.granule_query()
                   .short_name("VNP10A1F")
                   .version("2")
                   .bounding_box(*boundingBox)
                   .temporal(date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
                   .get(100))
        if results:
            ea.download(results, dataFolder)
        else:
            print(f"No VIIRS granules found for {date.strftime('%Y-%m-%d')}")
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
    if date.month >= 10:
        year = date.year + 1
        next_year = date.year
    else:
        year = date.year
        next_year = date.year 
    
    h_str = str(int(h)).zfill(2)
    v_str = str(int(v)).zfill(2)
    return os.path.join(dataRoot, f"WY{next_year}/VNP10A1F.A{dayOfYear}.h{h_str}v{v_str}.*.h5")

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
                           raster.values[0].astype(np.float32),  # float required so nodata=255 -> nan works
                           nodata=noData,
                           affine=raster.rio.transform(),  # required for passing numpy arrays
                           stats=['mean'],  # we only want the mean, others are available if needed
                           all_touched=True,  # include pixels touched by buffer edge, not just those with center inside
                           geojson_out=False,  # we will add the result back into a GeoDataFrame, so no need for GeoJSON
                           )

    gdf["VIIRS_SCA"] = [stat['mean'] for stat in stats]  # add the mean to the GeoDataFrame
    gdf["hasSnow"] = gdf["VIIRS_SCA"] >= threshold

    return gdf



def createMergedRxr(files: list[str]) -> xr.DataArray:
    """
        Creates a merged (mosaic-ed) rasterio dataset from a list of files.

        Parameters:
            files (list[str]): A list of filepaths to open and merge.

        Returns:
            merged (DataArray): A merged DataArray.
    """

    tifs = []
    for file in files:
        if not file:
            continue
        if str(file).endswith('.h5'):
            if os.path.getsize(file) < 500_000:
                os.remove(file)
                raise ValueError(f"Removed incomplete HDF5 file (< 500KB), will re-download: {file}")
            with rasterio.open(file) as src:
                sds = [s for s in src.subdatasets if 'CGF_NDSI_Snow_Cover' in s]
            if not sds:
                raise ValueError(f"CGF_NDSI_Snow_Cover not found in {file}")
            with h5py.File(file, 'r') as h5f:
                meta = h5f['HDFEOS INFORMATION/StructMetadata.0'][()].decode()
            ul_x, ul_y = map(float, re.search(r'UpperLeftPointMtrs=\(([^)]+)\)', meta).group(1).split(','))
            lr_x, lr_y = map(float, re.search(r'LowerRightMtrs=\(([^)]+)\)', meta).group(1).split(','))
            tif = rxr.open_rasterio(sds[0])
            ydim, xdim = tif.shape[1], tif.shape[2]
            x_coords = ul_x + ((lr_x - ul_x) / xdim) * (np.arange(xdim) + 0.5)
            y_coords = ul_y + ((lr_y - ul_y) / ydim) * (np.arange(ydim) + 0.5)
            tif = (tif.assign_coords(x=x_coords, y=y_coords)
                      .rio.write_crs("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"))
            tifs.append(tif)
        else:
            tifs.append(rxr.open_rasterio(file))
    if not tifs:
        raise FileNotFoundError("No valid VIIRS tile files found for this date/region")
    try:
        crs = next((t.rio.crs for t in tifs if t.rio.crs is not None), None)
        noLakes = [tif.where(tif != 237, other=0) for tif in tifs]  # replace all the lake values with 0
        noOceans = [tif.where(tif != 239, other=0) for tif in noLakes]  # replace all the ocean values with 0
        noErrors = [tif.where(tif <= 100, other=255) for tif in noOceans]  # replace all fill/error codes with nodata
        if crs is not None:
            noErrors = [da.rio.write_crs(crs) for da in noErrors]  # .where() drops grid_mapping attr
        merged = merge_arrays(noErrors, nodata=255)
        if merged.rio.crs and merged.rio.crs.to_epsg() != 4326:
            merged = merged.rio.reproject("EPSG:4326", nodata=255)
        return merged
    finally:
        for tif in tifs:
            tif.close()

def calculateGranuleExtent(boundingBox: list[float, float, float, float],
                               day: Union[datetime, str] = datetime(2018, 7, 7)):

    if not isinstance(day, datetime):
        day = datetime.strptime(day, "%Y-%m-%d")

    # Get params situated
    datasetName = "VNP10A1F"  # NPP-SUOMI VIIRS, but JPSS1 VIIRS also exists
    version = "2" #if day > datetime(2018, 1, 1) else "1"  # Looks like as of 5-10-2025 that v2 works for < 2018 now
    #print('VIIRS version: ', version)
    query = (ea.granule_query()
                .short_name(datasetName)
                .version(version)
                .bounding_box(*boundingBox)
                .temporal(day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d"))
                # Grab one day's worth of data, we only care about spatial extent
                )
    results = query.get(100)  # The Western CONUS is usually 7, so this is plenty
    #print(results)
    cells = []
    for result in results:
        geom_data = result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        if geom_data.get("GPolygons"):
            geometry = shapely.geometry.Polygon(
                [(x["Longitude"], x["Latitude"]) for x in
                    geom_data["GPolygons"][0]["Boundary"]["Points"]]
            )
        else:
            br = geom_data["BoundingRectangles"][0]
            geometry = shapely.geometry.box(
                br["WestBoundingCoordinate"], br["SouthBoundingCoordinate"],
                br["EastBoundingCoordinate"], br["NorthBoundingCoordinate"]
            )
        # search by name rather than hardcoded index — attribute order varies across granules
        attrs = {a["Name"].upper(): a["Values"][0] for a in result["umm"]["AdditionalAttributes"]}
        cell = {
            "h": attrs["HORIZONTALTILENUMBER"],
            "v": attrs["VERTICALTILENUMBER"],
            "geometry": geometry
        }
        cells.append(cell)
    geo = gpd.GeoDataFrame(cells, geometry="geometry", crs="EPSG:4326")
    return geo
