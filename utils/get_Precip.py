import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

#data packages
# import pydaymet as daymet ## Issues with this package and API -- skipping for now
# import pynldas2 as nldas
import pygridmet as gridmet
from pygridmet import GridMET
# import ee #pip install earthengine-api
# import utils.EE_funcs as EE_funcs
import urllib.request
import zipfile

#multiprocessing
from tqdm.auto import tqdm
import concurrent.futures as cf

#raster packages
import rasterio

import os
from datetime import datetime, timedelta

import boto3
import s3fs
# ee.Authenticate()
# ee.Initialize()
import warnings
warnings.filterwarnings("ignore")

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
    

def ProcessDates(args):
    date, WY_precip, Precippath = args
    ts = pd.to_datetime(str(date)) 
    d = ts.strftime('%Y-%m-%d')
    filed = d = ts.strftime('%Y%m%d')
    precipdf = WY_precip[WY_precip['datetime'] == d]
    precipdf.set_index('cell_id', inplace = True)
    precipdf.pop('datetime')
    precipdf['season_precip_cm'] = round(precipdf['season_precip_cm'],1)
    #Convert DataFrame to Apache Arrow Table
    table = pa.Table.from_pandas(precipdf)
    # Parquet with Brotli compression
    pq.write_table(table, f"{Precippath}/NLDAS_PPT_{filed}.parquet", compression='BROTLI')


def Make_Precip_DF(region, output_res, threshold, dataset):
    
    print(f"Adding precipitation features to ML dataframe for {region}.")
    Precippath = f"{HOME}/data/Precipitation/{region}/{output_res}M_{dataset}_Precip"

    #make precip df path
    if dataset == 'Daymet':
        PrecipDFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/Daymet_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
        DFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    ## daymet API currently deprecated, so skip to NLDAS from Veg
    elif dataset == 'NLDAS':
        PrecipDFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/NLDASDaymet_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
        DFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/DaymetVegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    # temporarily skipping to gridmet from veg -- issues with daymet and nldas APIs    
    elif dataset == 'gridMET':
        PrecipDFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/gridMETNLDASDaymet_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
        DFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    elif dataset == 'AORC':
        PrecipDFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/AORCgridMETNLDASDaymet_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
        DFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    if not os.path.exists(PrecipDFpath):
        os.makedirs(PrecipDFpath, exist_ok=True)

    #Get list of dataframes
    GeoObsDF_files = [filename for filename in os.listdir(DFpath) if filename.endswith('.parquet')]

    with cf.ProcessPoolExecutor(max_workers=CPUS) as executor:
        futures = {
            executor.submit(single_date_add_daymet_precip, (DFpath, Precippath, geofile, PrecipDFpath, region, dataset)): geofile
            for geofile in GeoObsDF_files
        }
        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Worker error ({futures[future]}): {e}")

    
def single_date_add_daymet_precip(args):
    training_df_path, precip_data_path, geofile, precip_df_path, WY, dataset = args
    #get date information
    date = geofile.split('_')[-1].split('.parquet')[0]
    region = geofile.split('_')[-2]
    region_date = f"{region}_{date}"
    year = date[:4]
    mon = date[4:6]
    day = date[6:]
    strdate = f"{year}-{mon}-{day}"
    print(f"Connecting precipitation to ASO observations for {WY} on {strdate} at {region}")
    
    if dataset == 'Daymet':
        var = 'Daymet'
    elif dataset == 'gridMET':
        var = 'gridMET'
    elif dataset == 'NLDAS':
        var = 'NLDAS'
    elif dataset == 'AORC':
        var = 'AORC'
    else:
        raise ValueError("dataset not recognized") 
    
    GDF = pd.read_parquet(os.path.join(training_df_path, geofile))

    # find the matching precip file
    pptfiles = [filename for filename in os.listdir(precip_data_path) if filename.endswith('.parquet')]
    ppt_filename = [filename for filename in pptfiles if region_date in filename]
    if not ppt_filename:
        print(f"No precip file found for {region_date}, skipping.")
        return
    ppt = pd.read_parquet(f"{precip_data_path}/{ppt_filename[0]}")

    # vectorized join on cell_id
    GDF = GDF.merge(
        ppt[['cell_id', 'season_precip_cm']].rename(columns={'season_precip_cm': var}),
        on='cell_id', how='left'
    )
    GDF[var] = GDF[var].round(1).fillna(0.0)
    
    #Convert DataFrame to Apache Arrow Table
    table = pa.Table.from_pandas(GDF)
    # Parquet with Brotli compression
    pq.write_table(table, f"{precip_df_path}/{dataset}_{geofile}", compression='BROTLI')
#          


def get_precip(WY,output_res,thresh,dataset):
    # get precip by grabbing bounding box from previous training DF for basin + date
    # set start date for precip obs to Oct 1 of previous year
    valid = ['daymet','gridmet','nldas']
    if dataset.lower() not in valid:
        raise ValueError("dataset must be one of %r." % valid)
    if dataset.lower() == 'daymet':
        dataset = 'Daymet'
    elif dataset.lower() == 'nldas':
        dataset = 'NLDAS'
    elif dataset.lower() == 'gridmet':
        dataset = 'gridMET'
        
    WY_start = datetime(WY-1, 10, 1)
    obs_start = WY_start.strftime('%Y-%m-%d')
    print("Water Year start date:",obs_start)
    
    # select basins, dates by training DF
    training_df_dir = f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/VIIRSGeoObsDFs/{thresh}_fSCA_Thresh"
    files = [filename for filename in os.listdir(training_df_dir)
             if filename.endswith(".parquet")
            ]
    # print(files)
    for file in tqdm(files, desc="Processing precip files"):
        filepath = f'{training_df_dir}/{file}'
        #Get timestamp
        timestamp = file.split('_')[-1].split('.')[0]
        #Get region
        region = file.split('_')[-2]
        # print(timestamp,region)
        obs_end = f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:]}'

        # check to see if precip data already downloaded, if not, then download
        precip_data_path = f"{HOME}/data/Precipitation/{WY}/{output_res}M_{dataset}_Precip"
        if not os.path.exists(precip_data_path):
            os.makedirs(precip_data_path, exist_ok=True)
        if not os.path.exists(f"{precip_data_path}/{dataset}_{region}_{timestamp}.parquet"):
        
            print(f"Getting precipitation data for {obs_end} at {region}")
            
            training_df = pd.read_parquet(filepath)
            # get bounding box by min/max coordinates
            left, right = training_df['cen_lon'].min(), training_df['cen_lon'].max()
            bottom, top = training_df['cen_lat'].min(), training_df['cen_lat'].max()
            # add some padding to bbox
            left -= 0.1
            bottom -= 0.1
            right += 0.1
            top += 0.1
            bbox = rasterio.coords.BoundingBox(left, bottom, right, top)
            # print(bbox)    
    
            # get precip from appropriate server from beginning of WY through observation date and reproject
            if dataset == 'Daymet':
                var = "prcp"
                obs_precip = daymet.get_bygeom(bbox,dates=(obs_start,obs_end),variables=var,crs="epsg:4326")
            elif dataset == 'gridMET':
                var = 'pr'
                obs_precip = gridmet.get_bygeom(bbox,dates=(obs_start,obs_end),variables=var,crs="epsg:4326")
            elif dataset == 'NLDAS':
                var = "prcp"
                obs_precip = nldas.get_bygeom(bbox,obs_start,obs_end,variables=var,geo_crs=4326)
            obs_precip_transformed = obs_precip.rio.reproject(rasterio.crs.CRS.from_epsg('4326'))   
            
            # extract metadata
            meta = training_df[['cell_id','cen_lat','cen_lon']]
            lats = xr.DataArray(meta['cen_lat'].values, dims='points')
            lons = xr.DataArray(meta['cen_lon'].values, dims='points')
            try:
                prcp_all = obs_precip_transformed[var].sel(x=lons, y=lats, method='nearest')
                raw_vals = prcp_all.values  # shape (time, npoints)
                season_precip_cm = np.round(raw_vals.sum(axis=0) / 10, 2)
            finally:
                obs_precip_transformed.close()
            precip_df = pd.DataFrame({
                'cell_id': meta['cell_id'].values,
                'cen_lat': meta['cen_lat'].values,
                'cen_lon': meta['cen_lon'].values,
                'precip': list(raw_vals.T),
                'season_precip_cm': season_precip_cm,
            })
            
            # save raw data for each basin and date
                
            table = pa.Table.from_pandas(precip_df)
            pq.write_table(table, f"{precip_data_path}/{dataset}_{region}_{timestamp}.parquet", compression='BROTLI')

## AORC functions adapted from CUAHSI HydroShare library - thank you!
def get_aorc_precip(WY,output_res,thresh):
    # define start of water year
    WY_start = datetime(WY-1, 10, 1)
    obs_start = WY_start.strftime('%Y-%m-%d')
    print("Water Year start date:",obs_start)

    ## Create a list of years to retrieve data 
    yrs = [WY-1, WY]
    
    ## Loading data (AORC data are organized by years, look at https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com/index.html)
    # Grab data for entire WY before clipping to each basin per training DFs 
    # Base URL
    base_url = f's3://noaa-nws-aorc-v1-1-1km'
    # Creating a connection to Amazon S3 bucket using the s3fs library (https://s3fs.readthedocs.io/en/latest/api.html).
    s3_out = s3fs.S3FileSystem(anon=True)              # access S3 as if it were a file system. 
    fileset = [s3fs.S3Map(                             # maps each year's Zarr dataset from S3 to a local-like object.
                root=f"s3://{base_url}/{yr}.zarr",     # Zarr dataset for each year
                s3=s3_out,                             # connection
                check=False                            # checking if the dataset exists before trying to load it
            ) for yr in yrs]                           # loops through each year
    
    ## Load data for specified years and variable of interest using the xarray library
    var = 'APCP_surface'
    ds_yrs = xr.open_mfdataset(fileset, engine='zarr')
    da_yrs_var = ds_yrs[var].rio.write_crs(4326,inplace=True).fillna(0)
    variable_long_name = da_yrs_var.attrs.get('long_name')
    # Temporal aggregation
    da_TimeAgg = da_yrs_var.resample(time='d').sum()
    
    # select basins, dates by training DF
    training_df_dir = f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/VIIRSGeoObsDFs/{thresh}_fSCA_Thresh"
    files = [filename for filename in os.listdir(training_df_dir)
             if filename.endswith(".parquet")
            ]
    precip_data_path = f"{HOME}/data/Precipitation/{WY}/{output_res}M_AORC_Precip"
    os.makedirs(precip_data_path, exist_ok=True)

    for file in tqdm(files, desc="Processing AORC precip files"):
        filepath = f'{training_df_dir}/{file}'
        #Get timestamp
        timestamp = file.split('_')[-1].split('.')[0]
        #Get region
        region = file.split('_')[-2]
        obs_end = f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:]}'

        out_path = f"{precip_data_path}/AORC_{region}_{timestamp}.parquet"
        if os.path.exists(out_path):
            continue

        print(f"Getting precipitation data for {obs_end} at {region}")

        training_df = pd.read_parquet(filepath)
        # get bounding box by min/max coordinates
        left, right = training_df['cen_lon'].min(), training_df['cen_lon'].max()
        bottom, top = training_df['cen_lat'].min(), training_df['cen_lat'].max()
        # add some padding to bbox
        left -= 0.1
        bottom -= 0.1
        right += 0.1
        top += 0.1
        da_WYagg = da_TimeAgg.loc[dict(time=slice(obs_start, obs_end))]
        da_bbox = da_WYagg.sel(latitude=slice(bottom, top), longitude=slice(left, right))

        obs_precip_transformed = da_bbox.rio.reproject(rasterio.crs.CRS.from_epsg('4326')).load()

        # extract metadata
        meta = training_df[['cell_id','cen_lat','cen_lon']]
        lats = xr.DataArray(meta['cen_lat'].values, dims='points')
        lons = xr.DataArray(meta['cen_lon'].values, dims='points')
        try:
            prcp_all = obs_precip_transformed.sel(x=lons, y=lats, method='nearest')
            raw_vals = prcp_all.values  # shape (time, npoints)
            season_precip_cm = np.round(raw_vals.sum(axis=0) / 10, 2)
        finally:
            obs_precip_transformed.close()
        precip_df = pd.DataFrame({
            'cell_id': meta['cell_id'].values,
            'cen_lat': meta['cen_lat'].values,
            'cen_lon': meta['cen_lon'].values,
            'precip': list(raw_vals.T),
            'season_precip_cm': season_precip_cm,
        })

        table = pa.Table.from_pandas(precip_df)
        pq.write_table(table, out_path, compression='BROTLI')
       
### next set of PRISM functions are adapted from Yen Yi Wu via CUAHSI - thank you! 
def _progress_hook(block_num, block_size, total_size, t):
    """
    Callback function to update tqdm progress bar during file download.
    """
    downloaded = block_num * block_size
    if total_size > 0:
        t.update(min(block_size, total_size - t.n))
    else:
        t.update(downloaded - t.n)

def prism_download(start,stop,path,var):
    base_url = "https://services.nacse.org/prism/data/get/us/4km/"
    while start <= stop:
        day = start.strftime("%Y%m%d")
        url = f"{base_url}/{var}/{day}?format=nc"
        output_file = os.path.join(path, day)

        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f'Downloading {day}') as t:
            urllib.request.urlretrieve(url, output_file, reporthook=lambda block_num, block_size, total_size: _progress_hook(block_num, block_size, total_size, t))

        start += timedelta(days=1)

def unzip_prism(start,stop,zipped_path,unzipped_path):
    while start <= stop:
        day = start.strftime("%Y%m%d")
        zip_file_path = os.path.join(zipped_path, day)

        # Check if the ZIP file exists
        if os.path.exists(zip_file_path):
            # Unzip the file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzipped_path)
                print(f"UNZIP file for {day} is completed.")
            os.remove(zip_file_path)
        else:
            print(f"ZIP file for {day} not found.")

        start += timedelta(days=1)

def get_prism(WY):
    prism_path = f'{HOME}/data/Precipitation/{WY}/prism_data'
    zipped_path = f'{prism_path}/zipped'
    unzipped_path = f'{prism_path}/unzipped'
    os.makedirs(zipped_path,exist_ok=True)
    os.makedirs(unzipped_path,exist_ok=True)

    var = "ppt"
    start = datetime.strptime(f"{WY-1}-10-01", "%Y-%m-%d")
    stop = datetime.strptime(f"{WY}-09-30", "%Y-%m-%d")

    if not os.path.exists(f'{prism_path}/{WY}.nc'):
        prism_download(start,stop,zipped_path,var)
        unzip_prism(start,stop,zipped_path,unzipped_path)

        date_idx = pd.Index(pd.date_range(start=start,end=stop),name='time')
        timeser = xr.open_mfdataset(f'{unzipped_path}/*.nc', 
                                        combine='nested',
                                        concat_dim=[date_idx,]
                                        )
        timeser.to_netcdf(path=f'{prism_path}/{WY}.nc')
    else:
        print(f'PRISM data already downloaded for {WY}')

def add_prism_df(WY,output_res,threshold):

    training_df_path = f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/AORCgridMETNLDASDaymet_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    df_path = f"{HOME}/data/TrainingDFs/{WY}/{output_res}M_Resolution/PRISM_AORCgridMETNLDASDaymet_Vegetation_Sturm_Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    os.makedirs(df_path,exist_ok=True)
    
    training_dfs = [filename for filename in os.listdir(training_df_path) if filename.endswith('.parquet')]

    obs_precip = xr.open_dataset(f'{HOME}/data/Precipitation/{WY}/prism_data/{WY}.nc')
    # reproject to WGS84
    obs_precip = obs_precip.rio.write_crs('EPSG:4269')
    obs_precip_transformed = obs_precip.rio.reproject(rasterio.crs.CRS.from_epsg('4326'))   

    WY_start = f'{WY-1}-10-01'

    for geofile in tqdm(training_dfs, desc="Processing PRISM files"):
        #get date information
        date = geofile.split('_')[-1].split('.parquet')[0]
        region = geofile.split('_')[-2]
        year = date[:4]
        mon = date[4:6]
        day = date[6:]
        strdate = f"{year}-{mon}-{day}"
        print(f"Connecting PRISM precipitation to ASO observations on {strdate} at {region}")

        # read training DF and add column for PRISM season precip
        GDF = pd.read_parquet(os.path.join(training_df_path, geofile))
        meta = GDF[['cell_id','cen_lat','cen_lon']]
        GDF.set_index('cell_id', inplace=True)

        # get bbox from training DF
        left, right = meta['cen_lon'].min(), meta['cen_lon'].max()
        bottom, top = meta['cen_lat'].min(), meta['cen_lat'].max()
        # add some padding to bbox
        left -= 0.1
        bottom -= 0.1
        right += 0.1
        top += 0.1
        obs_precip_date = obs_precip_transformed.loc[dict(time=slice(WY_start,strdate))]
        obs_precip_bbox = obs_precip_date.sel(y=slice(bottom, top), x=slice(left, right))

        lats = xr.DataArray(meta['cen_lat'].values, dims='points')
        lons = xr.DataArray(meta['cen_lon'].values, dims='points')
        pr_all = obs_precip_bbox['Band1'].sel(x=lons, y=lats, method='nearest')
        season_precip_arr = np.round(pr_all.values.sum(axis=0) / 10, 2)
        obs_precip_bbox.close()
        obs_precip_date.close()

        GDF['PRISM'] = season_precip_arr
        GDF.reset_index(inplace=True)

        #Convert DataFrame to Apache Arrow Table
        table = pa.Table.from_pandas(GDF)
        # Parquet with Brotli compression
        pq.write_table(table, f"{df_path}/PRISM_{geofile}", compression='BROTLI')
    obs_precip_transformed.close()
    