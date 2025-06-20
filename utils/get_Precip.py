import numpy as np
import pandas as pd
import ee #pip install earthengine-api
import utils.EE_funcs as EE_funcs
import os
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import concurrent.futures as cf
import pyarrow as pa
import pyarrow.parquet as pq
import pickle as pkl
import boto3
ee.Authenticate()
ee.Initialize()
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
    

def GetSeasonalAccumulatedPrecipSingleSite(args):
    Precippath, precip, output_res, lat, lon, cell_id, dates, WYs = args

    # See if file exists
    if os.path.exists(f"{Precippath}/NLDAS_PPT_{cell_id}.parquet") == False:
        #unit conversions and temporal frequency
        temporal_resample = 'D'
        kgm2_to_cm = 0.1

        # Gett lat/long from meta file
        poi = ee.Geometry.Point(lon, lat)


        #Get precipitation
        precip_poi = precip.getRegion(poi, output_res).getInfo()
        site_precip = EE_funcs.ee_array_to_df(precip_poi,['total_precipitation'])

        #Precipitation, do a different resample to get hourly precip and connect with temp to determine precip phase
        site_precip.set_index('datetime', inplace = True)
        site_precip = site_precip.resample(temporal_resample).sum()
        site_precip.reset_index(inplace = True)

        #make columns for cms
        site_precip['total_precipitation'] = site_precip['total_precipitation']*kgm2_to_cm
        site_precip.rename(columns={'total_precipitation':'daily_precipitation_cm'}, inplace = True)
        site_precip.pop('time')
        site_precip.set_index('datetime',inplace=True)


        #do precip cumulative per water year
        #get unique water years
        WYdict = {}
        WY_Precip = {}
        for year in WYs:
            WYdict[f"WY{year}"] = [d for d in dates if str(year) in d]
            WYdict[f"WY{year}"].sort()
            startdate = f"{year-1}-09-30"
            enddate =WYdict[f"WY{year}"][-1]

            #select dates within water year
            WY = site_precip.loc[startdate:enddate]
            WY.reset_index(inplace=True)

            #get seasonal accumulated precipitation for site
            WY['season_precip_cm'] = WY['daily_precipitation_cm'].cumsum()

            #be aware of file storage, save only dates lining up with ASO obs
            mask = WY['datetime'].isin(WYdict[f"WY{year}"])

            WY['cell_id'] = cell_id

            #select key columns
            cols = ['cell_id', 'datetime', 'season_precip_cm']
            WY = WY[cols]
            #save each year as a dic
            WY_Precip[f"WY{year}"] = WY[mask].reset_index(drop = True)

        df = pd.concat(WY_Precip.values(), ignore_index=True)
        table = pa.Table.from_pandas(df)
        # Parquet with Brotli compression
        pq.write_table(table, f"{Precippath}/NLDAS_PPT_{cell_id}.parquet", compression='BROTLI')
    
    

def get_precip_threaded(region, output_res, WYs):
    #  #ASO file path
    aso_swe_files_folder_path = f"{HOME}/data/ASO/{region}/{output_res}M_SWE_parquet/"

    #make directory for data 
    Precippath = f"{HOME}/data/Precipitation/{region}/{output_res}M_NLDAS_Precip/sites"

    if not os.path.exists(Precippath):
        os.makedirs(Precippath, exist_ok=True)
    
    #load metadata and get site info
    path = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/{region}_metadata.parquet"

    meta = pd.read_parquet(path)
    #reset index because we need cell_id as a column
    meta.reset_index(inplace=True)
    
    #set water year start/end dates based on ASO flights for the end date
    aso_swe_files = [filename for filename in os.listdir(aso_swe_files_folder_path)]
    aso_swe_files.sort()
    print(aso_swe_files[0])
    startyear = int(aso_swe_files[0][-16:-12])-1
    startdate = f"{startyear}-09-30"

    dates = []
    for file in aso_swe_files:
        month = file[-12:-10]
        day = file[-10:-8]
        year = file[-16:-12]
        enddate = f"{str(year)}-{month}-{day}"
        dates.append(enddate)
    dates.sort()
    enddate = dates[-1]
    enddate = pd.to_datetime(enddate)+pd.Timedelta(days=1)
    enddate =enddate.strftime('%Y-%m-%d')

    print(dates)

    #get only water years with ASO observations
    ASO_WYs = []
    WYdict = {}
    for year in WYs:
        try:
            WYdict[f"WY{year}"] = [d for d in dates if str(year) in d]
            WYdict[f"WY{year}"].sort()
            startdate_exception = f"{year-1}-10-01"
            enddate_exception =WYdict[f"WY{year}"][-1]
            ASO_WYs.append(year)
        except:
            print(f"No ASO observations for WY{year}")

    print(ASO_WYs, startdate, enddate)

    #NLDAS precipitation
    precip = ee.ImageCollection('NASA/NLDAS/FORA0125_H002').select('total_precipitation').filterDate(startdate, enddate)

    nsites = len(meta) #change this to all sites when ready


    print(f"Getting daily precipitation data for {nsites} sites")
    #create dictionary for year
    with cf.ThreadPoolExecutor(max_workers=6) as executor: #seems that they dont like when we shoot tons of threads to get data...
        {executor.submit(GetSeasonalAccumulatedPrecipSingleSite, (Precippath, precip, output_res, meta.iloc[i]['cen_lat'], meta.iloc[i]['cen_lon'], meta.iloc[i]['cell_id'], dates, ASO_WYs)):
                i for i in tqdm(range(nsites))}
        
    # for i in tqdm(range(nsites)): #trying for loop bc multithreader not working....
    #     args = Precippath, precip, output_res, meta.iloc[i]['cen_lat'], meta.iloc[i]['cen_lon'], meta.iloc[i]['cell_id'], dates, ASO_WYs
    #     GetSeasonalAccumulatedPrecipSingleSite(args)
    
    
    print(f"Job complete for getting precipiation datdata for WY{year}, processing dataframes for file storage")


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



def Make_Precip_DF(region, output_res, threshold):

    print(f"Adding precipitation features to ML dataframe for the {region} region.")
    Precippath = f"{HOME}/data/Precipitation/{region}/{output_res}M_NLDAS_Precip/sites/"
    DFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/VIIRSGeoObsDFs/{threshold}_fSCA_Thresh"

    #make precip df path
    PrecipDFpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/PrecipVIIRSGeoObsDFs/{threshold}_fSCA_Thresh"
    if not os.path.exists(PrecipDFpath):
        os.makedirs(PrecipDFpath, exist_ok=True)

    #Get list of dataframes
    GeoObsDF_files = [filename for filename in os.listdir(DFpath)]
    pptfiles = [filename for filename in os.listdir(Precippath)]

    with cf.ProcessPoolExecutor(max_workers=CPUS) as executor: 
        # Start the load operations and mark each future with its process function
        [executor.submit(single_date_add_precip, (DFpath, Precippath, geofile, PrecipDFpath, pptfiles, region)) for geofile in GeoObsDF_files]
    # for geofile in GeoObsDF_files:
    #     single_date_add_precip((DFpath, Precippath, geofile, PrecipDFpath, pptfiles, region))
#     except:
#         print(f"No ASO observations/dataframe to add NLDAS precipitation to in {region}, skipping")
       



#multiprocess this first step
def single_date_add_precip(args):
    DFpath, Precippath, geofile, PrecipDFpath, pptfiles, region = args
    #get date information
    date = geofile.split('VIIRS_GeoObsDF_')[-1].split('.parquet')[0]
    year = date[:4]
    mon = date[4:6]
    day = date[6:]
    strdate = f"{year}-{mon}-{day}"
    print(f"Connecting precipitation to ASO observations for {region} on {strdate}")

    GDF = pd.read_parquet(os.path.join(DFpath, geofile))
    GDF.set_index('cell_id', inplace = True)
    GDF['season_precip_cm'] = 0.0
    #get unique cells
    sites = list(GDF.index)
    for site in tqdm_notebook(sites):
        try:
            ppt = pd.read_parquet(f"{Precippath}/NLDAS_PPT_{site}.parquet")
            ppt.rename(columns={'datetime':'Date'}, inplace = True)
            GDF.loc[site,'season_precip_cm'] = round(ppt['season_precip_cm'][ppt['Date']== strdate].values[0],1)
  
        except:
           print(f"{site} is bad, delete file from folder and rerun the get precipitation script")

    #Convert DataFrame to Apache Arrow Table
    table = pa.Table.from_pandas(GDF)
    # Parquet with Brotli compression
    pq.write_table(table, f"{PrecipDFpath}/Precip_{geofile}", compression='BROTLI')
