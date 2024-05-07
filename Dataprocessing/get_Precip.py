import numpy as np
import pandas as pd
import ee #pip install earthengine-api
import EE_funcs
import os
from tqdm import tqdm
import concurrent.futures as cf
import pyarrow as pa
import pyarrow.parquet as pq
import pickle as pkl
ee.Authenticate()
ee.Initialize()
import warnings
warnings.filterwarnings("ignore")

HOME = os.path.expanduser('~')

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

        #Precipitation
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
            startdate = f"{year-1}-10-01"
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

        #PrecipDict[cell_id] = pd.concat(WY_Precip.values(), ignore_index=True)
        df = pd.concat(WY_Precip.values(), ignore_index=True)

        table = pa.Table.from_pandas(df)
        # Parquet with Brotli compression
        pq.write_table(table, f"{Precippath}/NLDAS_PPT_{cell_id}.parquet", compression='BROTLI')
        #print(f"{cell_id} done...")
    
    

def get_precip_threaded(region, output_res, WYs):
    #  #ASO file path
    aso_swe_files_folder_path = f"{HOME}/SWEMLv2.0/data/ASO/{region}/{output_res}M_SWE_parquet/"
    #make directory for data 
    Precippath = f"{HOME}/SWEMLv2.0/data/Precipitation/{region}/{output_res}M_NLDAS_Precip/sites"
    if not os.path.exists(Precippath):
        os.makedirs(Precippath, exist_ok=True)
    
    #load metadata and get site info
    path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution/{region}_metadata.parquet"
    meta = pd.read_parquet(path)
    #reset index because we need cell_id as a column
    meta.reset_index(inplace=True)
    
    #set water year start/end dates based on ASO flights for the end date
    aso_swe_files = [filename for filename in os.listdir(aso_swe_files_folder_path)]
    aso_swe_files.sort()
    startyear = int(aso_swe_files[0][-16:-12])-1
    startdate = f"{startyear}-10-01"

    dates = []
    for file in aso_swe_files:
        month = file[-12:-10]
        day = file[-10:-8]
        year = file[-16:-12]
        enddate = f"{str(year)}-{month}-{day}"
        dates.append(enddate)
    dates.sort()
    enddate = dates[-1]

    #NLDAS precipitation
    precip = ee.ImageCollection('NASA/NLDAS/FORA0125_H002').select('total_precipitation').filterDate(startdate, enddate)

    nsites = len(meta) #change this to all sites when ready


    print(f"Getting daily precipitation data for {nsites} sites")
    #create dictionary for year
    #PrecipDict ={}
    with cf.ThreadPoolExecutor(max_workers=None) as executor: #seems that they done like when we shoot tons of threads to get data...
        {executor.submit(GetSeasonalAccumulatedPrecipSingleSite, (Precippath, precip, output_res, meta.iloc[i]['cen_lat'], meta.iloc[i]['cen_lon'], meta.iloc[i]['cell_id'], dates, WYs)):
                i for i in tqdm(range(nsites))}
        
    # for i in tqdm(range(nsites)): #trying for loop bc multithreader not working....
    #     args = Precippath, precip, output_res, meta.iloc[i]['cen_lat'], meta.iloc[i]['cen_lon'], meta.iloc[i]['cell_id'], dates, WYs
    #     GetSeasonalAccumulatedPrecipSingleSite(args)
    
    
    print(f"Job complete for getting precipiation datdata for WY{year}, processing dataframes for file storage")
    # #combine all sites/obs
    # WY_precip = pd.concat(PrecipDict.values(), ignore_index=True)

    # #separate by date
    # datesds = WY_precip.datetime.unique()

    # print(f"Processing cells values into {len(datesds)} datetime files for reduced storage")
    # with cf.ProcessPoolExecutor(max_workers=None) as executor: 
    #     # Start the load operations and mark each future with its process function
    #     [executor.submit(ProcessDates, (date, WY_precip, Precippath)) for date in tqdm(datesds)]

    # print(f"Job complete, all precipitation data can be found in {Precippath}")



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
