import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import concurrent.futures as cf
import os
import warnings
import pickle as pkl
import boto3
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

# Module-level globals populated once per worker process by _worker_init
_ss_df = None
_n_snotel = None

def _worker_init(snotel_path, snotel_dict_path):
    """Load shared lookup data once per worker process instead of once per file."""
    global _ss_df, _n_snotel
    _ss_df = pd.read_parquet(snotel_path)
    with open(snotel_dict_path, 'rb') as f:
        _n_snotel = pkl.load(f)

# creat Seasonality features
#begin with day of year starting from October 1st

def DOS(date):
    Oct_Dec_days = 92
    date =date.strftime('%j')

    return int(date)+Oct_Dec_days

def site_anomaly(df):
    cols = ['ns_1','ns_2','ns_3','ns_4','ns_5','ns_6']
    for col in cols:
        newcol = f"{col}_anomaly"
        weekcol = f"{col}_week_mean"
        df[newcol] = df[col]/df[weekcol]
    #late season values cause division by 0 error. so far, all obs are 0/0, setting to 1
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(1, inplace = True)
    return df

def add_Seasonality(region, output_res, threshold):
    #load dataframe
    DFpath = f'{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/VIIRSGeoObsDFs/{threshold}_fSCA_Thresh'
    Savepath = f'{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/Seasonality_VIIRSGeoObsDFs/{threshold}_fSCA_Thresh'
    if not os.path.exists(Savepath):
        os.makedirs(Savepath, exist_ok=True)

    files = [f for f in os.listdir(DFpath) if f.endswith('.parquet')]
    print(f"Adding Day of Season, seasonal nearest monitoring site averages, and nearest monitoring site anomaly to averages to all {region} dataframes...")

    snotel_path = f'{HOME}/data/SNOTEL_Data/seasonal_snotel.parquet'
    snotel_dict_path = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution/nearest_SNOTEL.pkl"

    with cf.ProcessPoolExecutor(
        max_workers=CPUS,
        initializer=_worker_init,
        initargs=(snotel_path, snotel_dict_path),
    ) as executor:
        futures = {
            executor.submit(single_file_seasonality, (file, DFpath, Savepath)): file
            for file in files
        }
        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Worker error ({futures[future]}): {e}")
    
    
def single_file_seasonality(arg):
    file, DFpath, Savepath = arg
    df = pd.read_parquet(os.path.join(DFpath, file))
    #add day of season info
    df['DOS'] = df.apply(lambda df: DOS(df['Date']), axis=1)

    #add the in situ metrics here
    df = add_nearest_snotel_ave(df)

    #add swe anomaly for nearest sites to current obs
    df = site_anomaly(df)

    coldrop = ['week_id','year','Oct1st_weekid','EOY_weekid']
    df.drop(columns = coldrop, inplace =  True)
    #save dataframe
    table = pa.Table.from_pandas(df)
    # Parquet with Brotli compression
    pq.write_table(table, f"{Savepath}/Season_{file}", compression='BROTLI')

def match_nearest_snotel(cell_id, WY_Week, n_snotel, ss_df):
    nearest_sites = n_snotel[str(cell_id)]
    ss_df = ss_df[nearest_sites]
    ss_df = pd.DataFrame(ss_df.loc[WY_Week]).T
    ss_df.rename(columns = {nearest_sites[0]:'ns_1_week_mean',
                            nearest_sites[1]:'ns_2_week_mean',
                            nearest_sites[2]:'ns_3_week_mean',
                            nearest_sites[3]:'ns_4_week_mean',
                            nearest_sites[4]:'ns_5_week_mean',
                            nearest_sites[5]:'ns_6_week_mean'
                            }, inplace = True)
    
    #return ss_df
    arg =  ss_df['ns_1_week_mean'].values[0], ss_df['ns_2_week_mean'].values[0], ss_df['ns_3_week_mean'].values[0], ss_df['ns_4_week_mean'].values[0], ss_df['ns_5_week_mean'].values[0], ss_df['ns_6_week_mean'].values[0]

    return arg

def add_nearest_snotel_ave(df):
    #need to add WY week to ML dataframe
    df['Date'] = pd.to_datetime(df['Date'])
    df['week_id'] = df['Date'].dt.strftime("%V").astype(int)
    df['year'] = df['Date'].dt.year

    #adjust week id to reflect water year, typically 13-14 weeks
    df['Oct1st_weekid'] = df.apply(lambda df: Oct_water_week(df['year']), axis=1)
    df['EOY_weekid'] = df.apply(lambda df: EOY_water_week(df['year']), axis=1)
    df['WY_week'] = df.apply(lambda df: WY_week(df, df['week_id']), axis=1)

    #Get historical averages using data pre-loaded by _worker_init
    hist_site_ave = df.apply(
        lambda row: match_nearest_snotel(row['cell_id'], row['WY_week'], _n_snotel, _ss_df), axis=1
    )
    hist_site_ave = pd.DataFrame.from_records(hist_site_ave, columns=['ns_1_week_mean', 'ns_2_week_mean','ns_3_week_mean','ns_4_week_mean','ns_5_week_mean','ns_6_week_mean'])
    #merge with training DF
    df = pd.concat([df, hist_site_ave], axis=1)

    return df

def Oct_water_week(year):
    oct_date = pd.to_datetime(f"{year}-10-01")
    oct = oct_date.strftime('%V')

    return int(oct)


def EOY_water_week(year):
    eoy_date = pd.to_datetime(f"{year}-12-28")
    eoy = eoy_date.strftime('%V')

    return int(eoy)

def WY_week(row, week_id):
    if week_id < 39:
        return row['week_id'] - row['Oct1st_weekid'] + row['EOY_weekid']
    else:
        return row['week_id'] - row['Oct1st_weekid']


def seasonal_snotel():

    #DFpath = f'{HOME}/SWEMLv2.0/data/SNOTEL_Data'
    DFpath = f'{HOME}/data/SNOTEL_Data'
    snotel = pd.read_parquet(f"{DFpath}/ground_measures_dp.parquet")
    #snotel = snotel.T
    snotel.reset_index(inplace = True)
    snotel.rename(columns = {'dates':'date'}, inplace = True)
    #snotel.rename(columns = {'index':'date'}, inplace = True)
    snotel['date'] = pd.to_datetime(snotel['date'])
    snotel['week_id'] = snotel['date'].dt.strftime("%V").astype(int)
    snotel['year'] = snotel['date'].dt.year


    #adjust week id to reflect water year, typically 13-14 weeks
    snotel['Oct1st_weekid'] = snotel.apply(lambda snotel: Oct_water_week(snotel['year']), axis=1)
    snotel['EOY_weekid'] = snotel.apply(lambda snotel: EOY_water_week(snotel['year']), axis=1)
    snotel['WY_week'] = snotel.apply(lambda snotel: WY_week(snotel, snotel['week_id']), axis = 1)

    # remove
    coldrop = ['date','week_id', 'year', 'Oct1st_weekid', 'EOY_weekid']
    snotel.drop(columns = coldrop, inplace =  True)


    #get mean weighted station swe values
    ss_df = snotel.groupby('WY_week').mean()

    #make a normalized dataframe
    normalized_ss_df = ss_df/ss_df.max()

    table = pa.Table.from_pandas(ss_df)
    # Parquet with Brotli compression
    pq.write_table(table, f"{DFpath}/seasonal_snotel.parquet", compression='BROTLI')

    table = pa.Table.from_pandas(normalized_ss_df)
    # Parquet with Brotli compression
    pq.write_table(table, f"{DFpath}/normalized_seasonal_snotel.parquet", compression='BROTLI')