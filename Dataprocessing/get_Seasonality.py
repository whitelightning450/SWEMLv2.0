import datetime
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm, tqdm_notebook
import concurrent.futures as cf
import os
import warnings
import pickle as pkl
warnings.filterwarnings("ignore")

HOME = os.path.expanduser('~')

# creat Seasonality features
#begin with day of year starting from October 1st

def DOS(date):
    Oct_Dec_days = 92
    date =date.strftime('%j')

    return int(date)+Oct_Dec_days

def seasonal_site_rel(df):
    cols = ['ns_1','ns_2','ns_3','ns_4','ns_5','ns_6']
    for col in cols:
        newcol = f"Seasonal_{col}_rel"
        weekcol = f"{col}_week_mean"
        df[newcol] = df[col]/df[weekcol]
    #late season values cause division by 0 error. so far, all obs are 0/0, setting to 1
    df.fillna(1, inplace = True)
    return df

def add_Seasonality(region, output_res, threshold):
    #load dataframe
    DFpath = f'{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution/PrecipVIIRSGeoObsDFs/{threshold}_fSCA_Thresh'
    Savepath = f'{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution/Seasonality_PrecipVIIRSGeoObsDFs/{threshold}_fSCA_Thresh'
    if not os.path.exists(Savepath):
        os.makedirs(Savepath, exist_ok=True)

    files = [filename for filename in os.listdir(DFpath)]
    print(f"Adding Day of Season, seasonal nearest monitoring site averages, and seasonal nearest monitoring site relationship to averages to all {region} dataframes...")
    
    with cf.ProcessPoolExecutor(max_workers=None) as executor: 
        # Start the load operations and mark each future with its process function
        [executor.submit(single_file_seasonality, (file, DFpath, region, output_res, Savepath)) for file in tqdm_notebook(files)]
    # for file in tqdm_notebook(files):
    #     df = pd.read_parquet(os.path.join(DFpath, file))
    #     #add day of season info
    #     df['DOS'] = df.apply(lambda df: DOS(df['Date']), axis=1)

    #     #add the in situ metrics here
    #     df = add_nearest_snotel_ave(df, region, output_res)

    #     #add seasonal relationship for nearest sites to current obs
    #     df = seasonal_site_rel(df)

    #     coldrop = ['week_id','year','Oct1st_weekid','EOY_weekid']
    #     df.drop(columns = coldrop, inplace =  True)
    #     #save dataframe
    #     table = pa.Table.from_pandas(df)
    #     # Parquet with Brotli compression
    #     pq.write_table(table, f"{Savepath}/Season_{file}", compression='BROTLI')
    
    
def single_file_seasonality(arg):
    file, DFpath, region, output_res, Savepath = arg
    df = pd.read_parquet(os.path.join(DFpath, file))
    #add day of season info
    df['DOS'] = df.apply(lambda df: DOS(df['Date']), axis=1)

    #add the in situ metrics here
    df = add_nearest_snotel_ave(df, region, output_res)

    #add seasonal relationship for nearest sites to current obs
    df = seasonal_site_rel(df)

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

def add_nearest_snotel_ave(df, region, output_res):

    # set file paths
    DFpath = f'{HOME}/SWEMLv2.0/data/SNOTEL_Data'
    nearest_snotel_dict_path = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution"

    #need to add WY week to ML dataframe
    df.reset_index(inplace = True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['week_id'] = df['Date'].dt.strftime("%V").astype(int)
    df['year'] = df['Date'].dt.year

    #adjust week id to reflect water year, typically 13-14 weeks
    df['Oct1st_weekid'] = df.apply(lambda df: Oct_water_week(df['year']), axis=1)
    df['EOY_weekid'] = df.apply(lambda df: EOY_water_week(df['year']), axis=1)
    df['WY_week'] = df.apply(lambda df: WY_week(df, df['week_id']), axis = 1)

    #need to connect nearest sites with average snotel obs dataframes
    #load seasonal snotel data
    ss_df = pd.read_parquet(f"{DFpath}/seasonal_snotel.parquet")
    #load regional nearest snotel
    file =  open(f"{nearest_snotel_dict_path}/nearest_SNOTEL.pkl", 'rb')
    n_snotel =  pkl.load(file)
    #Get historical averages
    tqdm.pandas()
    hist_site_ave = df.progress_apply(lambda df: match_nearest_snotel(df['cell_id'], df['WY_week'], n_snotel, ss_df), axis=1)
    hist_site_ave = pd.DataFrame.from_records(hist_site_ave, columns=['ns_1_week_mean', 'ns_2_week_mean','ns_3_week_mean','ns_4_week_mean','ns_5_week_mean','ns_6_week_mean'])
    #merge with training DF
    df = pd.concat([df, hist_site_ave], axis =1)

    return df

def Oct_water_week(year):
    oct_date = pd.to_datetime(f"{year}-10-01")
    oct = oct_date.strftime('%V')

    return int(oct)


def EOY_water_week(year):
    eoy_date = pd.to_datetime(f"{year}-12-25")
    eoy = eoy_date.strftime('%V')

    return int(eoy)

def WY_week(df, week_id):
    if week_id <39:
        df['WY_week'] = df['week_id'] - df['Oct1st_weekid'] + df['EOY_weekid']
    else:
        df['WY_week'] = df['week_id'] - df['Oct1st_weekid']

    return df['WY_week']


def seasonal_snotel():

    DFpath = f'{HOME}/SWEMLv2.0/data/SNOTEL_Data'
    snotel = pd.read_parquet(f"{DFpath}/ground_measures.parquet")
    snotel = snotel.T
    snotel.reset_index(inplace = True)
    snotel.rename(columns = {'index':'date'}, inplace = True)
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
    weekids = list(snotel['WY_week'].unique())
    weekids.sort()

    ss_df = pd.DataFrame()
    for week in weekids:
        sdf = pd.DataFrame(snotel[snotel['WY_week']==week].mean(axis=0)).T
        ss_df = pd.concat([ss_df, sdf])

    ss_df.reset_index(drop = True, inplace = True)
    ss_df['WY_week'] = ss_df['WY_week'].astype(int)
    ss_df.set_index('WY_week', inplace = True)

    #make a normalized dataframe
    normalized_ss_df = ss_df/ss_df.max()

    table = pa.Table.from_pandas(ss_df)
    # Parquet with Brotli compression
    pq.write_table(table, f"{DFpath}/seasonal_snotel.parquet", compression='BROTLI')

    table = pa.Table.from_pandas(normalized_ss_df)
    # Parquet with Brotli compression
    pq.write_table(table, f"{DFpath}/normalized_seasonal_snotel.parquet", compression='BROTLI')