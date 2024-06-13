import pandas as pd
import numpy as np
import ulmo
import warnings
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import shapely.geometry
import concurrent.futures as cf
import threading
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import pyarrow as pa
import pyarrow.parquet as pq

from datetime import timedelta

warnings.filterwarnings("ignore")

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
#S3 = boto3.resource('S3', config=Config(signature_version=UNSIGNED))
BUCKET = S3.Bucket(BUCKET_NAME)


def get_SNOTEL_Threaded(args):
    SWE_df,sitecode, start_date, end_date = args
    
    # This is the latest CUAHSI API endpoint
    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'

    # Daily SWE
    variablecode = 'SNOTEL:WTEQ_D'

    #values_df = None
    # allows up to 3 attempts for getting site info, sometimes takes a few
    attempts = 0
    while attempts < 5:
        if attempts > 1:
            print(f"Attempt {attempts} for site {sitecode}")
        try:
            # Request data from the server
            site_values = ulmo.cuahsi.wof.get_values(wsdlurl, sitecode, variablecode, start=start_date, end=end_date)
            # end_date = end_date.strftime('%Y-%m-%d')
            # Convert to a Pandas DataFrame
            SNOTEL_SWE = pd.DataFrame.from_dict(site_values['values'])
            # Parse the datetime values to Pandas Timestamp objects
            SNOTEL_SWE['datetime'] = pd.to_datetime(SNOTEL_SWE['datetime'], utc=True).values
            SNOTEL_SWE.set_index('datetime', inplace = True)
            # Convert values to float and replace -9999 nodata values with NaN
            SNOTEL_SWE['value'] = pd.to_numeric(SNOTEL_SWE['value']).replace(-9999, np.nan)
            # Remove any records flagged with lower quality
            SNOTEL_SWE = SNOTEL_SWE[SNOTEL_SWE['quality_control_level_code'] == '1']
            SWE_df[end_date].loc[sitecode] = SNOTEL_SWE['value'].values[0]
            if attempts > 1:
                print(f"Attempt {attempts} successful for site {sitecode}")
            break

        except Exception as e:
            # end_date = end_date.strftime('%Y-%m-%d')
            attempts += 1
            SWE_df[end_date].loc[sitecode] = -9999
            print(f"Snotel data fail, {sitecode}")


def get_CDEC_Threaded(args): #https://ulmo.readthedocs.io/en/latest/api.html ulmo now has CDEC sites

    SWE_df, station_id, sensor_id, resolution, start_date, end_date = args
    #print(f"Station id: {station_id}, sensor id: {sensor_id}, resolution: {resolution}, start date: {start_date}, end date: {end_date}")
    url = 'https://cdec.water.ca.gov/dynamicapp/selectQuery?Stations=%s' % (station_id) + '&SensorNums=%s' % (
                sensor_id) + '&dur_code=%s' % (resolution) + '&Start=%s' % (start_date) + '&End=%s' % (end_date)
    # allows up to 3 attempts for getting site info, sometimes takes a few
    attempts = 0
    while attempts < 5:
        if attempts > 1:
            print(f"Attempt {attempts} for site {station_id}")

        try:
            CDEC_SWE = pd.read_html(url)[0]
            if CDEC_SWE.columns[0] != 'DATE':
                CDEC_SWE = pd.read_html(url)[1]
            CDEC_station_id = 'CDEC:' + station_id
            CDEC_SWE['station_id'] = CDEC_station_id
            CDEC_SWE = CDEC_SWE.set_index('station_id')
            CDEC_SWE = pd.DataFrame(CDEC_SWE.iloc[-1]).T
            cols = CDEC_SWE.columns
            CDEC_SWE.rename(columns={cols[1]: end_date}, inplace =  True)
            if CDEC_SWE[end_date].values[0] =='--':
                CDEC_SWE[end_date] = -9999
            SWE_df[end_date].loc[CDEC_station_id] = CDEC_SWE[end_date].values[0]

            if attempts > 1:
                print(f"Attempt {attempts} successful for site {station_id}")
            break
            

        except:
            attempts += 1
            url = 'https://cdec.water.ca.gov/dynamicapp/selectQuery?Stations=%s' % (station_id) + '&SensorNums=%s' % (
                sensor_id) + '&dur_code=%s' % (resolution) + '&Start=%s' % (start_date) + '&End=%s' % (end_date)
            CDEC_SWE = pd.DataFrame(-9999, columns=['station_id', end_date], index=[1])
            CDEC_station_id = 'CDEC:' + station_id
            CDEC_SWE['station_id'] = CDEC_station_id
            CDEC_SWE = CDEC_SWE.set_index('station_id')
            SWE_df[end_date].loc[CDEC_station_id] = CDEC_SWE[end_date]
            print(f"CDEC data fail, {url}")

def Get_Monitoring_Data_Threaded(dates):
    
    snotel_path = f"{HOME}/SWEMLv2.0/data/SNOTEL_Data"
    GM_template = pd.read_parquet(f"{snotel_path}/ground_measures.parquet")

    # Get all records, can filter later,
    CDECsites = list(GM_template.index)
    CDECsites = list(filter(lambda x: 'CDEC' in x, CDECsites))
    CDECsites_complete = CDECsites.copy()
    CDECsites = [x[-3:] for x in CDECsites]

    Snotelsites = list(GM_template.index)
    Snotelsites = list(filter(lambda x: 'SNOTEL' in x, Snotelsites))


    # Make SWE observation dataframe
    station_ids = CDECsites_complete + Snotelsites
    SWE_NA_fill = [-9999] * len(station_ids)
    SWE_df = pd.DataFrame(list(zip(station_ids)),
                                columns=['station_id'])
    SWE_df = SWE_df.set_index('station_id')


    for date in tqdm(dates):
        print(f"Getting SNOTEL and CDEC observations for {date}")
        date = pd.to_datetime(date)
        start_date = date - timedelta(days=1)
        start_date = start_date.strftime('%Y-%m-%d')
        date = date.strftime('%Y-%m-%d')
        SWE_df[date] = SWE_NA_fill

        resolution = 'D'
        sensor_id = '3'
    
        print(f"Getting California Data Exchange Center SWE data from {len(CDECsites)} sites...") 
        with cf.ThreadPoolExecutor(max_workers=None) as executor:
            {executor.submit(get_CDEC_Threaded, (SWE_df, site, sensor_id, resolution, start_date, date)): site for site in tqdm(CDECsites)}
            

        print(f"Getting NRCS SNOTEL SWE data from {len(Snotelsites)} sites...") 
        with cf.ThreadPoolExecutor(max_workers=6) as executor:
            {executor.submit(get_SNOTEL_Threaded, (SWE_df, site, start_date, date)): site for site in tqdm(Snotelsites)}

        SWE_df = SWE_df[~SWE_df.index.duplicated(keep='first')]
        SWE_df[date] = SWE_df[date].astype('float', errors='ignore')
        
        #fix na sites with regional average
        CDEC = SWE_df.loc['CDEC:ADM':'CDEC:WWC']
        meanSWE = CDEC[CDEC[date]>=0].mean().values[0]
        SWE_df[SWE_df[date]<-10]=meanSWE

        # remove -- from CDEC predictions and make df a float
        SWE_df[date] = SWE_df[date].astype(str)
        SWE_df[date] = SWE_df[date].replace(['--'], -9999)
        SWE_df[date] = pd.to_numeric(SWE_df[date], errors='coerce')
        SWE_df[date] = SWE_df[date].fillna(-9999)
        
        #Get state average SNOTEL to fill in nans
        states = ['WA', 'OR', 'CA', 'ID', 'NV', 'AZ', 'MT', 'WY', 'UT', 'CO','NM', 'SD']
        for state in states:
            s = SWE_df.copy()
            s.reset_index(inplace = True)
            stateSnotel = s[s['station_id'].str.contains(state)].copy()
            mean = stateSnotel[stateSnotel[date]>=0][date].mean()
            stateSnotel[date][stateSnotel[date] < 0] = mean
            stateSnotel.set_index('station_id', inplace = True)
            #update SWE_df
            SWE_df.update(stateSnotel)


        #change all slightly negative values to 0
        SWE_df[date][(SWE_df[date]<0) & (SWE_df[date]>-10)] = 0

    #add date to GM_template
    GM_template = pd.concat([GM_template, SWE_df], axis=1, join="inner")
    GM_template = GM_template.reindex(sorted(GM_template.columns), axis=1)
    #Convert DataFrame to Apache Arrow Table
    print('Updating local meta and saving.')
    table = pa.Table.from_pandas(GM_template)
    # Parquet with Brotli compression
    pq.write_table(table,f"{snotel_path}/ground_measures2.parquet", compression='BROTLI')

    return GM_template





def Get_Monitoring_Data_Threaded_dp(years, start_m_d, end_m_d, WY=True):
    
    snotel_path = f"{HOME}/SWEMLv2.0/data/SNOTEL_Data"
    GM_template = pd.read_parquet(f"{snotel_path}/ground_measures.parquet")

    # Get all records, can filter later,
    CDECsites = list(GM_template.index)
    CDECsites = list(filter(lambda x: 'CDEC' in x, CDECsites))
    CDECsites_complete = CDECsites.copy()
    CDECsites = [x[-3:] for x in CDECsites]

    Snotelsites = list(GM_template.index)
    Snotelsites = list(filter(lambda x: 'SNOTEL' in x, Snotelsites))


    # Make SWE observation dataframe
    station_ids = CDECsites_complete + Snotelsites
    #df = pd.DataFrame(columns=station_ids)

    for year in years:
        if WY == True:
            dates = list(pd.date_range(start=f"{year-1}-{start_m_d}",end=f"{year}-{end_m_d}"))

        else:
            dates = list(pd.date_range(start=f"{year}-{start_m_d}",end=f"{year}-{end_m_d}"))
        
        dates = [d.strftime("%Y-%m-%d") for d in dates]
        SWE_df = pd.DataFrame(columns= station_ids)
        SWE_df['dates'] = dates
        SWE_df.set_index('dates', inplace = True)
        SWE_df.fillna(-9999, inplace = True)

        print(f"Getting SNOTEL and CDEC observations for {year}")
        start_date = dates[0]
        end_date = dates[-1]
    

        resolution = 'D'
        sensor_id = '3'
        bad_sites = []
    
        print(f"Getting California Data Exchange Center SWE data from {len(CDECsites)} sites...") 
        with cf.ThreadPoolExecutor(max_workers=None) as executor:
            {executor.submit(get_CDEC_Threaded_dp, (SWE_df, site, sensor_id, resolution, start_date, end_date, bad_sites)): site for site in tqdm_notebook(CDECsites)}

        # for site in tqdm_notebook(CDECsites):
        #     args = SWE_df, site, sensor_id, resolution, start_date, end_date
        #     get_CDEC_Threaded_dp(args)
            
            
        print(f"Getting NRCS SNOTEL SWE data from {len(Snotelsites)} sites...") 
        with cf.ThreadPoolExecutor(max_workers=6) as executor:
            {executor.submit(get_SNOTEL_Threaded_dp, (SWE_df, site, start_date, end_date, bad_sites)): site for site in tqdm_notebook(Snotelsites)}

        for col in CDECsites_complete:
        # remove -- from CDEC predictions and make df a float
            SWE_df[col] = SWE_df[col].astype(str)
            SWE_df[col] = SWE_df[col].replace(['--'], '-9999')
            SWE_df[col] = pd.to_numeric(SWE_df[col], errors='coerce')
            SWE_df[col] = SWE_df[col].fillna(-9999)

    
        #convert to cm
        SWE_df = round(SWE_df*2.54,1)
        cols = SWE_df.columns
        #make all values close to 0, 0
        for col in cols:
            SWE_df[col][(SWE_df[col] <0.3) & (SWE_df[col]> -20)] = 0
            SWE_df[col][SWE_df[col] < -10] = -9999

        table = pa.Table.from_pandas(SWE_df)
        # Parquet with Brotli compression
        pq.write_table(table,f"{snotel_path}/{year}_ground_measures_dp.parquet", compression='BROTLI')


def get_SNOTEL_Threaded_dp(args):
    SWE_df,sitecode, start_date, end_date, bad_sites = args
    
    # This is the latest CUAHSI API endpoint
    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'

    # Daily SWE
    variablecode = 'SNOTEL:WTEQ_D'

    # allows up to 3 attempts for getting site info, sometimes takes a few
    attempts = 0
    while attempts < 5:
        if attempts > 1:
            print(f"Attempt {attempts} for site {sitecode}")
        try:
            # Request data from the server
            site_values = ulmo.cuahsi.wof.get_values(wsdlurl, sitecode, variablecode, start=start_date, end=end_date)
            # Convert to a Pandas DataFrame
            SNOTEL_SWE = pd.DataFrame.from_dict(site_values['values'])
            # Parse the datetime values to Pandas Timestamp objects
            SNOTEL_SWE['datetime'] = pd.to_datetime(SNOTEL_SWE['datetime'], utc=True).values
            SNOTEL_SWE.set_index('datetime', inplace = True)
            # Convert values to float and replace -9999 nodata values with NaN
            SNOTEL_SWE['value'] = pd.to_numeric(SNOTEL_SWE['value']).replace(-9999, np.nan)
            # Remove any records flagged with lower quality
            SNOTEL_SWE = SNOTEL_SWE[SNOTEL_SWE['quality_control_level_code'] == '1']
            SWE_df[sitecode] = SNOTEL_SWE['value']
            if attempts > 1:
                print(f"Attempt {attempts} successful for site {sitecode}")
            break

        except Exception as e:
            attempts += 1
            SWE_df[sitecode] = -9999
            print(f"Snotel data fail, {sitecode}")
            bad_sites.append(sitecode)


def get_CDEC_Threaded_dp(args): #https://ulmo.readthedocs.io/en/latest/api.html ulmo now has CDEC sites
    SWE_df, station_id, sensor_id, resolution, start_date, end_date, bad_sites = args
    #print(f"Station id: {station_id}, sensor id: {sensor_id}, resolution: {resolution}, start date: {start_date}, end date: {end_date}")
    url = 'https://cdec.water.ca.gov/dynamicapp/selectQuery?Stations=%s' % (station_id) + '&SensorNums=%s' % (
                sensor_id) + '&dur_code=%s' % (resolution) + '&Start=%s' % (start_date) + '&End=%s' % (end_date)
    # allows up to 3 attempts for getting site info, sometimes takes a few
    attempts = 0
    while attempts < 5:
        if attempts > 1:
            print(f"Attempt {attempts} for site {station_id}")

        try:
            CDEC_SWE = pd.read_html(url)[0]
            if CDEC_SWE.columns[0] != 'DATE':
                CDEC_SWE = pd.read_html(url)[1]
            CDEC_station_id = 'CDEC:' + station_id
            cols = CDEC_SWE.columns
        
            SWE_df[CDEC_station_id] = list(CDEC_SWE[cols[1]].values)

            if attempts > 1:
                print(f"Attempt {attempts} successful for site {station_id}")
            break
            

        except:
            attempts += 1
            CDEC_station_id = 'CDEC:' + station_id
            SWE_df[CDEC_station_id] = -9999
            print(f"CDEC data fail, {url}")
            bad_sites.append(CDEC_station_id)



def make_dates(years, start_m_d, end_m_d, WY = True):
    datelist = list()
    for year in years:
        if WY == True:
            dates = list(pd.date_range(start=f"{year-1}-{start_m_d}",end=f"{year}-{end_m_d}"))
            datelist = datelist+dates
        else:
            dates = list(pd.date_range(start=f"{year}-{start_m_d}",end=f"{year}-{end_m_d}"))
            datelist = datelist+dates

    datelist = [d.strftime("%Y-%m-%d") for d in datelist]

    return datelist

def combine_dfs(years):
    snotel_path = f"{HOME}/SWEMLv2.0/data/SNOTEL_Data"
    df = pd.DataFrame()
    for year in years:
        year_df = pd.read_parquet(f"{snotel_path}/{year}_ground_measures_dp.parquet")
        df = pd.concat([df, year_df])

    table = pa.Table.from_pandas(df)
    # Parquet with Brotli compression
    pq.write_table(table,f"{snotel_path}/ground_measures_dp.parquet", compression='BROTLI')


#def In_Situ_DataProcessing():
     # for site in tqdm_notebook(Snotelsites):
        #     args = SWE_df, site, start_date, end_date
        #     get_SNOTEL_Threaded_dp(args)

        # SWE_df = SWE_df[~SWE_df.index.duplicated(keep='first')]
        # SWE_df[date] = SWE_df[date].astype('float', errors='ignore')
        
        #fix na sites with regional average
        # CDEC = SWE_df.loc['CDEC:ADM':'CDEC:WWC']
        # meanSWE = CDEC[CDEC[date]>=0].mean().values[0]
        # SWE_df[SWE_df[date]<-10]=meanSWE

        # remove -- from CDEC predictions and make df a float
        # SWE_df[date] = SWE_df[date].astype(str)
        # SWE_df[date] = SWE_df[date].replace(['--'], -9999)
        # SWE_df[date] = pd.to_numeric(SWE_df[date], errors='coerce')
        # SWE_df[date] = SWE_df[date].fillna(-9999)
        
        # #Get state average SNOTEL to fill in nans
        # states = ['WA', 'OR', 'CA', 'ID', 'NV', 'AZ', 'MT', 'WY', 'UT', 'CO','NM', 'SD']
        # for state in states:
        #     s = SWE_df.copy()
        #     s.reset_index(inplace = True)
        #     stateSnotel = s[s['station_id'].str.contains(state)].copy()
        #     mean = stateSnotel[stateSnotel[date]>=0][date].mean()
        #     stateSnotel[date][stateSnotel[date] < 0] = mean
        #     stateSnotel.set_index('station_id', inplace = True)
        #     #update SWE_df
        #     SWE_df.update(stateSnotel)


        #change all slightly negative values to 0
        #SWE_df[date][(SWE_df[date]<0) & (SWE_df[date]>-10)] = 0