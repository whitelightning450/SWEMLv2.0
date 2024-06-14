import pandas as pd
import pickle as pkl
import os
#connecting to AWS
import warnings; warnings.filterwarnings("ignore")
import boto3
from botocore import UNSIGNED
from botocore.client import Config


#load access key
HOME = os.path.expanduser('~')
KEYPATH = "SWEMLv2.0/AWSaccessKeys.csv"
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


HOME = os.path.expanduser('~')

def modeldomain():
    dpath = f"{HOME}/SWEMLv2.0/data/PreProcessed"
    try:
        regions = pd.read_pickle(f"{dpath}/RegionVal.pkl")
    except:
        print('File not local, getting from AWS S3.')
        if not os.path.exists(dpath):
            os.makedirs(dpath, exist_ok=True)
        key = f"data/PreProcessed/RegionVal.pkl"            
        S3.meta.client.download_file(BUCKET_NAME, key,f"{dpath}/RegionVal.pkl")
        regions = pd.read_pickle(f"{dpath}/RegionVal.pkl")
    SWEMLv2regions = {}
    #regions['GBasin'].reset_index(inplace = True)
    #Reduce the number of regions for SWEMLv2.0
    #SW US
                
    SWEMLv2regions['Southwest'] = pd.concat([regions['S_Sierras'], 
                                            regions['N_Sierras'], 
                                            regions['GBasin'], 
                                            regions['Ca_Coast']])
    SWEMLv2regions['Southwest'].reset_index(drop = True, inplace = True)
    #NW us
    SWEMLv2regions['Northwest'] = pd.concat([regions['Wa_Coast'], 
                                            regions['N_Cascade'], 
                                            regions['S_Cascade'],
                                            regions['Or_Coast']])
    SWEMLv2regions['Northwest'].reset_index(drop = True, inplace = True)

    #Northern Rockies
    SWEMLv2regions['NorthernRockies'] = pd.concat([regions['E_WA_N_Id_W_Mont'], 
                                            regions['E_Or'], 
                                            regions['Sawtooth'],
                                            regions['Greater_Glacier'],
                                            regions['N_Yellowstone'],
                                            regions['Greater_Yellowstone'],
                                            regions['SW_Mont']])
    SWEMLv2regions['NorthernRockies'].reset_index(drop = True, inplace = True)

    
    #Southern Rockies
    SWEMLv2regions['SouthernRockies'] = pd.concat([regions['N_Co_Rockies'], 
                                            regions['SW_Co_Rockies'], 
                                            regions['N_Wasatch'],
                                            regions['S_Wasatch'],
                                            regions['SW_Mtns'],
                                            regions['S_Wyoming'],
                                            regions['SE_Co_Rockies']])
    SWEMLv2regions['SouthernRockies'].reset_index(drop = True, inplace = True)


    with open(f'{dpath}/SWEMLV2Regions.pkl', 'wb') as handle:
        pkl.dump(SWEMLv2regions, handle, protocol=pkl.HIGHEST_PROTOCOL)

    regionlist = list(SWEMLv2regions.keys())

    return regionlist