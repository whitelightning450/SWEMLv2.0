# hydrological packages
import hydroeval as he
from hydrotools.nwm_client import utils 

# basic packages
import numpy as np
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq


# system packages
from tqdm import tqdm_notebook
from datetime import datetime, date
import datetime
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")
import platform
import time

# data analysi packages
from scipy import optimize
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import joblib

# deep learning packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Shared/Utility scripts
import sys
import boto3
import s3fs
#sys.path.insert(0, '../..') #sys allows for the .ipynb file to connect to the shared folder files
from model_scripts import Simple_Eval

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
BUCKET_NAME = 'streamflow-app-data'
BUCKET = S3.Bucket(BUCKET_NAME)

#s3fs
fs = s3fs.S3FileSystem(anon=False, key=ACCESS['Access key ID'][0], secret=ACCESS['Secret access key'][0])


#def get_ML_Data(regionlist, output_res, DataFrame, fSCA_thresh, remove0swe, removeswe_thresh):
def get_ML_Data(regionlist, output_res, DataFrame, fSCA_thresh):

    #Get processed training data 
    regiondf = pd.DataFrame()
    for region in regionlist:
        filepath = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}/{DataFrame}/{fSCA_thresh}"
        #try:
        files = [filename for filename in os.listdir(filepath) if filename.endswith(".parquet")]
        print(f'Concatenating {len(files)} for the model dataframe development.')
        trainingDF = pd.DataFrame()
        for file in tqdm_notebook(files):
            datedf = pd.read_parquet(f"{filepath}/{file}")
            trainingDF = pd.concat([trainingDF,datedf])

        #reset index for below steps
        trainingDF.sort_values('Date', inplace = True)
        trainingDF.reset_index(inplace=True)
        # except:
        #     print("Data not found, retreiving from AWS S3")
        #     if not os.path.exists(datapath):
        #         os.makedirs(datapath, exist_ok=True)
        #     key = "SWEMLv2.0"+datapath.split("SWEMLv2.0",1)[1]+'/'+trainingfile     
        #     print(key)  
        #     S3.meta.client.download_file(BUCKET_NAME, key,filepath)
        #     df = pd.read_parquet(filepath)

        #convert swe from meters to cm to be consistent with in situ obs
        trainingDF['swe_cm'] = trainingDF['swe_m'] *100
        trainingDF.pop('swe_m')
        trainingDF['region'] = region
        #add region metric
        if region == 'Northwest':
            trainingDF['region_class'] = 3
        if region == 'Southwest':
            trainingDF['region_class'] = 2
        if region == 'SouthernRockies':
            trainingDF['region_class'] = 1

        print(f"There are {len(trainingDF)} datapoints for model training/testing in the {region} modeling domain.")
        #Add dataframes together
        regiondf = pd.concat([regiondf, trainingDF])

    #remove the large amounts of zero values
    # if remove0swe ==True:
    #     regiondf = regiondf[regiondf['swe_cm']>removeswe_thresh]

    print(f"There are {len(regiondf)} datapoints for model training/testing in the overall modeling domain.")

    regiondf.reset_index(inplace=True, drop=True)

    return regiondf

def remove0swe(df, remove0swe, removeswe_thresh):
    if remove0swe ==True:
        nonzeroSWE_df = df[df['swe_cm']>removeswe_thresh]

    print(f"There are {len(nonzeroSWE_df)} in the training dataset, removing {len(df)-len(nonzeroSWE_df)} zero values in the ASO dataset, VIIRS fSCA will capture these.")
    
    return nonzeroSWE_df