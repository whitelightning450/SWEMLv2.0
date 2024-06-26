# hydrological packages
import hydroeval as he
from hydrotools.nwm_client import utils 

# basic packages
import numpy as np
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
#import bz2file as bz2

# system packages
#from progressbar import ProgressBar
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
BUCKET_NAME = 'national-snow-model'
BUCKET = S3.Bucket(BUCKET_NAME)

#s3fs
fs = s3fs.S3FileSystem(anon=False, key=ACCESS['Access key ID'][0], secret=ACCESS['Secret access key'][0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def data_clean(df, regionlist): #add region to dataframe
    #clean data. For example, approximately 420,000 locations/timesteps in the Southern rockies domain show VIIRS as having snow but show 0" SWE. Likely an error and remove these sites.
    bad_low_ASO_data = df[(df['swe_cm']< 0.1) & (df['hasSnow'] == True)] #using VIIRS obs to "clean the data"
    bad_VIIRS_data = df[(df['swe_cm']> 0.1) & (df['hasSnow'] == False)]

    print(f"The provided data contains {len(df)} data points, of which {len(bad_low_ASO_data)} locations/timesteps show no SWE and VIIRS fsca > 20%")
    print(f"{len(bad_VIIRS_data)} locations/timesteps show SWE and VIIRS fsca < 20%")

    #remove unrealistically high values, values higher than below are most likely bad lidar obs
    bad_high_ASO_data = pd.DataFrame()
    for region in regionlist:
        if region == 'Southwest':
            value = 400
            bad_df = df[(df['region'] == region) & (df['swe_cm'] > value)] # This is ~150" of SWE
            print(f"{len(bad_high_ASO_data)} locations/timesteps show SWE greater than a realistic value ({value} cm) in the {region} domain")
        if region == 'Northwest':
            value = 800
            bad_df = df[(df['region'] == region) & (df['swe_cm'] > 800)] # This is ~300" of SWE
            print(f"{len(bad_high_ASO_data)} locations/timesteps show SWE greater than a realistic value ({value} cm) in the {region} domain")
        if region == 'SouthernRockies':
            value = 250
            bad_df = df[(df['region'] == region) & (df['swe_cm'] > 250)] # This is ~100" of SWE
            print(f"{len(bad_high_ASO_data)} locations/timesteps show SWE greater than a realistic value ({value} cm) in the {region} domain")
        
        bad_high_ASO_data = pd.concat([bad_high_ASO_data, bad_df])




    print('removing..')

    dfclean = pd.concat([df,bad_high_ASO_data, bad_low_ASO_data, bad_VIIRS_data])
    dfclean = dfclean[~dfclean.index.duplicated(keep=False)]
    dfclean.reset_index(inplace=True, drop=True)

    print(f"There are {len(dfclean)} datapoints for model training/testing.")

    return dfclean


def mlp_scaler(regionlist, df, years, splitratio, test_years, target, input_columns, model_path, scalertype = 'MinMax'):
    #check to make sure the model path exists
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # #Select training data
    # if years == True:
    #     #training years
    #     x_train = df[~df.datetime.dt.year.isin(test_years)]
    #     x_train.pop('Date')
    #     y_train = x_train[target]
    #     x_train.pop(target)
    #     x_train = x_train[input_columns]

    #     #testing years
    #     x_test = df[df.datetime.dt.year.isin(test_years)]
    #     x_test.pop('Date')
    #     y_test = x_test[target]
    #     x_test.pop(target)
    #     x_test = x_test[input_columns]

    #     #Convert dataframe to numpy, scale, save scalers
    #     y_train_np = y_train.to_numpy()
    #     x_train_np = x_train.to_numpy()
    #     x_test_np = x_test.to_numpy()
        
    # else:
    #take random sample of each region to ensure they are in the testing data
    x_train, y_train, x_test, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for region in regionlist:
        regionDF = df[df['region'] == region].copy()

        display(regionDF.head(5))

        X = regionDF[input_columns]
        y = regionDF[target]

        x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=splitratio, random_state=69)

        x_train = pd.concat([x_train, x_train_reg])
        x_test = pd.concat([x_test, x_test_reg])
        y_train = pd.concat([y_train, y_train_reg])
        y_test = pd.concat([y_test, y_test_reg])

    #convert to numpy
    y_train_np = y_train.to_numpy()
    x_train_np = x_train.to_numpy()
    x_test_np = x_test.to_numpy()

    scalerfilepath_x = f"{model_path}/scaler_x.save"
    scalerfilepath_y = f"{model_path}/scaler_y.save"

    if scalertype == 'MinMax':
        scaler = MinMaxScaler() #potentially change scalling...StandardScaler
        x_train_scaled = scaler.fit_transform(x_train_np)
        x_test_scaled = scaler.fit_transform(x_test_np)
        joblib.dump(scaler, scalerfilepath_x)

        scaler = MinMaxScaler() #potentially change scalling...StandardScaler
        y_train_scaled = scaler.fit_transform(y_train_np.reshape(-1, 1))
        joblib.dump(scaler, scalerfilepath_y)  

    if scalertype == 'Standard':
        scaler = StandardScaler() #potentially change scalling...StandardScaler
        x_train_scaled = scaler.fit_transform(x_train_np)
        joblib.dump(scaler, scalerfilepath_x)

        scaler = StandardScaler() #potentially change scalling...StandardScaler
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
        joblib.dump(scaler, scalerfilepath_y) 

    print(f"y train shape {y_train_scaled.shape}")
    print(f"x train shape {x_train_scaled.shape}")
    print(f"x test shape {x_test_scaled.shape}")

    # Convert to tensor for PyTorch
    x_train_scaled_t = torch.Tensor(x_train_scaled)
    y_train_scaled_t = torch.Tensor(y_train_scaled)
    x_test_scaled_t = torch.Tensor(x_test_scaled)
    #Make sure the tensors on are the respective device (cpu/gpu)
    x_train_scaled_t = x_train_scaled_t.to(device)
    y_train_scaled_t = y_train_scaled_t.to(device)
    x_test_scaled_t = x_test_scaled_t.to(device)

    return x_train_scaled_t, y_train_scaled_t, x_test_scaled_t, x_test, y_test



#Data processor for MLP model
def xgb_processor(regionlist, df, years, splitratio, test_years, target, input_columns, model_path, scalertype = 'MinMax'):
    #check to make sure the model path exists
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # #Select training data
    # if years == True:
    #     #training years
    #     x_train = df[~df.datetime.dt.year.isin(test_years)]
    #     x_train.pop('Date')
    #     y_train = x_train[target]
    #     x_train.pop(target)
    #     x_train = x_train[input_columns]

    #     #testing years
    #     x_test = df[df.datetime.dt.year.isin(test_years)]
    #     x_test.pop('Date')
    #     y_test = x_test[target]
    #     x_test.pop(target)
    #     x_test = x_test[input_columns]

    #     #Convert dataframe to numpy, scale, save scalers
    #     y_train_np = y_train.to_numpy()
    #     x_train_np = x_train.to_numpy()
    #     x_test_np = x_test.to_numpy()
        
    # else:
    #take random sample of each region to ensure they are in the testing data
    x_train, y_train, x_test, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for region in regionlist:
        regionDF = df[df['region'] == region].copy()

        display(regionDF.head(5))

        X = regionDF[input_columns]
        y = regionDF[target]

        x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=splitratio, random_state=69)

        x_train = pd.concat([x_train, x_train_reg])
        x_test = pd.concat([x_test, x_test_reg])
        y_train = pd.concat([y_train, y_train_reg])
        y_test = pd.concat([y_test, y_test_reg])

    #convert to numpy
    # y_train_np = y_train.to_numpy()
    # x_train_np = x_train.to_numpy()
    # x_test_np = x_test.to_numpy()

    # scalerfilepath_x = f"{model_path}/scaler_x.save"
    # scalerfilepath_y = f"{model_path}/scaler_y.save"

    # if scalertype == 'MinMax':
    #     scaler = MinMaxScaler() #potentially change scalling...StandardScaler
    #     x_train_scaled = scaler.fit_transform(x_train_np)
    #     x_test_scaled = scaler.fit_transform(x_test_np)
    #     joblib.dump(scaler, scalerfilepath_x)

    #     scaler = MinMaxScaler() #potentially change scalling...StandardScaler
    #     y_scaled_train = scaler.fit_transform(y_train_np.reshape(-1, 1))
    #     joblib.dump(scaler, scalerfilepath_y)  

    # if scalertype == 'Standard':
    #     scaler = StandardScaler() #potentially change scalling...StandardScaler
    #     x_train_scaled = scaler.fit_transform(x_train_np)
    #     joblib.dump(scaler, scalerfilepath_x)

    #     scaler = StandardScaler() #potentially change scalling...StandardScaler
    #     y_scaled_train = scaler.fit_transform(y_train.reshape(-1, 1))
    #     joblib.dump(scaler, scalerfilepath_y) 

    return x_train, y_train, x_test, y_test
