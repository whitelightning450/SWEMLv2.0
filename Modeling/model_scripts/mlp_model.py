# basic packages
import numpy as np
import pandas as pd
import os
import time
from hydrotools.nwm_client import utils 
from tqdm.notebook import tqdm_notebook


# system packages
from datetime import datetime, date
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")


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
import boto3
import s3fs
import sys
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



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

def model_arch(layers):
    input_shape, LD1, LD2, LD3, LD4, LD5, LD6 = layers
    # Build the model
    model = nn.Sequential(
        nn.Linear(input_shape, LD1),
        nn.ReLU(),
        nn.Linear(LD1, LD2),
        nn.ReLU(),
        nn.Linear(LD2, LD3),
        nn.ReLU(),
        nn.Linear(LD3, LD4),
        nn.ReLU(),
        nn.Linear(LD4, LD5),
        nn.ReLU(),
        nn.Linear(LD5, LD6),
        nn.ReLU(),
        nn.Linear(LD6, 1)
    ).to(DEVICE)

    return model


def mlp_train(x_train_scaled_t,y_train_scaled_t, layers, params, loss_func, model_path, modelname, shuffle = True):
    start_time = time.time()
    learning_rate, decay, epochs, batch_size = params

    # Create PyTorch datasets and dataloaders
    torch.manual_seed(69)
    train_dataset = TensorDataset(x_train_scaled_t, y_train_scaled_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle ) #add val loader here if we want to split the training data into training/valid/testing

    #load model
    model = model_arch(layers)

    # Define loss and optimizer
    criterion = loss_func
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    # Training loop
    for epoch in tqdm_notebook(range(epochs), desc= "Epochs completed"):
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    #save model
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)
    torch.save(model.state_dict(), f"{model_path}/{modelname}_model.pkl")

    print('finish')
    print("Run Time:" + " %s seconds " % (time.time() - start_time))



def mlp_predict(layers, model_path, modelname, x_test, x_test_scaled_t, y_test, Use_fSCA_Threshold):

    #load model
    model = model_arch(layers)
    model.load_state_dict(torch.load(f"{model_path}/{modelname}_model.pkl"))

    PredDF = x_test.copy()


    # l = len(y_test)
    # y_test = torch.Tensor(y_test.reshape(l,1))
    # y_test = y_test.to(DEVICE)

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions_scaled= model(x_test_scaled_t)

    # Invert scaling for actual
    #load scalers and scale
    scalername_y = "scaler_y.save"
    scalerfilepath_y = f"{model_path}/{scalername_y}"

    #load scalers
    scaler_y = joblib.load(scalerfilepath_y)

    #scale the testing data
    predictions = scaler_y.inverse_transform(predictions_scaled.to('cpu').numpy())
    predictions[predictions<0] = 0

    print('Model Predictions complete')

    #connect predictions with feature input dataframe
    predname = f"{modelname}_swe_cm"
    PredDF['ASO_swe_cm'] = y_test
    PredDF[predname] = predictions


    #change lines in predictions to reflect VIIRS hasSnow
    if Use_fSCA_Threshold == True:
        PredDF[predname][PredDF['hasSnow'] == False] = 0

    return PredDF


def mlp_optimization(search_params, 
                     x_train_scaled_t, 
                     y_train_scaled_t, 
                     loss_func, 
                     model_path, 
                     modelname, 
                     x_test_temp,
                     x_test_scaled, 
                     y_test_temp
                        ):

    epochs, batch_size, learning_rate, decay, L1, L2, L3, L4, L5, L6 = search_params
    
    GS_Eval_DF = pd.DataFrame()
    GS_Eval_dict = {}
    GS_Pred_dict = {}

    n_models = len(epochs)*len(batch_size)*len(learning_rate)*len(decay)*len(L1)*len(L2)*len(L3)*len(L4)*len(L5)*len(L6)
    print(f"Optimizing the {modelname} model by evaluating {n_models} models using grid search validation")

    counter = 1

    for e in epochs:
        for b in batch_size:
            for lr in learning_rate:
                for d in decay:
                    for l1 in L1:
                        for l2 in L2:
                            for l3 in L3:
                                for l4 in L4:
                                    for l5 in L5:
                                        for l6 in L6:
                                            #Train the model
                                            print(f"Training {counter} of {n_models} models")
                                            layers = x_train_scaled_t.shape[1], l1, l2, l3, l4, l5, l6
                                            params =  lr, d, e, b
                                            
                                            print(f"Parameters: {params}")
                                            print(f"Layers: {layers}")

                                            mlp_train(x_train_scaled_t,
                                                                y_train_scaled_t, 
                                                                layers, params, 
                                                                loss_func, 
                                                                model_path, 
                                                                modelname, 
                                                                shuffle = True)


                                            #Make a prediction for each location, save as compressed pkl file, and send predictions to AWS for use in CSES
                                            Preds_Dict = mlp_predict(
                                                                layers, 
                                                                model_path, 
                                                                modelname, 
                                                                x_test_temp,
                                                                x_test_scaled, 
                                                                y_test_temp
                                                              )

                                            #Evaluate model performance of the different models
                                            prediction_columns = ['swe_cm', f"{modelname}_swe_cm"]
                                            Eval_DF = Simple_Eval.Simple_Eval(Preds_Dict, 
                                                                            prediction_columns, 
                                                                            modelname, 
                                                                            plots = False, 
                                                                            keystats = False        
                                                                            )

                                            #create dataframe to store key model perf metrics, and inputs
                                            cols = [f"{modelname}_swe_kge", f"{modelname}_swe_cm_rmse", f"{modelname}_swe_mape", f"{modelname}_swe_pbias"]
                                            model_eval = Eval_DF[cols].copy()

                                            #Get mean scoring metrics for AOI - aver kge, mape, pbias
                                            model_eval = pd.DataFrame(model_eval.mean(axis=0)).T

                                            #Add model parameters
                                            parm_dict = {'Epochs': [e],
                                                        'Batchsize': [b],
                                                        'LR': [lr],
                                                        'Decay':[d],
                                                        'L1':[l1],
                                                        'L2':[l2],
                                                        'L3':[l3],
                                                        'L4':[l4],
                                                        'L5':[l5],
                                                        'L6':[l6]}
                                            params_df = pd.DataFrame.from_dict(parm_dict)

                                            #combine model eval df with params df
                                            model_df = pd.concat([model_eval, params_df], axis = 1)
                                            kge = round(model_df[f"{modelname}_swe_kge"].values[0],2)

                                            display(Eval_DF)

                                            #add to overall df
                                            GS_Eval_DF = pd.concat([GS_Eval_DF, model_df])
                                            GS_Eval_dict[kge] = Eval_DF
                                            GS_Pred_dict[kge] = Preds_Dict
                                            counter = counter +1
    #Sort by kge
    GS_Eval_DF.sort_values(by = f"{modelname}_swe_kge", ascending = False, inplace = True)
    GS_Eval_DF.reset_index(inplace=True, drop = True)

    return GS_Eval_DF, GS_Eval_dict, GS_Pred_dict

def Final_Model(GS_Eval_DF,
                x_train_scaled_t,
                y_train_scaled_t, 
                loss_func, 
                model_path, 
                modelname,
                test_years, 
                stations, 
                x_test_temp,
                x_test_scaled, 
                y_test_temp,
                StreamStats,
                station_index_list):

    #Train the model with optimized parameters
    # parameters
    epochs = GS_Eval_DF['Epochs'].values[0] # 
    batch_size = int(GS_Eval_DF['Batchsize'].values[0])
    learning_rate =  GS_Eval_DF['LR'].values[0]
    decay = GS_Eval_DF['Decay'].values[0]
    L1 = GS_Eval_DF['L1'].values[0]
    L2 = GS_Eval_DF['L2'].values[0]
    L3 = GS_Eval_DF['L3'].values[0]
    L4 = GS_Eval_DF['L4'].values[0]
    L5 = GS_Eval_DF['L5'].values[0]
    L6 = GS_Eval_DF['L6'].values[0]
    layers = x_train_scaled_t.shape[1], L1, L2, L3, L4, L5, L6
    params =  learning_rate, decay, epochs, batch_size

                                            
    print(f"Parameters: {params}")
    print(f"Layers: {layers}")
    loss_func = nn.MSELoss()

    #Train the model
    mlp_train(x_train_scaled_t,
                        y_train_scaled_t, 
                        layers, 
                        params, 
                        loss_func, 
                        model_path, 
                        modelname, 
                        shuffle = True)


    #Make a prediction for each location, save as compressed pkl file, and send predictions to AWS for use in CSES
    Preds_Dict = mlp_predict(test_years, 
                        layers, 
                        model_path, 
                        modelname, 
                        stations, 
                        x_test_temp,
                        x_test_scaled, 
                        y_test_temp,
                        StreamStats,
                        station_index_list)

    #Evaluate model performance of the different models, 'flow_cfs_pred', 
    prediction_columns = ['NWM_flow', f"{modelname}_flow"]
    Eval_DF = Simple_Eval.Simple_Eval(Preds_Dict, 
                                    prediction_columns, 
                                    modelname, 
                                    supply = False,
                                    plots = True, 
                                    keystats = False        
                                    )
    
    return Eval_DF, Preds_Dict