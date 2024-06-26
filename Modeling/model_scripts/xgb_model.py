# This file created on 02/20/2024 by savalan

import numpy as np
from hydrotools.nwm_client import utils 
import xgboost as xgb
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pickle as pkl
from xgboost import XGBRegressor
from scipy import optimize
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

# deep learning packages
import torch


#Shared/Utility scripts
import os
import boto3
import s3fs
import sys
sys.path.insert(0, '../..') #sys allows for the .ipynb file to connect to the shared folder files
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

class XGBoostRegressorCV:
    def __init__(self, params, path=None):
        self.params = params
        self.model = xgb.XGBRegressor()
        #self.model = xgb.XGBRegressor(tree_method="hist", device="cuda"
        self.best_model = None
        self.path = path

        if DEVICE =='cuda':
            self.model = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
            print(f"XGBoost model using GPU")


    def tune_hyperparameters(self, X, y, cv=3):
        """Performs GridSearchCV to find the best hyperparameters."""
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=self.params, 
                                   scoring='neg_mean_absolute_error', #change to mean squared error
                                   cv=cv, 
                                   n_jobs = -1,
                                   verbose=3)
        grid_search.fit(X, y)
        self.best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best RMSE: {grid_search.best_score_}")
        #save the model features
        pkl.dump(grid_search, open(self.path, "wb")) 

    def train(self, input_columns, X, y, parameters={}):
        """Trains the model using the best hyperparameters found."""
        if self.best_model:
            self.best_model.fit(X, y)

            # feature importance
            imp, feats = zip(*sorted(zip(self.best_model.feature_importances_, input_columns)))

            # plot
            pyplot.barh(feats, imp)
            pyplot.show()
        else:
            eta = parameters.best_params_['eta']
            max_depth =parameters.best_params_['max_depth']
            n_estimators = parameters.best_params_['n_estimators']
            self.model.fit(X, y, n_estimators=n_estimators, max_depth=max_depth, eta=eta, verbose=True)
            print("Please tune hyperparameters first.")

            # feature importance
            # feature importance
            imp, feats = zip(*sorted(zip(self.model.feature_importances_, input_columns), reverse=True))

            # plot
            pyplot.barh(feats, imp)
            pyplot.show()

    def predict(self, X):
        """Predicts using the trained XGBoost model on the provided data."""
        if self.best_model:
            return self.best_model.predict(X)
        else:
            print("Model is not trained yet. Please train the model first.")
            return None

    def evaluate(self, X, y):
        """Evaluates the trained model on a separate test set."""
        if self.best_model:
            
            # define model evaluation method
            cv = RepeatedKFold(n_splits=10, n_repeats=3)
            
            # evaluate model
            scores = cross_val_score(self.best_model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

            # Caculate model performance and force scores to be positive
            print('Mean MAE: %.3f (%.3f)' % (abs(scores.mean()), scores.std()) )

        else:
            print("Model is not trained yet. Please train the model first.")


def XGB_Train(model_path, input_columns, x_train, y_train, tries, hyperparameters, perc_data):
    start_time = time.time()

    # Start running the model several times. 
    for try_number in range(tries):
        print(f'Trial Number {try_number} ==========================================================')
        
        # # Set the optimizer, create the model, and train it. 
        xgboost_model = XGBoostRegressorCV(hyperparameters, f"{model_path}/best_model_hyperparameters.pkl")
        new_data_len = int(len(x_train) * perc_data) #determine hyperprams using 25% of the data
        print(f"Tuning hyperparametetrs on {perc_data*100}% of training data")
        x_hyper, y_hyper = x_train.iloc[:new_data_len], y_train.iloc[:new_data_len]
        xgboost_model.tune_hyperparameters(x_hyper, y_hyper)
        xgboost_model.evaluate(x_train.iloc[:new_data_len], y_train.iloc[:new_data_len])
        print('Training model with optimized hyperparameters')
        xgboost_model.train(input_columns, x_train, y_train)
        print('Saving Model')
        
        #adjust this to match changing models
        pkl.dump(xgboost_model, open(f"{model_path}/best_model.pkl", "wb"))  

    print('Run is Done!' + "Run Time:" + " %s seconds " % (time.time() - start_time))


def XGB_Predict(model_path, modelname, x_test, y_test, Use_fSCA_Threshold):

    PredDF = x_test.copy()

    #Load model
    xgboost_model = pkl.load(open(f"{model_path}/best_model.pkl", "rb"))
    predictions = xgboost_model.predict(x_test)
    predictions[predictions<0] = 0

    print('Model Predictions complete')

    #connect predictions with feature input dataframe
    predname = f"{modelname}_swe_cm"
    PredDF['ASO_swe_cm'] = y_test
    PredDF[predname] = predictions


    #change lines in predictions to reflect VIIRS hasSnow
    if Use_fSCA_Threshold == True:
        PredDF[predname][PredDF['hasSnow'] == False] = 0

    # #save predictions as compressed pkl file
    # pred_path = f"{HOME}/NWM_ML/Predictions/Hindcast/{modelname}/Multilocation"
    # file_path = f"{pred_path}/{modelname}_predictions.pkl"
    # if os.path.exists(pred_path) == False:
    #     os.makedirs(pred_path)
    # with open(file_path, 'wb') as handle:
    #     pkl.dump(Preds_Dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return PredDF
  