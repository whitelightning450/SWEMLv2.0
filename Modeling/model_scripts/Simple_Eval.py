#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error 
import hydroeval as he
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import os
HOME = os.path.expanduser('~')

def Parity_Evaluation_Plots(DF, regionlist, modelname, savefig, figname):

    fontsize = 12
    pred_col = f"{modelname}_swe_cm"

# Subplots.
    fig, ax = plt.subplots(1,1, 
                           figsize=(8, 4))
    plt.subplots_adjust(
                    wspace=0.4
                   )
    fig.patch.set_facecolor('white')

    #set min/max for y-axis of the predictions/observations
    ymin = min(DF['ASO_swe_cm'])*1.1
    ymax = max(DF['ASO_swe_cm'])*1.1
    
    #add color options
    colors = ['blue', 'orange', 'red','green']

    # Addscatter plot
    for region in np.arange(0, len(regionlist),1):
        regiondf = DF[DF['region']==regionlist[region]]
        ax.scatter(regiondf['ASO_swe_cm'], regiondf[pred_col],
                   c=colors[region], alpha=0.35, label=regionlist[region])

     # Add some parameters.
    ax.set_title('SWE Predictions', fontsize=fontsize)
    ax.set_xlabel('Observations (cm)', fontsize=fontsize-2)
    ax.set_ylabel('Predictions (cm)', fontsize=fontsize-2,)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(ymin, ymax)
    ax.legend(fontsize=fontsize-2, loc='upper right')
    
    #Add a 1:1 prediction:observation plot
    ax.plot((0,ymax),(0,ymax), linestyle = '--', color  = 'red')

    if savfig==True:
        plt.savefig(f"{HOME}SWEMLv2.0/Evaluation/Figures/{figname}.png", dpi =600, bbox_inches='tight')

    plt.show()
    
    
def TimeSeries_Evaluation_Plots(DF, predictions):

# Subplots.
    fig, ax = plt.subplots(1,1, figsize=(8, 7))
    fig.patch.set_facecolor('white')

    #set min/max for y-axis of the predictions/observations
    ymin = min(DF['ASO_swe_cm'])*1.1
    ymax = max(DF['ASO_swe_cm'])*1.1
    
    #add color options
    colors = ['blue', 'red','green']
    
    ax.plot(DF['DOY'], DF['ASO_swe_cm'],
                   c='orange', alpha=0.35, label= 'Observed')

    # Add predictions to plot
    for pred in np.arange(0, len(predictions),1):
        ax.plot(DF['DOY'], DF[predictions[pred]],
                   c=colors[pred], alpha=0.35, label=predictions[pred])

     # Add some parameters.
    ax.set_title('Streamflow Predictions', fontsize=16)
    ax.set_xlabel('Time (DOY)', fontsize=14)
    ax.set_ylabel('Streamflow (cfs)', fontsize=14,)
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=14, loc='upper right')

    plt.show()
    

#Define some key model performance metics: RMSE, PBias, MAE, MAPE
def RMSE(DF, predictions):
    eval_dict ={}
    for pred in predictions:
        rmse = mean_squared_error(DF['ASO_swe_cm'], DF[pred], squared=False)
        #print('RMSE for ', predictions[pred], ' is ', rmse, ' cfs')
        eval_dict[f"{pred}_rmse"] = rmse

    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval

def MAPE(DF, predictions):
    eval_dict ={}
    for pred in predictions:
        mape = round(mean_absolute_percentage_error(DF['ASO_swe_cm'], DF[pred])*100, 2)
        #print('Mean Absolute Percentage Error for ', predictions[pred], ' is ', mape, '%')
        eval_dict[f"{pred}_mape"] = mape
    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval
        
def PBias(DF, predictions):
    eval_dict ={}
    for pred in predictions:
        pbias = he.evaluator(he.pbias,  DF[pred], DF['ASO_swe_cm'])
        pbias = round(pbias[0],2)
        #print('Percentage Bias for ', predictions[pred], ' is ', pbias, '%')
        eval_dict[f"{pred}_pbias"] = pbias
    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval

def KGE(DF, predictions):
    eval_dict ={}
    for pred in predictions:
        kge, r, alpha, beta = he.evaluator(he.kge,  DF[pred], DF['ASO_swe_cm'])
        kge = round(kge[0],2)
        #print('Kling-Glutz Efficiency for ', predictions[pred], ' is ', kge)
        eval_dict[f"{pred}_kge"] = kge
    
    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval

def Key_Stats(DF, predictions):  
    eval_dict = {}
    eval_dict['min_storage'] = min(DF['storage'])
    eval_dict['max_storage'] = max(DF['storage'])
    eval_dict['min_swe'] = min(DF['swe'])
    eval_dict['max_swe'] = max(DF['swe'])
    eval_dict['min_obs_flow'] = min(DF['ASO_swe_cm'])
    eval_dict['max_obs_flow'] = max(DF['ASO_swe_cm'])
    for pred in predictions:
        eval_dict[f"{pred}_min"] = min(DF[pred])
        eval_dict[f"{pred}_fmax"] = max(DF[pred])

    eval = pd.DataFrame.from_dict(eval_dict, orient = 'index').T
    return eval

#function for randomly sampling predictions
def SamplePreds(regionlist, PredsDF, df,n_samples):
    print(n_samples)
    PredsDF.insert(0, 'cell_id', df['cell_id'])
    PredsDF.insert(1, 'Date', df['Date'])
    PredsDF.insert(2, 'region', df['region'])

    sampleDF = pd.DataFrame()
    for region in regionlist:
        DF = PredsDF[PredsDF['region']==region].copy()
        DF = DF.sample(n=n_samples, random_state=69)

        sampleDF = pd.concat([sampleDF, DF])

    return sampleDF, PredsDF

def Simple_Eval(regionlist, PredsDF, prediction_columns, modelname, savfig, figname, plots = False, keystats = False):
    
    plotdf = PredsDF.copy()
    plotdf.reset_index(inplace = True, drop = False)
    Parity_Evaluation_Plots(plotdf, regionlist, modelname, savfig, figname)

    #put the below into a DF so we can compare all sites..determine overall and regional skill
    region = pd.DataFrame()
    region['region'] = ['Overall']

    #Get RMSE from the model
    rmse = round(RMSE(PredsDF, prediction_columns))

    #Get Mean Absolute Percentage Error from the model
    mape = round(MAPE(PredsDF, prediction_columns),2)

    #Get Percent Bias from the model
    pbias = round(PBias(PredsDF, prediction_columns), 2)

    #Get Kling-Gutz Efficiency from the model
    kge = round(KGE(PredsDF, prediction_columns),2)

    EvalDF = pd.DataFrame()
    EvalDF = pd.concat([EvalDF, region, kge,rmse,mape,pbias],axis = 1)


    for region in regionlist:
        regionDF = PredsDF[PredsDF['region']==region].copy()
        reg = pd.DataFrame()
        reg['region'] = [region]

        #Get RMSE from the model
        rmse = round(RMSE(regionDF, prediction_columns))

        #Get Mean Absolute Percentage Error from the model
        mape = round(MAPE(regionDF, prediction_columns),2)

        #Get Percent Bias from the model
        pbias = round(PBias(regionDF, prediction_columns), 2)

        #Get Kling-Gutz Efficiency from the model
        kge = round(KGE(regionDF, prediction_columns),2)

        evaldf = pd.DataFrame()
        evaldf = pd.concat([evaldf, reg, kge,rmse,mape,pbias],axis = 1)
        EvalDF = pd.concat([EvalDF, evaldf])


    display(EvalDF)

    return EvalDF
        

