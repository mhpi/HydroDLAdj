# This code is written by Yalan Song from MHPI group, Penn State Univerity
# Purpose: This code solves ODEs of hydrological models with Adjoint
import torch
import numpy as np
import os
import sys

sys.path.append('../..')
from HydroDLAdj.nnModel import train_module
from HydroDLAdj.nnModel import test_module
from HydroDLAdj.nnModel import lstmModel_module
from HydroDLAdj.data import func

import random
import glob
import re
import pickle
import pandas as pd
##Set the random numbers
randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Set the GPU machine to use
gpuid = 7
torch.cuda.set_device(gpuid)
device = torch.device("cuda")
dtype=torch.float32

## To create the pickle file for CAMELS data, you can use the code the following link:
# https://colab.research.google.com/drive/1oq5onUpekCjuInTlnEdT3TGR39SSjr8F?usp=sharing#scrollTo=FGViqzC__BCw
## Or use the following command lines to directly get training_file and validation_file, but in this way, the input variables and training/test periods are fixed
#Ttrain = [19801001, 19951001] #training period
#valid_date = [19951001, 20101001]  # Testing period
#!pip install gdown
#!gdown 1HrO-8A0l7qgVVz6pIRFfqZin2ZxAG72B
#!gdown 1ZPI_ypIpF9o-YzZnC9mc-Ti2t-c6g7F5
#!gdown 1VhjjKE7KYcGIeWlihOP9fQtZ71E7ufjl
## Load the data
datapath = "/data/yxs275/hydroDL/example/notebook/datatest/"
train_file = datapath+'training_file'

## Path to save your model
saveFolder = "/data/yxs275/NROdeSolver/output/HBVtest_module_hbv_1_2_13_dynamic_rout_static_no_threshold/"
# Load X, Y, C from a file
with open(train_file, 'rb') as f:
    train_x, train_y, train_c = pickle.load(f)  # Adjust this line based on your file format
##Forcings from the pickle file are precipitaion and temperature from Daymet
## PET prepared by MHPI group from [19801001, 20101001]
data_PET_all = np.load(datapath+"PET.npy" )
time_PET = pd.date_range('1980-10-01', f'2010-09-30', freq='d')
data_PET = data_PET_all[:,time_PET.get_loc('1980-10-01'):time_PET.get_loc('1995-10-01')]

##List of attributes
attrLst = [ 'p_mean','pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
            'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
            'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
            'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
            'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
            'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']



basinarea  = train_c[:,np.where(np.array(attrLst)=='area_gages2')[0]]

streamflow_data = func.basinNorm_mmday(train_y,  basinarea, toNorm=True)
##List of forcing used in the hydrological model
forcing_lst = ['prcp','tmean','PET']


xTrain = np.concatenate((train_x,np.expand_dims(data_PET,axis=-1)),axis = -1)


#Data normalization for inputs of NN
log_norm_cols = [ "prcp" ]

scaler = func.HydroScaler(attrLst=attrLst, seriesLst=forcing_lst, xNanFill=0.0, log_norm_cols=log_norm_cols)

attri_norm, xTrain_norm = scaler.fit_transform(train_c, xTrain.copy())
attri_norm[np.isnan(attri_norm)] = 0.0
xTrain_norm[np.isnan(xTrain_norm)] = 0.0

attri_norm = np.expand_dims(attri_norm, axis=1)
attri_norm = np.repeat(attri_norm, xTrain_norm.shape[1], axis=1)
data_norm = np.concatenate((xTrain_norm, attri_norm), axis=-1)


#Hyperparameters
bs = 100   ##batch size
nS = 5   ## number of state variables
nEpoch = 50   ##number of epochs
alpha = 0.25    ##aplha in the loss function
rho = 365   ###time length of batch
buffTime = 365   ##length of warmup period
delta_t  = torch.tensor(1.0).to(device = device,dtype = dtype)  ## Time step (one day)

nflux = 1  ## Number of target fluxes, only one : streamflwo


ninv = data_norm.shape[-1]
nfea = 13     ## number of physical parameters in the hyfrological models
routn = 15  ## Length of the routing window
nmul = 16    ## number of components
hiddeninv = 256   ## hidden size of NN
drinv = 0.5     ## dropout rate
model = lstmModel_module.lstmModel(ninv, nfea, nmul, hiddeninv, drinv)   ##Initializate the NN (LSTM)
tdlst = [1,2, 13]   ## index of the dynamic parameter
tdRepS = [str(ix) for ix in tdlst]

startEpoch = 0   ## Start from epoch 0
rerun = False     ## Swtich for continuously rerun the model

if rerun:

    weights_filenames = []
    for fidx, f in enumerate(glob.glob(saveFolder+"model_Ep*")):
        weights_filenames.append(f)
        print(re.split("(\d+)",f))
    weights_filenames.sort(key = lambda x: int(re.split("(\d+)",x)[1]))


    path_model = saveFolder+weights_filenames[-1]
    print("Reading ", path_model)
    model = torch.load(path_model, map_location=f"cuda:{gpuid}")
    startEpoch = int(re.split("(\d+)",weights_filenames[-1])[1])+1

if os.path.exists(saveFolder) is False:
    os.mkdir(saveFolder)

model_name = "HBV_Module"


train_module.trainModel(xTrain,
           streamflow_data,
           data_norm,
           nS,
           nflux,
           nfea,
           nmul,
           model,
           delta_t,
           alpha,
           tdlst,
           startEpoch=startEpoch,
           nEpoch=nEpoch,
           miniBatch=[bs, rho],
           buffTime=buffTime,
           saveFolder=saveFolder,
           routn=routn,
           model_name=model_name,
           useAD_efficient = False,
               )


## Model validation
## Load the trained model
testepoch = 50
testbs = 200  #batchsize for testing
model_file = saveFolder + f'/model_Ep{testepoch}.pt'
print("Reading ", model_file)
model = torch.load(model_file, map_location=f"cuda:{gpuid}")


validation_file = datapath+'validation_file'
with open(validation_file, 'rb') as g:
    val_x, val_y, val_c = pickle.load(g)

basinarea_val  = val_c[:,np.where(np.array(attrLst)=='area_gages2')[0]]

streamflow_data_val = func.basinNorm_mmday(val_y,  basinarea, toNorm=True)


data_PET_val = data_PET_all[:,time_PET.get_loc('1995-10-01'):time_PET.get_loc('2010-09-30')+1]
xTrain_val = np.concatenate((val_x,np.expand_dims(data_PET_val,axis=-1)),axis = -1)

attri_norm_val = scaler.transform(val_c, attrLst)
xTrain_norm_val = scaler.transform(xTrain_val, forcing_lst)
attri_norm_val[np.isnan(attri_norm_val)] = 0.0
xTrain_norm_val[np.isnan(xTrain_norm_val)] = 0.0

attri_norm_val= np.expand_dims(attri_norm_val, axis=1)
attri_norm_val = np.repeat(attri_norm_val, xTrain_norm_val.shape[1], axis=1)
data_norm_val = np.concatenate((xTrain_norm_val, attri_norm_val), axis=-1)

warmuplength = 730 ## Use 2 years training data to warmup the simulation in validation
xTrain_val = np.concatenate((xTrain[:,-warmuplength:,:], xTrain_val), axis= 1)
data_norm_val = np.concatenate((data_norm[:,-warmuplength:,:], data_norm_val), axis= 1)


test_module.testModel(xTrain_val,
                      streamflow_data_val,
                      data_norm_val,
                      nS,
                      nflux,
                      nfea,
                      nmul,
                      model,
                      delta_t,
                      tdlst,
                      bs = testbs,
                      saveFolder=saveFolder,
                      routn=routn,
                      model_name=model_name,
                      useAD_efficient = False,
                      )
