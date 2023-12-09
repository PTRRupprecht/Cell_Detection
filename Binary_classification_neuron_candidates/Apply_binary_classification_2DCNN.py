#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:18:48 2023

@author: Peter Rupprecht, ptrrupprecht+celldetection@gmail.com

Code to apply pretrained models for cell classification (cell vs. non-cell) from a 3D local volume (31x31x91 pixels)


Please change "folder_name" according to the local paths on your computer

"""


folder_name = '/home/helios/Desktop/Cell_Detection/Binary_classification_neuron_candidates/'


"""

Import required packages

"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import os, glob
import numpy as np

import zipfile
import copy


"""

Load data for which the classifier will be applied

The output of this section is a numpy array ("input_data")

This is a 4D array with (number of samples) x 1 x (91 pixelx) x (31 pixels) x (31 pixels)


"""

os.chdir(os.path.join(folder_name,'ground_truth_data'))

labeled_files = glob.glob('*selected_cells.npy')
zips = glob.glob('*selected_cells.zip')

if len(labeled_files) == 0:
    print('Extracting zip files of ground truth data ...')
    for zip in zips:
        with zipfile.ZipFile(zip, mode="r") as archive:
            archive.extractall()
    print("Unzipping finished.")

raw_data_all = np.zeros((91,31,31,100*len(labeled_files)))
labels_all = np.zeros((100*len(labeled_files,)))
for i,labeled_file in enumerate(labeled_files):
  index = (i-1)*100
  raw_data = np.load(labeled_file)
  raw_data_all[:,:,:,(i*100):(i+1)*100] = raw_data
  labels = np.load(labeled_file[:-4]+'_scored.npy')
  labels_all[(i*100):(i+1)*100] = labels

labels_allX = np.zeros((labels_all.shape[0],2))
labels_allX[:,0] = labels_all == 0
labels_allX[:,1] = labels_all == 1

temp = np.expand_dims(raw_data_all,4)
temp = np.moveaxis(temp,3,0)
input_data = np.moveaxis(temp,4,1)






# Define Convolutional Neural Networks

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, (7,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (7,3))
        self.fc1 = nn.Linear(1728, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_xy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



"""

Set up model folder

"""

model_folder = os.path.join(folder_name,'trained_models')



"""

Test existing models on loaded data ("input_data")

1. Go through all different models (xy, xz, yz)
2. Go through all 5 replicas of each model (index variable "kk")
3. Go through all samples in "input_data" (index variable "k") (inference is done in parallel)
4. Compare with ground truth annotation (if available) and compute "performance" metric

"""



net_xy = Net_xy()
models_xy = glob.glob(os.path.join(model_folder,'Model_xy*'))

performance = np.zeros((len(models_xy),1))
for kk,model in enumerate(models_xy):
  
  net_xy.load_state_dict(torch.load(model))
  
  input_data_xy = copy.deepcopy(input_data[:,:,45,:,:])
  for k in np.arange(input_data_xy.shape[0]):
    input_data_xy[k,:,:,:] = (input_data_xy[k,:,:,:] - np.nanmean(input_data_xy[k,:,:,:]))/np.nanstd(input_data_xy[k,:,:,:])
  inputs = torch.from_numpy(input_data_xy[:,:,:,:])
  labels = torch.from_numpy(labels_allX[:,:].astype(int) )

  net_xy.eval()
  outputs = net_xy(inputs.float())
  
  if kk == 0:
    outputs_test_all = outputs.detach().numpy()
  else:
    outputs_test_all += outputs.detach().numpy()
  
  # Quantify performance (% correct decisions)
  resultoutputs_test = np.zeros((outputs.shape[0],2))
  for j in np.arange(outputs.shape[0]):
    resultoutputs_test[j,0] = outputs[j,0] > outputs[j,1]
    resultoutputs_test[j,1] = labels[j,0] > labels[j,1]
  performance[kk] = np.sum(resultoutputs_test[:,0] == resultoutputs_test[:,1])/resultoutputs_test.shape[0]


net_xz = Net()
models_xz = glob.glob(os.path.join(model_folder,'Model_xz*'))

performance = np.zeros((len(models_xz),1))
for kk,model in enumerate(models_xz):
  
  net_xz.load_state_dict(torch.load(model))
  
  input_data_xz = copy.deepcopy(input_data[:,:,:,15,:])
  for k in np.arange(input_data_xz.shape[0]):
    input_data_xz[k,:,:,:] = (input_data_xz[k,:,:,:] - np.nanmean(input_data_xz[k,:,:,:]))/np.nanstd(input_data_xz[k,:,:,:])   
  inputs = torch.from_numpy(input_data_xz[:,:,:,:])
  labels = torch.from_numpy(labels_allX[:,:].astype(int) )

  net_xz.eval()
  outputs = net_xz(inputs.float())
  
  if 0:
    outputs_test_all = outputs.detach().numpy()
  else:
    outputs_test_all += outputs.detach().numpy()
  
  # Quantify performance (% correct decisions)
  resultoutputs_test = np.zeros((outputs.shape[0],2))
  for j in np.arange(outputs.shape[0]):
    resultoutputs_test[j,0] = outputs[j,0] > outputs[j,1]
    resultoutputs_test[j,1] = labels[j,0] > labels[j,1]
  performance[kk] = np.sum(resultoutputs_test[:,0] == resultoutputs_test[:,1])/resultoutputs_test.shape[0]


net_yz = Net()
models_yz = glob.glob(os.path.join(model_folder,'Model_yz*'))

performance = np.zeros((len(models_yz),1))
for kk,model in enumerate(models_yz):
  
  net_yz.load_state_dict(torch.load(model))
  
  input_data_yz = copy.deepcopy(input_data[:,:,:,:,15])
  for k in np.arange(input_data_yz.shape[0]):
    input_data_yz[k,:,:,:] = (input_data_yz[k,:,:,:] - np.nanmean(input_data_yz[k,:,:,:]))/np.nanstd(input_data_yz[k,:,:,:])    
  inputs = torch.from_numpy(input_data_yz[:,:,:,:])
  labels = torch.from_numpy(labels_allX[:,:].astype(int) )

  net_yz.eval()
  outputs = net_yz(inputs.float())
  
  if 0:
    outputs_test_all = outputs.detach().numpy()
  else:
    outputs_test_all += outputs.detach().numpy()
  
  # Quantify performance (% correct decisions)
  resultoutputs_test = np.zeros((outputs.shape[0],2))
  for j in np.arange(outputs.shape[0]):
    resultoutputs_test[j,0] = outputs[j,0] > outputs[j,1]
    resultoutputs_test[j,1] = labels[j,0] > labels[j,1]
  performance[kk] = np.sum(resultoutputs_test[:,0] == resultoutputs_test[:,1])/resultoutputs_test.shape[0]






""" Quantify performance (% correct decisions) """


  
resultoutputs_test = np.zeros((outputs.shape[0],2))
evaluation_metric = np.zeros((outputs.shape[0],1))
for j in np.arange(outputs.shape[0]):
  evaluation_metric[j] = outputs_test_all[j,1] - outputs_test_all[j,0]
  resultoutputs_test[j,0] = evaluation_metric[j] < 0
  resultoutputs_test[j,1] = labels[j,0] > labels[j,1]
performance_ensemble = np.sum(resultoutputs_test[:,0] == resultoutputs_test[:,1])/resultoutputs_test.shape[0]


# results_file_binarization = os.path.join(folder_name,'Evaluation_metrics.npy')
# np.save(results_file_binarization,evaluation_metric)





