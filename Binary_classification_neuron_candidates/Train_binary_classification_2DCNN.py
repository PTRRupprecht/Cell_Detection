#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 01:10:07 2023

@author: Peter Rupprecht, ptrrupprecht+celldetection@gmail.com

Code to read in training data for cell classification (cell vs. non-cell) from a 3D local volume (31x31x91 pixels)

Code for the 2D CNN architecture was inspired by this classic model: https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py

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

Load training and validation data

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

Train models for xy

"""

augmentation = 0

while len(glob.glob(os.path.join(model_folder,'Model_xy_aug_*'))) < 5:

  # Initialize network, loss function and optimizer
  net_0 = Net_xy()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adagrad(net_0.parameters())
  
  # Training and validation data
  training_data =  np.random.choice(1900,1800,replace=False)
  validation_data = np.setxor1d(np.arange(1900),training_data)
  
  # Track loss and performance
  track_loss0 = np.zeros((2500,2))
  counter = 0
  aborting = 0
  # Run while loop until the performance is good enough 
  while aborting == 0:
      
      # Extract training data
      selected_samples = np.random.randint(0,len(training_data),10)
      input_data_xy = copy.deepcopy(input_data[training_data[selected_samples],:,45,:,:])
      for k in np.arange(input_data_xy.shape[0]):
        input_data_xy[k,:,:,:] = (input_data_xy[k,:,:,:] - np.nanmean(input_data_xy[k,:,:,:]))/np.nanstd(input_data_xy[k,:,:,:])
        if augmentation:
          if np.random.rand() > 0.75:
            input_data_xy[k,:,:,:] = input_data_xy[k,:,::-1,:]
          if np.random.rand() > 0.75:
            input_data_xy[k,:,:,:] = input_data_xy[k,:,:,::-1]
      inputs = torch.from_numpy(input_data_xy[:,:,:,:])
      labels = torch.from_numpy(labels_allX[training_data[selected_samples],:].astype(int) )
      
      # Zero the parameter gradients
      optimizer.zero_grad()
      
      # Train network with backprop
      net_0.train()
      outputs = net_0(inputs.float())
      loss = criterion(outputs.double(), labels.double())
      loss.backward()
      optimizer.step()
      
      # Get statistics for training loss
      running_loss = loss.item()
      
      # Extract validation data
      input_data_xy_test = copy.deepcopy(input_data[validation_data,:,45,:,:])
      for k in np.arange(input_data_xy_test.shape[0]):
        input_data_xy_test[k,:,:,:] = (input_data_xy_test[k,:,:,:] - np.nanmean(input_data_xy_test[k,:,:,:]))/np.nanstd(input_data_xy_test[k,:,:,:])
      inputs_test = torch.from_numpy(input_data_xy_test[:,:,:,:])
      labels_test = torch.from_numpy(labels_allX[validation_data,:].astype(int) )
      
      # Validate on entire validation dataset
      net_0.eval()
      outputs_test = net_0(inputs_test.float())
      loss_test = criterion(outputs_test.double(), labels_test.double())
      
      # Quantify performance (% correct decisions)
      resultoutputs_test = np.zeros((outputs_test.shape[0],2))
      for j in np.arange(outputs_test.shape[0]):
        resultoutputs_test[j,0] = outputs_test[j,0] > outputs_test[j,1]
        resultoutputs_test[j,1] = labels_test[j,0] > labels_test[j,1]
      performance = np.sum(resultoutputs_test[:,0] == resultoutputs_test[:,1])/resultoutputs_test.shape[0]
      
      # Print performance metrics
      if np.mod(counter,100) == 0:
        print(str(running_loss)+' '+str(loss_test)+' '+str(performance))
      
      # Keep track of while loop iterations
      counter += 1
      
      # Save performance metrics for later inspection
      track_loss0[counter,0] = loss_test
      track_loss0[counter,1] = performance
      
      # Stop while loop under certain conditions
      if counter > 1500: # Training not successful
        
        aborting = 1
        
        print('No successful model generated!\n')
        
      if counter > 50 and performance > 0.93: # Training successful
        
        model_count = len(glob.glob(os.path.join(model_folder,'Model_xy_aug_*')))
        
        torch.save(net_0.state_dict(),os.path.join(model_folder,'Model_xy_aug_'+str(model_count)+'.pt'))
        
        aborting = 1
        
        print('Generated model successfully!\n')


"""  Evaluate ensemble performance

performanceX = np.zeros((5,))
for kk in np.arange(5):
  net_0.load_state_dict(torch.load(os.path.join(model_folder,'Model_xy_'+str(kk)+'.pt')))
  
  net_0.eval()
  outputs_test = net_0(inputs_test.float())
  
  if kk == 0:
    outputs_test_all = outputs_test.detach().numpy()
  else:
    outputs_test_all += outputs_test.detach().numpy()
  
  loss_test = criterion(outputs_test.double(), labels_test.double())
  
  # Quantify performance (% correct decisions)
  resultoutputs_test = np.zeros((outputs_test.shape[0],2))
  for j in np.arange(outputs_test.shape[0]):
    resultoutputs_test[j,0] = outputs_test[j,0] > outputs_test[j,1]
    resultoutputs_test[j,1] = labels_test[j,0] > labels_test[j,1]
  performance = np.sum(resultoutputs_test[:,0] == resultoutputs_test[:,1])/resultoutputs_test.shape[0]
  
  performanceX[kk] = performance


# Quantify performance (% correct decisions)
resultoutputs_test = np.zeros((outputs_test.shape[0],2))
for j in np.arange(outputs_test.shape[0]):
  resultoutputs_test[j,0] = outputs_test[j,0] > outputs_test[j,1]
  resultoutputs_test[j,1] = labels_test[j,0] > labels_test[j,1]
performance_ensemble = np.sum(resultoutputs_test[:,0] == resultoutputs_test[:,1])/resultoutputs_test.shape[0]

"""


""" Model xz"""



while len(glob.glob(os.path.join(model_folder,'Model_xz_aug_*'))) < 5:

  # Initialize network, loss function and optimizer
  net = Net()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adagrad(net.parameters())
  
  # Training and validation data
  training_data =  np.random.choice(1900,1800,replace=False)
  validation_data = np.setxor1d(np.arange(1900),training_data)
  
  # Track loss and performance
  track_loss0 = np.zeros((2500,2))
  counter = 0
  aborting = 0
  # Run while loop until the performance is good enough 
  while aborting == 0:
      
      # Extract training data
      selected_samples = np.random.randint(0,len(training_data),10)
      input_data_xz = copy.deepcopy(input_data[training_data[selected_samples],:,:,15,:])
      for k in np.arange(input_data_xz.shape[0]):
        input_data_xz[k,:,:,:] = (input_data_xz[k,:,:,:] - np.nanmean(input_data_xz[k,:,:,:]))/np.nanstd(input_data_xz[k,:,:,:])
        if augmentation:
          if np.random.rand() > 0.75:
            input_data_xz[k,:,:,:] = input_data_xz[k,:,:,::-1]
      inputs = torch.from_numpy(input_data_xz[:,:,:,:])
      labels = torch.from_numpy(labels_allX[training_data[selected_samples],:].astype(int) )
      
      # Zero the parameter gradients
      optimizer.zero_grad()
      
      # Train network with backprop
      net.train()
      outputs = net(inputs.float())
      loss = criterion(outputs.double(), labels.double())
      loss.backward()
      optimizer.step()
      
      # Get statistics for training loss
      running_loss = loss.item()
      
      input_data_xz_test = copy.deepcopy(input_data[validation_data,:,:,15,:])
      for k in np.arange(input_data_xz_test.shape[0]):
        input_data_xz_test[k,:,:,:] = (input_data_xz_test[k,:,:,:] - np.nanmean(input_data_xz_test[k,:,:,:]))/np.nanstd(input_data_xz_test[k,:,:,:])

      inputs_test = torch.from_numpy(input_data_xz_test[:,:,:,:])
      labels_test = torch.from_numpy(labels_allX[validation_data,:].astype(int) )
      
      # Validate on entire validation dataset
      net.eval()
      outputs_test = net(inputs_test.float())
      loss_test = criterion(outputs_test.double(), labels_test.double())
      
      # Quantify performance (% correct decisions)
      resultoutputs_test = np.zeros((outputs_test.shape[0],2))
      for j in np.arange(outputs_test.shape[0]):
        resultoutputs_test[j,0] = outputs_test[j,0] > outputs_test[j,1]
        resultoutputs_test[j,1] = labels_test[j,0] > labels_test[j,1]
      performance = np.sum(resultoutputs_test[:,0] == resultoutputs_test[:,1])/resultoutputs_test.shape[0]
      
      # Print performance metrics
      if np.mod(counter,100) == 0:
        print(str(running_loss)+' '+str(loss_test)+' '+str(performance))
      
      # Keep track of while loop iterations
      counter += 1
      
      # Save performance metrics for later inspection
      track_loss0[counter,0] = loss_test
      track_loss0[counter,1] = performance
      
      # Stop while loop under certain conditions
      if counter > 1500: # Training not successful
        
        aborting = 1
        
        print('No successful model generated!\n')
        
      if counter > 50 and performance > 0.93: # Training successful
        
        model_count = len(glob.glob(os.path.join(model_folder,'Model_xz_aug_*')))
        
        torch.save(net.state_dict(),os.path.join(model_folder,'Model_xz_aug_'+str(model_count)+'.pt'))
        
        aborting = 1
        
        print('Generated model successfully!\n')
        
        
        
        
        
""" Model yz"""



while len(glob.glob(os.path.join(model_folder,'Model_yz_aug_*'))) < 5:

  # Initialize network, loss function and optimizer
  net = Net()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adagrad(net.parameters())
  
  # Training and validation data
  training_data =  np.random.choice(1900,1800,replace=False)
  validation_data = np.setxor1d(np.arange(1900),training_data)
  
  # Track loss and performance
  track_loss0 = np.zeros((2500,2))
  counter = 0
  aborting = 0
  # Run while loop until the performance is good enough 
  while aborting == 0:
      
      # Extract training data
      selected_samples = np.random.randint(0,len(training_data),10)
      input_data_xz = copy.deepcopy(input_data[training_data[selected_samples],:,:,:,15])
      for k in np.arange(input_data_xz.shape[0]):
        input_data_xz[k,:,:,:] = (input_data_xz[k,:,:,:] - np.nanmean(input_data_xz[k,:,:,:]))/np.nanstd(input_data_xz[k,:,:,:])
        if augmentation:
          if np.random.rand() > 0.75:
            input_data_xz[k,:,:,:] = input_data_xz[k,:,:,::-1]
      inputs = torch.from_numpy(input_data_xz[:,:,:,:])
      labels = torch.from_numpy(labels_allX[training_data[selected_samples],:].astype(int) )
      
      # Zero the parameter gradients
      optimizer.zero_grad()
      
      # Train network with backprop
      net.train()
      outputs = net(inputs.float())
      loss = criterion(outputs.double(), labels.double())
      loss.backward()
      optimizer.step()
      
      # Get statistics for training loss
      running_loss = loss.item()
      
      input_data_xz_test = copy.deepcopy(input_data[validation_data,:,:,:,15])
      for k in np.arange(input_data_xz_test.shape[0]):
        input_data_xz_test[k,:,:,:] = (input_data_xz_test[k,:,:,:] - np.nanmean(input_data_xz_test[k,:,:,:]))/np.nanstd(input_data_xz_test[k,:,:,:])

      inputs_test = torch.from_numpy(input_data_xz_test[:,:,:,:])
      labels_test = torch.from_numpy(labels_allX[validation_data,:].astype(int) )
      
      # Validate on entire validation dataset
      net.eval()
      outputs_test = net(inputs_test.float())
      loss_test = criterion(outputs_test.double(), labels_test.double())
      
      # Quantify performance (% correct decisions)
      resultoutputs_test = np.zeros((outputs_test.shape[0],2))
      for j in np.arange(outputs_test.shape[0]):
        resultoutputs_test[j,0] = outputs_test[j,0] > outputs_test[j,1]
        resultoutputs_test[j,1] = labels_test[j,0] > labels_test[j,1]
      performance = np.sum(resultoutputs_test[:,0] == resultoutputs_test[:,1])/resultoutputs_test.shape[0]
      
      # Print performance metrics
      if np.mod(counter,100) == 0:
        print(str(running_loss)+' '+str(loss_test)+' '+str(performance))
      
      # Keep track of while loop iterations
      counter += 1
      
      # Save performance metrics for later inspection
      track_loss0[counter,0] = loss_test
      track_loss0[counter,1] = performance
      
      # Stop while loop under certain conditions
      if counter > 1500: # Training not successful
        
        aborting = 1
        
        print('No successful model generated!\n')
        
      if counter > 50 and performance > 0.93: # Training successful
        
        model_count = len(glob.glob(os.path.join(model_folder,'Model_yz_aug_*')))
        
        torch.save(net.state_dict(),os.path.join(model_folder,'Model_yz_aug_'+str(model_count)+'.pt'))
        
        aborting = 1
        
        print('Generated model successfully!\n')
        
        
        
        



