#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Tue Feb 28 22:43:03 2023

@author: Peter Rupprecht, ptrrupprecht+celldetection@gmail.com

# ============================================================================
# Script for merging (elimination of splitter cells)
# 
# The script takes each detected point as a candidate and looks for neighboring
# pixels that are brighter. It continues to do so (gradient ascent) until a peak is reached
# This peak is taken as the new location of the cell
# Then, duplicate cell locations (same peak reached) are taken as a single cell
# Finally, a deep-learning based classifier is applied to only keep cells and discard non-cells
# =============================================================================


Please change "folder_name" according to the local paths on your computer

Also, change the filenames for "parent_folder_patients" and "filenames_patients" so that they match the raw data and the list of cells detected via ClearMap.

"""


folder_name = '/home/helios/Desktop/Cell_Detection/Binary_classification_neuron_candidates'



# import packages
import numpy as np
import glob, os, time, tifffile, copy
from  scipy import ndimage
from skimage import segmentation
from skimage.morphology import binary_dilation, ball
from skimage import measure
#from numba import jit

import matplotlib.pyplot as plt


# dependencies for the classifier
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import os, glob
import numpy as np

import copy

"""

Define Convolutional Neural Networks


"""


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


# indicate folder and file names

parent_folder_patients = '/media/helios/SpeedDisk/FCD_Stitched/FCD_SampleAnalysis_Crop/FCD_Patients/'
filenames_patients = glob.glob(parent_folder_patients+'/FCD_CX_2.*_NeuN-Cy3/*NeuN-*-*.tif')

filenames_all = filenames_patients

for file_index,filename in enumerate(filenames_all):
    
  BaseDirectory = os.path.dirname(filename)
    
  cFosFile = os.path.join(filename)
  point_list = cFosFile[:-4]+'_cells-allpoints.npy'
  
  file_size = os.path.getsize(cFosFile)
  print('Processing the next file. Progress: '+str(100*(file_index+1)/len(filenames_all))+'%')
  file_size = np.round(file_size/1024**3)
  print("File Size is :", file_size, "GB")
  
  os.chdir(BaseDirectory)
  print('Processing the following folder: '+BaseDirectory)
  
  if os.path.isfile(point_list) and file_index >= 0 and file_size < 45:
  
    raw_data_filename = cFosFile
  
    
    detected_points_filename = point_list # result from ClearMap
    
    # load data
    raw_data = np.array(tifffile.imread(raw_data_filename))
    detected_points = np.load(detected_points_filename)
    
    detected_points_old = copy.deepcopy(detected_points)
    detected_points_new = detected_points
    
    print('Process '+str(len(detected_points))+' points with gradient-based peak-finder.')
    
    
    
    # gradient search takes about 80 s for 50k points, i.e., typically around 20-40 min for a sample with 1M-2M  detected points
    
    # go through all cell candidates
    for points in np.arange(detected_points.shape[0]):
        
      if np.mod(points,50000) == 0:
        print(str(points)+' out of '+str(detected_points.shape[0]))
      
      starting_point0 = detected_points[points,:].astype(int)[::-1]
      
      cube_size = 15
      low_z0 = max(0,starting_point0[0]-cube_size*3)  # anisotropy of search environment^
      high_z0 = min(starting_point0[0]+cube_size*3+1,raw_data.shape[0])
      low_y0 = max(0,starting_point0[1]-cube_size)
      high_y0 = min(starting_point0[1]+cube_size+1,raw_data.shape[1])
      low_x0 = max(0,starting_point0[2]-cube_size)
      high_x0 = min(starting_point0[2]+cube_size+1,raw_data.shape[2])
      larger_environment = np.array(raw_data[low_z0:high_z0,low_y0:high_y0,low_x0:high_x0]).astype(float)
      
      
      larger_environmentX = ndimage.gaussian_filter(larger_environment.astype(float),[2.5,1,1])  # anisotropy of smoothing
      
      focus_point = np.zeros((3,),dtype='int')
      for dimension in np.arange(3):
          if dimension == 0: # anisotropy of search environment^
              scale = 3
          else:
              scale = 1
          if starting_point0[dimension] - cube_size*scale < 0:
              focus_point[dimension] = starting_point0[dimension]
          else:
              focus_point[dimension] = cube_size*scale
      
      focus_point_initial = copy.deepcopy(focus_point)
      
      # until no further increase of fluorescence seen

      stop_criterion = 0
      while not stop_criterion:
      
        # extract local environment of seed point
        
        low_z = max(0,focus_point[0]-1)
        high_z = min(focus_point[0]+2,larger_environmentX.shape[0])
        low_y = max(0,focus_point[1]-1)
        high_y = min(focus_point[1]+2,larger_environmentX.shape[1])
        low_x = max(0,focus_point[2]-1)
        high_x = min(focus_point[2]+2,larger_environmentX.shape[2])

        environment = np.array(larger_environmentX[low_z:high_z,low_y:high_y,low_x:high_x]).astype(float)

        # find location of maximum value in neighborhood
        indices = np.array(np.unravel_index(np.argmax(environment), environment.shape))
        
        # backup
        old_focus_point = copy.deepcopy(focus_point)
        # update
        focus_point = ([low_z,low_y,low_x] + indices).astype(int)
        
        
        # if stopping criterion reached
        if larger_environmentX[focus_point[0],focus_point[1],focus_point[2]] == larger_environmentX[old_focus_point[0],old_focus_point[1],old_focus_point[2]]:
          stop_criterion = 1
      
      
      # write back in reverse order (z,y,x) --> (x,y,z)
      detected_points_new[points,:] = focus_point[::-1] - focus_point_initial[::-1] + starting_point0[::-1]
    
#      print(np.sqrt(np.sum((focus_point - focus_point_initial)**2)))
    
    # discard duplicates (cell seeds that arrive at the same local maximum)
    detected_points_new = np.unique(detected_points_new,axis = 0)
    print('Found and kept '+str(len(detected_points_new))+' points.')
    
    
    # Go through batches and apply pretrained binary classifier
    accept_cell_candidates = np.zeros((detected_points_new.shape[0],))
    confidence_acceptance_all = np.zeros((detected_points_new.shape[0],))
    batch_size = 5000
    nb_chunks = np.ceil(len(detected_points_new)/batch_size)
    
    for batch in np.arange(nb_chunks):
      
      print('Batch '+str(int(batch+1))+' out of '+str(int(nb_chunks)))
      
      # get indices of neuron candidates for this batch
      indices = np.arange(int(batch*batch_size),int(min(batch*batch_size + batch_size,len(detected_points_new)) ))
      
      # pre-allocate array
      all_environments = np.zeros((len(indices),91,31,31))
      
      for k in np.arange(len(indices)):
          
        starting_point0 = detected_points_new[indices[k],:].astype(int)[::-1]
        
        # get boundaries of the cube
        cube_size = 15
        low_z0 = max(0,starting_point0[0]-cube_size*3)  # anisotropy of search environment^
        high_z0 = min(starting_point0[0]+cube_size*3+1,raw_data.shape[0])
        low_y0 = max(0,starting_point0[1]-cube_size)
        high_y0 = min(starting_point0[1]+cube_size+1,raw_data.shape[1])
        low_x0 = max(0,starting_point0[2]-cube_size)
        high_x0 = min(starting_point0[2]+cube_size+1,raw_data.shape[2])

        larger_environment = np.array(raw_data[low_z0:high_z0,low_y0:high_y0,low_x0:high_x0]).astype(float)
        
        # in case the cube is cut off by the stack boundaries, pad with a reflection
        lz_plus = -min(0,starting_point0[0]-cube_size*3)
        ly_plus = -min(0,starting_point0[1]-cube_size)
        lx_plus = -min(0,starting_point0[2]-cube_size)
        hz_plus = -min(0,raw_data.shape[0] - (starting_point0[0]+cube_size*3+1))
        hy_plus = -min(0,raw_data.shape[1] - (starting_point0[1]+cube_size+1))
        hx_plus = -min(0,raw_data.shape[2] - (starting_point0[2]+cube_size+1))

        # perform padding
        larger_environment_padded = np.pad(larger_environment, [(lz_plus,hz_plus), (ly_plus,hy_plus), (lx_plus,hx_plus)], 'reflect')
        
        all_environments[k,:,:,:] = larger_environment_padded
      
      # adapt dimensionality of the matrix to create a suited input for the network
      all_environments = np.expand_dims(all_environments,1)
    
      # folder where pretrained networks are stored
      model_folder = os.path.join(folder_name,'trained_models')

      # get classification from the xy-view
      net_xy = Net_xy()
      models_xy = glob.glob(os.path.join(model_folder,'Model_xy*'))
      
      for kk,model in enumerate(models_xy):
        
        net_xy.load_state_dict(torch.load(model))
        
        input_data_xy = copy.deepcopy(all_environments[:,:,45,:,:])
        for k in np.arange(input_data_xy.shape[0]):
          input_data_xy[k,:,:,:] = (input_data_xy[k,:,:,:] - np.nanmean(input_data_xy[k,:,:,:]))/np.nanstd(input_data_xy[k,:,:,:])    
        inputs = torch.from_numpy(input_data_xy[:,:,:,:])
      
        net_xy.eval()
        outputs = net_xy(inputs.float())
        
        if kk == 0:
          outputs_test_all = outputs.detach().numpy()
        else:
          outputs_test_all += outputs.detach().numpy()
      
      # get classification from the xz-view
      net_xz = Net()
      models_xz = glob.glob(os.path.join(model_folder,'Model_xz*'))
      
      for kk,model in enumerate(models_xz):
        
        net_xz.load_state_dict(torch.load(model))
        
        input_data_xz = copy.deepcopy(all_environments[:,:,:,15,:])
        for k in np.arange(input_data_xz.shape[0]):
          input_data_xz[k,:,:,:] = (input_data_xz[k,:,:,:] - np.nanmean(input_data_xz[k,:,:,:]))/np.nanstd(input_data_xz[k,:,:,:])    
        inputs = torch.from_numpy(input_data_xz[:,:,:,:])
      
        net_xz.eval()
        outputs = net_xz(inputs.float())
        
        if 0:
          outputs_test_all = outputs.detach().numpy()
        else:
          outputs_test_all += outputs.detach().numpy()
      
      # get classification from the yz-view
      net_yz = Net()
      models_yz = glob.glob(os.path.join(model_folder,'Model_yz*'))
      
      performance = np.zeros((len(models_yz),1))
      for kk,model in enumerate(models_yz):
        
        net_yz.load_state_dict(torch.load(model))
        
        input_data_yz = copy.deepcopy(all_environments[:,:,:,:,15])
        for k in np.arange(input_data_yz.shape[0]):
          input_data_yz[k,:,:,:] = (input_data_yz[k,:,:,:] - np.nanmean(input_data_yz[k,:,:,:]))/np.nanstd(input_data_yz[k,:,:,:])    
        inputs = torch.from_numpy(input_data_yz[:,:,:,:])
      
        net_yz.eval()
        outputs = net_yz(inputs.float())
        
        if 0:
          outputs_test_all = outputs.detach().numpy()
        else:
          outputs_test_all += outputs.detach().numpy()

      # combine classifications and use cutoff (chosen by manual inspection)
      confidence_acceptance = outputs_test_all[:,1] - outputs_test_all[:,0]
      
      accept_cell_candidates_this_batch = confidence_acceptance > 30
      
      accept_cell_candidates[indices] = accept_cell_candidates_this_batch
      confidence_acceptance_all[indices] = confidence_acceptance

    # keep only points classified as neurons by the 2D-CNN
    
    confidence_acceptance_all_selected = confidence_acceptance_all[accept_cell_candidates.astype(np.bool)]
    detected_points_new_reduced = detected_points_new[accept_cell_candidates.astype(np.bool)]
    print('Kept '+str(len(detected_points_new_reduced))+' points after thresholding.')
        
    
    points = detected_points_new_reduced
    
    
    # remove duplicates (adjacent labels of the same neuron)
    
    for point_index in np.arange(len(points)):
      
      if np.mod(point_index,50000) == 0:
        print('Progress for proximity pruning: '+str(point_index/len(points)*100))  
      
      point = points[point_index,:]
      
      dist = np.linalg.norm(points-point,axis=1)
      discard = np.sum(np.logical_and(dist<=5,dist>0.5))
      if discard > 0:
        points[point_index,:] = [-10,-10,-10]
    
    correct_indices = points[:,1] > -10
    
    points = points[correct_indices,:]
    confidence_acceptance_all_selected = confidence_acceptance_all_selected[correct_indices,]
    
    
    # make a test stack for visualization
    if 1:
        print('Make a tif-file to check quality of cell detection.')
    
        # make a tif file to check back
        detected_points_new_int = points.astype(int)
        points_data = np.zeros((raw_data.shape))
        for k in np.arange(detected_points_new_int.shape[0]):
          points_data[detected_points_new_int[k,2],detected_points_new_int[k,1],detected_points_new_int[k,0]] += 1
        
        points_data = points_data.astype(raw_data.dtype) * raw_data.max();
        raw_data.shape = raw_data.shape + (1,);
        points_data.shape =  points_data.shape + (1,);
        points_data = np.concatenate((raw_data, points_data), axis  = 3);
        
        import tifffile as tiff
        
        filename = cFosFile[:-4]+'check_cells_gradient_plus_classifier_undoubled.tif'
        tiff.imsave(filename, points_data.transpose([0,1,2,3]), photometric = 'minisblack',  planarconfig = 'contig', bigtiff = True)
  
        print('Save detected cells to a numpy file.')

    else:
        print('No file saved.')
    
    try:
        del points_data
        del raw_data
        del raw_dataX
        del X
        del marker_stack
    except:
        pass
    # save results to a numpy or mat file
    print('Save detected cells to a numpy file.')

    
    
    filename_points_after_gradient_corrected = cFosFile[:-4]+'_points_gradient_plus_classifier_corrected.mat'
    
    import scipy.io as sio

    sio.savemat(filename_points_after_gradient_corrected,{'points':points,'confidence_acceptance_all_selected':confidence_acceptance_all_selected})
  
    print('Done with this stack.\n')


  else:

    print('These files have not been processed with Ilastik before; or were manually excluded from processing.\n')


  


