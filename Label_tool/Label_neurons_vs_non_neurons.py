#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:10:09 2023

@author: Peter Rupprecht, ptrrupprecht+celldetection@gmail.com

Code for a handy user interface to annotate cell candidates

Please change "folder_name" according to the local paths on your computer

"""


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import zipfile



folder_name = '/home/helios/Desktop/Cell_Detection/Binary_classification_neuron_candidates'

os.chdir(os.path.join(folder_name,'ground_truth_data'))

labeled_files = glob.glob('*selected_cells.npy')
zips = glob.glob('*selected_cells.zip')

if len(labeled_files) == 0:
    print('Extracting zip files of ground truth data ...')
    for zip in zips:
        with zipfile.ZipFile(zip, mode="r") as archive:
            archive.extractall()
    print("Unzipping finished.")


"""
Define user interface
"""

class IndexTracker(object):
    def __init__(self, ax, X, kkk):
        
        self.selection_done = 0
        self.show_cell = 1
        self.active_window = 1
        self.keep = 1
        
        self.progress = kkk
        
        self.ax = ax[0]
        self.ax2 = ax[1]
#        ax[0].set_title('use scroll wheel to navigate images')

        self.X = X
        self.rows, self.cols, self.slices = X.shape
        
        self.ind = self.slices//2

        self.im = ax[0].imshow(self.X[:, :, self.ind])

        self.ind2 = self.rows//2

        self.im2 = ax[1].imshow(self.X[self.ind2, :, :])

        self.update()
        
    def key_press(self, event):
        if event.key =='b':
            self.active_window  = np.abs(self.active_window - 1)
        if event.key =='d':
            self.keep  = 0
            self.selection_done = 1
        if event.key =='g':
            self.keep  = 1
            self.selection_done = 1
        
    def onscroll(self, event):
#        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            if self.active_window > 0.5:
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind2 = (self.ind2 + 1) % self.rows
        else:
            if self.active_window > 0.5:
                self.ind = (self.ind - 1) % self.slices
            else:
                self.ind2 = (self.ind2 - 1) % self.rows
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind, weight='bold')
        if self.ind == 15:
            self.ax.yaxis.label.set_color('red')
        else:
            self.ax.yaxis.label.set_color('black')
        self.im2.set_data(self.X[self.ind2, :, :])
        self.ax2.set_ylabel('slice %s' % self.ind2, weight='bold')
        if self.ind2 == 45:
            self.ax2.yaxis.label.set_color('red')
        else:
            self.ax2.yaxis.label.set_color('black')
        
        self.ax2.set_xlabel('Index '+str(kkk), weight='bold')
        
        self.im.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()
        
        

"""
Go through data and save results back to a label file
"""

    
os.chdir('/media/helios/SpeedDisk/FCD_Stitched/Cell_detection_binary_ground_truth/')


filenames = glob.glob('*selected_cells.npy')


for filename in filenames:
    
    if not os.path.exists(filename[:-4]+'_scored_by_person_X.npy'):
    
        all_cell_environments = np.load(filename)
        fig, ax = plt.subplots(1, 2)
        keep_all = np.zeros((all_cell_environments.shape[3],))
        counter = 0
        for kkk in np.arange(all_cell_environments.shape[3]):
            
            environemt = all_cell_environments[:,:,:,kkk]
            
            if counter > 20:
                plt.close()
                plt.pause(0.5)
                fig, ax = plt.subplots(1, 2)
                counter = 0
            
            tracker = IndexTracker(ax, environemt,kkk)
        
            
            cid1 = fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
            cid2 = fig.canvas.mpl_connect('key_press_event', tracker.key_press)
            plt.show()
            
            while tracker.selection_done == 0:
                plt.pause(0.1)
            
            keep_all[kkk] = tracker.keep
            
            manual_labels = keep_all
            counter += 1
        np.save(filename[:-4]+'_scored_by_person_X.npy',manual_labels)
    


    
## Browse through neuron candidates
        

    
    