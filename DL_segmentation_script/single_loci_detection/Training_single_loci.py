#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:37:21 2021

@author: fiche
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
# matplotlib inline
# config InlineBackend.figure_format = 'retina'
import os
import sys

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D

from augmend import Augmend, FlipRot90, Elastic, Identity, IntensityScaleShift, AdditiveNoise, Scale

from tensorflow import keras
from keras.callbacks import TensorBoard  #Visulization of Accuracy and loss

np.random.seed(42)
lbl_cmap = random_label_cmap()

# Define the number of trainings to be performed
# ----------------------------------------------

data_dir_all = ['/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-11_10-54/Training_data_thresh_2_Deconvolved',
                '/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-11_10-54/Training_data_thresh_4_Deconvolved',
                '/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-11_10-54/Training_data_thresh_2_Deconvolved',
                '/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-11_10-54/Training_data_thresh_4_Deconvolved']
            
model_name_all = ['stardist_20210618_simu_deconvolved_thresh_2_01',
                  'stardist_20210618_simu_deconvolved_thresh_4_01',
                  'stardist_20210618_simu_deconvolved_thresh_2_02',
                  'stardist_20210618_simu_deconvolved_thresh_4_02']


for n_training in range(len(data_dir_all)):
        
    data_dir = data_dir_all[n_training]
    model_name = model_name_all[n_training]
        
    os.chdir(data_dir)
    
    # Load the training data
    # ----------------------
    
    X = sorted(glob('train/raw/*raw*.tif'))
    Y = sorted(glob('train/label/*label*.tif'))
    
    X_trn = list(map(imread,X))
    Y_trn = list(map(imread,Y))
    n_channel = 1 if X_trn[0].ndim == 3 else X_trn[0].shape[-1]
    
    # Load the validation/testing data
    # --------------------------------
    
    X = sorted(glob('test/raw/*raw*.tif'))
    Y = sorted(glob('test/label/*label*.tif'))
    
    X_val = list(map(imread,X))
    Y_val = list(map(imread,Y))
    n_channel = 1 if X_val[0].ndim == 3 else X_val[0].shape[-1]
    
    # Normalize the data and calculate the anisotropy of the data
    # -----------------------------------------------------------
    
    axis_norm = (0,1,2)   # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
        sys.stdout.flush()
    
    X_trn = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_trn)]
    Y_trn = [fill_label_holes(y) for y in tqdm(Y_trn)]
    
    X_val = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X_val)]
    Y_val = [fill_label_holes(y) for y in tqdm(Y_val)]
    
    extents = calculate_extents(Y_trn)
    anisotropy = tuple(np.max(extents) / extents)
    
    # Configure the network
    # ---------------------
    
    # 96 is a good default choice (see 1_data.ipynb)
    n_rays = 32
    
    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = True and gputools_available()
    
    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = (1,1,1)
    
    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)
    
    conf = Config3D (
        rays             = rays,
        grid             = grid,
        anisotropy       = anisotropy,
        use_gpu          = use_gpu,
        n_channel_in     = n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size = (30,64,64),
        train_batch_size = 4,
        backbone = 'resnet',
        train_epochs = 500,
        train_steps_per_epoch = 100
    )
    
    # Configurate the gpu
    # -------------------
    
    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory
        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        #limit_gpu_memory(0.8, total_memory=11019)
        # alternatively, try this:
        limit_gpu_memory(None, allow_growth=True)
        
    # Define the network model according to the selected parameters and check that
    # the perceptive field of the network is bigger than the average size of the 
    # objetcs.
    # --------
        
    model = StarDist3D(conf, name=model_name, basedir='models')
    
    median_size = calculate_extents(Y_trn, np.median)
    fov = np.array(model._axes_tile_overlap('ZYX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")
        
        
    # Define the parameters for the data augmentation
    # -----------------------------------------------
    
    elastic_kwargs = dict(axis=(1,2), amount=7, use_gpu=model.config.use_gpu)
    #scale_kwargs = dict(axis=1, amount=1.2, use_gpu=model.config.use_gpu)
    aug = Augmend()
    #aug.add([Scale(order=0,**scale_kwargs),Scale(order=0,**scale_kwargs)], probability=0.25)
    aug.add([FlipRot90(axis=(1,2)),FlipRot90(axis=(1,2))])
    aug.add([Elastic(order=0,**elastic_kwargs),Elastic(order=0,**elastic_kwargs)], probability=0.5)
    aug.add([IntensityScaleShift(scale=(.6,2),shift=(-.2,.2)),Identity()])
    aug.add([AdditiveNoise(sigma=(0.05,0.05)),Identity()], probability=0.25)
    
    def augmenter(x,y):
        return aug([x,y])
    
    
    # Launch the training of the model
    # --------------------------------
    
    model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)   
    model.optimize_thresholds(X_val, Y_val)
    
    
    # # Test the model using the validation data and save the statistics
    # # ----------------------------------------------------------------
    
    # Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
    #           for x in tqdm(X_val)]
    # taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # stats = [matching_dataset(Y_val, Y_val_pred, thresh=t, show_progress=False) for t in tqdm(taus)]

    # fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    # for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
    #     ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    # ax1.set_xlabel(r'IoU threshold $\tau$')
    # ax1.set_ylabel('Metric value')
    # ax1.grid()
    # ax1.legend()
    
    # for m in ('fp', 'tp', 'fn'):
    #     ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    # ax2.set_xlabel(r'IoU threshold $\tau$')
    # ax2.set_ylabel('Number #')
    # ax2.grid()
    # ax2.legend();
    
    # fig_name = data_dir + '/models/' + model_name + '/' + 'Statistics.png'
    # fig.savefig(fig_name)
