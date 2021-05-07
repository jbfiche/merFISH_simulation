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


np.random.seed(42)
lbl_cmap = random_label_cmap()

data_dir = '/home/fiche/Data_pancreas/Annotated_data/data_pancreas/'
model_name = 'stardist_17032021'
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
n_rays = 96

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = True and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
grid = tuple(1 if a > 1.5 else 4 for a in anisotropy)

# Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

conf = Config3D (
    rays             = rays,
    grid             = grid,
    anisotropy       = anisotropy,
    use_gpu          = use_gpu,
    n_channel_in     = n_channel,
    # adjust for your data below (make patch size as large as possible)
    train_patch_size = (28,128,128),
    train_batch_size = 2,
    backbone = 'unet',
    train_epochs = 400,
    train_steps_per_epoch = 100
)

# Configurate the gpu
# -------------------

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    #limit_gpu_
    #memory(0.8)
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

elastic_kwargs = dict(axis=(1,2), amount=10, use_gpu=model.config.use_gpu)
#scale_kwargs = dict(axis=1, amount=1.5, use_gpu=model.config.use_gpu)
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