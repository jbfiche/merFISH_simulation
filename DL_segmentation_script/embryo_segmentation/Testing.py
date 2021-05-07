#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:13:21 2021

@author: fiche
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import os

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.data import PadAndCropResizer
from csbdeep.utils.tf import limit_gpu_memory

from stardist import random_label_cmap
from stardist.models import StarDist3D

np.random.seed(6)
lbl_cmap = random_label_cmap()

# Define the folders where the trained model and the test data are saved
# ----------------------------------------------------------------------

model_dir = '/mnt/PALM_dataserv/DATA/JB/2021/Data_embryo_3D_DAPI/data_embryos/models'
test_dir = '/mnt/PALM_dataserv/DATA/JB/2021/Data_embryo_3D_DAPI/Test'
model_name = 'stardist_12032021'

# Load the model
# --------------

model = StarDist3D(None, name=model_name, basedir=model_dir)
limit_gpu_memory(None, allow_growth=True)

# Look for the data. Depending whether the images are large (>1000x1000pix) or 
# small, the procedure for the image reconstruction is not the same. 
# ------------------------------------------------------------------

os.chdir(test_dir)
X = sorted(glob('*.tif'))

axis_norm = (0,1,2)

for n_im in range(len(X)):
    im_name = Path(X[n_im]).stem
    im = imread(X[n_im])
    im = normalize(im,1,99.8,axis=axis_norm)
    Lx = im.shape[1]
    
    if Lx<1000:
        labels, details = model.predict_instances(im)
        
    else:
        resizer = PadAndCropResizer()
        axes = 'ZYX'
        
        im = resizer.before(im, axes, model._axes_div_by(axes))
        labels, polys = model.predict_instances(im, n_tiles=(1,8,8))
        labels = resizer.after(labels, axes)
    
    mask = np.array(labels>0, dtype=int)
    
    label_name = im_name + '_label.tif'
    save_tiff_imagej_compatible(label_name, labels, axes='ZYX')
    
    mask_name = im_name + '_mask.tif'
    save_tiff_imagej_compatible(mask_name, mask, axes='ZYX')