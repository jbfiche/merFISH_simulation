#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:13:21 2021

@author: fiche
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from glob import glob
from tqdm import tqdm
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

model_dir = '/mnt/PALM_dataserv/DATA/JB/2021/Data_single_loci/Simulation_3D/Simulated_data/Simulation_06_05_21/models/'
test_dir = '/mnt/PALM_dataserv/DATA/JB/2021/Data_single_loci/Data_test_small/deconvolved/'
model_name = 'stardist_20210506_simu_deconvolved'

repeat = 10

# Load the model
# --------------

model = StarDist3D(None, name=model_name, basedir=model_dir)
limit_gpu_memory(None, allow_growth=True)

# Look for the data. Depending whether the images are large (>1000x1000pix) or 
# small, the procedure for the image reconstruction is not the same. 
# ------------------------------------------------------------------

os.chdir(test_dir)
X = sorted(glob('*_raw.tif'))
axis_norm = (0,1,2)

t_av = np.zeros(len(X))
t_std = np.zeros(len(X))
nobjects = np.zeros(len(X))

for n_im in range(len(X)):
    im_name = Path(X[n_im]).stem
    print(im_name)
    
    im = imread(X[n_im])
    im = normalize(im,1,99.8,axis=axis_norm)
    Lx = im.shape[1]
    t = np.zeros(repeat)
    
    for n_repeat in tqdm(range(repeat)):
        t0= time.time()
        
        if Lx<1000:
            labels, details = model.predict_instances(im)
            
        else:
            resizer = PadAndCropResizer()
            axes = 'ZYX'
            
            im = resizer.before(im, axes, model._axes_div_by(axes))
            labels, polys = model.predict_instances(im, n_tiles=(1,8,8))
            labels = resizer.after(labels, axes)
            
        t1 = time.time() - t0
        t[n_repeat]=t1
        
    print("Time elapsed: " + str(np.median(t)) + "+/-" + str(np.std(t)))
    print(t)
    
    nobjects[n_im] = np.max(np.unique(labels))
    t_av[n_im] = np.median(t)
    t_std[n_im] = np.std(t)

    mask = np.array(labels>0, dtype=int)
    
    label_name = im_name + '_label.tif'
    save_tiff_imagej_compatible(label_name, labels, axes='ZYX')
    
    mask_name = im_name + '_mask.tif'
    save_tiff_imagej_compatible(mask_name, mask, axes='ZYX')
    
#print("The average computation time is " + str(np.mean(t)) + " +/- " + str(np.std(t)))

idx = np.argsort(nobjects)

plt.errorbar(nobjects[idx], t_av[idx], t_std[idx], label='both limits (default)')
plt.xlabel('N detected objects')
plt.ylabel('Computation time (s)')

fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Computation_time.png')