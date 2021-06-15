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
import shutil

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

model_dir_all = ['/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-11_10-54/Training_data_Deconvolved/models']

model_name_all = ['stardist_20210611_simu_deconvolved_thresh_2']

test_dir_all = [['/mnt/grey/DATA/users/JB/Simulations_3D/Test_data/Real_data/deconvolved']]

#test_dir_all = [['/mnt/grey/DATA/users/JB/Simulations_3D/Test_data/Embryo_0_DPP/deconvolved',
             # '/mnt/grey/DATA/users/JB/Simulations_3D/Test_data/Embryo_2_DOC/deconvolved',
             # '/mnt/grey/DATA/users/JB/Simulations_3D/Test_data/Simulated_data/deconvolved']]

# Indicate whether half of the planes should be removed
# -----------------------------------------------------

Remove_planes = True

# For each model, calculate the segmented images
# ----------------------------------------------

for n_model in range(len(model_dir_all)):

    model_dir = model_dir_all[n_model]
    model_name = model_name_all[n_model]
    print(model_name)

    for n_test in range(len(test_dir_all[n_model])):

        # For the selected model, define the folder where the test data are saved.
        # For each model, creates a specific folder where the results will be saved.
        # --------------------------------------------------------------------------

        test_dir = test_dir_all[n_model][n_test]
        print(test_dir)
        os.chdir(test_dir)

        dest_dir = 'Test_' + model_name
        if os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir)
        os.mkdir(dest_dir)

        # Load the model
        # --------------

        model = StarDist3D(None, name=model_name, basedir=model_dir)
        limit_gpu_memory(None, allow_growth=True)

        # Look for the data. Depending whether the images are large (>1000x1000pix) or
        # small, the procedure for the image reconstruction is not the same.
        # ------------------------------------------------------------------

        X = sorted(glob('*.tif'))
        for n in range(len(X)):
            X[n] = os.path.abspath(X[n])

        axis_norm = (0, 1, 2)

        t_av = np.zeros(len(X))
        t_std = np.zeros(len(X))
        nobjects = np.zeros(len(X))

        # Define the destination folder as the main folder
        # ------------------------------------------------

        os.chdir(dest_dir)

        for n_im in range(len(X)):
            im_name = Path(X[n_im]).stem
            print(im_name)

            im = imread(X[n_im])
            
            if Remove_planes:
                nplanes = im.shape[0]
                im = im[0:nplanes:2,:,:]
            
            print(im.shape)
            im = normalize(im, 1, 99.8, axis=axis_norm)
            Lx = im.shape[1]

            t0 = time.time()
            if Lx < 1000:
                labels, details = model.predict_instances(im)

            else:
                resizer = PadAndCropResizer()
                axes = 'ZYX'

                im = resizer.before(im, axes, model._axes_div_by(axes))
                labels, polys = model.predict_instances(im, n_tiles=(1, 8, 8))
                labels = resizer.after(labels, axes)

            t1 = time.time() - t0
            print("Time elapsed: " + str(t1))

            nobjects[n_im] = np.max(np.unique(labels))
            mask = np.array(labels > 0, dtype=int)

            raw_name = im_name + '.tif'
            save_tiff_imagej_compatible(raw_name, im, axes='ZYX')

            label_name = im_name + '_label.tif'
            save_tiff_imagej_compatible(label_name, labels, axes='ZYX')

            mask_name = im_name + '_mask.tif'
            save_tiff_imagej_compatible(mask_name, mask, axes='ZYX')

        # print("The average computation time is " + str(np.mean(t)) + " +/- " + str(np.std(t)))
        #
        # idx = np.argsort(nobjects)
        #
        # plt.errorbar(nobjects[idx], t_av[idx], t_std[idx], label='both limits (default)')
        # plt.xlabel('N detected objects')
        # plt.ylabel('Computation time (s)')
        #
        # fig1 = plt.gcf()
        # plt.show()
        # plt.draw()
        # fig1.savefig('Computation_time.png')
