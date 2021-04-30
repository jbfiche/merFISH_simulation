#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:24:18 2021

@author: jb
"""

import numpy as np
import tifffile
import os
from glob2 import glob
import matplotlib.pyplot as plt


psf_folder = ["/mnt/PALM_dataserv/DATA/JB/2021/Data_single_loci/3D_PSF_examples/Airy-scan",
              "/mnt/PALM_dataserv/DATA/JB/2021/Data_single_loci/3D_PSF_examples/RAMM/psf_samples"]

destination_folder = "/mnt/PALM_dataserv/DATA/JB/2021/Data_single_loci/3D_PSF_examples/psf_analyze_npy/"

for folder in psf_folder:

    os.chdir(folder)
    psf_files = glob('psf_*.tif')

    for file in psf_files:

        im = tifffile.imread(file)

        # Normalize the psf intensity
        # ---------------------------

        im_shape = im.shape
        I = np.reshape(im, im.shape[0]*im.shape[1]*im.shape[2])
        I = np.sort(I)

        I_min = np.median(I[0:len(I)//1000])
        I_max = max(I)

        psf = (im - I_min)/I_max.astype(float)

        # Calculate the coordinates of the brightest pixel in order to define the
        # center of the psf.
        # ------------------

        Z_max, X_max, Y_max = np.unravel_index(np.argmax(im), im.shape)

        # Recenter the psf as a 101x101x101 pixel and save the numpy file for the simulations
        # -----------------------------------------------------------------------------------

        psf_new = psf[Z_max-50:Z_max+50, X_max-50:X_max+50, Y_max-50:Y_max+50]

        psf_file_name = destination_folder + os.path.splitext(file)[0]
        np.save(psf_file_name, psf_new)

        # Calculate the MIP for each psf and save it
        # ------------------------------------------

        mip_XY = np.max(psf_new, axis=0)
        mip_XZ = np.max(psf_new, axis=2)

        name = 'mipXY_' + os.path.splitext(file)[0] + '.tif'
        with tifffile.TiffWriter(name) as tf:
            tf.save((mip_XY*65000).astype(np.uint16))

        name = 'mipXZ_' + os.path.splitext(file)[0] + '.tif'
        with tifffile.TiffWriter(name) as tf:
            tf.save((mip_XZ*65000).astype(np.uint16))
