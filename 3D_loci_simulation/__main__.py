#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:37:43 2021

@author: jb
"""

import yaml
import os
from datetime import datetime
import os.path as path
import matplotlib.pyplot as plt
import glob
import numpy as np
from shutil import copyfile

# Open the configuration file
# ---------------------------

with open(path.join(path.dirname(__file__), "Config.yaml"), "r") as f:
    config_parameters = yaml.load(f, Loader=yaml.FullLoader)

destinationfolder = config_parameters["destinationfolder"]
mainfolder = config_parameters["mainfolder"]
psf_folder = config_parameters["psffolder"]

# Look for psf data
# -----------------

os.chdir(psf_folder)
psf_files = glob.glob('psf*.npy')

for n in range(len(psf_files)):
    psf_files[n] = os.path.abspath(psf_files[n])

# Define the current working directory
# ------------------------------------

os.chdir(mainfolder)
print(os.getcwd())
from generate_loci import Loci
from simulate_movie import SimulateData

# Test whether the destination folder exist and create a folder with the data
# and time of the day
# -------------------

if not path.exists(destinationfolder):
    os.mkdir(destinationfolder)

simulationdir = os.path.join(
    destinationfolder, datetime.now().strftime("%Y-%m-%d_%H-%M")
)

os.mkdir(simulationdir)
os.chdir(simulationdir)

# Create the folder structure for the deconvolution
# -------------------------------------------------

os.mkdir('To_deconvolve')
os.mkdir('To_deconvolve/folder_1')
os.mkdir('To_deconvolve/folder_1/folder_2')

nROI = config_parameters["acquisition_data"]["nROI"]
for roi in range(nROI):

    print('simulate ROI #{} :'.format(roi))

    # Generate the loci coordinates
    # -----------------------------

    _loci = Loci(config_parameters)

    N_detection = config_parameters["detection"]["number_detections_per_image"]
    n_locus = int(np.random.normal(N_detection, 20, 1))
    loci_coordinates = _loci.define_locus_coordinates(n_locus)

    N_false_positive = config_parameters["detection"]["number_false_positive_data"]
    n_fp = int(np.random.normal(N_false_positive, 1000, 1))
    if np.random.binomial(1, 0.8) == 0:
        print('Homogeneous')
        fp_coordinates = _loci.define_homogeneous_bkg_coordinates(n_fp)
    else:
        print('Inhomogeneous')
        fp_coordinates = _loci.define_inhomogeneous_bkg_coordinates(n_fp)

    # Generate the images
    # -------------------

    _stack = SimulateData(config_parameters, n_locus, loci_coordinates, n_fp, fp_coordinates, psf_files)
    _stack.create_bkg_stack()
    movie_name = _stack.simulate_raw_stack(roi)
    _stack.create_ellipsoid_template()
    _stack.simulate_ground_truth(roi)

    # Make a copy of the stack in a folder for the deconvolution
    # ----------------------------------------------------------

    folder_name = 'ROI_' + str(roi)
    new_dir = os.path.join(simulationdir, 'To_deconvolve/folder_1/folder_2', folder_name)
    os.mkdir(new_dir)

    new_path = new_dir + '/' + movie_name
    copyfile(movie_name, new_path)
