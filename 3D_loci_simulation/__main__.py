#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:37:43 2021

@author: jb
"""

import yaml
import os, datetime
from datetime import datetime
import os.path as path
import matplotlib.pyplot as plt
import glob
import numpy as np

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

# Generate the loci coordinates
# -----------------------------

_loci = Loci(config_parameters)

N_detection = config_parameters["detection"]["number_detections_per_image"]
n_locus = int(np.random.normal(N_detection, 20, 1))
loci_coordinates = _loci.define_locus_coordinates(n_locus)

N_false_positive = config_parameters["detection"]["number_false_positive_data"]
n_fp = int(np.random.normal(N_false_positive, 1000, 1))
fp_coordinates = _loci.define_false_positive_coordinates(n_fp)

# Generate the images
# -------------------

_stack = SimulateData(config_parameters, n_locus, loci_coordinates, n_fp, fp_coordinates, psf_files)
_stack.create_bkg_stack()
_stack.simulate_raw_stack()
_stack.create_ellipsoid_template()
_stack.simulate_ground_truth()

