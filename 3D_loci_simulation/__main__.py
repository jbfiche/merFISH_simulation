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

if path.exists(destinationfolder) == False:
    os.mkdir(destinationfolder)

simulationdir = os.path.join(
    destinationfolder, datetime.now().strftime("%Y-%m-%d_%H-%M")
)

os.mkdir(simulationdir)
os.chdir(simulationdir)

# Generate the loci coordinates
# -----------------------------

_loci = Loci(config_parameters)
loci_coordinates = _loci.define_coordinates()

# Generate the images
# -------------------

_stack = SimulateData(config_parameters, loci_coordinates, psf_files)
_stack.create_bkg_stack()
# _stack.add_locus_image(1)
_stack.create_loci_stack()
