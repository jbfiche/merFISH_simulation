#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:22:12 2020

@author: jb
"""

import yaml
import os, datetime
from datetime import datetime
import os.path as path
import matplotlib.pyplot as plt

# Open the configuration file
# ---------------------------

with open(path.join(path.dirname(__file__), "CONFIG.yml"), "r") as f:
    config_parameters = yaml.load(f, Loader=yaml.FullLoader)

codebookfolder = config_parameters["codebook"]["codebookfolder"]
destinationfolder = config_parameters["destinationfolder"]
mainfolder = config_parameters["mainfolder"]

# Define the current working directory
# ------------------------------------

os.chdir(mainfolder)
print(os.getcwd())

# Test whether the destination folder exist and create a folder with the data
# and time of the day
# -------------------

if path.exists(destinationfolder) == False:
    os.mkdir(destinationfolder)

simulationdir = os.path.join(
    destinationfolder, datetime.now().strftime("%Y-%m-%d_%H-%M")
)

os.mkdir(simulationdir)
        
# Test to generate the fiducial coordinates
# -----------------------------------------

from Detections import Fiducial, Readout

fiducial = Fiducial(config_parameters)
fiducial.define_coordinates()


readout = Readout(config_parameters)
readout.read_codebook()
readout.define_coordinates()

# Test to generate and save the movies
# ------------------------------------

from Simulate_movie import SimulateData

simu = SimulateData(config_parameters, fiducial.fid_coordinates, readout.probe_coordinates)
simu.create_fiducial_image(3, 500)

plt.imshow(simu.fiducial_im)

