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

# Test to open the codebook file
# ------------------------------

import csv

ProbeName = []
ProbeCode = []

with open(codebookfolder, newline="") as codebook:

    csv_reader = csv.reader(codebook, delimiter=",", quotechar="|")
    csv_dic = csv.DictReader(codebook)

    for row in csv_reader:

        if row[0] != "Blank-01":

            ProbeName.append(row[0])
            ProbeCode.append(row[2:17])

        else:
            break
        
# Test to generate the fiducial coordinates
# -----------------------------------------

from Fiducial import Fiducial

fiducial = Fiducial(config_parameters)
fiducial.define_coordinates()
print(fiducial.fid_coordinates)
