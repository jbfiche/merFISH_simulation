#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:20:55 2020

@author: jb
"""

import numpy
import csv

class Fiducial:
    
    def __init__(self, param):
        """

        Parameters
        ----------
        param : dictionnary containing all the general parameters for the
        experiment.

        Returns
        -------
        None.

        """
        
        self.nROI = param["acquisition_data"]["nROI"]
        self.nfiducial = param["fiducial"]["number"]
        self.width = param["image"]["width"]
        self.height = param["image"]["height"]
        
        
    def define_coordinates(self):
        
        fid_coordinates = numpy.empty([self.nROI*self.nFiducial,3], dtype=float)
        
        for roi in range(self.nROI):
            for n in range(self.nfiducial):
                x = numpy.random.rand(1)*self.width
                y = numpy.random.rand(1)*self.height

                fid_coordinates[self.nROI*roi+n,0:3] = [roi, x[0], y[0]]
                # fid_coordinates[self.nROI*roi+n,0] = roi
                # fid_coordinates[self.nROI*roi+n,1] = x[0]
                # fid_coordinates[self.nROI*roi+n,2] = y[0]
                
        self.fid_coordinates = fid_coordinates
        
    
class Readout:
    
    def __init__(self, param):
        
        self.nROI = param["acquisition_data"]["nROI"]
        self.nprobe = param["readout_probe"]["number"]
        self.width = param["image"]["width"]
        self.height = param["image"]["height"]
        self.codebookfolder = param["codebook"]["codebookfolder"]
        
    def read_codebook(self):
        
        self.probe_name = []
        self.probe_code = []

        with open(self.codebookfolder, newline="") as codebook:

            csv_reader = csv.reader(codebook, delimiter=",", quotechar="|")
        
            for row in csv_reader:
        
                if row[0] != "Blank-01":
        
                    self.probe_name.append(row[0])
                    self.probe_code.append(row[2:17])
        
                else:
                    break
                