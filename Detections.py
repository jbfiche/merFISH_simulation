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
        """
        Define for each ROI the coordinates of the fluorescent beads. These coordinates will be the ones used for the very first image, later drift will be added to the process.

        Returns
        -------
        None.

        """

        fid_coordinates = numpy.empty([self.nROI * self.nfiducial, 3], dtype=float)

        for roi in range(self.nROI):
            for n in range(self.nfiducial):
                x = numpy.random.rand(1) * self.width
                y = numpy.random.rand(1) * self.height

                fid_coordinates[self.nfiducial * roi + n, 0:3] = [roi, x[0], y[0]]
                # fid_coordinates[self.nROI*roi+n,0] = roi
                # fid_coordinates[self.nROI*roi+n,1] = x[0]
                # fid_coordinates[self.nROI*roi+n,2] = y[0]

        self.fid_coordinates = fid_coordinates


class Readout:
    def __init__(self, param):

        self.nROI = param["acquisition_data"]["nROI"]
        self.nprobe = param["readout_probe"]["number_detections_per_image"]
        self.width = param["image"]["width"]
        self.height = param["image"]["height"]
        self.codebookfolder = param["codebook"]["codebookfolder"]
        self.nloci = param["readout_probe"]["number_selected_loci"]

    def read_codebook(self):
        """
        Read the codebook.csv file, taking into account only the field that
        are not called 'blank'. In the process three variables are created :
            - probe_name is a list containing all the names of the loci
            - probe_code is a numpy array containing all the 16bit words associated to each loci
            - max_nloci returns the maximum number of loci in the csv file.

        Returns
        -------
        None.

        """

        self.probe_name = []
        self.probe_code = []
        self.max_nloci = 0

        with open(self.codebookfolder, newline="") as codebook:

            csv_reader = csv.reader(codebook, delimiter=",", quotechar="|")
            next(csv_reader)

            for row in csv_reader:

                if row[0] != "Blank-01":
                    row_convert = [int(i) for i in row[2:18]]
                    self.probe_code.append(row_convert)
                    self.probe_name.append(row[0])
                    self.max_nloci += 1

                else:
                    break

        self.probe_code = numpy.asarray(self.probe_code)

    def define_coordinates(self):
        """
        As for the fiducial, the coordinated of each loci is calculated here.
        The loci are randomly assigned, according to the number of loci we
        want (define initialy in the parameter file by nloci). The coordinates
        are saved in the numpy array probe_coordinates.

        Returns
        -------
        None.

        """

        probe_coordinates = numpy.empty([self.nROI * self.nprobe, 4], dtype=float)

        if self.nloci <= self.max_nloci:
            nloci = self.nloci
        else:
            nloci = self.max_nloci

        for roi in range(self.nROI):
            for nprobe in range(self.nprobe):

                x = numpy.random.rand(1) * self.width
                y = numpy.random.rand(1) * self.height
                loci = numpy.rint(numpy.random.rand(1) * (nloci - 1))

                probe_coordinates[self.nprobe * roi + nprobe, 0:4] = [
                    roi,
                    x[0],
                    y[0],
                    69]
                    # loci[0],
                # ]

        self.probe_coordinates = probe_coordinates
