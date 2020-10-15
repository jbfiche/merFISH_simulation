#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:26:32 2020

@author: jb
"""

import numpy
import tifffile
import os
import matplotlib.pyplot as plt


class SimulateData:
    def __init__(self, param, fid_coordinates, probe_coordinates, probe_code, simulationdir):

        self.destfolder = simulationdir

        self.nROI = param["acquisition_data"]["nROI"]
        self.nprobe = param["readout_probe"]["number"]
        self.width = param["image"]["width"]
        self.height = param["image"]["height"]
        self.destfolder = param["destinationfolder"]
        self.bkg_fiducial = param["fiducial"]["background_intensity"]
        self.bkg_readout = param["readout_probe"]["background_intensity"]
        self.nround = param["acquisition_data"]["rounds"]

        psf_FWHM = param["image"]["psf_FWHM_nm"]
        pixel_size = param["image"]["pixel_size_nm"]
        self.s = psf_FWHM / (2.3 * pixel_size)

        self.fid_coordinates = fid_coordinates
        self.probe_coordinates = probe_coordinates
        self.probe_code = probe_code

        self.gain = numpy.random.normal(
            loc=2.0, scale=0.1, size=(self.width, self.height)
        )
        self.offset = numpy.random.randint(
            95, high=106, size=(self.width, self.height)
        ).astype(numpy.float)
        self.read_noise = numpy.random.normal(
            loc=1.5, scale=0.1, size=(self.width, self.height)
        )

    def create_single_bkg_image(self, bkg_intensity):

        image = self.gain * numpy.random.poisson(
            lam=bkg_intensity, size=(self.width, self.height)
        )

        # Read Noise - Normal distribution
        image += numpy.random.normal(
            scale=self.read_noise, size=(self.width, self.height)
        )

        # Camera baseline.
        image += self.offset

        self.bkg_im = image

    def create_fiducial_image(self, roi, I):

        self.create_single_bkg_image(self.bkg_fiducial)
        loc = numpy.zeros((self.width, self.height))

        for row in self.fid_coordinates:
            if row[0] == roi:

                x = row[1]
                y = row[2]
                
                x0 = int(numpy.round(x))
                y0 = int(numpy.round(y))

                for i in range(-7, 7):
                    for j in range(-7, 7):
                        X = x0 + i
                        Y = y0 + j
                        if X > 0 and X < self.width and Y > 0 and Y < self.height:
                            d = (x - x0 - i) ** 2 + (y - y0 - j) ** 2
                            loc[x0 + i, y0 + j] = I * numpy.exp(-d / (2 * self.s ** 2))

        self.fiducial_im = loc + self.bkg_im

    def create_readout_image(self, roi, n, I):

        self.create_single_bkg_image(self.bkg_readout)
        loc = numpy.zeros((self.width, self.height))

        for row in self.probe_coordinates:
            if row[0] == roi:

                x = row[1]
                y = row[2]
                loci = int(row[3])
                
                x0 = int(numpy.round(x))
                y0 = int(numpy.round(y))

                if self.probe_code[loci, n] == 1:

                    for i in range(-7, 7):
                        for j in range(-7, 7):
                            X = x0 + i
                            Y = y0 + j
                            if X > 0 and X < self.width and Y > 0 and Y < self.height:
                                d = (x - x0 - i) ** 2 + (y - y0 - j) ** 2
                                loc[x0 + i, y0 + j] = I * numpy.exp(
                                    -d / (2 * self.s ** 2)
                                )

        self.readout_im = loc + self.bkg_im

    def simulate_data(self):

        for roi in range(self.nROI):
            for n in range(self.nround):
                
                print(roi,n)

                name = "Test_" + str(roi) + "_" + str(n) + ".tif"
                with tifffile.TiffWriter(name) as tf:
                    
                    self.create_readout_image(roi, n, 500)
                    tf.save(numpy.round(self.readout_im).astype(numpy.uint16))

                    self.create_fiducial_image(roi, 1000)
                    tf.save(numpy.round(self.fiducial_im).astype(numpy.uint16))
