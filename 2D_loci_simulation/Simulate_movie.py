#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:26:32 2020

@author: jb
"""

import numpy
import tifffile


class SimulateData:
    def __init__(self, param, fid_coordinates, probe_coordinates, probe_code):

        self.nROI = param["acquisition_data"]["nROI"]
        self.nprobe = param["readout_probe"]["number_detections_per_image"]
        self.width = param["image"]["width"]
        self.height = param["image"]["height"]
        self.bkg_fiducial = param["fiducial"]["background_intensity"]
        self.bkg_readout = param["readout_probe"]["background_intensity"]
        self.nround = param["acquisition_data"]["rounds"]
        self.drift = param["fiducial"]["average_drift_px"]
        self.Ifid = param["fiducial"]["average_beads_intensity"]
        self.Ireadout = param["readout_probe"]["average_readout_intensity"]

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
        """
        Simple code to create a typical background image for a sCMOS camera. 
        This code was adapted from codes found on the Zhuang lab github.

        Parameters
        ----------
        bkg_intensity : int
        Define the background intensity for the image (based on a Poisson law
        for the shot noise)

        Returns
        -------
        None.

        """

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

    def create_fiducial_image(self, roi, I0, drift):
        """
        For a specific ROI, the code is adding on a background image the
        localizations of the fluorescent beads, according to the positions
        calculated in the class Fiducial.


        Parameters
        ----------
        roi : int - indicate to which roi this image will be assigned.
        I0 : int - average intensity of the fluorescent beads.
        drift : int - average number of pixels for the drift. It is used as
        the standard deviation of a normal law centered on zero.

        Returns
        -------
        None.

        """

        self.create_single_bkg_image(self.bkg_fiducial)
        loc = numpy.zeros((self.width, self.height))

        for row in self.fid_coordinates:
            if row[0] == roi:

                x = row[1] + drift[0, 0]
                y = row[2] + drift[0, 1]

                x0 = int(numpy.round(x))
                y0 = int(numpy.round(y))

                I = numpy.random.poisson(I0, 1)

                for i in range(-7, 7):
                    for j in range(-7, 7):
                        X = x0 + i
                        Y = y0 + j
                        if X > 0 and X < self.width and Y > 0 and Y < self.height:
                            d = (x - x0 - i) ** 2 + (y - y0 - j) ** 2
                            loc[x0 + i, y0 + j] = loc[x0 + i, y0 + j] + I * numpy.exp(-d / (2 * self.s ** 2))

        self.fiducial_im = loc + self.bkg_im

    def create_readout_image(self, roi, n, I0, drift):
        """
        For a specific ROI, the code is adding on a background image the
        localizations of the readout probes, according to the positions
        calculated in the class Readout.

        Parameters
        ----------
        roi : int - indicate to which roi this image will be assigned.
        n : int - indicate to which injection round the image will be assigned.
        I0 : int - average intensity of the fluorescent beads.
        drift : int - average number of pixels for the drift. It is used as
        the standard deviation of a normal law centered on zero.

        Returns
        -------
        None.

        """

        self.create_single_bkg_image(self.bkg_readout)
        loc = numpy.zeros((self.width, self.height))

        for row in self.probe_coordinates:
            if row[0] == roi:

                x = row[1] + drift[0, 0]
                y = row[2] + drift[0, 1]
                loci = int(row[3])

                x0 = int(numpy.round(x))
                y0 = int(numpy.round(y))

                I = numpy.random.poisson(I0, 1)

                if self.probe_code[loci, n] == 1:

                    for i in range(-7, 7):
                        for j in range(-7, 7):
                            X = x0 + i
                            Y = y0 + j
                            if X > 0 and X < self.width and Y > 0 and Y < self.height:
                                d = (x - x0 - i) ** 2 + (y - y0 - j) ** 2
                                loc[x0 + i, y0 + j] = loc[x0 + i, y0 + j] + I * numpy.exp(
                                    -d / (2 * self.s ** 2)
                                )

        self.readout_im = loc + self.bkg_im

    def simulate_data(self):
        """
        Calculate and save the simulated data, according to the number of roi
        and rounds for the experiment.
        Note that the drift is defined as a numpy array containing two values, 
        one for the x-drift, and the other for the y-drift. For the first round, 
        the drift is automatically set to [0,0]. Then, for each round, a drift
        is added according to a normal law with an average defined by the value
        of "drift" in the CONFIG file. 

        Returns
        -------
        None.

        """

        for roi in range(self.nROI):
            
            drift = numpy.zeros((1, 2))
            drift_array = numpy.zeros((self.nround,2))

            for n in range(self.nround):

                print(roi, n)

                name = "Test_" + str(roi) + "_" + str(n) + ".tif"
                with tifffile.TiffWriter(name) as tf:

                    self.create_readout_image(roi, n, self.Ireadout, drift)
                    tf.save(numpy.round(self.readout_im).astype(numpy.uint16))

                    self.create_fiducial_image(roi, self.Ifid, drift)
                    tf.save(numpy.round(self.fiducial_im).astype(numpy.uint16))

                    drift_array[n,:] = drift
                    drift = drift + numpy.random.normal(0, self.drift, (1, 2))
                    
            txt_name = "Drift_ROI_" + str(roi) + ".txt"
            numpy.savetxt(txt_name,drift_array)
