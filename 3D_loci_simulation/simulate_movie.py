#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:50:41 2021

@author: jb
"""

import tifffile
import numpy as np
import matplotlib.pyplot as plt


class SimulateData:

    def __init__(self, param, loci_coordinates, psf_files):

        self.nROI = param["acquisition_data"]["nROI"]
        self.nlocus = param["detection"]["number_detections_per_image"]

        self.I = param["image"]["intensity_single_probe"]

        self.Lx = param["image"]["width"]
        self.Ly = param["image"]["height"]
        self.Lz = param["image"]["number_z_planes"]

        self.Lx_psf = param["psf"]["psf_width"]
        self.Ly_psf = param["psf"]["psf_height"]
        self.Lz_psf = param["psf"]["psf_planes"]

        self.Dx = self.Lx + self.Lx_psf
        self.Dy = self.Ly + self.Ly_psf
        self.Dz = self.Lz

        self.pixel_size = param["image"]["pixel_size_nm"]
        self.dz_planes = param["image"]["z_spacing_nm"]
        self.dz_planes_psf = param["psf"]["z_spacing_psf_nm"]
        self.r = self.dz_planes / self.dz_planes_psf  # ratio of plane interspace between the simulated stack and the psf

        self.bkg = param["image"]["background_intensity"]

        self.loci_coordinates = loci_coordinates
        self.psf_files = psf_files

        # Define the properties of the simulated sCMOS camera
        # ---------------------------------------------------
        self.gain = np.random.normal(
            loc=10.36, scale=0.25, size=(self.Dx, self.Dy)
        )
        self.offset = np.random.normal(
            loc=100, scale=1.2, size=(self.Dx, self.Dy)
        ).astype(np.float)

        self.offset = np.round(self.offset)

        self.read_noise = np.random.normal(
            loc=1.5, scale=0.1, size=(self.Dx, self.Dy)
        )

    def create_single_bkg_image(self, bkg_intensity, Dx, Dy):
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

        image = self.gain * np.random.poisson(
            lam=bkg_intensity, size=(Dx, Dy)
        )

        # Read Noise - Normal distribution
        image += np.random.normal(
            scale=self.read_noise, size=(Dx, Dy)
        )

        # Camera baseline.
        image += self.offset
        return image

    def create_bkg_stack(self):

        stack = np.zeros((self.Dx, self.Dy, self.Dz))

        for n_plane in range(self.Dz):
            stack[:, :, n_plane] = self.create_single_bkg_image(self.bkg, self.Dx, self.Dy)

        self.stack = stack

    def add_locus_image(self, nlocus):

        # Load the coodinates of the probes in the locus
        # ----------------------------------------------

        key = "Locus_" + str(nlocus)
        coordinates = self.loci_coordinates[key]

        # Pick the psf images
        # -------------------

        n_psf = np.random.randint(0, len(self.psf_files))
        psf = np.load(self.psf_files[n_psf])
        psf_width, psf_height, psf_planes = psf.shape

        # Apply a random transformation (flip) to the psf
        # -----------------------------------------------

        if np.random.randint(0, 2):
            psf = np.flip(psf, axis=0)

        if np.random.randint(0, 2):
            psf = np.flip(psf, axis=1)

        # Load the coordinates of each probe in the locus
        # ------------------------------------------------

        for n_probe in range(coordinates.shape[0]):

            x = coordinates[n_probe, 0]
            y = coordinates[n_probe, 1]
            z = coordinates[n_probe, 2]

            if x != 0 and y != 0 and z != 0:

                # Convert the coordinates into pixel/plane
                # -----------------------------------------

                x = x * 1000 / self.pixel_size + self.Lx_psf / 2
                y = y * 1000 / self.pixel_size + self.Ly_psf / 2
                z = z * 1000 / self.dz_planes

                # Select the central plane in the stack - np.floor(z) is 
                # indicating which plane of the stack will be used as the central
                # plane for the psf. 
                # Since the distance between the planes is smaller for the psf, 
                # dz is indicating which plane of the psf stack will be used as
                # central plane (50+dz).
                # ---------------------

                dz_psf = (z - np.floor(z)) * self.r
                dz_psf = np.round(dz_psf)

                x0 = np.round(x)
                y0 = np.round(y)
                z0 = np.floor(z)

                dzmin = - self.Lz_psf / 2 / self.r
                dzmax = self.Lz_psf / 2 / self.r

                for dz in range(int(dzmin), int(dzmax), 1):

                    # Select the stack plane and the psf plane
                    # ----------------------------------------
                    z = z0 + dz
                    z_psf = self.Lz_psf / 2 + dz * self.r + dz_psf

                    if 0 <= z <= self.Lz - 1 and 0 <= z_psf <= self.Lz_psf - 1:
                        im = self.stack[:, :, int(z)]
                        im_psf = psf[int(50 + dz)] * np.random.poisson(self.I)

                        x_roi_min = int(x0 - self.Lx_psf / 2)
                        x_roi_max = int(x0 + self.Lx_psf / 2)
                        y_roi_min = int(y0 - self.Ly_psf / 2)
                        y_roi_max = int(y0 + self.Ly_psf / 2)

                        try:
                            im[x_roi_min:x_roi_max, y_roi_min:y_roi_max] = im_psf + im[x_roi_min:x_roi_max, y_roi_min:y_roi_max]
                            self.stack[:, :, int(z)] = im
                        except ValueError:
                            print(x_roi_min,x_roi_max, y_roi_min, y_roi_max)

    def create_loci_stack(self):

        for n in range(self.nlocus):
            self.add_locus_image(n)

        name = "Test_.tif"
        with tifffile.TiffWriter(name) as tf:

            for n in range(self.Lz):
                im = self.stack[:, :, n]
                im = im[int(self.Lx_psf / 2): int(self.Lx + self.Lx_psf / 2),
                     int(self.Ly_psf / 2): int(self.Ly + self.Ly_psf / 2)]
                tf.save(np.round(im).astype(np.uint16))

                # fig, ax = plt.subplots()
                # plt.imshow(im)
                # plt.show()

                # fig, ax = plt.subplots()
                # plt.imshow(im_psf)
                # plt.show()
