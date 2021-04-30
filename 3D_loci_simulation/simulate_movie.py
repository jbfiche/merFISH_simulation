#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:50:41 2021

@author: jb
"""

import tifffile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class SimulateData:

    def __init__(self, param, n_locus, loci_coordinates, n_fp, fp_coordinates, psf_files):

        self.nROI = param["acquisition_data"]["nROI"]
        self.Intensity = param["image"]["intensity_single_probe"]
        self.n_locus = n_locus
        self.n_fp = n_fp

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

        self.a = param["ground_truth"]["simulated_psf_width"] # in nm
        self.c = param["ground_truth"]["simulated_psf_height"]  # in nm

        self.bkg = param["image"]["background_intensity"]

        self.loci_coordinates = loci_coordinates
        self.fp_coordinates = fp_coordinates
        self.psf_files = psf_files

        # Define the properties of the simulated sCMOS camera - the parameter for the offset and the gain where
        # indicated based on measured on the Orca Flash 4
        # ------------------------------------------------

        self.gain = np.random.normal(loc=10.36, scale=0.25, size=(self.Dx, self.Dy))

        self.offset = np.random.normal(loc=100, scale=1.2, size=(self.Dx, self.Dy)).astype(np.float)
        self.offset = np.round(self.offset)

        self.read_noise = np.random.normal(loc=0.5, scale=0.1, size=(self.Dx, self.Dy))

        # Define the template for the simulated and the ground-truth data
        # ---------------------------------------------------------------

        self.stack = np.zeros((self.Dx, self.Dy, self.Dz))
        self.gt_stack = np.zeros((self.Lx, self.Ly, self.Lz))

    def create_single_bkg_image(self, bkg_intensity, dx, dy):
        """
        Simple code to create a typical background image for a sCMOS camera. 
        This code was adapted from codes found on the Zhuang lab github.

        Parameters
        ----------
        bkg_intensity : int Define the background intensity for the image (based on a Poisson law for the shot noise)
        dx, dy : int Size of the simulated image in pixel

        Returns
        -------
        image : the simulated background image

        """

        image = self.gain * np.random.poisson(lam=bkg_intensity, size=(dx, dy))

        # Read Noise - Normal distribution
        image += np.random.normal(scale=self.read_noise, size=(dx, dy))

        # Camera baseline.
        image += self.offset
        return image

    def create_bkg_stack(self):

        bkg = np.random.randint(self.bkg[0], self.bkg[1])
        for n_plane in range(self.Dz):
            self.stack[:, :, n_plane] = self.create_single_bkg_image(bkg, self.Dx, self.Dy)

    def add_locus_image(self, coordinates):

        # Pick the psf images
        # -------------------

        n_psf = np.random.randint(0, len(self.psf_files))
        psf = np.load(self.psf_files[n_psf])

        # Apply a random transformation (flip) to the psf
        # -----------------------------------------------

        if np.random.binomial(1, 0.5) == 1:
            psf = np.flip(psf, axis=1)

        if np.random.binomial(1, 0.5) == 1:
            psf = np.flip(psf, axis=2)

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
                        im_psf = psf[int(z_psf)] * np.random.poisson(self.Intensity)

                        x_roi_min = int(x0 - self.Lx_psf / 2)
                        x_roi_max = int(x0 + self.Lx_psf / 2)
                        y_roi_min = int(y0 - self.Ly_psf / 2)
                        y_roi_max = int(y0 + self.Ly_psf / 2)

                        try:
                            im[x_roi_min:x_roi_max, y_roi_min:y_roi_max] = im_psf + im[x_roi_min:x_roi_max,
                                                                                    y_roi_min:y_roi_max]
                            self.stack[:, :, int(z)] = im
                        except ValueError:
                            print(x_roi_min, x_roi_max, y_roi_min, y_roi_max)

    def simulate_raw_stack(self):

        for n in tqdm(range(self.n_locus)):
            key = "Locus_" + str(n)
            coordinates = self.loci_coordinates[key]
            self.add_locus_image(coordinates)

        # print('')
        # print('Simulate false positive :')
        # for n in tqdm(range(self.n_fp)):
        #     key = "FP_" + str(n)
        #     coordinates = self.fp_coordinates[key]
        #     self.add_locus_image(coordinates)

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

    def simulate_ground_truth(self):

        print('')
        print('Simulate ground truth :')

        for n in tqdm(range(self.n_locus)):
            key = "Locus_" + str(n)
            coordinates = self.loci_coordinates[key]

            # calculate the mean position
            x0 = np.median(coordinates[:, 0]) * 1000/self.pixel_size
            y0 = np.median(coordinates[:, 1]) * 1000/self.pixel_size
            z0 = np.median(coordinates[:, 2]) * 1000/self.pixel_size

            # simulate the psf as an ellipsoid
            for x, y, z in self.ellipsoid_coordinates:
                self.gt_stack[int(x+x0), int(y+y0), int(z+z0)] = n

        name = "Test_GT.tif"
        with tifffile.TiffWriter(name) as tf:

            for n in range(self.Lz):
                im = self.gt_stack[:, :, n]
                tf.save(np.round(im).astype(np.uint16))

    def create_ellipsoid_template(self):

        # simulate the psf as an ellipsoid
        a = self.a / self.pixel_size
        c = self.c / self.pixel_size
        r_a = np.arange(-np.ceil(a), np.ceil(a)+1, 1)
        r_c = np.arange(-np.ceil(c), np.ceil(c)+1, 1)
        ellipsoid_coordinates = np.zeros((len(r_a) * len(r_a) * len(r_c), 3))

        print(r_a)
        n = 0
        for x in r_a:
            for y in r_a:
                for z in r_c:

                    r = (x/a)**2 + (y/a)**2 + (z/c)**2
                    if r <= 1:
                        ellipsoid_coordinates[n, 0] = x
                        ellipsoid_coordinates[n, 1] = y
                        ellipsoid_coordinates[n, 2] = z
                        n += 1

        self.ellipsoid_coordinates = ellipsoid_coordinates[0:n, :]

