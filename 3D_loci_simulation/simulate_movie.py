#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:50:41 2021

@author: jb

The class SimulateData is used to calculate a 3D stack of images for the HiM experiment. During initialization, the
class is receiving the following input parameters :
- param : parameters read from config.yaml
- n_locus : the number of locus that will be simulated
- loci-coordinates : (x,y,z,I) of each locus
- n_fp : number of false positive probes
- fp_coordinates : (x,y,z,I) of each emitter
- psf_files : list of all the available 3D templates for the psf

"""

import tifffile
import numpy as np
# from astropy.modeling import models, fitting


# from tqdm import tqdm
# import matplotlib.pyplot as plt


class SimulateData:

    def __init__(self, param, n_locus, loci_coordinates, n_fp, fp_coordinates, psf_files):

        self.nROI = param["acquisition_data"]["nROI"]
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

        self.a = param["ground_truth"]["simulated_psf_width"]  # in nm
        self.c = param["ground_truth"]["simulated_psf_height"]  # in nm

        self.bkg = param["image"]["background_intensity"]

        self.loci_coordinates = loci_coordinates
        self.fp_coordinates = fp_coordinates

        # Define the properties of the simulated sCMOS camera - the parameter for the offset and the gain where
        # indicated based on measures performed on the Orca Flash 4
        # ---------------------------------------------------------

        self.gain = np.random.normal(loc=10.36, scale=0.25, size=(self.Dx, self.Dy))

        self.offset = np.random.normal(loc=100, scale=1.2, size=(self.Dx, self.Dy)).astype(np.float)
        self.offset = np.round(self.offset)

        self.read_noise = np.random.normal(loc=0.5, scale=0.1, size=(self.Dx, self.Dy))
        self.read_noise = np.absolute(self.read_noise)
        
        # Load all the psf files and keep the templates in a dictionary
        # -------------------------------------------------------------
        
        self.n_psf = len(psf_files)
        self.psf = dict()
        for n_psf in range(self.n_psf):
            key = "psf_" + str(n_psf)
            self.psf[key] = np.load(psf_files[n_psf])

        # Define the template for the simulated and the ground-truth data
        # ---------------------------------------------------------------

        self.stack = np.zeros((self.Dx, self.Dy, self.Dz))
        self.bkg_stack = np.zeros((self.Dx, self.Dy, self.Dz))

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
        """
        Create a stack of background images. PSF will be added to the background later to create the simulation.
        """

        bkg = np.random.randint(self.bkg[0], self.bkg[1])
        for n_plane in range(self.Dz):
            self.stack[:, :, n_plane] = self.create_single_bkg_image(bkg, self.Dx, self.Dy)

    def add_locus_image(self, coordinates):
        """
        From the stack of background images and the list of coordinates for the simulated detections, detections are
        simulated by adding a 3D psf measured on a microscope (from a list of available psf previously normalized and
        centered). For each coordinates, the psf is randomly selected from a set of available npy files. The psf is
        randomly flip in the x/y direction, not along the z axis. From coordinates, the intensity associated to each
        emitter was also defined.

        Parameters
        ----------
        coordinates: numpy array containing the coordinates of each single proche (x, y, z) as well as its intensity
        """

        # Select the psf images
        # -------------------

        n_psf = np.random.randint(0, self.n_psf)
        key = "psf_" + str(n_psf)
        psf = self.psf[key]

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
            I = coordinates[n_probe, 3]

            # Convert the coordinates into pixel/plane
            # -----------------------------------------

            x = x * 1000 / self.pixel_size + self.Lx_psf / 2
            y = y * 1000 / self.pixel_size + self.Ly_psf / 2
            z = z * 1000 / self.dz_planes
                
            # Add the psf image to the simulated stack
            # ----------------------------------------
            
            self.add_psf_images(x, y, z, I, psf)
            

    def add_psf_images(self, x, y, z, I, psf):
        """
        For each emitter, the 3D images of the psf are added to the simulated
        stack

        Parameters
        ----------
        x , y, z : float - coordinates of the probe
            x_roi_min = int(x0 - (self.Lx_psf-1) / 2)
        x_roi_max = int(x0 + (self.Lx_psf-1) / 2 + 1) # Need to add 1, since python is not taking into account the last index
        y_roi_min = int(y0 - (self.Ly_psf-1) / 2)
        y_roi_max = int(y0 + (self.Ly_psf-1) / 2 + 1)    I : float - intensity associated to the probe. 
        psf : np array - image of the selected psf, after transformation
        """
        
        # Return the closest plane value according to the position z of the probe
        # -----------------------------------------------------------------------
        x0 = np.floor(x)
        y0 = np.floor(y)
        z0 = np.floor(z)
        
        # Define the boundaries of the crop
        # ---------------------------------
        
        x_roi_min = int(x0 - (self.Lx_psf-1) / 2)
        x_roi_max = int(x0 + (self.Lx_psf-1) / 2 + 1) # Need to add 1, since python is not taking into account the last index
        y_roi_min = int(y0 - (self.Ly_psf-1) / 2)
        y_roi_max = int(y0 + (self.Ly_psf-1) / 2 + 1)
        
        # z corresponds to the central position of the psf. Since the inter-plane
        # distance is not the same for the psf and the simulated images, dz_psf 
        # is used to define which psf plane (z0_psf) is the closest to z.
        # --------------------------------------------------------------
        dz_psf = (z - z0) * self.r
        dz_psf = np.round(dz_psf)
        z0_psf = (self.Lz_psf-1) / 2 + dz_psf
        
        # According to the central psf plane and the total number of images
        # available for the psf and the simulated movie, define the minimum and
        # maximum values for the planes
        # ----------------------------
        if z0_psf / self.r < z0:
            zmin = np.ceil(z0 - z0_psf / self.r)
        else:
            zmin = 0
            
        if (self.Lz_psf - 1 - z0_psf) / self.r < self.Lz - 1 - z0:
            zmax = np.floor(z0 + (self.Lz_psf - 1 - z0_psf) / self.r)
        else:
            zmax = self.Lz - 1

        zmin = int(zmin)
        zmax = int(zmax)
            
        # Add the psf images to the stack
        # -------------------------------
        for z in range(zmin, zmax+1, 1):
            
            try :
                z_psf = z0_psf - (z0 - z)*self.r
                im = self.stack[:, :, int(z)]
                im_psf = psf[int(z_psf)] * I
                im[x_roi_min:x_roi_max, y_roi_min:y_roi_max] = im_psf + im[x_roi_min:x_roi_max,
                                                                                    y_roi_min:y_roi_max]
                self.stack[:, :, int(z)] = im
            except ValueError:
                print(x_roi_min, x_roi_max, y_roi_min, y_roi_max, z, z_psf, self.Lz, self.Lz_psf)
                print(ValueError)
                break
            

    def simulate_raw_stack(self, nROI, path):
        """
        Create the stack of simulated images. The simulation is performed in two steps:
        - the false positive are first added to the background images
        - the psf associated to each emitters is then added.
        At the end of the process, the stack is saved as a tiff file.

        Parameters
        ----------
        nROI : the number of the ROI. Used to create a unique name for the output stack.

        Return
        ------
        name : the name of the simulated movie
        """

        # Calculate the background images with the false positive events
        # --------------------------------------------------------------
        
        print('simulate {} false positive'.format(self.n_fp))
        coordinates = np.zeros((1, 4))
        for n in range(self.n_fp):
            coordinates[0, :] = self.fp_coordinates[n, :]
            self.add_locus_image(coordinates)

        # Keep the background image for the ground truth calculation
        # ----------------------------------------------------------
        
        self.bkg_stack = self.stack[int(self.Lx_psf / 2): int(self.Lx + self.Lx_psf / 2),
                   int(self.Ly_psf / 2): int(self.Ly + self.Ly_psf / 2), :]

        # Simulate the loci
        # -----------------
        
        print('simulate {} true positive'.format(self.n_locus))
        for n in range(self.n_locus):
            key = "Locus_" + str(n)
            coordinates = self.loci_coordinates[key]
            if len(coordinates) > 0:
                self.add_locus_image(coordinates)

        # Save the final stack of images
        # ------------------------------
        
        self.stack = self.stack[int(self.Lx_psf / 2): int(self.Lx + self.Lx_psf / 2),
                                    int(self.Ly_psf / 2): int(self.Ly + self.Ly_psf / 2), :]
        
        name = "ROI_" + str(nROI) + ".tif"
        full_name = path + "/" + name
        with tifffile.TiffWriter(full_name) as tf:

            for n in range(self.Lz):
                im = self.stack[:, :, n]
                tf.save(np.round(im).astype(np.uint16))

                # fig, ax = plt.subplots()
                # plt.imshow(im)
                # plt.show()

                # fig, ax = plt.subplots()
                # plt.imshow(im_psf)
                # plt.show()

        return name


    def simulate_ground_truth(self, nROI, threshold, path):
        """
        Simulate the ground truth images associated to the set of emitters coordinates. The process is performed in
        three steps :
        - from the set of coordinates associated to each locus, keep all the non-zero coordinates and calculate the
        centroid position of the locus.
        - calculate the sum of all the emitters intensity and check that the final intensity is above a specific
         threshold with respect to the background
        - each locus is then modeled as a single ellipsoid

        At the end, two stack of images are saved :
        - one where each locus is instanciated with a specific ID n>0 = Ground truth
        - one where all pixel belonging to a locus are assigned the value 1 = MASK

        Parameters
        ----------
        nROI : the number of the ROI. Used to create a unique name for the output stack.
        """

        print('Simulate ground truth with threshold = {}'.format(threshold))
        gt_stack = np.zeros((self.Lx, self.Ly, self.Lz))

        for n in range(self.n_locus):
            
            key = "Locus_" + str(n)
            coordinates = self.loci_coordinates[key]
            
            if len(coordinates) > 0:

                # Calculate the centroid position of the locus
                # --------------------------------------------
                
                x0 = np.average(coordinates[:, 0], weights=coordinates[:, 3]) * 1000 / self.pixel_size
                y0 = np.average(coordinates[:, 1], weights=coordinates[:, 3]) * 1000 / self.pixel_size
                z0 = np.average(coordinates[:, 2], weights=coordinates[:, 3]) * 1000 / self.dz_planes
                int_locus = sum(coordinates[:, 3])
    
                # Calculate the background around the locus position
                # --------------------------------------------------
                
                SNR = self.calculate_SNR(x0, y0, z0, int_locus)
    
                # check there is enough signal to allow a proper detection of the locus
                if SNR > threshold:
        
                    # simulate the psf as an ellipsoid
                    for x, y, z in self.ellipsoid_coordinates:
                        if x + x0 < self.Lx and y + y0 < self.Ly and z + z0 < self.Lz:
                            gt_stack[int(x + x0), int(y + y0), int(z + z0)] = n + 1

        # Save the ground truth
        # ---------------------        

        name = path + "/GT_ROI_" + str(nROI) + ".tif"
        with tifffile.TiffWriter(name) as tf:

            for n in range(self.Lz):
                im = gt_stack[:, :, n]
                tf.save(np.round(im).astype(np.uint16))
                
        # Save the projection of the mask (since it is not used for the training)
        # ----------------------------------------------------------------------

        name = path + "/mask_ROI_" + str(nROI) + ".tif"
        mask = np.sum(gt_stack, axis=2)
        mask[mask > 0] = 1
        
        with tifffile.TiffWriter(name) as tf:
            tf.save(np.round(mask).astype(np.uint8))
                
                
    def calculate_SNR(self, x0, y0, z0, I):
        """
        Estimate SNR following Serge et al. method. The calculation is perfomed
        on multiple planes around the focal plane, in order to have a more 
        accurate estimation of the SNR

        Parameters
        ----------
        x0, y0, z0 : int - average position of the locus
        I : float - total intensity of the locus
        
        Returns
        -------
        SNR : float - return the estimated value of the SNR for the locus
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.calculate_roi_limits(x0, y0, z0)
        dz = zmax-zmin
        
        snr = np.zeros((dz+1,))
        
        for n in range(dz+1):
            z = zmin + n
            bkg = self.bkg_stack[xmin:xmax, ymin:ymax, z]
            im = self.stack[xmin:xmax, ymin:ymax, z]
            
            bkg_std = np.std(bkg)
            bkg_mean = np.median(bkg)
            
            # I = np.amax(im) - bkg_mean 
            im = np.sort(im, axis = None)
            I = np.sum(im[len(im)-5:len(im)]) - 5*bkg_mean 
            
            snr[n] = 10*np.log10(I**2 / bkg_std**2)
        
        SNR = np.mean(snr)
        return SNR
        

    def create_ellipsoid_template(self):
        """
        Create a template ellipsoid for the ground truth. According to the parameters defined in the config file, the
        method is calculating the coordinates of all the pixels of an ellipsoid centered around (0,0,0).
        """

        # simulate the psf as an ellipsoid
        a = self.a / self.pixel_size
        c = self.c / self.pixel_size
        r_a = np.arange(-np.ceil(a), np.ceil(a) + 1, 1)
        r_c = np.arange(-np.ceil(c), np.ceil(c) + 1, 1)
        ellipsoid_coordinates = np.zeros((len(r_a) * len(r_a) * len(r_c), 3))

        n = 0
        for x in r_a:
            for y in r_a:
                for z in r_c:

                    r = (x / a) ** 2 + (y / a) ** 2 + (z / c) ** 2
                    if r <= 1:
                        ellipsoid_coordinates[n, 0] = x
                        ellipsoid_coordinates[n, 1] = y
                        ellipsoid_coordinates[n, 2] = z
                        n += 1

        self.ellipsoid_coordinates = ellipsoid_coordinates[0:n, :]

    def calculate_roi_limits(self, x, y, z):
        """
        Using the position of the simulated locus, calculate the limits of an
        ROI around the x,y,z coordinates according to the specified dimensions
        dxy & dz.

        Parameters
        ----------
        x : float - x position of the simulated locus
        y : float - y position of the simulated locus
        z : float - z position of the simulated locus

        Returns
        -------
        Boundaries of the ROI.
        """

        dxy = 6
        dz = 1

        if x > dxy:
            xmin = x - dxy
        else:
            xmin = 0
        if x < self.Lx - 1 - dxy:
            xmax = x + dxy
        else:
            xmax = self.Lx - 1

        if y > dxy:
            ymin = y - dxy
        else:
            ymin = 0
        if y < self.Ly - 1 - dxy:
            ymax = y + dxy
        else:
            ymax = self.Ly - 1

        if z > dz:
            zmin = z - dz
        else:
            zmin = 0
        if z < self.Lz - 1 - dz:
            zmax = z + dz
        else:
            zmax = self.Lz - 1

        return np.array([xmin, xmax, ymin, ymax, zmin, zmax]).astype('int16')

