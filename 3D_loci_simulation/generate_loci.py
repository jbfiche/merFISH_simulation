#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:04:25 2021

@author: jb
"""

import numpy as np


class Loci:

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
        # size of th the field of view where the localizations can be simulated (in µm). The parameters are calculated
        # according to the shape of the simulated stack of images and the pixel size.

        self.Lx = param["image"]["width"] * param["image"]["pixel_size_nm"] / 1000
        self.Ly = param["image"]["height"] * param["image"]["pixel_size_nm"] / 1000
        self.Lz = param["image"]["number_z_planes"] * param["image"]["z_spacing_nm"] / 1000
        self.dz = param["detection"]["minimum_depth_um"]

        self.N_probes = param["detection"]["number_probes"]
        self.D_probes = param["detection"]["loci_size_kb"]

        self.Intensity = param["image"]["intensity_single_probe"]

        # The 3D physical distance is related to the genomic distance (in kb)
        # following a power law (see Cattoni et al. 2017. Nat. Communications). In
        # order to generate the loci, the average size of the loci will calculated
        # according to the parameters indicated in the paper. The power law is
        # d_3D = g*d_kb^b.
        # Below, the minimum and maximum values indicated in the paper are indicated.
        # A uniform random law will be used to select the values for each locus.

        self.bmin = param["detection"]["beta_min"]
        self.bmax = param["detection"]["beta_max"]
        self.gmin = param["detection"]["gamma_min"]
        self.gmax = param["detection"]["gamma_max"]

        # In the embryos, the genome is diploid. In most of the cases, the loci are
        # paired together which makes the loci twice brighter. The probability of two
        # loci being paired in defined by the parameter P_paired and the average
        # distance is D_diploid.

        self.P_paired = param["detection"]["diploid_pair_probability"]
        self.D_diploid = param["detection"]["average_paired_distance_nm"]  # average distance separating two paired loci

        # For the inhomogeneous background simulation

        self.step_xy = param["image"]["pixel_size_nm"] / 1000
        self.step_z = param["image"]["z_spacing_nm"] / 1000
        self.bkg_step = param["image"]["background_step"]

        # Finally, the probability for each probe to hybridize is also defined as
        # P_hybridization

        self.P_hybridization = param["detection"]["probe_hybridization_probability"]

    def define_locus_coordinates(self, n_detections):
        """
        Calculate the 3D position of each detection. The calculation is performed in
        steps :
        1- define for each locus the position of the first probe, following a uniform 
        distribution in XYZ. In the case of the Z position, the centroid is however 
        constrained between planes 10-50.  
        2- the distance between the first and last probes of the locus is calculated.
        The positions of each probe is then calculated using spherical coordinates
        only if the probe is hybridizing.
        3- the process is repeated in the case of diploid loci. In that case, the position
        of the second locus is estimated using a Khi-square probability law.

        Returns
        -------
        Loci : dictionary containing all the 3D positions of the hybridized probes
        """

        loci = dict()
        for n in range(n_detections):
            
            # Using a uniform probability distribution, generate the 3D
            # coordinates of the locus as well as the probability for a probe
            # to hybridize to this locus. 

            diploid = False
            x0 = np.random.uniform(0, self.Lx, 1)
            y0 = np.random.uniform(0, self.Ly, 1)
            z0 = np.random.uniform(self.dz, self.Lz - self.dz, 1)
            p_hyb = np.random.uniform(self.P_hybridization[0], self.P_hybridization[1], 1)
            
            # Calculate the size of the locus as well as the coordinates and
            # intensity of all the probes hybridized to it
            
            dx, dy, dz = self.calculate_locus_size()
            locus_coordinates = self.calculate_probes_coordinates(x0, y0, z0, dx, dy, dz, p_hyb)

            # Since the genome is diploid and similar loci have a very high probability (>0.9) to be paired together,
            # the same calculation is performed for the locus copy. A Bernouilli process is used to decide whether the locus
            # is diploid. If yes, the position of the paired locus is recalculated by estimating the distance between
            # the two loci (chi-square distribution).
            
            diploid_coord, diploid = self.calculate_diploid_coordinates(x0, y0, z0)
            if diploid:
                dx, dy, dz = self.calculate_locus_size()
                locus_coordinates = np.concatenate((locus_coordinates,self.calculate_probes_coordinates(diploid_coord[0],
                                                                                                        diploid_coord[1],
                                                                                                        diploid_coord[2],
                                                                                                        dx, dy, dz, p_hyb)))

            key = "Locus_" + str(n)
            loci[key] = locus_coordinates
         
        return loci
    
    def calculate_locus_size(self):
        """
        From Cattoni et al. 2O17 - Nat. Communications, it is possible to infer
        from the genomic distance the physical distance. The size of the locus 
        is calculated based on a power law, the parameters b/g are generated 
        according to the values published.
        The size of the locus should be small compared to the persistance length
        of the chromatine. The locus is therefore simulated as a straight rod.
        It orientation is randomly computed using 3D spherical coordinates
        (theta & phi) and finally converted to cartesian coordinates. 

        Returns
        -------
        dx, dy, dz = extremities of the locus
        """
        b = np.random.uniform(self.bmin, self.bmax, 1)
        g = np.random.uniform(self.gmin, self.gmax, 1)
        r = (g * self.D_probes ** b) / 1000  # returns the estimated size of the loci in um

        theta = np.random.uniform(0, np.pi, 1)
        phi = np.random.uniform(0, 2 * np.pi, 1)

        dx = r * np.sin(theta) * np.cos(phi)
        dy = r * np.sin(theta) * np.sin(phi)
        dz = r * np.cos(theta)
        
        return dx, dy, dz

    def calculate_probes_coordinates(self, x0, y0, z0, dx, dy, dz, p_hyb):
        """
        According the the positions of the locus and its length, the 3D 
        coordinates and the intensity of the probes are calculated.

        Parameters
        ----------
        x0, y0, z0 : float - positions of the locus
        dx, dy, dz : float - length of the locus 
        p_hyb : float - probability for a probe to hybridize to this locus

        Returns
        -------
        locus_coordinates : np array - contains all the coordinates of the 
        probes. If the coordinates are all zeros, it means the probe is not
        hybridized. The fourth column corresponds to the intensity associated
        to the probe

        """
        
        # Check the locus is in the range of coordinates allowed
        if (0 < x0+dx < self.Lx) and (0 < y0+dy < self.Ly) and (self.dz < z0+dz < self.Lz - self.dz):
            
            locus_coordinates = np.zeros((self.N_probes, 4))
            for n_probe in range(self.N_probes):

                if np.random.binomial(1, p_hyb) == 1:
                    locus_coordinates[n_probe, 0] = x0 + n_probe * dx / self.N_probes
                    locus_coordinates[n_probe, 1] = y0 + n_probe * dy / self.N_probes
                    locus_coordinates[n_probe, 2] = z0 + n_probe * dz / self.N_probes
                    locus_coordinates[n_probe, 3] = np.random.poisson(self.Intensity)
                    
            locus_coordinates = locus_coordinates[~np.all(locus_coordinates == 0, axis=1)]
                    
        else:
            locus_coordinates = []
            
        return locus_coordinates
    
    def calculate_diploid_coordinates(self, x0, y0, z0):
        """
        Generate the coodinates of the diploid locus

        Parameters
        ----------
        x0, y0, z0 : float - coordinates of the first locus

        Returns
        -------
        x, y, z :  float - coordinates of the diploid locus
        diploid : boolean - indicates whether the locus is diploid or not
        """

        diploid_coord = np.zeros(3,)
        if np.random.binomial(1, self.P_paired) == 1:

            r_diploid = np.random.chisquare(1) * self.D_diploid + self.D_diploid
            theta = np.random.uniform(0, np.pi, 1)
            phi = np.random.uniform(0, 2 * np.pi, 1)

            diploid_coord[0] = x0 + r_diploid * np.sin(theta) * np.cos(phi)
            diploid_coord[1] = y0 + r_diploid * np.sin(theta) * np.sin(phi)
            diploid_coord[2] = z0 + r_diploid * np.cos(theta)

            # Check the coordinates of the paired locus are still in the border defined by the parameters

            if (0 < diploid_coord[0] < self.Lx) and (0 < diploid_coord[1] < self.Ly) and (self.dz < diploid_coord[2] < self.Lz - self.dz):
                diploid = True
            else:
                diploid = False
                
        else:
            diploid = False
            
        return diploid_coord, diploid

    def define_homogeneous_bkg_coordinates(self, n_detections):
        """
        Calculate the 3D position of each false positive detection. The calculation is performed in
        the same way than for the locus, though this time, only a single probe is used.

        Returns
        -------
        fp : array containing all the 3D positions of the single probes
        """

        fp_coordinates = np.zeros((n_detections, 4))

        fp_coordinates[:, 0] = np.random.uniform(0, self.Lx, n_detections)
        fp_coordinates[:, 1] = np.random.uniform(0, self.Ly, n_detections)
        fp_coordinates[:, 2] = np.random.uniform(self.dz, self.Lz - self.dz, n_detections)
        fp_coordinates[:, 3] = np.random.poisson(self.Intensity, n_detections)

        return fp_coordinates

    def define_inhomogeneous_bkg_coordinates(self, n_detections):
        """
        Calculate an inhomogeneous background based on a 3D random walk

        Parameters
        ----------
        n_detections : int - define the number of false positive probes

        Returns
        -------
        coordinates : np array - return the coordinates and intensity of all
        false positive probes defining the background
        """

        bkg_step = np.random.uniform(self.bkg_step[0], self.bkg_step[1], 1)
        x0 = np.random.uniform(0, self.Lx)
        y0 = np.random.uniform(0, self.Ly)
        z0 = np.random.uniform(self.dz, self.Lz - self.dz)
        step = np.random.normal(loc=0,
                                scale=bkg_step,
                                size=(3, n_detections))
        intensity = np.random.poisson(self.Intensity, size=(n_detections+1,))
        coordinates = np.zeros((4, n_detections+1))
        coordinates[0, 0] = x0
        coordinates[1, 0] = y0
        coordinates[2, 0] = z0
        coordinates[3, 0] = intensity[0]

        for n in range(n_detections):
            coordinates[0, n+1] = step[0, n] * self.step_xy + coordinates[0, n]
            coordinates[1, n+1] = step[1, n] * self.step_xy + coordinates[1, n]
            coordinates[2, n+1] = step[2, n] * self.step_z + coordinates[2, n]
            coordinates[3, n+1] = intensity[n+1]

        x = coordinates[0, :]
        idx = np.where(x < 0)
        coordinates[0, idx] = coordinates[0, idx] + self.Lx
        idx = np.where(x > self.Lx)
        coordinates[0, idx] = coordinates[0, idx] - self.Lx

        y = coordinates[1, :]
        idx = np.where(y < 0)
        coordinates[1, idx] = coordinates[1, idx] + self.Ly
        idx = np.where(y > self.Ly)
        coordinates[1, idx] = coordinates[1, idx] - self.Ly

        z = coordinates[2, :]
        idx = np.where(z < 0)
        coordinates[2, idx] = coordinates[2, idx] + self.Lz
        idx = np.where(z > self.Lz)
        coordinates[2, idx] = coordinates[2, idx] - self.Lz

        coordinates = np.transpose(coordinates)

        return coordinates


