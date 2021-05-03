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

            diploid = False
            x0 = np.random.uniform(0, self.Lx, 1)
            y0 = np.random.uniform(0, self.Ly, 1)
            z0 = np.random.uniform(self.dz, self.Lz - self.dz, 1)
            p_hyb = np.random.uniform(self.P_hybridization[0], self.P_hybridization[1], 1)

            # Since the genome is diploid and similar loci have a very high probability (>0.9) to be paired together,
            # the calculation is performed in two steps. The first one corresponds to the locus itself. The second step
            # to its copy. The position of the copy is usually very close to the initial locus.

            for n_locus in range(2):

                if n_locus == 1:

                    # Bernouilli process to decide whether the locus is diploid. If yes, the position of the paired
                    # locus is recalculated by estimating the distance between the two loci (chi-square distribution).
                    # The paired locus is accepted only if it is localized within the border of the image.

                    if np.random.binomial(1, self.P_paired) == 1:

                        r_diploid = np.random.chisquare(1) * self.D_diploid + self.D_diploid
                        theta = np.random.uniform(0, np.pi, 1)
                        phi = np.random.uniform(0, 2 * np.pi, 1)

                        x0 = x0 + r_diploid * np.sin(theta) * np.cos(phi)
                        y0 = y0 + r_diploid * np.sin(theta) * np.sin(phi)
                        z0 = z0 + r_diploid * np.cos(theta)

                        if 0 < x0 < self.Lx and 0 < y0 < self.Ly and self.dz < z0 < self.Lz - self.dz:
                            diploid = True
                        else:
                            break
                    else:
                        break

                b = np.random.uniform(self.bmin, self.bmax, 1)
                g = np.random.uniform(self.gmin, self.gmax, 1)
                r = (g * self.D_probes ** b) / 1000  # returns the estimated size of the loci in um

                theta = np.random.uniform(0, np.pi, 1)
                phi = np.random.uniform(0, 2 * np.pi, 1)

                dx = r * np.sin(theta) * np.cos(phi)
                dy = r * np.sin(theta) * np.sin(phi)
                dz = r * np.cos(theta)

                if not diploid:
                    locus_coordinates = np.zeros((self.N_probes, 4))
                else:
                    locus_coordinates_0 = locus_coordinates

                for n_probe in range(self.N_probes):

                    if np.random.binomial(1, p_hyb) == 1:
                        locus_coordinates[n_probe, 0] = x0 + n_probe * dx / self.N_probes
                        locus_coordinates[n_probe, 1] = y0 + n_probe * dy / self.N_probes
                        locus_coordinates[n_probe, 2] = z0 + n_probe * dz / self.N_probes
                        locus_coordinates[n_probe, 3] = np.random.poisson(self.Intensity)

            if diploid:
                locus_coordinates = np.concatenate((locus_coordinates_0, locus_coordinates))
            key = "Locus_" + str(n)
            loci[key] = locus_coordinates

        return loci

    def define_false_positive_coordinates(self, n_detections):
        """
        Calculate the 3D position of each false positive detection. The calculation is performed in
        the same way than for the locus, though this time, only a single probe is used.

        Returns
        -------
        FP : dictionary containing all the 3D positions of the single probes
        """

        fp = dict()

        for n in range(n_detections):

            x0 = np.random.uniform(0, self.Lx, 1)
            y0 = np.random.uniform(0, self.Ly, 1)
            z0 = np.random.uniform(self.dz, self.Lz - self.dz, 1)

            fp_coordinates = np.zeros((1, 4))
            fp_coordinates[0, 0] = x0
            fp_coordinates[0, 1] = y0
            fp_coordinates[0, 2] = z0
            fp_coordinates[0, 3] = np.random.poisson(self.Intensity)

            key = "FP_" + str(n)
            fp[key] = fp_coordinates

        return fp
