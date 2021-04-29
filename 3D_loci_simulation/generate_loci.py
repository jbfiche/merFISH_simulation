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
        
        self.Lx = param["detection"]["width_um"]
        self.Ly = param["detection"]["height_um"]
        self.Lz = param["detection"]["depth_um"]
        self.dz = param["detection"]["minimum_depth_um"]
        
        self.N_probes = param["detection"]["number_probes"]
        self.D_probes = param["detection"]["loci_size_kb"]

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
        self.D_diploid = param["detection"]["average_paired_distance_nm"] # average distance separating two paired loci

# Finally, the probability for each probe to hybridize is also defined as
# P_hybridization

        self.P_hybridization = param["detection"]["probe_hybridization_probability"]

# Define the average number of probes / images

        self.N_detection = param["detection"]["number_detections_per_image"]
        

    def define_coordinates(self):
        """
        Calculate the 3D position of each detection. The calculation is performed in
        steps :
        1- define for each locus the position of the first probe, following a uniform 
        distribution in XYZ. In the case of the Z position, the centroid is however 
        constrained between planes 10-50.  
        2- the distance between the first and last probes of the locus is calculated.
        The positions of each probe is then calculated using spherical coordinates
        only if the probe is hybridizing.

        Returns
        -------
        Loci : dictionnary containing all the 3D positions of the hybridized probes

        """

        loci = dict()
        
        for n in range(self.N_detection):
            
            x0 = np.random.uniform(0, self.Lx, 1)
            y0 = np.random.uniform(0, self.Ly, 1)
            z0 = np.random.uniform(self.dz, self.Lz - self.dz, 1)
        
            b = np.random.uniform(self.bmin, self.bmax, 1)
            g = np.random.uniform(self.gmin, self.gmax, 1)
            r = (g * self.D_probes**b)/1000 # returns the estimated size of the loci in um
            
            theta = np.random.uniform(0, np.pi, 1)
            phi = np.random.uniform(0, 2*np.pi, 1)
            
            dx = r * np.sin(theta) * np.cos(phi)
            dy = r * np.sin(theta) * np.sin(phi)
            dz = r * np.cos(theta)
            
            locus_coordinates = np.zeros((self.N_probes,3))
            
            for n_probe in range(self.N_probes):
                
                p = np.random.uniform(0, 1, 1)
                if p < self.P_hybridization:
                    
                    locus_coordinates[n_probe, 0] = x0 + n_probe*dx/self.N_probes
                    locus_coordinates[n_probe, 1] = y0 + n_probe*dy/self.N_probes
                    locus_coordinates[n_probe, 2] = z0 + n_probe*dz/self.N_probes
                
            p = np.random.uniform(0, 1, 1)
            if p < self.P_paired:    
                r = np.random.chisquare(1)*self.D_diploid
                
                XXXXX
                    
            key = "Locus_" + str(n)
            loci[key] = locus_coordinates
            
        return loci
            



