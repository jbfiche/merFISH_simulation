#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:47:16 2021

@author: fiche
"""

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import os

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, download_and_extract_zip_file

from stardist import relabel_image_stardist3D, Rays_GoldenSpiral, calculate_extents
from stardist import fill_label_holes, random_label_cmap
from stardist.matching import matching_dataset

np.random.seed(42)
lbl_cmap = random_label_cmap()


data_dir = '/mnt/grey/DATA/users/JB/Simulations_3D/2021-06-11_10-54/Training_data_thresh_4_Raw'

os.chdir(data_dir)

# Load the training data
# ----------------------

X = sorted(glob('train/raw/*raw*.tif'))
Y = sorted(glob('train/label/*label*.tif'))

X_trn = list(map(imread,X))
Y_trn = list(map(imread,Y))
n_channel = 1 if X_trn[0].ndim == 3 else X_trn[0].shape[-1]

# Load the validation/testing data
# --------------------------------

X = sorted(glob('test/raw/*raw*.tif'))
Y = sorted(glob('test/label/*label*.tif'))

X_val = list(map(imread,X))
Y_val = list(map(imread,Y))
n_channel = 1 if X_val[0].ndim == 3 else X_val[0].shape[-1]

# Calculate the anisotropy of the data
# ------------------------------------

extents = calculate_extents(Y_val)
anisotropy = tuple(np.max(extents) / extents)
print('empirical anisotropy of labeled objects = %s' % str(anisotropy))

# Calculate the total number of objects for the training and validation data
# --------------------------------------------------------------------------

n = 0
for i in range(len(Y_trn)):
    y = Y_trn[i]
    n = n + len(np.unique(y))

print('The total number of objects segmented for the training data is ' + str(n))

n = 0
for i in range(len(Y_val)):
    y = Y_val[i]
    n = n + len(np.unique(y))

print('The total number of objects segmented for the validation data is ' + str(n))

# Test the ground truth data against the star-convex model
# --------------------------------------------------------

def reconstruction_scores(n_rays, Y, anisotropy):
    scores = []
    for r in tqdm(n_rays):
        rays = Rays_GoldenSpiral(r, anisotropy=anisotropy)
        Y_reconstructed = [relabel_image_stardist3D(lbl, rays) for lbl in Y]
        mean_iou = matching_dataset(Y, Y_reconstructed, thresh=0, show_progress=False).mean_true_score
        scores.append(mean_iou)
    return scores

n_rays = [8, 16, 32, 64, 96, 128]
scores_iso   = reconstruction_scores(n_rays, Y_val, anisotropy=None)
scores_aniso = reconstruction_scores(n_rays, Y_val, anisotropy=anisotropy)

fig = plt.figure(figsize=(8,5))
plt.plot(n_rays, scores_iso,   'o-', label='Isotropic')
plt.plot(n_rays, scores_aniso, 'o-', label='Anisotropic')
plt.xlabel('Number of rays for star-convex polyhedra')
plt.ylabel('Reconstruction score (mean intersection over union)')
plt.legend()

fig_name = data_dir + '/' + 'Star_convex_fit.png'
fig.savefig(fig_name)

             