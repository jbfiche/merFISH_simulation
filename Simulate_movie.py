#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:26:32 2020

@author: jb
"""

import numpy
import numpy.random
import tifffile
import os
import matplotlib.pyplot as plt

os.chdir("/home/jb/Workspace/Python/HiM_simulation/")
print(os.getcwd())

n_frames = 1
x_size = 512
y_size = 512
gain = numpy.random.normal(loc=2.0, scale=0.1, size=(x_size, y_size))
offset = numpy.random.randint(95, high=106, size=(x_size, y_size)).astype(numpy.float)
read_noise = numpy.random.normal(loc=1.5, scale=0.1, size=(x_size, y_size))
c_data = [
    [0.0, "dark"],
    [1000.0, "light1"],
    [2000.0, "light2"],
    [3000, "light3"],
    [4000, "light4"],
]

print("Mean offset", numpy.mean(offset))
print("Mean read noise", numpy.mean(read_noise))


def makeMovie(name, average_intensity):
    with tifffile.TiffWriter(name) as tf:
        for i in range(n_frames):
            # Signal - Poisson distribution
            image = gain * numpy.random.poisson(
                lam=average_intensity, size=(x_size, y_size)
            )

            # Read Noise - Normal distribution
            image += numpy.random.normal(scale=read_noise, size=(x_size, y_size))

            # Camera baseline.
            image += offset

            # Add localizations
            loc = localization(10, 20, 2000, 1.3)
            image += loc

            # 16 bit camera.
            tf.save(numpy.round(image).astype(numpy.uint16))
    print("Made", name)


def localization(x, y, I, s):
    test = numpy.zeros((x_size, y_size))
    x0 = numpy.round(x)
    y0 = numpy.round(y)

    for i in range(-7, 7):
        for j in range(-7, 7):
            d = (x - x0 - i) ** 2 + (y - y0 - j) ** 2
            test[x0 + i, y0 + j] = I * numpy.exp(-d / (2 * s ** 2))

    return test


for elt in c_data:
    localization(10, 10, 2000, 2.5)
    makeMovie(elt[1] + ".tif", elt[0])
