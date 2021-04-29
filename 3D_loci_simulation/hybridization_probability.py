#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:53:44 2021

@author: jb
"""

from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

N_probe = 45
N_limit = 10

Prob = np.zeros((100,))

for n in range(100):
    
    P = 0
    
    for x in range(N_limit):
        
        f = x/N_probe
        p = n/100
        P = P + binom.pmf(x, N_probe, p)
        
    Prob[n] = P
    
fig, ax = plt.subplots()
ax.plot(np.arange(0,1,0.01),Prob)
ax.set(xlabel='p value', ylabel='probability')
ax.grid()
# fig.savefig("/home/jb/Desktop/pulse_stability_I.png", dpi=500)
plt.show()
