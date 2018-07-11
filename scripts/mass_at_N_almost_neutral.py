# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:19:54 2018

@author: philipp.merz
"""


import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(context='talk', style='whitegrid', color_codes=True)
# sns.set_palette("Blues")
mass_at_N = []

for l in range(1, 400):
    G = bd.np.array([[1, l], [1, l]])
    moran = bd.MarkovChain(100, s=1, game=G)
    moran.fixation_prob_calc()
    mass_at_N.append(np.mean(moran.rho_i))
    del moran
    
    
    
    
fig, ax = plt.subplots()
ax.plot(range(1, 400), mass_at_N, label="mass at N")
ax.set_xlabel(r"$\beta$")
ax.legend(loc="best") 
plt.show()
