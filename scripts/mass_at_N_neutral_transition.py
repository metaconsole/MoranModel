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
x_axis = np.linspace(-0.999, 0.999, 10000)
for l in x_axis:
    G = bd.np.array([[1, 1], [1, 1+l]])
    moran = bd.MarkovChain(100, s=1, game=G)
    moran.fixation_prob_calc()
    mass_at_N.append(np.mean(moran.rho_i))
    del moran
    
    
    
    
fig, ax = plt.subplots()
ax.plot(x_axis, mass_at_N, label="mass at N")
ax.set_xlabel(r"$\beta$")
ax.legend(loc="best") 
plt.show()
