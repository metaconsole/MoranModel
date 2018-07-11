# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:30:31 2018

@author: philipp.merz
"""

import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(context='talk', style='whitegrid', color_codes=True)
# sns.set_palette("Blues")
G = bd.np.array([[1.1, 2], [2.1, 1]])
# dominated strat 3, 0.001 5, 1 becaus of rounding error
moran = bd.MarkovChain(54, s =1, game= G)

Tau_unif = []

for N in range(5, 200):
    moran = bd.MarkovChain(N, s=1, game=G)
    Tau_unif.append(np.mean(moran.expected_visits))
    del moran
    
fig, ax = plt.subplots()
ax.plot(range(5, 200), Tau_unif, label="average fixation time")
ax.set_xlabel(r"$N$-population size")
ax.legend(loc="best") 
plt.show()
