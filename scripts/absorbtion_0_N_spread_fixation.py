# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:27:26 2018

@author: philipp.merz
"""


import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sympy as sp
import pandas as pd


sns.set(context='talk', style='whitegrid')


G = bd.np.array([[3, 1], [1, 2]])

spread = []

for N_ in range(10, 800):
    moran = bd.MarkovChain(N_, s = 1, game= G)
    moran.fixation_prob_calc()  
    chaos_range = np.where(np.logical_and(moran.rho_i>=0.001, 
                                          moran.rho_i<=0.999))[0]
    spread.append(chaos_range[-1]- chaos_range[0])

data = pd.DataFrame({"x":list(range(10, 800)), "spread":spread})

sns.lmplot(x="x", y="spread", data= data, order=2, ci=None,
           scatter_kws={"s":80})




#for i in range(1, moran.N):
#    moran.start_path(i, 1, moran)
#    moran.paths[1].long_run(1, 10)
#    a, b, c = moran.paths[1].absorbed_mass
#    absorp_0.append(a)
#    absorp_N.append(b)


    
#fig, ax = plt.subplots()
#
#ax.plot(moran.states, absorp_N, label=r"$\rho_i$")
#ax.scatter(33, absorp_N[33])
#sns.
#ax.legend()
#plt.show()
