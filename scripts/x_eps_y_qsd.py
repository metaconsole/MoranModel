# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 21:35:22 2018

@author: philipp.merz
"""

import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



sns.set(context='talk', style='whitegrid', color_codes=True)
qsd_s = []
mass_N = []
start = bd.np.zeros(49)
start[24] = 1

for eps in np.linspace(0.001, 2, num= 100):
    G = bd.np.array([[1, 1+eps], [1, 1]])    
    moran = bd.MarkovChain(50, s =1, game= G)
    moran.fixation_prob_calc()
    mass_N.append(np.mean(moran.rho_i))
    moran.qsd_approx_script(start, precision= 10)
    qsd_s.append(moran.qsd_q_power)
    
for qs in qsd_s:
    plt.plot(qs)
    
    
    