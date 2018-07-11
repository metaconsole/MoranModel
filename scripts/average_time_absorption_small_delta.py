# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:20:32 2018

@author: philipp.merz
"""

import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set(context='talk', style='whitegrid', color_codes=True)
# sns.set_palette("Blues")

Tau_v= []
mass_at_N = []

for delta in np.linspace(0, 0.99, num= 1000):
    G = bd.np.array([[1, 1+ delta], [1, 1-delta]])
    G_string = str(G)[1:-1]
    moran = bd.MarkovChain(100, s=1, game=G)
    Tau_v.append(np.mean(moran.expected_visits))
    moran.fixation_prob_calc()
    mass_at_N.append(np.mean(moran.rho_i))
    del moran
    
