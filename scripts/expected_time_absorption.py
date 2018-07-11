# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:22:16 2018

@author: philipp.merz
"""

import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='paper', style='whitegrid')

# Moran model mit N = 199 (symetrie), selektion 0.3 und spiel G

G = bd.np.array([[1, 2], [2, 1]])
moran = bd.MarkovChain(30, s =1, game= G)

# moran.start_path(1, 1, moran)
Tau_1 = []
for _ in range(20):
    a, b, c = bd.sample_path(moran, start=1, step=10**7)
    Tau_1.append(len(b)-1)
    
