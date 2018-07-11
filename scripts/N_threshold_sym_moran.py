# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 16:37:57 2018

@author: philipp.merz
"""

import birthdeathchain as bd
import matplotlib.pyplot as pl
import numpy as np

Game = bd.np.array([[1, 1.01], [1.01, 1]])
mass = {}
runs = {}


for i in range(520, 600, 16):
    moran = bd.MarkovChain(i, s= 0, game = Game)
    moran.start_path(i//2, str(i)+'_run', moran)
    moran.paths[str(i)+'_run'].completeRun(10)
    mass[str(i)+'_run_abs_mass'] = moran.paths[str(i)+'_run'].absorbed_mass
    print(i, moran.paths[str(i)+'_run'].absorbed_mass)
    del moran
    













# test for really high iterations of transition Matrix, inconclusive: looked 
# like nothing changed even for absurd numbers when we renormalised
#
#for i in range(1, 9):
#    print(i)
#    mor_temp = bd.MarkovChain(i*50, game = Game)
#    mor_temp.start_path(i*25, 'dum', mor_temp)
#    mor_temp.paths['dum'].completeRun(50)
#    mass[str(i*50)+'abs_mass'] = mor_temp.paths['dum'].absorbed_mass
#    runs[str(i*50)+'qsd_app'] = mor_temp.paths['dum'].compRun[:,-1]


