# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:35:32 2018

@author: philipp.merz
"""


import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sympy as sp
# from sympy.solvers.solveset import nonlinsolve 
from scipy.optimize import fsolve

sns.set(context='talk', style='whitegrid')


G = np.array([[1, 3], [3, 1]])
moran = bd.MarkovChain(100, s = 1, game= G)
start = np.zeros((moran.N-1,))
start[moran.N//2] = 1
moran.qsd_approx_script(start, precision= 12)

def quasi_stat_dist(q, mc, col=3):
    Q = mc.Q
    Q[:, col] = np.ones((mc.N-1))
    q_ = q* sum(q.dot(Q))
    q_[col] =1
    res = -np.subtract(q.dot(Q), q_)
    return res

def quasi_stat_dist_no_sub(q, mc):
    Q = mc.Q
    res = np.subtract(q.dot(Q), q*sum(q.dot(Q)))
    res = np.append(res, [sum(q)-1])
    return res

def quad(x):
    return( x[0]*x[1], x[0]**2*x[1])

x_0 = np.ones((moran.N-1,))/(moran.N-1)

qsd_solved = fsolve(quasi_stat_dist_no_sub, start, moran)

fig, (ax1, ax2) = plt.subplots(nrows = 2)
ax1.plot(moran.trans, qsd_solved)

ax2.plot(moran.trans, moran.qsd_q_power)
plt.show()

