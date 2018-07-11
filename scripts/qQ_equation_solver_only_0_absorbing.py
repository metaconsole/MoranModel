# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:50:52 2018

@author: philipp.merz
"""

import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sympy as sp
from sympy.solvers.solveset import nonlinsolve 

sp.init_printing()
sns.set(context='talk', style='whitegrid')


G = bd.np.array([[1, 3], [3, 1]])
moran = bd.MarkovChain(3, s = 1, game= G)
# N gleich 17 ist schon nicht mehr abzuwarten
N = moran.N


q_string = ""

for i in range(1, moran.N+1):
    q_string += "q_"+str(i)+" "

q = sp.symbols(q_string)
b_i = [1/4]*3
# np.copy(moran.b_i[1:])
d_i = [1/4, 1/4, 1/2, 1/2]
# np.copy(moran.d_i)
#d_i[-1] = 0.02
r_i = [1/2]*4
# np.copy(moran.r_i[1:])
#r_i[-1] = 0.98

c = 1- q[0]*d_i[0]

equ = [-q[0]*c + q[0]*r_i[0]+ q[1]*d_i[1], 
       -q[-1]*c + q[-1]*r_i[-1]+ q[-2]*b_i[-2]]

for l in range(1, moran.N-1):
    equ.append(-q[l]*c +q[l-1]*b_i[l-1]+ q[l]*r_i[l]+ q[l+1]*d_i[l+1])

# i_to_delete = 3

# equ.pop(i_to_delete)
# equ.append(sum(q)-1)

result = nonlinsolve(equ, q)
print(result)