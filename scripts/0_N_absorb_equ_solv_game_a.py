# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:04:27 2018

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

N = 4
a = sp.Symbol("a")
game =[[1, a], [a, 1]]
s = 1

b_i = []
r_i = []
d_i = []
for i in range(1, N):
    F = a*(i-1)+a*(N-i)
    G = i + (N-i-1)
    birth = ((N-i)*i*(1-s+s*F))/((i*(1-s+s*F)+(N-i)*(1-s +s*G))*N)
    death = ((N-i)*i*(1-s+s*G))/((i*(1-s+s*F)+(N-i)*(1-s +s*G))*N)
    b_i.append(birth)
    d_i.append(death)
    r_i.append(1- death - birth)
    

q_string = ""

for i in range(1,N):
    q_string += "q_"+str(i)+" "

q = sp.symbols(q_string)

c = 1- q[0]*d_i[0] - q[-1]*b_i[-1]

equ = [-q[0]*c + q[0]*r_i[0]+ q[1]*d_i[1], 
       -q[-1]*c + q[-1]*r_i[-1]+ q[-2]*b_i[-2]]

for l in range(1, N-2):
    equ.append(-q[l]*c +q[l-1]*b_i[l-1]+ q[l]*r_i[l]+ q[l+1]*d_i[l+1])

equ.append(sum(q)-1)
unknowns = list(q)+[a]
result = nonlinsolve(equ, unknowns)
print(result)