# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:48:07 2017

@author: Nero
"""

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as pl
import sympy
import scipy.stats as st





def moran_matrix(N, s = 0.3, a= 2, b = 5, c = 4, d = 1):
    """transition Matrix for Moran modell with game [[a, b][c, d]]
     a= 2, b = 5, c = 4, d = 1 ergab stabiles mixed equ 
     N... population size
     s... importantcy of the game
     a, b, c, d parameters of the game """

    T = np.zeros((N+1,N+1))
    T[0][0] = 1
    T[N][N] = 1
    for i in range(1, N):
        F = a*(i-1)+b*(N-i)
        G = c*i + d*(N-i-1)
        birth = ((N-i)*i*(1-s+s*F))/((i*(1-s+s*F)+(N-i)*(1-s +s*G))*N)
        death = ((N-i)*i*(1-s+s*G))/((i*(1-s+s*F)+(N-i)*(1-s +s*G))*N)
        T[i][i-1] = death
        T[i][i+1] = birth
        T[i][i] = 1- death - birth        
    return(T)
        
def dist_n(n, q, P):
    """making n steps with chain P starting at q"""
    return(q.dot(la.matrix_power(P, n)))

def stat_dist(P):
    """computes the stationary distribution of a Markov Chain.
    raises singular Matrix exception if Matrix is not invertible."""
    n = np.shape(P)[0]
    b = np.zeros((n,))
    b[-1] = 1
    tem = (P-np.eye(n)).transpose()
    tem[-1, :] = np.ones((n,))
    res = la.solve(tem, b)
    return(res)
    
def quasi_stat(P):
    n = np.shape(P)[0]
    b = np.zeros((n,))
    b[-1] = 1
    
    return None

def mean_rec(P):
    return 0

def states(P):
    """returns the equivalence classes for the states, ie. classifies states
    as transient or recurrent"""
    return 0
    
def dist_plot(q):
    """plots a curve showing the distribution over the states """
    states = [x for x in range(len(q))]
    weights = [x for x in q]
 #   pl.scatter(states, weights)
    pl.plot(states, weights, "b-")
    pl.plot(0, q[0], "gs")
    pl.plot(states[-1], q[-1], "gs")
    pl.grid(True)
    pl.show()

###########MAINMAINMAIN##############


"""starting distribution"""

moran = moran_matrix(200)
N = moran.shape[0]
q_moran = moran[1:-1, 1:-1]

q = np.zeros((N,))
q[100] = 1
q_n = dist_n(2000, q, moran)
v = stat_dist(q_moran)

uni = np.ones((N,))/N

mor_2 = moran_matrix(200, s=0)
q_mor_2 = dist_n(2000, q, mor_2)
v_2 = stat_dist(mor_2[1:-1, 1:-1])
q_mor_3 = dist_n(100000, q, mor_2)

# no globals
""" transition Matrix for symetric random walk with abs barriers 0, N"""
srw = np.zeros((N,N))

for i in range(N-2):
    srw[i+1][i] = 1/2
    srw[i+1][i+2] = 1/2
srw[0][0] = 1
srw[N-1][N-1] = 1

# no globals
""" transition Matrix for symetric random walk with reflecting
     barriers at 0, N"""
srw_re = np.zeros((N,N))

for i in range(N-2):
    srw_re[i+1][i] = 1/2
    srw_re[i+1][i+2] = 1/2
srw_re[0][1] = 1
srw_re[N-1][N-2] = 1

# no globals
""" transition Matrix for cyclic symetric random walk """
srw_cy = np.zeros((N,N))

for i in range(N-2):
    srw_cy[i+1][i] = 1/2
    srw_cy[i+1][i+2] = 1/2
srw_cy[0][N-1] = 0.5
srw_cy[0][1] = 0.5
srw_cy[N-1][0] = 0.5
srw_cy[N-1][N-2] = 0.5


