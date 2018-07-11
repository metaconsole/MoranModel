# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 07:36:20 2017

@author: philipp.merz
"""

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as pl
import sympy as sp

class MarkovChain:
    def moran_matrix(N, s, game):
        """transition Matrix for Moran modell with game = [[a, b][c, d]]
        a= 2, b = 5, c = 4, d = 1 ergab stabiles mixed equ 
        N... population size
        s... importance of the game
        a, b, c, d parameters of the game """
    
        T = np.zeros((N+1,N+1))
        T[0][0] = 1
        T[N][N] = 1
        a, b, c, d = game.flatten()
        for i in range(1, N):
            F = a*(i-1)+b*(N-i)
            G = c*i + d*(N-i-1)
            birth = ((N-i)*i*(1-s+s*F))/((i*(1-s+s*F)+(N-i)*(1-s +s*G))*N)
            death = ((N-i)*i*(1-s+s*G))/((i*(1-s+s*F)+(N-i)*(1-s +s*G))*N)
            T[i][i-1] = death
            T[i][i+1] = birth
            T[i][i] = 1- death - birth        
        return(T)
    
    def __init__(self, N, s= 0.3, game = np.array([[2, 0], [4, 1]])):
        self.states = np.arange(N+1)
        self.traps = (0, N)
        self.trans = tuple(range(1, N))
        self.game = game
        self.selection = s
        self.T = MarkovChain.moran_matrix(N, s, game)
        self.Q = self.T[1:-1, 1:-1]
    
    def stat_dist(self):
        n = np.shape(self.T)[0]
        b = np.zeros((n,))
        b[-1] = 1
        tem = (self.T-np.eye(n)).transpose()
        tem[-1, :] = np.ones((n,))
        res = la.solve(tem, b)
        self.v = res
        return(res)
        
    def qsd_approx(self):
        n = np.shape(self.Q)[0]
        b = np.zeros((n,))
        b[-1] = 1
        tem = (self.Q-np.eye(n)).transpose()
        tem[-1, :] = np.ones((n,))
        res = la.solve(tem, b)
        self.qsd1 = res
        return(res)

mor_1 = MarkovChain(200)
mor_2 = MarkovChain(1000)

