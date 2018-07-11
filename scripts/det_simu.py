# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:15:58 2018

@author: philipp.merz
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import birthdeathchain as bd

sns.set(context='talk', style='whitegrid')


def rep_moran(game, start, h= 0.01, steps = 1000):
    a, b, c, d = game.flatten()
    
    y_t = [start]
    
    for i in range(steps):
        y_t.append(y_t[-1]+h*y_t[-1]*(1-y_t[-1])*(y_t[-1]*\
                   (a-c)+(1-y_t[-1])*(b-d)))
    
    time = np.arange(0, h*(steps+1), 0.01)
    return time, y_t

def x_star(game):
    a, b, c, d = game.flatten()
    return (d-b)/(a-b-c+d)

#def deterministic_simu(game, start, k= 10, h= 0.01, steps= 100):
#    x_st = x_star(game)*np.ones((steps+1))
#    for i in range(10):
  
G = np.array([[1.1, 2], [2.1, 1]])
      
fig, ax = plt.subplots()
starts = [0.01, 0.2, 0.5, 0.9,  0.99]
for i in starts:
    t, y = rep_moran(G, i)
    ax.plot(t, y, label="start at "+str(i))

ax.legend(loc='best')
plt.show()
 

x_result = []
x_moran = []
x_interior = []
x_axis =np.linspace(-1.0001, 0.9999, num = 5000)
for l in x_axis:
    G_temp = np.array([[1, 1+l], [1, 1]])
    x_result.append(x_star(G_temp))
    moran = bd.MarkovChain(100, s=1, game=G_temp)
    x_moran.append(np.argmin(abs(np.subtract(moran.b_i[1:], 
                                             moran.d_i[:-1])))+1)
    x_interior.append(moran.x_interior)
    del moran

fig, ax = plt.subplots()
ax.plot(x_axis, np.divide(x_moran, 100), label=r"interior state $\frac{i}{100}$")
ax.plot(x_axis, x_result, label=r"$x^*$")
ax.legend(loc='best')
ax.set_xlabel(r"$\gamma$")
plt.show()


    

   