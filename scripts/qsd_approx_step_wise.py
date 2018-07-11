# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:05:41 2018

@author: philipp.merz
"""


import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='talk', style='whitegrid')


G = bd.np.array([[1, 5], [5, 1]])
moran = bd.MarkovChain(100, s = 1, game= G)
# start = bd.np.ones((39,))/39
start = bd.np.random.rand(moran.N-1)
start /= sum(start)
# moran.start_path(1, 1, moran)
qsd_approxs= []

fig, ax = plt.subplots()
ax.plot(moran.trans, start, label= 'start')

for i in [20, 50, 200]:
    a, b, c = moran.qsd_approx_script(start, iterations = i) 
    qsd_approxs.append((a, b, c))
    ax.plot(moran.trans, a, label = 'steps '+str(c))

a, b, c = moran.qsd_approx_script(start, precision = 6)
ax.plot(moran.trans, a, label = 'steps '+str(c))


ax.legend(loc='upper right', shadow=True)
plt.show()