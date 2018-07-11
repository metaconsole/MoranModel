# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:49:30 2017

@author: philipp.merz
"""
import birthdeathchain as bd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(context='talk', style='whitegrid', color_codes=True)
# sns.set_palette("Blues")
G = bd.np.array([[1.1, 2], [2.1, 1]])
# dominated strat 3, 0.001 5, 1 becaus of rounding error
moran = bd.MarkovChain(100, s =1, game= G)

starting_points =  bd.np.random.randint(0, moran.N, 4)
#[2, 20, 50, 90, 97]
#### plotting 4 different mc paths #####

fig, ax = plt.subplots()

for k in starting_points:
    a, b, c = bd.sample_path(moran, start = k, step= 10**3 )
    ax.plot(a, b, label='start at '+str(k))


legend = ax.legend(loc='best', shadow=True)
# legend.get_frame().set_facecolor('#dce2e2')
# show the results
plt.show()
############

uniform = bd.np.ones(moran.N-1,)/(moran.N-1)
moran.start_path("uniform", 1, moran)

# qsd_simulated_comp, steps, comp = moran.paths[1].complete_run(10, 18)
qsd_simulated, kl_long_run, *steps = moran.paths[1].long_run(1, 6)


# script approximation of qsd
# start = bd.np.ones(moran.N-1)
# start = start/sum(start)
start = bd.np.zeros(moran.N-1)
start[moran.N//2] = 1
qsd_script, kl_qsd, count = moran.qsd_approx_script(start, precision = 12)
pseudo, count2 = moran.qsd_approx_p(start, precision = 12)

###########
sns.set_palette("muted")
sns.set_context("poster")
fig, ax = plt.subplots()
# ax.plot(trans, moran.qsd_eigen_approx, label ='left eigenv. of Q, approx of QSD')
ax.plot(moran.states, qsd_simulated, label ='LTD')
ax.plot(moran.trans, pseudo, label='PSD')
ax.plot(moran.trans, qsd_script, label ='QSD')
legend = ax.legend(loc='best')
# ax.set_xlabel("states")
plt.show()

absorb_mass = moran.paths[1].absorbed_mass
############
fig, ax = plt.subplots()
ax.plot(moran.trans, moran.expected_visits, label="average fixation time ")
ax.legend(loc='best')
plt.show()
##########
fig, ax = plt.subplots()
ax.plot(moran.states[:-1], moran.b_i, label="birth prob")
ax.plot(moran.states[1:], moran.d_i, label="death prob")
ax.plot(moran.states, moran.r_i, label="remain prob")
ax.legend(loc='best')
plt.show()
############

fig, ax = plt.subplots()
moran.fixation_prob_calc()
ax.plot(moran.states, moran.rho_i, label=r"$\rho_i$")
ax.legend(loc='best')