# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 07:36:20 2017

@author: philipp.merz
"""

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as pl

#np.random.seed(19680801)

def sample_path(mc, start = None, step = 50):
    '''a module function that takes mc object to draw a sample path, using
    start as a starting point and draws one possible mc trajectorie
    mc object needs to have mc.P transition matrix'''
    x = np.arange(step+1)
    rands = np.random.rand(step+1)
    y = [start]
    P_cum = np.cumsum(mc.P, axis= 1)
    for i in range(step):
        y.append(len(P_cum[y[-1],:][P_cum[y[-1],:]<rands[i]]))
        if y[-1] in mc.traps:
            rest = [y[-1]]*(step-i-1)
            y = y + rest
            break
        
    return x, y, rands

def d_kl(x, y, distance = None):
    '''returns the kulback leibler- entropy between two distributions x, y. 
    Aslong as distance is None it returns it element wise.
    Distance from y to x'''
    res = np.multiply(x, np.log(np.divide(x, y)))
    if distance is not None:
        res = np.sum(res)
    return res

def expcted_dist(moran, dist):
    '''returns the expected distribution given a distribution and a moran chain
    Used to give stability criterion, ie if we are close to expected state 
    then we are stable. 
    defined on T, transient states'''
#    exp_dis = 1/moran.N*(np.multiply(np.arange(2, moran.N+1), moran.b_i[1:])+ \
#            np.multiply(np.arange(1, moran.N), moran.r_i[1:-1]) + \
#            np.multiply(np.arange(moran.N-1), moran.d_i[:-1]))
    N = moran.N
    exp_dis = (N-1)/N*dist + dist.dot(moran.Q)*1/N
    return(exp_dis)
        

class MarkovChain:
    def moran_matrix(N, s, game):
        """transition Matrix for Moran modell with game = [[a, b][c, d]]
        a= 2, b = 5, c = 4, d = 1 ergab stabiles mixed equ 
        i... Number of A individuals
        N-i..Number of B individuals
        N... population size
        s... importance of the game
        a, b, c, d parameters of the game """
    
        P = np.zeros((N+1,N+1))
        P[0][0] = 1
        P[N][N] = 1
        a, b, c, d = game.flatten()
        for i in range(1, N):
            F = a*(i)+b*(N-i)
            G = c*i + d*(N-i)
            #F = a*(i-1)+b*(N-i)
            #G = c*i + d*(N-i-1)
            birth = ((N-i)*i*(1-s+s*F))/((i*(1-s+s*F)+(N-i)*(1-s +s*G))*N)
            death = ((N-i)*i*(1-s+s*G))/((i*(1-s+s*F)+(N-i)*(1-s +s*G))*N)
            P[i][i-1] = death
            P[i][i+1] = birth
            P[i][i] = 1- death - birth        
        return(P)
    
    def __init__(self, N, s= 0.3, game = np.array([[2, 3], [4, 1]])):
        self.states = np.arange(N+1)
        self.N = N
        self.traps = (0, N)
        self.trans = tuple(range(1, N))
        self.game = game
        self.selection = s
        self.P = MarkovChain.moran_matrix(N, s, game)
        self.b_i = self.P.diagonal(1)
        self.d_i = self.P.diagonal(-1)
        self.r_i = self.P.diagonal()
        self.Q = self.P[1:-1, 1:-1]
        self.paths = {}
        self.expected_visits = np.sum(la.inv(np.eye(N-1)-self.Q), axis = 1)
        self.x_star_state()
        
    
    def stat_dist(self):
        '''calculates the stationary distribution of a MC, i.e.
        a left eigenvector to eigenvalue 1. res.dot(P) = res where res also 
        sums to 1.
        Fails if Matrix is singular'''
        
        n = np.shape(self.P)[0]
        b = np.zeros((n,))
        b[-1] = 1
        tem = (self.P-np.eye(n)).transpose()
        tem[-1, :] = np.ones((n,))
        res = la.solve(tem, b)
        self.v = res
        return(res)
        
    def qsd_eigen_approx(self, col = -1):
        '''calculates an approximation to the Quasi stationary distribution
        of a MC. It solves the equation res.dot(Q) = res where Q is P without
        the transient states, aditionally res sums to 1. 
        
        col.... which column should be discarded during the calculation. 
                Defaults to the last column.'''
        
        n = np.shape(self.Q)[0]
        b = np.zeros((n,))
        b[col] = 1
        tem = (self.Q-np.eye(n)).transpose()
        tem[col, :] = np.ones((n,))
        res = la.solve(tem, b)
        self.qsd_eigen_approx = res
        return(res)
        
    def qsd_left_eigen(self):
        eig, eig_vecs = np.linalg.eig(self.Q)
        n = np.argmax(eig)
        left = eig_vecs[n]
        left_val = eig[n]
        self.left = left
        return left, left_val
        
    def qsd_approx_script(self, start, iterations = None, precision = 6):
        '''calculates the Quasi stationary distribution of a MC starting from a
        given Vector. It multiplies it with Q and then renorms it the result to 
        be stochastic again.
        This will be iterated until the distribution stops changing or a given
        number of iterations is reached.'''
        n = np.shape(self.Q)[0]     
        if np.shape(start)[0] !=  n:
            return 'start vector has wrong dimensions.\n'\
                    'Need '+str(n)+ ' entries.'
        count = 0
        if iterations is None:
            cond = True
        else:
            cond = iterations > count 
        
        while cond:   
            step = start.dot(self.Q)
            step /= sum(step)
            if la.norm(start- step) < 10**-precision:
                break
            start = np.copy(step)
            count += 1
            if iterations is not None:
                cond = count < iterations                
        self.qsd_q_power = step
        kl_qsd = d_kl(step, start)
        return step, kl_qsd, count
    
    def qsd_approx_p(self, start, iterations = None, precision = 6):
        '''calculates the Quasi stationary distribution of a MC starting from a
        given Vector. It multiplies it with Q and then renorms it the result to 
        be stochastic again.
        This will be iterated until the distribution stops changing or a given
        number of iterations is reached.'''
        n = np.shape(self.Q)[0]     
        if np.shape(start)[0] !=  n:
            return 'start vector has wrong dimensions.\n'\
                    'Need '+str(n)+ ' entries.'
        count = 0
        if iterations is None:
            cond = True
        else:
            cond = iterations > count 
        
        while cond:   
            step = start.dot(self.Q.T)
            step /= sum(step)
            if la.norm(start- step) < 10**-precision:
                break
            start = np.copy(step)
            count += 1
            if iterations is not None:
                cond = count < iterations                
        self.qsd_p_power = step
        
        self.pseudo_qp = np.multiply(step, self.qsd_q_power)/ \
        sum(np.multiply(step, self.qsd_q_power)) 
        return step, count
    
    def x_star_state(self):
        a, b, c, d = self.game.flatten()
        self.x_interior = self.N*(d-b)/(a-b-c+d) -d +a
    
    def start_path(self, start, name, mc):
        '''method to initialize a Path Object and store it in an MarkovChain 
        Object.'''
        self.paths[name] = Path(start, name, mc)
    
    def plot_path(self, name, start = 0, stop = None, step = None):
        pl.plot(self.paths[name].compRun) #slicing dazu ansonsten unötige methode
        
    def fixation_prob_calc(self):
        '''method to calculate the fixation probability for every state in S
        Uses the explicit formula for rho_i'''
        prob_absorb_N = np.zeros((self.N+1,))
        prob_absorb_N[-1]= 1
        lamb = np.divide(self.d_i[:-1], self.b_i[1:])
        cum_lamb = np.cumprod(lamb)
        denom = np.sum(cum_lamb)
        denom += 1
        for j in range(1, self.N):
            nume = np.sum(cum_lamb[:j])
            nume += 1
            prob_absorb_N[j] = nume/denom
        
        self.rho_i = prob_absorb_N
            
class RPS_chain:
    def trans_prob(row_state, col_state, N, s, game):
        '''calculates the probability to transition from (i, j) to (i_2, j_2)'''
        i, j = row_state
        i_2, j_2 = col_state
        pop_ij = np.array([i/N, j/N, (N-i-j)/N])
        R_fit = (1 -s + s*game[0,:].dot(pop_ij))*i
        P_fit = (1- s + s*game[1,:].dot(pop_ij))*j
        S_fit = (1- s + s*game[2,:].dot(pop_ij))*(N-i-j)
        sum_fit = R_fit + P_fit + S_fit
        R_fit /= sum_fit
        P_fit /= sum_fit
        S_fit /= sum_fit
        if i_2 - i == 0:
            if j_2 - j == 0: # 0 0
                return (i/N)*R_fit + (j/N)*P_fit + ((N-i-j)/N)*S_fit
            elif j_2 -j == 1: # 0 +1
                return ((N-i-j)/N)*P_fit
            else: # 0 -1
                return (j/N)*S_fit
        elif i_2 - i == 1:
            if j_2 - j== 0: # 1 0
                return ((N-i-j)/N)*R_fit
            elif j_2 - j == -1: # 1 -1
                return (j/N)*R_fit
            else: # 1 1
                return 0
        else:
            if j_2 - j == 0: # -1 0
                return (i/N)*S_fit
            elif j_2 - j == 1: # -1 1
                return (i/N)*P_fit
            else: # -1 -1
                return 0
    
    def transition_matrix(N, s, game):
        '''transition matrix for RPS chain with M states and i R players, 
        j P players and N-i-j S players.'''
        M = int((N+1)*(N+2)/2)
        P = np.zeros((M, M))
        states_map = [(i, j) for i in range(N+1) 
        for j in range(N+1) if i + j <= N]
        traps = np.array([0, N, M-1])
        P[traps, traps] = np.ones((3,))
        
        for row in range(M):
            if row not in traps:
                for col in range(M):
                    if abs(states_map[row][0]- states_map[col][0]) < 2 and \
                    abs(states_map[row][1] - states_map[col][1]) < 2:
                        P[row][col] = RPS_chain.trans_prob(states_map[row], 
                                                           states_map[col], 
                                                           N, s, game)
    
        return M, states_map, P, traps

    def __init__(self, N, s= 0.3, game = 
                 np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])):
        '''constructor for a Rock Paper Scissors type Markov Chain
        M... number of possible (i, j) states
        states_map... array associating a (i, j) state with a k state in M '''
        
        M, states_map, P, traps = RPS_chain.transition_matrix(N, s, game)
        
        self.M = M
        self.N = N
        self.states_map = states_map
        self.P = P
        self.traps = traps
        self.selection = s
        self.game = game
        self.paths = {}
        self.states = np.arange(M)

    
    def start_path(self, start, name, mc):
        '''method to initialize a Path Object and store it in an MarkovChain 
        Object.'''
        self.paths[name] = Path(start, name, mc)
    
    def plot_states(self, path = None, start = None, step = None ):
        if path is not None:
            dist = self.paths[path].compRun[:,-1]
        elif start is not None and step is not None:
            pos = np.zeros((self.M,))
            pos[start] = 1
            dist = pos.dot(np.linalg.matrix_power(self.P, step))
        pic = np.ones((self.N +1, self.N +1))*(-1) # how to mark points that are not valid
        for i in range(self.M):
            pic[self.states_map[i]] = dist[i]
        
        pl.imshow(pic,  origin = 'lower')

class Path:
    def __init__(self, start, name, mc):
        if type(start) is int:
            arr = np.zeros(mc.states.shape[0])
            arr[start] = 1 
            self.start = arr
        elif(start== "uniform"):
            arr = np.ones(mc.states.shape[0])/mc.states.shape[0]
            self.start = arr
        elif(start == "random"):
            arr = np.random.rand(mc.states.shape[0])
            arr = arr / sum(arr)
            self.start = arr
        else:
            self.start = start

        
        self.absorbed_mass = [0, 0, 0]
        self.mc = mc
        
    def steps(self, n):
        return(self.start.dot(la.matrix_power(self.mc.P, n)))
        
    def complete_run(self, k, precision = 6):
        M = la.matrix_power(self.mc.P, k)
        q = self.start.dot(M)
        p = np.copy(q)
        i = 0
        compRun = np.ones((self.mc.states.shape[0], 10**6+3))
        compRun[:, 0] = self.start
        compRun[:, 1] = q
        while i <= 10**6:
            q = q.dot(M)
            if la.norm(q-p) < 10**-precision: # quality of approximation ( 9 is high)
                break
            p = np.copy(q)
            i += 1
            compRun[:, i+1] = q
        self.compRun = compRun[:, :i+2]
        self.steps_made = (i-1)*k
        self.absorbed_mass = [q[0], q[-1], q[0]+q[-1]]
        self.end_M = la.matrix_power(M, self.steps_made)
        
        states_mean= np.mgrid[0:i, 0:len(q)][1]
        self.run_mean = np.mean(np.multiply(states_mean, 
                                            compRun[:,i+2].T), 
                                axis=1)
        return(q, i, compRun[:, :i+2])
        
    def long_run(self, k, precision = 6):
        M = la.matrix_power(self.mc.P, k)
        q = self.start.dot(M)
        p = np.copy(q)
        i = 0
        states = self.mc.states
        mean_run = [np.sum(np.multiply(q, states))]
        
        while i <= 10**6:
            q = q.dot(M)
            if abs(sum(q)- 1) > 10**(-7):
                q /= sum(q)
            mean_run.append(np.sum(np.multiply(q, states)))
            if la.norm(q-p) < 10**-precision: # quality of approximation ( 9 is high)
                break
            p = np.copy(q)
            i += 1
        kl_long_run = d_kl(q, p)
        self.run_mean = mean_run
        self.steps_made = (i-1)*k
        self.absorbed_mass = [q[0], q[-1], q[0]+q[-1]]
        self.end_point = la.matrix_power(M, self.steps_made)
        return(q, i, kl_long_run)
        
        
        
        
        
        
        
        
        
        