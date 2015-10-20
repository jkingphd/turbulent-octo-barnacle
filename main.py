# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:52:03 2015
Diffusion_sim.py
@author: jkk3
"""

import numpy as np

def sim_diff(L1, L2, L3, D1, D2, dt, t_final, n_trials):
    '''Simulate 1-D Brownian motion of n particles in two-zone region.'''
    t_final = t_final/dt
    a = L1; b = L1 + L2;
    L_total = L1 + L2 + L3
    sigma_1 = np.sqrt(2*D1*dt)
    sigma_2 = np.sqrt(2*D2*dt)
    
    pop_data = np.zeros(t_final)
    hist_data = np.zeros((t_final, np.int(L_total/L2)-1))
    bins = np.arange(0, L_total, L2)
    pos_i = np.random.uniform(0., L1, n_trials)
    pos_f = np.zeros(np.shape(pos_i))
    hist_data[0] = np.histogram(pos_i, bins = bins)[0]

    for t in np.arange(1, t_final):
        
        # Have each particle take a normally distributed step based on current position
        pos_f = pos_i + np.random.normal(scale = sigma_1, size = n_trials)
        
        # Case 1: ((a < pos_i) & (pos_i < b))
        idx_1 = (pos_i > a) & (pos_i < b)
        pos_f[idx_1] = pos_i[idx_1] + np.random.normal(scale = sigma_2, size = idx_1.sum())
        
        # Case 2: ((a < pos_i) & (pos_i < b)) & (a > pos_f)
        # Particle starts inside of the membrane and exits to the left
        idx_2 = ((a < pos_i) & (pos_i < b)) & (a > pos_f)
        s_2 = 1 - np.abs(a - pos_i[idx_2])/np.abs(pos_f[idx_2] - pos_i[idx_2])
        pos_f[idx_2] = a + np.random.normal(scale = sigma_1, size = idx_2.sum())*s_2
    
        
        # Case 3: ((a < pos_i) & (pos_i < b)) & (b < pos_f)
        # Particle starts inside of the membrane and exits to the right
        idx_3 = ((a < pos_i) & (pos_i < b)) & (b < pos_f)
        s_3 = 1 - np.abs(b - pos_i[idx_3])/np.abs(pos_f[idx_3] - pos_i[idx_3])
        pos_f[idx_3] = b + np.random.normal(scale = sigma_1, size = idx_3.sum())*s_3
        
        # Case 4: pos_i < a & pos_f > a
        # Particle starts the left of the membrane and enters membrane
        idx_4 = (pos_i < a) & (pos_f > b)
        s_4 = 1 - np.abs(a - pos_i[idx_4])/np.abs(pos_f[idx_4] - pos_i[idx_4])
        pos_f[idx_4] = a + np.random.normal(scale = sigma_2, size = idx_4.sum())*s_4
    
        # Case 5: pos_i > b & pos_f < b
        # Particle starts the right of the membrane and enters membrane
        idx_5 = (pos_i > b) & (pos_f < b)
        s_5 = 1 - np.abs(b - pos_i[idx_5])/np.abs(pos_f[idx_5] - pos_i[idx_5])
        pos_f[idx_5] = b + np.random.normal(scale = sigma_2, size = idx_5.sum())*s_5
        
        # Check boundaries
        pos_f[pos_f < 0] = np.abs(pos_f[pos_f < 0])
        pos_f[pos_f > L_total] = L_total - np.abs(pos_f[pos_f > L_total] - pos_i[pos_f > L_total])
        
        # Calculate fraction of population in target region and store
        pop_data[t] = np.sum(pos_f > L1 + L2)/np.float(n_trials)
        hist_data[t] = np.histogram(pos_f, bins = bins)[0]
        
        # Throw out old data
        pos_i = pos_f
        
    return pop_data, hist_data
    
L1 = np.array([1000., 2000., 4000., 6000., 8000.])
L2 = 10.
L3 = 7000.
D1 = 590.
D2 = D1*0.03
dt = 1.0
t_final = 3600
n_trials = 100000

pop = np.zeros((L1.size, t_final))
for i in range(L1.size):
    pop[i] = sim_diff(L1[i], L2, L3, D1, D2, dt, t_final, n_trials)[0]
    
np.save('pop.npy', pop)