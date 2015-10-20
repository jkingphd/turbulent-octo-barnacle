# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:36:41 2015

@author: jkk3
"""

import matplotlib.pyplot as plt
import numpy as np

pop = np.load('pop.npy')*100
t = np.arange(0,3600)

fig = plt.figure('Population', (9,6))
ax = fig.add_subplot(111)
ax.plot(t, pop[0], color = 'red', lw = 1.5, label = '1 mm')
ax.plot(t, pop[1], color = 'orange', lw = 1.5, label = '2 mm')
ax.plot(t, pop[2], color = 'green', lw = 1.5, label = '4 mm')
ax.plot(t, pop[3], color = 'blue', lw = 1.5, label = '6 mm')
ax.plot(t, pop[4], color = 'violet', lw = 1.5, label = '8 mm')
ax.legend()
ax.grid(True)
ax.axis([t[0], t[-1], 0, 100.0])
ax.set_ylabel("Population (%)", size = 16)
ax.set_xlabel("Time (s)", size = 16)
ax.set_title("Population vs. Time")

plt.show()