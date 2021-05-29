# -*- coding: utf-8 -*-
"""
porosity_field_generation_function
Created on Mon May 24 17:04:29 2021

@author: Czahasky
"""


import numpy as np
import matplotlib.pyplot as plt



k = np.logspace(0, 5, 5)

# phi = (np.log(k) + np.log(0.1))/0.2
# phi2 = (np.log(k) + np.log(0.1))/0.325
# phi3 = (np.log(k) + np.log(0.1))/0.4

# a = 0.25-1.0
# b = 2-20
# phi = (log(k)/a + b)/100
phi4 = ((np.log(k)/0.25)+5)/100
phi5 = ((np.log(k)/0.325)+10)/100
phi6 = ((np.log(k)/0.25)+20)/100

# plt.plot(phi, k)
# plt.plot(phi2, k)
# plt.plot(phi3, k)

plt.plot(phi4, k)
plt.plot(phi5, k)
plt.plot(phi6, k)

plt.yscale('log')
