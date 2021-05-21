# -*- coding: utf-8 -*-
"""
gsfield_generation_demo
Created on Wed Apr 28 12:55:06 2021
@author: Christopher Zahasky


"""
# pip install scikit-optimize
import numpy as np
import gstools as gs
import matplotlib.pyplot as plt
from skopt.sampler import Lhs
import time

lhs = Lhs(lhs_type="classic", criterion=None)

# log_var, log_mD(10 mD to 20 D), x len, y len, z len, rotation x, rotation y, rotation z
Ps = lhs.generate([(-4., -0.5), (1.0, 4.3), (1.0, 50.), (1.0, 50.0), (1.0, 50.0), \
                  (0.0, 1.57), (0.0, 1.57), (0.0, 1.57), (1,3)], 15)

# example of how to efficiently save data to text file    
# np.savetxt('parameter_space_26k.csv', Ps , delimiter=',', fmt='%.3e')
    
start_td = time.time() # start a timer
# x = y = np.arange(100)
x = np.arange(20)
y = np.arange(20)
z = np.arange(40)

# log_var = 0.3 # max
# log_var = 0.0001 # min
log_var = -1.0
log_mD = 2

# model = gs.Exponential(dim=3, var=10**log_var, len_scale=[5.0, 10.0, 1.0], angles=[0, 3.14/2, 0.0])

for i in range(0, 10):
    p = Ps[i]
    
    model = gs.Exponential(dim=3, var=10**p[0], len_scale=[p[2], p[3], p[4]], angles=[p[5], p[6], p[7]])
    srf = gs.SRF(model, seed=20170519)
    
    field = 10**(srf.structured([x, y, z]) + p[1])
    
    end_td = time.time() # end timer
    print('Sec to run generate field: ', (end_td - start_td)) # show run time
    print(np.log10(np.max(field)/np.min(field)))
    # srf.plot()
    
    # plt.figure(figsize=(5, 4), dpi=200)
    # plt.pcolor(np.log10(field[:, :, 0]))
    # plt.gca().set_aspect('equal')
    # cbar = plt.colorbar()
    # cbar.set_label('log10(mD)')
    
    # plt.figure(figsize=(5, 4), dpi=200)
    # plt.pcolor(np.log10(field[:, 0, :]))
    # plt.gca().set_aspect('equal')
    
    plt.figure(figsize=(5, 4), dpi=200)
    plt.pcolor(field[0, :, :])
    plt.gca().set_aspect('equal')
    cbar = plt.colorbar()
    cbar.set_label('mD')
    
    plt.figure(figsize=(5, 4), dpi=200)
    plt.pcolor(field[:, :, 0])
    plt.gca().set_aspect('equal')
    cbar = plt.colorbar()
    cbar.set_label('mD')

# histogram analysis
# a = np.zeros((100,1))
# n=0
# for list in x:
#     a[n]=list[0]
#     n=n+1
    
# hist, bin_edges = np.histogram(a, 50)
# plt.plot(bin_edges[:-1], hist )