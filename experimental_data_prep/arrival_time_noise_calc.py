# -*- coding: utf-8 -*-
"""
arrival_time_noise_calc

Created on Thu Apr  8 17:32:51 2021

@author: Christopher Zahasky (czahasky@wisc.edu)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from exp_arrival_3d_functions import plot_2d
from matplotlib.ticker import AutoMinorLocator


def load_arrival_data(arrival_data_filename):
    # data directory
    data_dir_arrival = os.path.join('.', 'pet_arrival_time_data')
    # data dimensions
    nslice = 40
    nrow = 20
    ncol = 20
    # print loaded filename
    print(arrival_data_filename + '.csv loaded')
    # Import data
    arrival_data = np.loadtxt(data_dir_arrival + '\\' + arrival_data_filename + '.csv', delimiter=',')
    # remove last value (average perm)
    arrival_data = arrival_data[0:-1]
    arrival_data = arrival_data.reshape(nrow, ncol, nslice)
    return arrival_data    

all_arrival_data = np.array([load_arrival_data('Berea_C1_1ml_2_3mm_at_norm'), \
                              load_arrival_data('Berea_C1_2ml_2_3mm_at_norm'), \
                              load_arrival_data('Berea_C1_3ml_2_3mm_at_norm'), \
                              load_arrival_data('Berea_C1_4ml_2_3mm_at_norm'), \
                              load_arrival_data('Berea_C1_6ml_2_3mm_at_norm')])
    
# all_arrival_data = np.array([load_arrival_data('Berea_C1_1ml_2_3mm_at_norm'), \
#                               load_arrival_data('Berea_C1_2ml_2_3mm_at_norm'), \
#                               load_arrival_data('Berea_C1_3ml_2_3mm_at_norm'), \
#                               load_arrival_data('Berea_C1_6ml_2_3mm_at_norm')])
    
n, r, c, s = all_arrival_data.shape

arrival_std = np.nanstd(all_arrival_data, axis=0)
arrival_mean = np.nanmean(all_arrival_data, axis=0)
arrival_mean[arrival_mean==0] = np.nan

# calculate the difference from the mean of each voxel in each experiment and
# save in a single vector
for i in range(0,n):
    print(i)
    diff = arrival_mean - all_arrival_data[i,:,:,:]
    d = diff.flatten()
    if i == 0:
        D = d
    else:
        D = np.append(D,d)
        D.shape
            
# remove nans
D = D[~np.isnan(D)]
# calculate histogram
histd, bin_edgesd = np.histogram(D, 50)
# generate synthetic noise
s = np.random.normal(0, np.std(D), len(D)+25000)
histn, bin_edgesn = np.histogram(s, 50)

# plot histogram
fs = 14 # fontsize
fig = plt.figure(figsize=(5, 4), dpi=200)
ax = fig.gca()
ax.plot(bin_edgesd[2:], histd[1:], 'k')
ax.plot(bin_edgesn[2:], histn[1:], 'r')
ax.set_title('Histogram of arrival time noise', fontsize=fs+2)
ax.set_ylabel('Counts (5 repeated images)', fontsize=fs)
ax.set_xlabel('Difference from voxel mean [-]', fontsize=fs)
ax.set_xlim((-0.06, 0.06))
# ax.xaxis.set_minor_locator(AutoMinorLocator(), direction="in")
# ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(direction="in", bottom=True, top=True, left=True, right=True, labelsize=fs-2)

print(np.mean(D))
print(np.std(D))



#dsd calculation
dsum = np.sum(D)
dsum2 = np.sum(D**2)
n = len(D)
dsd = np.sqrt((n*dsum2 - (dsum**2))/(n*(n-1)))
print(dsd)


# double check that there isn't any trend in noise across core
plot_2d(arrival_std[:,11,:], 0.2388, 0.2329, 'arrival std', cmap='bwr')
plot_2d(arrival_std[:,:,11], 0.2388, 0.2329, 'arrival std', cmap='bwr')

plot_2d(all_arrival_data[1,:,11,:],  0.2388, 0.2329, 'at norm_diff', cmap='bwr')

# arrival_std[arrival_std==0] = np.nan
# mean_std = np.nanmean(arrival_std)

# print(mean_std)
# hist, bin_edges = np.histogram(arrival_std, 50)

# plt.figure(figsize=(6, 4), dpi=200)
# plt.plot(bin_edges[2:], hist[1:])