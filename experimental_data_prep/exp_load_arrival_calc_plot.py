# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:10:03 2020

@author: zahas
"""


import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import RegularGridInterpolator

from exp_arrival_3d_functions import plot_2d, exp_arrival_map_function, axial_linear_interp_3D, quantile_calc
from temporal_moment_testing import mean_exp_bt_map


# Navajo
# data_filename = 'Navajo_2ml_1_2_3mm_cropped_nan'
# phi = 19.8/((3.1415*2.54**2)*10)
# km2 = 28.5*9.869233E-13/1000
# ketton
# data_filename = 'Ketton_2ml_2_3mm_cropped_nan'
# phi = 48.7/((3.1415*2.54**2)*10)
# km2 = 1920*9.869233E-13/1000
# Bentheimer
# data_filename = 'Bentheimer_2ml_2_3mm_cropped_nan'
# phi = 52.4/((3.1415*2.54**2)*10.3)
# km2 = 1722*9.869233E-13/1000
# Berea
# data_filename = 'Berea_C1_6ml_2_3mm_cropped_nan'
# phi = 42.0/((3.1415*2.54**2)*10)
# km2 = 23.2*9.869233E-13/1000
# Edwards
# data_filename = 'Edwards_2ml_2_3mm_cropped_nan'
# phi = 85.6/((3.1415*2.54**2)*10.3)
# km2 = 132*9.869233E-13/1000
# Estaillades
# data_filename = 'Estaillades_3ml_2_3mm_cropped_nan'
# phi = 52.2/((3.1415*2.54**2)*10.3)
# km2 = 608*9.869233E-13/1000
# Indiana
data_filename = 'Indiana_4ml_2_3mm_cropped_nan'
phi = 34.9/((3.1415*2.54**2)*10.3)
km2 = 98.3*9.869233E-13/1000

timestep = 4
# =============================================================================
# LOAD SELECTED EXAMPLE DATA 
# =============================================================================
# Set path to experimental data
data_dir = os.path.join('.', 'pet_data')
# data_dir = os.path.join('.')

# Import data
all_data = np.loadtxt(data_dir + '\\' + data_filename + '.csv', delimiter=',')

# PET_size(1); PET_size(2); PET_size(3); PET_size(4); timestep_length; q; inj_pv; vox_size(:)])
dz = all_data[-1] # voxel size in z direction (parallel to axis of core)
dy = all_data[-2] # voxel size in y direction
dx = all_data[-3] # voxel size in x direction
tracer_volume = all_data[-4] # tracer injected (ml)
q = all_data[-5] # flow rate (ml/min)
tstep = all_data[-6] # timstep length (sec)
ntime = int(all_data[-7])
nslice = int(all_data[-8])
nrow = int(all_data[-10])
ncol = int(all_data[-9])
# calculate tracer injection duration in seconds
tracer_inj_duration = tracer_volume/q*60 

# Expected advection velocity based on estimated pore volume
v = q/(3.1415*2.54**2)/60/phi
print('Expected advection velocity = ' + str(v))

# crop off dataset information
pet_data = all_data[0:-10]
# reshape from imported column vector to 4D matrix
pet_data = pet_data.reshape(nrow, ncol, nslice, ntime)
# crop edges
pet_data = pet_data[1:-1, 1:-1, :, :]

# Plot at a few timesteps to make sure import is correct
plot_2d(pet_data[:,11,:,timestep], dz, dy, 'concentration', cmap='OrRd')

# call function for calculating arrival time quantiles
at_array, at_array_norm, at_diff_norm = exp_arrival_map_function(pet_data, tstep, [dx, dy, dz], 0.5, 1)

plot_2d(at_array_norm[:,11,:], dz, dy, 'arrival time', cmap='YlGn')
plot_2d(at_diff_norm[:,:,20], dx, dy, 'arrival time', cmap='bwr')
plot_2d(at_diff_norm[:,11,:], dz, dy, 'arrival time', cmap='bwr')
plot_2d(at_diff_norm[11,:,:], dz, dy, 'arrival time', cmap='bwr')

## 3D linear interpolation, this is done after arrival time calculation
at_interp_3d, dz_interp = axial_linear_interp_3D(at_diff_norm, dx, dy, dz, 40)
plot_2d(at_interp_3d[:,11,:], dz_interp, dy, 'interp arrival time', cmap='bwr')

# save normalized breakthrough data
save_filename_atdn = 'arrival_time_data'  + '\\' + data_filename[:-12] + '_at_norm.csv'
save_data = np.append(at_interp_3d.flatten('C'), [km2])
np.savetxt(save_filename_atdn, save_data, delimiter=',')



# bt_array, bt_array_norm, bt_diff_norm, M0, M1, M2 = mean_exp_bt_map(pet_data, tstep, [dx, dy, dz], v)
# plot_2d(bt_diff_norm[:,11,:], dz, dy, 'interp arrival time', cmap='bwr')   
# noise = np.nanmean(np.nanmean(pet_data[:,:,:,0], 0), 0)

# pet_cut_noise = pet_data
# indices = np.argwhere(pet_cut_noise < (np.mean(noise)+2*np.std(noise)))







