# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:10:03 2020

@author: zahas
"""


import os
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import RegularGridInterpolator

from exp_bt_3d_functions import mean_exp_bt_map, plot_2d, axial_linear_interp_3D


data_filename = 'Bentheimer_4ml_2_3mm_vox_nan'
phi = 52.4/((3.1415*2.54**2)*10.3)

timestep = 3
# =============================================================================
# LOAD SELECTED EXAMPLE DATA 
# =============================================================================
# Set path to experimental data
# data_dir = os.path.join('.', 'Data')
data_dir = os.path.join('.')

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

# Expected advection velocity assuming 25% porosity
v = q/(3.1415*2.54**2)/60/phi

# crop off dataset information
pet_data = all_data[0:-10]
# reshape from imported column vector to 4D matrix
pet_data = pet_data.reshape(nrow, ncol, nslice, ntime)
# crop edges
pet_data = pet_data[1:-1, 1:-1, :, :]

# Plot at a few timesteps to make sure import is correct
plot_2d(pet_data[:,:,6,timestep], dx, dy, 'concentration', 0, np.nanmax(pet_data[:,:,6,timestep]), cmap='OrRd')
plot_2d(pet_data[:,11,:,timestep], dz, dy, 'concentration', 0, np.nanmax(pet_data[:,:,6,timestep]), cmap='OrRd')

# call function for performing moment analysis
bt_array, M0, M1, M2 = mean_exp_bt_map(pet_data, tstep, tracer_inj_duration, [dx, dy, dz], v)


plot_2d(bt_array[:,:,0], dx, dy, 'bt', -60, 60, cmap='bwr')
plot_2d(bt_array[:,11,:], dz, dy, 'bt', 0, 600, cmap='OrRd')

# plot_2d(advection_array[:,11,:], dz, dy, 'advection', 0, v)
# plot_2d(dispersion_array[:,11,:], dz, dy, 'dispersion', -0.001, 0.05)

print('Expected advection velocity = ' + str(v))

oned = np.nansum(np.nansum(pet_data, 0), 0)
timearray = np.arange(tstep/2, tstep*42, tstep)

m0 = np.trapz(oned[0,:], timearray)
# calculate first moment of grid cell 
m1 = np.trapz(oned[0,:]*timearray, timearray)

plt.plot(timearray, oned[0,:])
plt.plot([m1/m0, m1/m0], [0, 20])
plt.plot(timearray, oned[-1,:])

mean_bt_inlet = np.sum(bt_array[:,:,0])/324
print(mean_bt_inlet)
print(m1/m0)

## 3D linear interpolation, this is done after breakthrough time calculation
# pet_data_t = pet_data[:,:,:,timestep]
# pet_interp_3d, dz_interp = axial_linear_interp_3D(pet_data_t, dx, dy, dz, 40)
# plot_2d(pet_interp_3d[:,11,:], dz_interp, dy, 'concentration')




