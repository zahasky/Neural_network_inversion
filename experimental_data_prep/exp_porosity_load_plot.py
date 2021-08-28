# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:10:03 2020

@author: zahas
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
import torch
# from scipy.interpolate import RegularGridInterpolator

from exp_arrival_3d_functions import *
from temporal_moment_testing import *



# Berea
# data_filename = 'Berea_porosity_uncoarsened_nan'
# Navajo
# data_filename = 'Navajo_porosity_coarsened'
data_filename = 'Navajo_porosity_uncoarsened'

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
nslice = int(all_data[-4])
print(nslice)
nrow = int(all_data[-5])
ncol = int(all_data[-6])


# crop off dataset information
data = all_data[0:-6]
# reshape from imported column vector to 4D matrix
data = data.reshape(nrow, ncol, nslice)
# Flip data to orient top of cores to top of plot
# data = np.flip(data, 0)


# Plot at a few timesteps to make sure import is correct
# plot_2d(data[:,10,:], dz, dy, 'porosity', cmap='viridis')
plot_2d(data[:,:,0], dx, dy, 'porosity', cmap='viridis')

# remove nans
data[np.isnan(data)]=0
data[data<0]=0.001
data[data>0.5]=0.5

por_tensor = torch.from_numpy(data.copy())
por_tensor = por_tensor.reshape(1, nrow, ncol, nslice)

# pytorch downsample
# downsample along axis of core
por_tensor_coarse = interpolate(por_tensor, size=[ncol,40], mode='bilinear')
# permute and downsample plane perpendicular to long axis of core
por_tensor_coarse = por_tensor_coarse.permute(0, 3, 1, 2)
por_tensor_coarse = interpolate(por_tensor_coarse, size=([20, 20]), mode='bilinear')
# permute back to original orientation 
por_tensor_coarse = por_tensor_coarse.permute(0, 2, 3, 1)

# Hard coded mask for 20x20 grid cross-section
mp = np.array([7, 5, 3, 2, 2, 1, 1])
mask_corner = np.ones((10,10))
for i in range(len(mp)):
    mask_corner[i, 0:mp[i]] = 0
    
mask_top = np.concatenate((mask_corner, np.fliplr(mask_corner)), axis=1)
mask = np.concatenate((mask_top, np.flipud(mask_top)))

# print(mask)

# plot comparison of before and after
plot_2d(mask, dx, dy, 'mask', cmap='gray')
plot_2d(por_tensor[0, :,:,22], dx, dy, 'porosity', cmap='viridis')
plt.clim(0.05, 0.2)
plot_2d(por_tensor_coarse[0,:,:,11], dx*(nrow/20), dy*(ncol/20), 'porosity', cmap='viridis')
plt.clim(0.05, 0.2)

for col in range(40):
    por_tensor_coarse[0,:,:,col] = np.multiply(por_tensor_coarse[0,:,:,col], mask)
    
plot_2d(por_tensor_coarse[0,:,:,11], dx*(nrow/20), dy*(ncol/20), 'porosity', cmap='viridis')
plt.clim(0.05, 0.2)

plot_2d(por_tensor_coarse[0,:,10,:], dx*(nrow/20), dy*(ncol/20), 'porosity', cmap='viridis')
plt.clim(0.05, 0.2)