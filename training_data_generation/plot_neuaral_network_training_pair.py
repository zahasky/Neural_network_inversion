# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:10:03 2020

@author: zahas
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# training data iteration
td = 10


# layer to plot
ilayer = 0
# fontsize
fs = 18
hfont = {'fontname':'Arial'}

# number of layers 
nlay = 1
# Grid cell size
grid_size = [0.25, 0.25, 0.25] # selected units [cm]


# =============================================================================
# LOAD SELECTED EXAMPLE DATA 
# =============================================================================
# Set path to perm maps
perm_field_dir = os.path.join('.', 'matlab_perm_fields\\k_training_data')

# Set path to training data output
workdir = os.path.join('.', 'Tdata_2D\\td_1000')
# Import permeability map
tdata_km2 = np.loadtxt(perm_field_dir + '\\td_km2_' + str(td) +'.csv', delimiter=',')

nrow = int(tdata_km2[-2]) # number of rows / grid cells
ncol = int(tdata_km2[-1]) # number of columns (parallel to axis of core)
tdata_km2 = tdata_km2[0:-2]
raw_km2 = tdata_km2.reshape(nlay, nrow, ncol)


model_out_filename_sp = workdir + '\\td' + str(td) + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
tdata_ex = np.loadtxt(model_out_filename_sp, delimiter=',')
load_lx = tdata_ex[-1]
load_dp = tdata_ex[-2]
tdata_ex = tdata_ex[0:-2]
tdata_ex = tdata_ex.reshape(nlay, nrow, ncol)

# Generate pressure input with same dimensions as breakthrough time input
p_input = np.zeros((np.shape(tdata_ex)), dtype=np.float)
# Set inlet slice equal to pressure drop in kPa
p_input [:,:,0] = load_dp/1000


# =============================================================================
# PLOT DATA 
# =============================================================================
# Define grid    
# Describe grid for results    
Lx = (ncol - 1) * grid_size[1]   # length of model in selected units 
Ly = (nrow - 1) * grid_size[0]   # length of model in selected units 
y, x = np.mgrid[slice(0, Ly + grid_size[0], grid_size[0]),
                slice(0, Lx + grid_size[1], grid_size[1])]


# Second figure with head and breakthrough time difference maps
fig1 = plt.figure(figsize=(18, 10))
ax0 = fig1.add_subplot(2, 2, 2, aspect='equal')
imp = plt.pcolor(x, y, raw_km2[0,:,:], cmap='gray', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('[m^2]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
ax0.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Training Data #' + str(td) + ' Permeability Map', fontsize=fs+2, **hfont)

ax2 = fig1.add_subplot(2, 2, 1, aspect='equal')
imp = plt.pcolor(x, y, tdata_ex[ilayer,:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs)
plt.title('Breakthrough time difference map', fontsize=fs+2)

ax2 = fig1.add_subplot(2, 2, 3, aspect='equal')
imp = plt.pcolor(x, y, p_input[ilayer,:,:], cmap='BuPu', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('[kPa]', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs)
plt.title('Known grid cell pressure', fontsize=fs+2)
