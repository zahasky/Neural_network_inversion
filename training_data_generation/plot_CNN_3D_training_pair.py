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
td = 2


# layer to plot
ilayer = 0
# fontsize
fs = 18
hfont = {'fontname':'Arial'}

# number of layers 
# nlay = 1
# Grid cell size
grid_size = [0.25, 0.25, 0.25] # selected units [cm]


# =============================================================================
# LOAD SELECTED EXAMPLE DATA 
# =============================================================================
# Set path to perm maps
perm_field_dir = os.path.join('.', '3D_fields\\Examples')

# Set path to training data output
workdir = os.path.join('.', '3D_fields\\Examples')
# Import permeability map
kdata = np.loadtxt(perm_field_dir + '\\td_3dk_m2_' + str(td) +'.csv', delimiter=',')

nlay = int(kdata[-3]) # number of rows / grid cells
nrow = int(kdata[-2]) # number of rows / grid cells
ncol = int(kdata[-1]) # number of columns (parallel to axis of core)
kdata = kdata[0:-3]
raw_km2 = kdata.reshape(nlay, nrow, ncol)


model_out_filename_sp = workdir + '\\norm_td' + str(td) + '_bt_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
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
Lx = ncol * grid_size[2]   # length of model in selected units 
Ly = nrow * grid_size[1]   # length of model in selected units 
y, x = np.mgrid[slice(0, Ly + grid_size[1], grid_size[1]),
                 slice(0, Lx + grid_size[2], grid_size[2])]


# Second figure with head and breakthrough time difference maps
fig1 = plt.figure(figsize=(18, 10))
ax0 = fig1.add_subplot(2, 2, 1, aspect='equal')
imp = plt.pcolor(x, y, raw_km2[round(nlay/2),:,:], cmap='gray', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('[m$^2$]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
ax0.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Training Data #' + str(td) + ' Permeability Layer 10', fontsize=fs+2, **hfont)
plt.clim(np.min(raw_km2), np.max(raw_km2))

ax2 = fig1.add_subplot(2, 2, 2, aspect='equal')
imp = plt.pcolor(x, y, tdata_ex[round(nlay/2),:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Breakthrough time difference Layer 10', fontsize=fs+2, **hfont)
plt.clim(np.min(tdata_ex)*0.7, np.max(tdata_ex)*0.7)


ax2 = fig1.add_subplot(2, 2, 3, aspect='equal')
imp = plt.pcolor(x, y, raw_km2[:,round(nrow/2),:], cmap='gray', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('[m$^2$]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
ax0.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Training Data #' + str(td) + ' Permeability Row 10', fontsize=fs+2, **hfont)
plt.clim(np.min(raw_km2), np.max(raw_km2))

ax2 = fig1.add_subplot(2, 2, 4, aspect='equal')
imp = plt.pcolor(x, y, tdata_ex[:,round(nrow/2),:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Breakthrough time difference Row 10', fontsize=fs+2, **hfont)
plt.clim(np.min(tdata_ex)*0.7, np.max(tdata_ex)*0.7)