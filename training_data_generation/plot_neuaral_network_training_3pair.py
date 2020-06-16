# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:10:03 2020

@author: zahas
"""


import os
import numpy as np
import matplotlib.pyplot as plt


# training data iteration
# td = 2603


# layer to plot
# ilayer = 0
# fontsize
fs = 23
hfont = {'fontname':'Arial'}

# number of layers 
nlay = 1
# Grid cell size
grid_size = [0.25, 0.25, 0.25] # selected units [cm]

# directory_name = 'Tdata_2D_g10k_horz'
# datadir = os.path.join('..', directory_name, 'mt3d_test', 'mt3dms')
# workdir = os.path.join('D:\\training_data\\Tdata_2D_g10k_horz')




# =============================================================================
# LOAD SELECTED EXAMPLE DATA 
# =============================================================================
def get_perm_and_btmap(td): # Set path to perm maps
    ilayer = 0;
    # perm_field_dir = os.path.join('.', 'matlab_perm_fields\\k_training_data_rot')
    perm_field_dir = os.path.join('D:\\training_data\\gauss_fields\\no_rotation')
    
    # Set path to training data output
    # workdir = os.path.join('.', 'Tdata_2D\\td_10k_rot')
    workdir = os.path.join('D:\\training_data\\Tdata_2D_g10k_horz')
    # Import permeability map
    tdata_km2 = np.loadtxt(perm_field_dir + '\\tdg_km2_' + str(td) +'.csv', delimiter=',')
    
    nrow = int(tdata_km2[-2]) # number of rows / grid cells
    ncol = int(tdata_km2[-1]) # number of columns (parallel to axis of core)
    tdata_km2 = tdata_km2[0:-2]
    raw_km2 = tdata_km2.reshape(nlay, nrow, ncol)
    perm_field = raw_km2[ilayer,:,:]/9.869233E-13
    
    
    model_out_filename_sp = workdir + '\\td' + str(td) + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    tdata_ex = np.loadtxt(model_out_filename_sp, delimiter=',')
    load_lx = tdata_ex[-1]
    load_dp = tdata_ex[-2]
    tdata_ex = tdata_ex[0:-2]
    tdata_ex = tdata_ex.reshape(nlay, nrow, ncol)
    bt_field = tdata_ex[ilayer,:,:]
    
    # viscosity of water in pascal
    mu = 8.9e-4 # [pa.s]
    q = 0.024669065517374814 # [cm/min]
    u = q/60/100 #[m/sec]
    # q = 0.5/100**3/60
    mean_perm = (load_lx/100)*mu*u*nrow/load_dp
    
    # print('Model length: '+ str(load_lx) + ' cm')
    print('Pressure drop: '+ str(load_dp) + ' pascals')
    print('Pressure-based mean perm: '+ str(mean_perm) + ' m^2')
    print('Field-based mean perm: '+ str(np.mean(tdata_km2)) + ' m^2')
    
    return perm_field, bt_field, ncol, nrow

# =============================================================================
# PLOT DATA 
# =============================================================================
td = 9

perm_field, bt_field, ncol, nrow = get_perm_and_btmap(td)
# Define grid    
# Describe grid for results    
Lx = (ncol - 1) * grid_size[1]   # length of model in selected units 
Ly = (nrow - 1) * grid_size[0]   # length of model in selected units 
y, x = np.mgrid[slice(0, Ly + grid_size[0], grid_size[0]),
                slice(0, Lx + grid_size[1], grid_size[1])]


# perm_field, bt_field = get_perm_and_btmap(10, 0)

# Second figure with head and breakthrough time difference maps
fig1 = plt.figure(figsize=(25, 8.5))

ax0 = fig1.add_subplot(2, 3, 1, aspect='equal')
imp = plt.pcolor(x, y, perm_field, cmap='gray', edgecolors='k', linewidths=0.2)
plt.xticks(np.arange(0, 10, step=2))
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('[Darcy]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
# ax0.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Training data #' + str(td) + ' permeability', fontsize=fs+2, **hfont)

ax2 = fig1.add_subplot(2, 3, 4, aspect='equal')
imp = plt.pcolor(x, y, bt_field, cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
plt.xticks(np.arange(0, 10, step=2))
cbar = plt.colorbar()
cbar.set_label('[PV]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Breakthrough time difference', fontsize=fs+2, **hfont)

td = 4032
perm_field, bt_field, ncol, nrow = get_perm_and_btmap(td)
ax0 = fig1.add_subplot(2, 3, 2, aspect='equal')
imp = plt.pcolor(x, y, perm_field, cmap='gray', edgecolors='k', linewidths=0.2)
plt.xticks(np.arange(0, 10, step=2))
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('[Darcy]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
# ax0.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
# plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Training data #' + str(td) + ' permeability', fontsize=fs+2, **hfont)

ax2 = fig1.add_subplot(2, 3, 5, aspect='equal')
imp = plt.pcolor(x, y, bt_field, cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
plt.xticks(np.arange(0, 10, step=2))
cbar = plt.colorbar()
cbar.set_label('[PV]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
# plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Breakthrough time difference', fontsize=fs+2, **hfont)

td = 8017
perm_field, bt_field, ncol, nrow = get_perm_and_btmap(td)
ax0 = fig1.add_subplot(2, 3, 3, aspect='equal')
imp = plt.pcolor(x, y, perm_field, cmap='gray', edgecolors='k', linewidths=0.2)
plt.xticks(np.arange(0, 10, step=2))
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('[Darcy]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
# ax0.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
# plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Training data #' + str(td) + ' permeability', fontsize=fs+2, **hfont)

ax2 = fig1.add_subplot(2, 3, 6, aspect='equal')
imp = plt.pcolor(x, y, bt_field, cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
plt.xticks(np.arange(0, 10, step=2))
cbar = plt.colorbar()
cbar.set_label('[PV]', fontsize=fs, **hfont)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs, **hfont)
ax2.tick_params(axis='both', which='major', labelsize=fs)
# plt.ylabel('Distance [cm]', fontsize=fs, **hfont)
plt.title('Breakthrough time difference', fontsize=fs+2, **hfont)
