# -*- coding: utf-8 -*-
"""
flopy_bt_time_3d_diff_generation.py
Created on Fri Jun  5 08:23:38 2020

@author: Christopher Zahasky

This script is used to generate 3D breakthrough time maps using synthetically 
generated 3D permeability fields. This script calls functions from the python
script titled 'flopy_bt_3d_functions.py'
"""

# Only packages called in this script need to be imported
import sys
import os
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# import math
import time

# Import custom functions to run flopy stuff, the calls should be structured as:
# from file import function
from flopy_arrival_time_3d_functions import mt3d_pulse_injection_sim, flopy_arrival_map_function

# # run installed version of flopy or add local path
# try:
#     import flopy
# except:
#     fpth = os.path.abspath(os.path.join('..', '..'))
#     sys.path.append(fpth)
#     import flopy

## LAPTOP PATHS
# names of executable with path IF NOT IN CURRENT DIRECTORY
# exe_name_mf = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
# exe_name_mt = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# DELL 419 PATHS
# names of executable with path IF NOT IN CURRENT DIRECTORY
exe_name_mf = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
exe_name_mt = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# directory to save data
directory_name = 'arrival_time_maps'
# workdir = os.path.join('.', '3D_fields', directory_name)
workdir = os.path.join('D:\\Training_data_generation_3D\\Tdata_3D_20K_with_arrival', directory_name)

# uncomment if you want the information about numpy, matplotlib, and flopy printed    
# print(sys.version)
# print('numpy version: {}'.format(np.__version__))
# print('matplotlib version: {}'.format(mpl.__version__))
# print('flopy version: {}'.format(flopy.__version__))

# Set path to perm maps
# perm_field_dir = os.path.join('.')
# perm_field_dir = os.path.join('.', '3D_perm')
perm_field_dir = os.path.join('D:\\Training_data_generation_3D\\Tdata_3D_20K_with_arrival\\syn_core_perm_maps')

# Import core shape mask
core_mask = np.loadtxt('core_template.csv', delimiter=',')
core_mask = core_mask.reshape(20, 20)

# =============================================================================
# VARIABLES THAT DON'T CHANGE
# =============================================================================
# grid_size = [grid size in direction of Lx (layer thickness), 
    # Ly (left to right axis when looking down the core), Lz (long axis of core)]
grid_size = [0.23291, 0.23291, 0.2388] # selected units [cm]
# Output control for MT3dms
# nprs (int):  the frequency of the output. If nprs > 0 results will be saved at 
# the times as specified in timprs (evenly allocated between 0 and sim run length); 
# if nprs = 0, results will not be saved except at the end of simulation; if NPRS < 0, simulation results will be 
# saved whenever the number of transport steps is an even multiple of nprs. (default is 0).
nprs = 150 
# period length in selected units (for steady state flow it can be set to anything)
perlen_mf = [1., 90]
# Numerical method flag
mixelm = -1

# Call function and time it
start_td = time.time() # start a timer

# value of back up realizations to draw from
replacement_counter = 40

td = 876
while td < 3404:
    
    print('TRAINING DATASET: ' + str(td))
       
    # Import permeability map
    tdata_km2 = np.loadtxt(perm_field_dir + '\\core_k_3d_m2_' + str(td) +'.csv', delimiter=',')
    # Resave data in smaller format
    np.savetxt(perm_field_dir + '\\core_k_3d_m2_' + str(td)+'.csv', tdata_km2 , delimiter=',', fmt='%.3e')
    
    # extract geometric information
    nlay = int(tdata_km2[-3]) # number of layers / grid cells
    nrow = int(tdata_km2[-2]) # number of columns 
    ncol = int(tdata_km2[-1]) # number of slices (parallel to axis of core)
    # crop data
    tdata_km2 = tdata_km2[0:-3]

    # Reshape to 3D array
    raw_km2 = tdata_km2.reshape(nlay, nrow, ncol)
    # this is the command needed if data isn't permuted before converting from matrix to vector in matlab
    # raw_km2 = tdata_km2.reshape(nlay, nrow, ncol, order='F')
    
    # if defined, set permeability values outside of core to zero with core mask
    # if 'core_mask' in locals():
    #     for col in range(ncol):
    #         raw_km2[:,:,col] = np.multiply(raw_km2[:,:,col], core_mask)
    #         # save cropped hk data
    #         save_cropped_perm_filename = perm_field_dir + '\\core_k_3d_m2_' + str(td) +'.csv'
    #         save_data = np.append(raw_km2.flatten('C'), [nlay, nrow, ncol])
    #         np.savetxt(save_cropped_perm_filename, save_data, delimiter=',')
        
    # Convert permeabiltiy (in m^2) to hydraulic conductivity in cm/min
    raw_hk = raw_km2*(1000*9.81*100*60/8.9E-4)
    
    # Describe grid for results    
    Lx = (ncol) * grid_size[2]   # length of model in selected units 
    Ly = (nrow) * grid_size[1]   # length of model in selected units 
    
    # Model workspace and new sub-directory
    model_dirname = ('td'+ str(td))
    model_ws = os.path.join(workdir, model_dirname)
    
    mf, mt, conc, timearray, km2_mean = mt3d_pulse_injection_sim(model_dirname, model_ws, raw_hk, grid_size, perlen_mf, nprs, mixelm, exe_name_mf, exe_name_mt)
    # print('Core average perm: '+ str(km2_mean) + ' m^2')
    
    # Option to plot and calculate geometric mean to double check that core average perm in close
    # raw_km2_array = raw_km2.flatten()
    # index = np.argwhere(raw_km2_array > 0) 
    # geo_mean = np.exp(np.sum(np.log(raw_km2_array[index]))/index.size)
    # print('Geometric mean perm: ' + str(geo_mean) + ' m^2')
    
    
    # calculate quantile arrival time map from MT3D simulation results
    at_array, at_array_norm, at_diff_norm = flopy_arrival_map_function(conc, np.array(timearray), grid_size, 0.5, 0.1)
    
    # In some cases the models may fail to run or there are issues with calculating arrival times
    # When this happens the realization is replaced with a random realization 
    # from a second sampling. This occurs less than 50 times in the entire 
    # 20,000 data initially generated
    if isinstance(at_diff_norm, int):
        # Import permeability map
        tdata_km2 = np.loadtxt('D:\\Training_data_generation_3D\\Tdata_3D_20K_with_arrival\\backup_core_maps' + '\\core_td_3dk_m2_' + str(replacement_counter) +'.csv', delimiter=',')
        print('Training dataset replaced with: ' + '\\core_td_3dk_m2_' + str(replacement_counter))
        # Replace bad realization with new dataset
        np.savetxt(perm_field_dir + '\\core_k_3d_m2_' + str(td)+'.csv', tdata_km2 , delimiter=',', fmt='%.3e')
        # update replacement counter
        replacement_counter += 1
        continue
        
    
    # =============================================================================
    # SAVE DATA 
    # =============================================================================
    # save normalized arrival time difference data
    save_filename_btdn = workdir + '\\' + 'arrival_norm_diff_' + model_dirname + '_'  + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    save_data = np.append(at_diff_norm.flatten('C'), [km2_mean])
    np.savetxt(save_filename_btdn, save_data, delimiter=',', fmt='%.3e')
    
    # save unnormalized breakthrough data
    save_filename_btd = workdir + '\\' + 'arrival_norm_' + model_dirname + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    save_data = np.append(at_array_norm.flatten('C'), [km2_mean])
    np.savetxt(save_filename_btd, save_data, delimiter=',', fmt='%.3e')
    
    # Try to delete the previous folder of MODFLOW and MT3D files
    if td > 1:
        old_model_ws = os.path.join(workdir, ('td'+ str(td-1)))
        # Try to remove tree; if failed show an error using try...except on screen
        try:
            shutil.rmtree(old_model_ws)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
    
    # Update td counter        
    td += 1
    
# Print final run time
end_td = time.time() # end timer
print('Minutes to run 20,000 training simulations: ', (end_td - start_td)/60) # show run time
    
# =============================================================================
# PLOT DATA 
# =============================================================================
# Define grid    
y, x = np.mgrid[slice(0, Ly + grid_size[1], grid_size[1]),
                  slice(0, Lx + grid_size[2], grid_size[2])]
# layer to plot
ilayer = 8
# # fontsize
fs = 18

# # Crossection of core
# xc, yc = np.mgrid[slice(0, Ly + grid_size[1], grid_size[1]),
#                   slice(0, Ly + grid_size[0], grid_size[0])]
# fig1 = plt.figure(figsize=(10, 5))
# ax0 = fig1.add_subplot(1, 1, 1, aspect='equal')
# imp = plt.pcolor(xc, yc, raw_km2[:,:,0], cmap='gray', edgecolors='k', linewidths=0.2)
# cbar = plt.colorbar()
# # plt.clim(0,1) 
# cbar.set_label('[m^2]', fontsize=fs)
# cbar.ax.tick_params(labelsize= (fs-2)) 
# ax0.tick_params(axis='both', which='major', labelsize=fs)
# plt.title('Permeability', fontsize=fs+2)

# Load breakthrough time data
tdata_norm = np.loadtxt(save_filename_btdn, delimiter=',')
load_km2 = tdata_norm[-1]
tdata_norm = tdata_norm[0:-1]
tdata_norm = tdata_norm.reshape(nlay, nrow, ncol)
    
# tdata_raw = np.loadtxt(save_filename_btd, delimiter=',')
# load_km2 = tdata_raw[-1]
# tdata_raw = tdata_raw[0:-1]
# tdata_raw = tdata_raw.reshape(nlay, nrow, ncol)


fig1 = plt.figure(figsize=(10, 5))
ax0 = fig1.add_subplot(1, 1, 1, aspect='equal')
imp = plt.pcolor(x, y, raw_km2[ilayer,:,:], cmap='gray', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('[m^2]', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Permeability', fontsize=fs+2)


# First figure with concentration data
fig1 = plt.figure(figsize=(10, 15))
ax0 = fig1.add_subplot(3, 1, 1, aspect='equal')
imp = plt.pcolor(x, y, conc[5,ilayer,:,:], cmap='OrRd', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0,0.4) 
cbar.set_label('Solute concentration', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Time: %1.1f min' %timearray[5], fontsize=fs+2)

ax1 = fig1.add_subplot(3, 1, 2, aspect='equal')
imp = plt.pcolor(x, y, conc[20,ilayer,:,:], cmap='OrRd', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0,0.4) 
cbar.set_label('Solute concentration', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax1.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Time: %1.1f min' %timearray[20], fontsize=fs+2)

ax2 = fig1.add_subplot(3, 1, 3, aspect='equal')
imp = plt.pcolor(x, y, conc[35,ilayer,:,:], cmap='OrRd', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0,0.4) 
cbar.set_label('Solute concentration', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Time: %1.1f min' %timearray[35], fontsize=fs+2)
plt.ylabel('Distance [cm]', fontsize=fs)

# Second figure with head and breakthrough time difference maps
fig2 = plt.figure(figsize=(10, 15))
ax0 = fig2.add_subplot(3, 1, 1, aspect='equal')
dp_pressures = at_array[ilayer,:,:]
imp = plt.pcolor(x, y, dp_pressures, cmap='BuGn', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0, np.max(dp_pressures)) 
cbar.set_label('Time [min]', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
plt.title('0.50 Quantile Arrival Time [-]', fontsize=fs+2)

ax1 = fig2.add_subplot(3, 1, 2, aspect='equal')
imp = plt.pcolor(x, y, at_array_norm[ilayer,:,:], cmap='YlGn', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('PV [-]', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax1.tick_params(axis='both', which='major', labelsize=fs)
plt.title('0.50 Quantile Arrival Time [-]', fontsize=fs+2)

ax2 = fig2.add_subplot(3, 1, 3, aspect='equal')
imp = plt.pcolor(x, y, tdata_norm[ilayer,:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs)
plt.title('Quantile Arrival Difference', fontsize=fs+2)