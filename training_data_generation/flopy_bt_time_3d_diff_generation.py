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
from flopy_bt_3d_functions import mt3d_pulse_injection_sim, mean_bt_map

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
directory_name = 'Testdata_3D_cores_4k'
# datadir = os.path.join('..', directory_name, 'mt3d_test', 'mt3dms')
# workdir = os.path.join('.', '3D_fields', directory_name)
workdir = os.path.join('D:\\Training_data_generation_3D\\', directory_name)

# uncomment if you want the information about numpy, matplotlib, and flopy printed    
# print(sys.version)
# print('numpy version: {}'.format(np.__version__))
# print('matplotlib version: {}'.format(mpl.__version__))
# print('flopy version: {}'.format(flopy.__version__))

# Set path to perm maps
# perm_field_dir = os.path.join('.', '3D_perm')
perm_field_dir = os.path.join('D:\\Training_data_generation_3D\\Testdata_3D_perm_4k')

# Import core shape mask
core_mask = np.loadtxt('core_template.csv', delimiter=',')
# raw_km2 = tdata_km2.reshape(nlay, nrow, ncol, order='F') # this is the command needed if data isn't permuted before converting from matrix to vector in matlab
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

for td in range(3, 4001):
    
    print('TRAINING DATASET: ' + str(td))
       
    # Import permeability map
    tdata_km2 = np.loadtxt(perm_field_dir + '\\trd_3dk_m2_' + str(td) +'.csv', delimiter=',')
    
    nlay = int(tdata_km2[-3]) # number of layers / grid cells
    nrow = int(tdata_km2[-2]) # number of columns 
    ncol = int(tdata_km2[-1]) # number of slices (parallel to axis of core)
    tdata_km2 = tdata_km2[0:-3]
    # raw_km2 = tdata_km2.reshape(nlay, nrow, ncol, order='F') # this is the command needed if data isn't permuted before converting from matrix to vector in matlab
    raw_km2 = tdata_km2.reshape(nlay, nrow, ncol)
    
    # if defined, set permeability values outside of core to zero with core mask
    if 'core_mask' in locals():
        for col in range(ncol):
            raw_km2[:,:,col] = np.multiply(raw_km2[:,:,col], core_mask)
            # save cropped hk data
            save_cropped_perm_filename = perm_field_dir + '\\core_td_3dk_m2_' + str(td) +'.csv'
            save_data = np.append(raw_km2.flatten('C'), [nlay, nrow, ncol])
            np.savetxt(save_cropped_perm_filename, save_data, delimiter=',')
        
    # Convert permeabiltiy (in m^2) to hydraulic conductivity in cm/min
    raw_hk = raw_km2*(1000*9.81*100*60/8.9E-4)
    
    # Describe grid for results    
    Lx = (ncol) * grid_size[2]   # length of model in selected units 
    Ly = (nrow) * grid_size[1]   # length of model in selected units 
    
    # Model workspace and new sub-directory
    model_dirname = ('td'+ str(td))
    model_ws = os.path.join(workdir, model_dirname)
    
    mf, mt, conc, timearray, pressures, dp = mt3d_pulse_injection_sim(model_dirname, model_ws, raw_hk, grid_size, perlen_mf, nprs, mixelm, exe_name_mf, exe_name_mt)
    # print pressure drop
    print('Pressure drop: '+ str(dp/1000) + ' kPa')
    
    # process the simulation data
    bt_array, bt_diff, bt_diff_norm = mean_bt_map(conc, raw_hk, grid_size, perlen_mf, timearray)
    
    # =============================================================================
    # SAVE DATA 
    # =============================================================================
    # save normalized breakthrough data
    save_filename_btdn = workdir + '\\' + 'norm_' + model_dirname + '_bt' + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    save_data = np.append(bt_diff_norm.flatten('C'), [dp, Lx])
    np.savetxt(save_filename_btdn, save_data, delimiter=',')
    
    # save unnormalized breakthrough data
    save_filename_btd = workdir + '\\' + model_dirname + '_bt' + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    save_data = np.append(bt_array.flatten('C'), [dp, Lx])
    np.savetxt(save_filename_btd, save_data, delimiter=',')
    
    # Try to delete the previous folder of MODFLOW and MT3D files
    if td > 1:
        old_model_ws = os.path.join(workdir, ('td'+ str(td-1)))
        # Try to remove tree; if failed show an error using try...except on screen
        try:
            shutil.rmtree(old_model_ws)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))


# Print final run time
end_td = time.time() # end timer
print('Minutes to run 4,000 training simulations: ', (end_td - start_td)/60) # show run time
    
# =============================================================================
# PLOT DATA 
# =============================================================================
# Define grid    
y, x = np.mgrid[slice(0, Ly + grid_size[1], grid_size[1]),
                  slice(0, Lx + grid_size[2], grid_size[2])]
# layer to plot
ilayer = 9
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
load_lx = tdata_norm[-1]
load_dp = tdata_norm[-2]
tdata_norm = tdata_norm[0:-2]
tdata_norm = tdata_norm.reshape(nlay, nrow, ncol)
    
tdata_raw = np.loadtxt(save_filename_btd, delimiter=',')
load_lx = tdata_raw[-1]
load_dp = tdata_raw[-2]
tdata_raw = tdata_raw[0:-2]
tdata_raw = tdata_raw.reshape(nlay, nrow, ncol)


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
dp_pressures = (pressures[ilayer,:,:]/1000)-70
imp = plt.pcolor(x, y, dp_pressures, cmap='Blues', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0, np.max(dp_pressures)) 
cbar.set_label('kPa', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Pressure Drop', fontsize=fs+2)

ax1 = fig2.add_subplot(3, 1, 2, aspect='equal')
imp = plt.pcolor(x, y, tdata_raw[ilayer,:,:], cmap='YlGn', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('Time [min]', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax1.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Mean Breakthrough Time [min]', fontsize=fs+2)

ax2 = fig2.add_subplot(3, 1, 3, aspect='equal')
imp = plt.pcolor(x, y, tdata_norm[ilayer,:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs)
plt.title('Breakthrough time difference map', fontsize=fs+2)
