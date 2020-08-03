# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:23:38 2020

@author: zahas
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
    
# from flopy.utils.util_array import read1d

# set figure size, call mpl.rcParams to see all variables that can be changed
# mpl.rcParams['figure.figsize'] = (8, 8)

## CD paths
# exe_name_mf = 'mf2005'
# exe_name_mt = 'mt3dms'

## LAPTOP PATHS
# names of executable with path IF NOT IN CURRENT DIRECTORY
exe_name_mf = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
exe_name_mt = 'C:\\Users\\zahas\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# DELL 419 PATHS
# names of executable with path IF NOT IN CURRENT DIRECTORY
# exe_name_mf = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mf2005'
# exe_name_mt = 'D:\\Dropbox\\Research\\Simulation\\modflow\\executables\\mt3dms'

# directory to save data
directory_name = 'Tdata_3D_10k'
# datadir = os.path.join('..', directory_name, 'mt3d_test', 'mt3dms')
workdir = os.path.join('.', '3D_fields\\GRFS_v2', directory_name)
# workdir = os.path.join('D:\\training_data', directory_name)

# uncomment if you want the information about numpy, matplotlib, and flopy printed    
# print(sys.version)
# print('numpy version: {}'.format(np.__version__))
# print('matplotlib version: {}'.format(mpl.__version__))
# print('flopy version: {}'.format(flopy.__version__))

# Set path to perm maps
perm_field_dir = os.path.join('.', '3D_fields\\GRFS_v2')
# perm_field_dir = os.path.join('D:\\training_data\\gauss_fields\\no_rotation')

# =============================================================================
# VARIABLES THAT DON'T CHANGE
# =============================================================================
# grid_size = [grid size in direction of Lx (layer thickness), 
    # Ly (left to right axis when looking down the core), Lz (long axis of core)]
grid_size = [0.25, 0.25, 0.25] # selected units [cm]
# Output control for MT3dms
# nprs (int):  the frequency of the output. If nprs > 0 results will be saved at 
# the times as specified in timprs (evenly allocated between 0 and sim run length); 
# if nprs = 0, results will not be saved except at the end of simulation; if NPRS < 0, simulation results will be 
# saved whenever the number of transport steps is an even multiple of nprs. (default is 0).
nprs = 200
# viscosity of water in pascal
mu = 8.9e-4 # [pa.s]
# Injection flux per cell
q = 0.024669065517374814 # [cm/min]
u = q/60/100 #[m/sec]
    
# period length in selected units (for steady state flow it can be set to anything)
perlen_mf = [1., 20]
# Numerical method flag
mixelm = -1

# Call function and time it
start_td = time.time() # start a timer

for td in range(1, 2):
    
    print('TRAINING DATASET: ' + str(td))
       
    # Import permeability map
    tdata_km2 = np.loadtxt(perm_field_dir + '\\td_3dk_m2_' + str(td) +'.csv', delimiter=',')
    
    nlay = int(tdata_km2[-3]) # number of layers / grid cells
    nrow = int(tdata_km2[-2]) # number of columns 
    ncol = int(tdata_km2[-1]) # number of slices (parallel to axis of core)
    tdata_km2 = tdata_km2[0:-3]
    raw_km2 = tdata_km2.reshape(nlay, nrow, ncol, order='F')
    # Convert permeabiltiy (in m^2) to hydraulic conductivity in cm/min
    raw_hk = raw_km2*(1000*9.81*100*60/8.9E-4)
    # Describe grid for results    
    Lx = (ncol - 1) * grid_size[2]   # length of model in selected units 
    Ly = (nrow - 1) * grid_size[1]   # length of model in selected units 
    
    # Model workspace and new sub-directory
    model_dirname = ('td'+ str(td))
    model_ws = os.path.join(workdir, model_dirname)
    
    mf, mt, conc, timearray, pressures = mt3d_pulse_injection_sim(model_dirname, model_ws, raw_hk, grid_size, perlen_mf, nprs, mixelm, exe_name_mf, exe_name_mt)
    # calculate pressure drop
    dp = np.mean(pressures[:,:,0]) - np.mean(pressures[:,:,-1])
    print('Pressure drop: '+ str(dp) + ' pascals')
    
    # process the simulation data
    bt_array, bt_diff, bt_diff_norm = mean_bt_map(conc, grid_size, perlen_mf, timearray)
    
    # print('Mean bt_diff inlet: ' + str(np.mean(bt_diff_norm[:,:,0])))
    # print('Mean bt_diff outlet: ' + str(np.mean(bt_diff_norm[:,:,-1])))
    
    # mean_perm = (Lx/100)*mu*u*nrow/dp
 
    # print('Pressure-based mean perm: '+ str(mean_perm/9.869233E-13) + ' D')
    # print('Field-based mean perm: '+ str(np.mean(tdata_km2)/9.869233E-13) + ' D')
    
    # =============================================================================
    # SAVE DATA 
    # =============================================================================
    # save file name and short path
    save_filename_sp = workdir + '\\' + model_dirname + '_' + str(nlay) + '_' + str(nrow) + '_' + str(ncol) +'.csv'
    save_data = np.append(bt_diff_norm.flatten('C'), [dp, Lx])
    np.savetxt(save_filename_sp, save_data, delimiter=',')
    
    tdata_ex = np.loadtxt(save_filename_sp, delimiter=',')
    load_lx = tdata_ex[-1]
    load_dp = tdata_ex[-2]
    tdata_ex = tdata_ex[0:-2]
    tdata_ex = tdata_ex.reshape(nlay, nrow, ncol)
    
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
# print('Minutes to run 10,000 training simulations: ', (end_td - start_td)/60) # show run time
    
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
plt.clim(0,1) 
cbar.set_label('Solute concentration', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Time: %1.1f min' %timearray[5], fontsize=fs+2)

ax1 = fig1.add_subplot(3, 1, 2, aspect='equal')
imp = plt.pcolor(x, y, conc[30,ilayer,:,:], cmap='OrRd', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0,1) 
cbar.set_label('Solute concentration', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax1.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Time: %1.1f min' %timearray[30], fontsize=fs+2)

ax2 = fig1.add_subplot(3, 1, 3, aspect='equal')
imp = plt.pcolor(x, y, conc[60,ilayer,:,:], cmap='OrRd', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
plt.clim(0,1) 
cbar.set_label('Solute concentration', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax1.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Time: %1.1f min' %timearray[60], fontsize=fs+2)



# Second figure with head and breakthrough time difference maps
fig2 = plt.figure(figsize=(10, 15))
ax0 = fig2.add_subplot(3, 1, 1, aspect='equal')
dp_pressures = pressures - np.mean(pressures[:,:,-1])
imp = plt.pcolor(x, y, dp_pressures[ilayer,:,:], cmap='Blues', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('Pascals', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax0.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Pressure Drop', fontsize=fs+2)

ax1 = fig2.add_subplot(3, 1, 2, aspect='equal')
imp = plt.pcolor(x, y, bt_array[ilayer,:,:], cmap='YlGn', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
# plt.clim(0,1) 
cbar.set_label('Time [min]', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax1.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Mean Breakthrough Time [min]', fontsize=fs+2)

ax2 = fig2.add_subplot(3, 1, 3, aspect='equal')
imp = plt.pcolor(x, y, tdata_ex[ilayer,:,:], cmap='RdYlBu_r', edgecolors='k', linewidths=0.2)
cbar = plt.colorbar()
cbar.set_label('Pore Volumes', fontsize=fs)
cbar.ax.tick_params(labelsize= (fs-2)) 
ax2.set_xlabel('Distance from inlet [cm]', fontsize=fs)
ax2.tick_params(axis='both', which='major', labelsize=fs)
plt.ylabel('Distance [cm]', fontsize=fs)
plt.title('Breakthrough time difference map', fontsize=fs+2)