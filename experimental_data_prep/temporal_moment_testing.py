# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:35:15 2021

@author: zahas
"""

# All packages called by functions should be imported
import numpy as np

# Function to calculate the mean breakthrough time
def mean_exp_bt_map(conc, timestep, grid_size, v):
    # also output the mean breakthrough time map
    conc_size = conc.shape
    finite_vox_per_slice = (np.size(conc[:,:,0,0]) - np.sum(np.isnan(conc[:,:,0,0])))
    print(finite_vox_per_slice)
    # define array of times based on timestep size (in seconds)
    # Note that we are referencing the center of the imaging timestep since a 
    # given frame is an average of the detected emission events during that period
    timearray = np.arange(timestep/2, timestep*conc_size[3], timestep)
    
    # Mean arrival time calculation in inlet and outlet slice
    oned = np.nansum(np.nansum(conc, 0), 0)
    # Find mean arrival of inlet slice time
    cprofile = oned[0,:]
    max_index = np.argmax(cprofile)
    # First index greater than half of max
    mid_index = np.argmax(cprofile> cprofile[max_index]/2)
    # linearly interpolate
    m = (cprofile[mid_index] - cprofile[mid_index-1])/(timearray[mid_index] - timearray[mid_index-1])
    b = cprofile[mid_index-1] - m*timearray[mid_index-1]
    t_mean_in =  ((cprofile[max_index]/2) - b)/m
    
    # Find mean arrival of outlet slice time
    cprofile = oned[-1,:]
    max_index = np.argmax(cprofile)
    # First index greater than half of max
    mid_index = np.argmax(cprofile> cprofile[max_index]/2)
    # linearly interpolate
    m = (cprofile[mid_index] - cprofile[mid_index-1])/(timearray[mid_index] - timearray[mid_index-1])
    b = cprofile[mid_index-1] - m*timearray[mid_index-1]
    t_mean_out =  ((cprofile[max_index]/2) - b)/m

    # core length
    core_length = grid_size[2]*conc_size[2]
    # array of grid cell centers before interpolation
    z_coord_pet = np.arange(grid_size[2]/2, core_length, grid_size[2])
    # z_coord_pet2 = np.arange(0, core_length+grid_size[2], grid_size[2])
    # print(z_coord_pet2)
    
    # Preallocate breakthrough time array
    bt_array = np.zeros((conc_size[0], conc_size[1], conc_size[2]), dtype=np.float)
    M0 = np.zeros((conc_size[0], conc_size[1], conc_size[2]), dtype=np.float)
    M1 = np.zeros((conc_size[0], conc_size[1], conc_size[2]), dtype=np.float)
    M2 = np.zeros((conc_size[0], conc_size[1], conc_size[2]), dtype=np.float)
    
    for xc in range(0, conc_size[0]):
        for yc in range(0, conc_size[1]):
            for zc in range(0, conc_size[2]):
                # Check if outside core
                if np.isfinite(conc[xc, yc, zc, 0]):
                    # extract voxel breakthrough curve
                    cell_btc = conc[xc, yc, zc, :]
                    # check to make sure tracer is in grid cell
                    if cell_btc.sum() > 0:
                        # calculate zero moment
                        M0[xc, yc, zc] = np.trapz(cell_btc, timearray)
                        # calculate first moment of grid cell 
                        M1[xc, yc, zc] = np.trapz(cell_btc*timearray, timearray)
                        # calculate second moment of grid cell 
                        M2[xc, yc, zc] = np.trapz(cell_btc*(timearray**2), timearray)
                        # calculate center of mass in time (mean breakthrough time)
                        bt_array[xc, yc, zc] = M1[xc, yc, zc]/M0[xc, yc, zc]
                        # advection along axis of core
                        # advection_array[xc, yc, zc] = z_coord_pet[zc]/((m1/m0) - t0/2)
                        # longitudinal dispersion (Wolff 1978)
                        # u2 = m2/m0 - (m1/m0)**2
                        # dispersion_array[xc, yc, zc] = (u2- (t0**2)/12)* v**3/(2*z_coord_pet[zc])
                        
                        # longitudinal dispersion (Valocchi 1985)
                        # T = timearray*v/z_coord_pet[zc]
                        # # solute moments
                        # mT0 = np.trapz(cell_btc, T)
                        # mT1 = np.trapz(T*cell_btc, T)
                        # u1 = mT1/mT0
                        # u2 = np.trapz(((T-u1)**2)*cell_btc, T)/mT0
                        # dispersion_array[xc, yc, zc] = (u2/u1)*(v*z_coord_pet[zc]/2)
                        

    
    v = (core_length-grid_size[2])/(t_mean_out - t_mean_in)
    print('advection velocity: ' + str(v))
    
    # Normalize bt times
    bt_array_norm = (bt_array-t_mean_in)/(t_mean_out - t_mean_in)
    
    # vector of ideal mean arrival time based average v
    bt_ideal = z_coord_pet/z_coord_pet[-1]
    # print(bt_ideal)

    # Turn this vector into a matrix so that it can simple be subtracted from
    bt_ideal_array = np.tile(bt_ideal, (conc_size[0], conc_size[1], 1))

    # % Arrival time difference map
    bt_diff_norm = (bt_ideal_array - bt_array_norm)
    
    # # alternative normalization
    # # vector of ideal mean arrival time based average v
    # bt_ideal = z_coord_pet/v
    # # print(bt_ideal)

    # # Turn this vector into a matrix so that it can simple be subtracted from
    # bt_ideal_array = np.tile(bt_ideal, (conc_size[0], conc_size[1], 1))
    # bt_diff = ((bt_array-t_mean_in) - bt_ideal_array)
    
    # bt_diff_norm2 = bt_diff*(v/core_length)
    
    # Replace nans with zeros
    bt_array[np.isnan(bt_array)] = 0
    # Replace nans with zeros
    bt_array_norm[np.isnan(bt_array_norm)] = 0
    # Replace nans with zeros
    # bt_diff_norm[np.isnan(bt_diff_norm)] = 0
    return bt_array, bt_array_norm, bt_diff_norm, M0, M1, M2
    # return bt_array, bt_diff, bt_diff_norm