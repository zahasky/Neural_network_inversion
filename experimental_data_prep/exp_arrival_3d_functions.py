# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:01:50 2021

@author: zahas
"""
# All packages called by functions should be imported
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import time

def quantile_calc(btc_1d, timearray, quantile, t_increment):
    # find length of time array
    ntime = timearray.shape[0]
    # calculate zero moment
    M0 = np.trapz(btc_1d, timearray)
    # reset incremental quantile numerator tracker
    M0im = 0

    for i in range(1, ntime):
        # numerically integrate from the beginning to the end of the dataset
        M0i = np.trapz(btc_1d[:i], timearray[:i])
        # check to see if the quantile has been surpassed, if so then linearly 
        # interpolate between the current measurement and the previous measurement
        if M0i/M0 > quantile:
            # recalculate the integral of the previous measurement (i-1)
            M0im = np.trapz(btc_1d[:i-1], timearray[:i-1])
            # linear interpolation
            m = (btc_1d[i-1] - btc_1d[i-2])/(timearray[i-1] - timearray[i-2])
            b = btc_1d[i-2] - m*timearray[i-2]
            # now search through the space between the two measurements to find 
            # when the area under the curve is equal to the desired quantile
            for xt in np.arange(timearray[i-2], timearray[i-1] +t_increment, t_increment):
                # calculate the linearly interpolated area under the curve
                M0int = M0im + np.trapz([btc_1d[i-2], m*xt+b], [timearray[i-2], xt])
            
                if M0int/M0 > quantile:
                    tau = xt
                    break # the inside FOR loop
            
            break # the outside FOR loom
            
    return tau
            
# Function to calculate the quantile arrival time map
def exp_arrival_map_function(conc, timestep, grid_size, quantile, t_increment):
    # start timer
    tic = time.perf_counter()
    
    # determine the size of the data
    conc_size = conc.shape
    
    # define array of times based on timestep size (in seconds)
    # Note that we are referencing the center of the imaging timestep since a 
    # given frame is an average of the detected emission events during that period
    timearray = np.arange(timestep/2, timestep*conc_size[3], timestep)
    
    # sum of slice concentrations for calculating inlet and outlet breakthrough
    oned = np.nansum(np.nansum(conc, 0), 0)
    
    # arrival time calculation in inlet slice
    tau_in = quantile_calc(oned[0,:], timearray, quantile, t_increment/10)
    
    # arrival time calculation in outlet slice
    tau_out = quantile_calc(oned[-1,:], timearray, quantile, t_increment/10)

    # core length
    core_length = grid_size[2]*conc_size[2]
    # array of grid cell centers before interpolation
    z_coord_pet = np.arange(grid_size[2]/2, core_length, grid_size[2])
    
    # Preallocate arrival time array
    at_array = np.zeros((conc_size[0], conc_size[1], conc_size[2]), dtype=np.float)
    
    for xc in range(0, conc_size[0]):
        for yc in range(0, conc_size[1]):
            for zc in range(0, conc_size[2]):
                # Check if outside core
                if np.isfinite(conc[xc, yc, zc, 0]):
                    # extract voxel breakthrough curve
                    cell_btc = conc[xc, yc, zc, :]
                    # check to make sure tracer is in grid cell
                    if cell_btc.sum() > 0:
                        # call function to find quantile of interest
                        at_array[xc, yc, zc] = quantile_calc(cell_btc, timearray, quantile, t_increment)

    v = (core_length-grid_size[2])/(tau_out - tau_in)
    print('advection velocity: ' + str(v))
    
    # Normalize arrival times
    at_array_norm = (at_array-tau_in)/(tau_out - tau_in)
    
    # vector of ideal mean arrival time based average v
    at_ideal = z_coord_pet/z_coord_pet[-1]

    # Turn this vector into a matrix so that it can simply be subtracted from
    at_ideal_array = np.tile(at_ideal, (conc_size[0], conc_size[1], 1))

    # Arrival time difference map
    at_diff_norm = (at_ideal_array - at_array_norm)
    
    # Replace nans with zeros
    at_array[np.isnan(conc[:,:,:,0])] = 0
    # Replace nans with zeros
    at_array_norm[np.isnan(conc[:,:,:,0])] = 0
    # Replace nans with zeros
    at_diff_norm[np.isnan(conc[:,:,:,0])] = 0
    # stop timer
    toc = time.perf_counter()
    print(f"Function runtime is {toc - tic:0.4f} seconds")
    return at_array, at_array_norm, at_diff_norm
                

    
def plot_2d(map_data, dx, dy, colorbar_label, cmap):
    # fontsize
    fs = 18
    hfont = {'fontname':'Arial'}
    r, c = np.shape(map_data)
    # Define grid
    # Lx = c * dx   # length of model in selected units 
    # Ly = r * dy   # length of model in selected units
    # x, y = np.mgrid[slice(0, Lx + dx, dx), slice(0, Ly + dy, dy)]
    
  
    x_coord = np.linspace(dx/2, dx*c, c)
    y_coord = np.linspace(dy/2, dy*r, r)
    
    X, Y = np.meshgrid(x_coord, y_coord)
    # print(slice(0, Ly + dy, dy))
    # print(c)
    # print(slice(0, Lx + dx, dx))
    # print(r)
    
    # fig, ax = plt.figure(figsize=(10, 10) # adjust these numbers to change the size of your figure
    # ax.axis('equal')          
    # fig2.add_subplot(1, 1, 1, aspect='equal')
    # Use 'pcolor' function to plot 2d map of concentration
    # Note that we are flipping map_data and the yaxis to so that y increases downward
    plt.figure(figsize=(12, 4), dpi=200)
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'nearest', edgecolor ='k', linewidth = 0.01)
    plt.gca().set_aspect('equal')  
    # add a colorbar
    cbar = plt.colorbar() 
    # plt.clim(cmin, cmax) 
    # label the colorbar
    cbar.set_label(colorbar_label, fontsize=fs, **hfont)
    # make colorbar font bigger
    cbar.ax.tick_params(labelsize= (fs-2)) 
    # make axis fontsize bigger!
    plt.tick_params(axis='both', which='major', labelsize=fs)
    # Label x-axis
    plt.gca().invert_yaxis()
    
def axial_linear_interp_3D(data_3d, dx, dy, dz, cnnslice):
    xd, yd, zd = np.shape(data_3d)
    # for neural network the grid needs to be regridded to a size of 20 x 20 x 40
    core_length = dz*zd
    # array of grid cell centers before interpolation
    z_coord_pet = np.arange(dz/2, core_length, dz)
    x_coord_pet = np.linspace(dx/2, dx*xd, xd)
    y_coord_pet = np.linspace(dy/2, dy*yd, yd)
    # array of grid cell centers after interpolation
    # new grid size in CNN model
    dz_interp = core_length/cnnslice
    z_coord_interp = np.arange(dz_interp/2, dz_interp*cnnslice, dz_interp)
    # aquire interpolation function
    pet_interpolating_function = RegularGridInterpolator((x_coord_pet, y_coord_pet, z_coord_pet), data_3d)
    # Define coordinate list as 3 column matrix for locations of interpolated values
    X, Y, Z = np.meshgrid(x_coord_pet, y_coord_pet, z_coord_interp, indexing='ij')
    coord_list = np.transpose(np.vstack([X.ravel(), Y.ravel(), Z.ravel()]))
    # run interpolation
    pet_interp = pet_interpolating_function(coord_list)
    # reshape to correct size
    pet_interp_3d = pet_interp.reshape(xd, yd, cnnslice)
    return pet_interp_3d, dz_interp
    