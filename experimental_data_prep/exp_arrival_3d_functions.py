# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:01:50 2021

@author: zahas
"""
# All packages called by functions should be imported
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy import integrate
import time

# Much faster quantile calculation 
def quantile_calc(btc_1d, timearray, quantile):
    # calculate cumulative amount of solute passing by location
    M0i = integrate.cumtrapz(btc_1d, timearray)
    # normalize by total to get CDF
    quant = M0i/M0i[-1]
    # calculate midtimes
    mid_time = (timearray[1:] + timearray[:-1]) / 2.0
    
    # now linearly interpolate to find quantile
    gind = np.argmax(quant > quantile)
    m = (quant[gind] - quant[gind-1])/(mid_time[gind] - mid_time[gind-1])
    b = quant[gind-1] - m*mid_time[gind-1]
    
    tau = (quantile-b)/m
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
    tau_in = quantile_calc(oned[0,:], timearray, quantile)
    
    # arrival time calculation in outlet slice
    tau_out = quantile_calc(oned[-1,:], timearray, quantile)

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
                        at_array[xc, yc, zc] = quantile_calc(cell_btc, timearray, quantile)

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
    
  
    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    
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
    plt.pcolormesh(X, Y, map_data, cmap=cmap, shading = 'auto', edgecolor ='k', linewidth = 0.01)
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
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r)) 
    # Label x-axis
    # plt.gca().invert_yaxis()
    
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
    
