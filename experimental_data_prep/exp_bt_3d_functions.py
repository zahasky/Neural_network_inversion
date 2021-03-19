# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:01:50 2021

@author: zahas
"""
# All packages called by functions should be imported
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator



# Function to calculate the mean breakthrough time
def mean_exp_bt_map(conc, timestep, t0, grid_size, v):
    # also output the mean breakthrough time map
    conc_size = conc.shape
    finite_vox_per_slice = (np.size(conc[:,:,0,0]) - np.sum(np.isnan(conc[:,:,0,0])))
    print(finite_vox_per_slice)
    # define array of times based on timestep size (in seconds)
    # Note that we are referencing the center of the imaging timestep since a 
    # given frame is an average of the detected emission events during that period
    timearray = np.arange(timestep/2, timestep*conc_size[3], timestep)

    # core length
    core_length = grid_size[2]*conc_size[2]
    # array of grid cell centers before interpolation
    z_coord_pet = np.arange(grid_size[2]/2, core_length, grid_size[2])
    z_coord_pet2 = np.arange(0, core_length+grid_size[2], grid_size[2])
    print(z_coord_pet2)
    
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
                        

    # 1D moment
    # np.sum(conc
    # If image discritization was aligned perfect with injection then 
    # half the injection time should be subtracted. 
    # half_inj_time = (inj_pv/2)
    # Instead we shift breakthrough time so that mean of first slice is 
    # half-way through the pulse injection
    # mean_bt_inlet = np.sum(bt_array[:,:,0])/finite_vox_per_slice
    # print(mean_bt_inlet)
    # bt_array -= mean_bt_inlet 

    # Arrival time difference map calculation
    # new mean breakthrough time at the inlet
    mean_bt_inlet = np.sum(bt_array[:,:,0])/finite_vox_per_slice
    print(mean_bt_inlet)
    # calculate mean arrival time at the outlet
    mean_bt_outlet = np.sum(bt_array[:,:,-1])/finite_vox_per_slice
    # Calculate velocity from difference in mean arrival time between inlet and outlet
# # STOP # # model_length = grid_size[2]*(conc_size[3]-1)
    v = (core_length-grid_size[2])/(mean_bt_outlet - mean_bt_inlet)
    print('advection velocity: ' + str(v))
    # vector of ideal mean arrival time based average v
    # bt_ideal = z_coord_pet/v
    # print(bt_ideal)

    # # Turn this vector into a matrix so that it can simple be subtracted from
    # bt_ideal_array = np.tile(bt_ideal, (conc_size[1], conc_size[2], 1))

    # # % Arrival time difference map
    # bt_diff = (bt_ideal_array - bt_array)
    
    # # normalized
    # bt_diff_norm = bt_diff*(v/model_length)
    
    # # Replace nans with zeros
    # bt_array[np.isnan(bt_array)] = 0
    # # Replace nans with zeros
    # bt_diff[np.isnan(bt_diff)] = 0
    # # Replace nans with zeros
    # bt_diff_norm[np.isnan(bt_diff_norm)] = 0
    return bt_array, M0, M1, M2
    # return bt_array, bt_diff, bt_diff_norm

# From Matlab
# % convert seconds to minutes in mean arrival time
# Xt = Xt./60;
# % shift mean arrival time so that mean of first slice is mean at x=0
# inj_t = inj_pv/q;
# mean_arrival_inlet = nanmean(nanmean(Xt(:,:,1)));
# Xt = Xt -(mean_arrival_inlet - (inj_t/2));
    
def plot_2d(map_data, dx, dy, colorbar_label, cmin, cmax, cmap):
    # fontsize
    fs = 18
    hfont = {'fontname':'Arial'}
    r, c = np.shape(map_data)
    # Define grid
    Lx = c * dx   # length of model in selected units 
    Ly = r * dy   # length of model in selected units
    y, x = np.mgrid[slice(0, Ly + dy, dy), slice(0, Lx + dx, dx)]
    # print(x)
    
    # fig, ax = plt.figure(figsize=(10, 10) # adjust these numbers to change the size of your figure
    # ax.axis('equal')          
    # fig2.add_subplot(1, 1, 1, aspect='equal')
    # Use 'pcolor' function to plot 2d map of concentration
    # Note that we are flipping map_data and the yaxis to so that y increases downward
    plt.figure()
    plt.pcolor(x, y, map_data, cmap=cmap, edgecolors='k', linewidths=0.2)
    plt.gca().set_aspect('equal')  
    # add a colorbar
    cbar = plt.colorbar() 
    plt.clim(cmin, cmax) 
    # label the colorbar
    cbar.set_label(colorbar_label, fontsize=fs, **hfont)
    # make colorbar font bigger
    cbar.ax.tick_params(labelsize= (fs-2)) 
    # make axis fontsize bigger!
    plt.tick_params(axis='both', which='major', labelsize=fs)
    # Label x-axis
    # plt.ylabel('Soil column depth [cm]', fontsize=fs)
#     plt.gca().invert_yaxis()
    
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
    
