# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:25:15 2021

@author: Czahasky
"""


# Only packages called in this script need to be imported
import os
import numpy as np
import matplotlib.pyplot as plt

# dry scans
path2dry = 'Z:\\Experimental_data\\Stanford_data\\PET_CT\\Navajo_ss_cores_strathclyde\\BS21\\CT_data\\June_SI_exp\\BH21-si-dry'
dry_list = [7, 8 , 9]
# wet scans
path2wet = 'Z:\\Experimental_data\\Stanford_data\\PET_CT\\Navajo_ss_cores_strathclyde\\BS21\\CT_data\\June_SI_exp\\Bh21_wet'
wet_list = [13, 14, 15]
filename_root = '_156_156_88_16bit.raw'

img_dim = [156, 156, 88]
vox_size = [0.031250*4, 0.031250*4, 0.125]


def plot_2d(map_data, dx, dy, colorbar_label, cmap, *args):

    r, c = np.shape(map_data)

    x_coord = np.linspace(0, dx*c, c+1)
    y_coord = np.linspace(0, dy*r, r+1)
    
    X, Y = np.meshgrid(x_coord, y_coord)
    
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
    if args:
        plt.clim(cmin, cmax) 
    # label the colorbar
    cbar.set_label(colorbar_label)
    # make colorbar font bigger
    # cbar.ax.tick_params(labelsize= (fs-2)) 
    # make axis fontsize bigger!
    plt.tick_params(axis='both', which='major')
    plt.xlim((0, dx*c)) 
    plt.ylim((0, dy*r)) 
    
def load_average_data(path, list_of_scans, filename_root, img_dim):
    All_data = np.zeros((img_dim[0], img_dim[1], img_dim[2], len(list_of_scans)))
    for i in range(0, len(list_of_scans)):
        data = np.fromfile((path + '\\' + str(list_of_scans[i]) + filename_root), dtype=np.uint16)
        data = np.reshape(data, (img_dim[2], img_dim[0], img_dim[1]))
        data = np.transpose(data, (1, 2, 0))
        All_data[:,:,:,i] = data
        Avg = np.mean(All_data, axis=3)
    return Avg


def coarsen_slices(array3d, coarseness):
    array_size = array3d.shape
    coarse_array3d = np.zeros((int(array_size[0]/coarseness), int(array_size[1]/coarseness), int(array_size[2])))
    for z in range(0, array_size[2]):
        array_slice = array3d[:,:, z]
        temp = array_slice.reshape((array_size[0] // coarseness, coarseness,
                                array_size[1] // coarseness, coarseness))
        coarse_array3d[:,:, z] = np.mean(temp, axis=(1,3))
    return coarse_array3d


Dry = load_average_data(path2dry, dry_list, filename_root, img_dim)

Wet = load_average_data(path2wet, wet_list, filename_root, img_dim)


# plot_2d(data[:,77,:], vox_size[2], vox_size[0], 'HU', cmap='gray')
# plot_2d(Dry[:,:,20], vox_size[0], vox_size[1], 'HU', cmap='gray')
plot_2d(Dry[:,77,:], vox_size[2], vox_size[0], 'HU', cmap='gray')
plt.clim(1000, 1600) 

Dry_coarse = coarsen_slices(Dry[2:-2,2:-2,:], 4)
plot_2d(Dry_coarse[:,:,22], vox_size[0], vox_size[1], '[-]', cmap='gray')
plt.clim(1000, 1600) 

plot_2d(Wet[:,77,:], vox_size[2], vox_size[0], 'HU', cmap='gray')
plt.clim(1000, 1600) 

Wet_coarse = coarsen_slices(Wet[2:-2,2:-2,:], 4)
plot_2d(Wet_coarse[:,:,22], vox_size[0], vox_size[1], '[HU]', cmap='gray')
plt.clim(1000, 1600) 

Por = (Wet_coarse - Dry_coarse) /(1000)
# crop porosity
Por = Por[:,:,4:-5]


plot_2d(Por[:,:,22], vox_size[0], vox_size[1], '[-]', cmap='viridis')
plt.clim(0.05, 0.2)

plot_2d(Por[:,22,:], vox_size[2], vox_size[1]*4, '[-]', cmap='viridis')
plt.clim(0.1, 0.2)

plot_2d(Wet[:,:,22], vox_size[0], vox_size[1], '[-]', cmap='gray')
plot_2d(Dry[:,:,22], vox_size[0], vox_size[1], '[-]', cmap='gray')
plot_2d(Por[:,:,22], vox_size[0], vox_size[1], '[-]', cmap='viridis')

data_size = Por.shape
save_filename = 'D:\\Dropbox\\Codes\\Deep_learning\\Neural_network_inversion\\experimental_data_prep\\pet_data'  + '\\' 'Navajo_porosity_coarsened.csv'
save_data = np.append(Por.flatten('C'), [data_size, vox_size])
np.savetxt(save_filename, save_data, delimiter=',')

