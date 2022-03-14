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
# Navajo
# path2dry = 'Z:\\Experimental_data\\Stanford_data\\PET_CT\\Navajo_ss_cores_strathclyde\\BS21\\CT_data\\June_SI_exp\\BH21-si-dry'
# filename_root = '_156_156_88_16bit.raw'
# dry_list = [7, 8 , 9]
# img_dim = [156, 156, 88]
# vox_size = [0.031250, 0.031250, 0.125]
# coarse_vox_size = [0.031250*4, 0.031250*4, 0.125]
# coarse_factor = 4
# # end crop: Dry[:,:,4:-5]
# core_max = 1700

# Berea
path2dry = 'D:\\Dropbox\\Writing\\PET_CNN_inversion_paper\\data\\raw_ct\\berea'
filename_root = '_berea_CT_scan_152_152_96_16b.raw'
dry_list = [10, 13, 16]
img_dim = [152, 152, 96]
vox_size = [0.031250, 0.031250, 0.125]
coarse_factor = 4
coarse_vox_size = [0.031250*coarse_factor, 0.031250*coarse_factor, 0.125]
# end crop: Dry[:,:,8:-8]


# Edwards
# path2dry = 'D:\\Dropbox\\Codes\\Deep_learning\\Neural_network_inversion\\experimental_data_prep\\pet_data\\raw_ct\\edwards'
# filename_root = 'Edwards_dry_424_424_115_16b.raw'
# dry_list = [1]
# img_dim = [424, 424, 115]
# vox_size = [0.0117, 0.0117, 0.1]
# coarse_vox_size = [0.0117*8, 0.0117*8, 0.1]
# #end crop: Dry[:,:,6:-8]
# coarse_factor = 8
# core_max = 1100
# core_min =500

# Indiana
# path2dry = 'D:\\Dropbox\\Codes\\Deep_learning\\Neural_network_inversion\\experimental_data_prep\\pet_data\\raw_ct\\indiana'
# filename_root = 'Indiana_dry_392_392_120_16b.raw'
# dry_list = [1]
# img_dim = [392, 392, 120]
# vox_size = [0.012, 0.012, 0.1]
# coarse_vox_size = [0.012*8, 0.012*8, 0.1]
# #end crop: Dry[:,:,7:-12]
# coarse_factor = 8
# core_max = 2100
# core_min =500

# Ketton
# path2dry = 'D:\\Dropbox\\Codes\\Deep_learning\\Neural_network_inversion\\experimental_data_prep\\pet_data\\raw_ct\\ketton'
# filename_root = 'Ketton_dry_106_106_105_16b.raw'
# dry_list = [1]
# img_dim = [106, 106, 105]
# vox_size = [0.046875, 0.046875, 0.1] #cm
# coarse_vox_size = [0.046875*2, 0.046875*2, 0.1]
# #end crop: Dry[:,:,4:102]
# coarse_factor = 2
# core_max = 2100
# core_min =500


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
# crop ends
Dry = Dry[:,:,8:-8]
# flip upside down to correctly orient
Dry = np.flip(Dry, 0)

Dry_coarse = coarsen_slices(Dry, coarse_factor)



# Try normalizing CT data to porosity range
# core_min = Dry_coarse[10:28, 10:28, :].min()
core_min = np.percentile(Dry_coarse[10:28, 10:28, :], 1)
# core_max = Dry_coarse[10:28, 10:28, :].max()
core_max = np.percentile(Dry_coarse[10:28, 10:28, :], 99)
# core_min = Dry_coarse[20:33,20:33, :].min()
# core_min = 1500

# plot_2d(data[:,77,:], vox_size[2], vox_size[0], 'HU', cmap='gray')
plot_2d(Dry_coarse[:,25,:], coarse_vox_size[2], coarse_vox_size[0], 'HU', cmap='gray')
plt.clim(core_min,core_max)

plot_2d(Dry[:,25*coarse_factor,:], vox_size[2], vox_size[0], 'HU', cmap='gray')
plt.clim(core_min,core_max)



Dry_coarse[Dry_coarse>core_max] = core_max
Dry_coarse[Dry_coarse<core_min] = core_min
norm_ct = np.absolute(Dry_coarse - core_max)/(core_max - core_min)
# por_guess = (norm_ct*0.10) + 0.15

plot_2d(norm_ct[:,25,:], coarse_vox_size[2], coarse_vox_size[0], 'HU', cmap='gray')
# plt.clim(core_min,core_max)

# por_guess = np.flip(por_guess, 0)
# plot_2d(por_guess[:,25,:], vox_size[2], coarse_vox_size[0], '[-]', cmap='viridis')
# plt.clim(0.15, 0.25)

# data_size = por_guess.shape

save_filename = 'D:\\Dropbox\\Codes\\Deep_learning\\Neural_network_inversion\\experimental_data_prep\\pet_data'  + '\\' 'Ketton_scaled_CT_coarsened.csv'
# save_data = np.append(por_guess.flatten('C'), [data_size, coarse_vox_size])
# np.savetxt(save_filename, save_data, delimiter=',')

data_size = norm_ct.shape
save_filename = 'D:\\Dropbox\\Codes\\Deep_learning\\Neural_network_inversion\\experimental_data_prep\\pet_data'  + '\\' 'berea_dry_norm.csv'
save_data = np.append(norm_ct.flatten('C'), [data_size, coarse_vox_size])
np.savetxt(save_filename, save_data, delimiter=',')

# range a = 1-4.5, b = 15-25 in form np.exp(norm_ct.flatten()*a)+b
# range a = 5 - 45, = 23-30 in form norm_ct.flatten()*a)+ (b-a/2)
    