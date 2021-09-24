# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 11:10:03 2020

@author: zahas
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
import torch
# from scipy.interpolate import RegularGridInterpolator

from exp_arrival_3d_functions import *
from temporal_moment_testing import *



# Berea
# data_filename = 'Berea_porosity_uncoarsened_nan'
# data_filename = 'Berea_scaled_CT_coarsened'
# carbonates
# data_filename = 'Edwards3Dporo_nan'
# data_filename = 'Indiana3Dporo_nan'
# data_filename = 'Ketton3Dporo_nan'
data_filename = 'Ketton_scaled_CT_coarsened'

# =============================================================================
# LOAD SELECTED EXAMPLE DATA 
# =============================================================================
# Set path to experimental data
data_dir = os.path.join('.', 'pet_data')
# data_dir = os.path.join('.')

# Import data
all_data = np.loadtxt(data_dir + '\\' + data_filename + '.csv', delimiter=',')

# PET_size(1); PET_size(2); PET_size(3); PET_size(4); timestep_length; q; inj_pv; vox_size(:)])
dz = all_data[-1] # voxel size in z direction (parallel to axis of core)
dy = all_data[-2] # voxel size in y direction
dx = all_data[-3] # voxel size in x direction
nslice = int(all_data[-4])
print(nslice)
nrow = int(all_data[-5])
ncol = int(all_data[-6])


# crop off dataset information
data = all_data[0:-6]
# reshape from imported column vector to 4D matrix
data = data.reshape(nrow, ncol, nslice)
# Flip data to orient top of cores to top of plot
# data = np.flip(data, 0)

# save_data = np.append(data.flatten('C'), [ncol, nrow, nslice, dx, dy, dz])
# np.savetxt(data_dir + '\\' + data_filename + '2.csv', save_data, delimiter=',')

# Plot at a few timesteps to make sure import is correct
# plot_2d(data[:,10,:], dz, dy, 'porosity', cmap='viridis')
plot_2d(data[:,:,0], dx, dy, 'porosity', cmap='viridis')

# remove nans
data[np.isnan(data)]=0
data[data<0]=0.001
data[data>0.5]=0.5

por_tensor = torch.from_numpy(data.copy())
por_tensor = por_tensor.reshape(1, nrow, ncol, nslice)

# pytorch downsample
# downsample along axis of core
por_tensor_coarse = interpolate(por_tensor, size=[ncol,40], mode='bilinear')
# permute and downsample plane perpendicular to long axis of core
por_tensor_coarse = por_tensor_coarse.permute(0, 3, 1, 2)
por_tensor_coarse = interpolate(por_tensor_coarse, size=([20, 20]), mode='bilinear')
# permute back to original orientation 
por_tensor_coarse = por_tensor_coarse.permute(0, 2, 3, 1)

# Hard coded mask for 20x20 grid cross-section
mp = np.array([7, 5, 3, 2, 2, 1, 1])
mask_corner = np.ones((10,10))
for i in range(len(mp)):
    mask_corner[i, 0:mp[i]] = 0
    
mask_top = np.concatenate((mask_corner, np.fliplr(mask_corner)), axis=1)
mask = np.concatenate((mask_top, np.flipud(mask_top)))

# print(mask)

# plot comparison of before and after
# plot_2d(mask, dx, dy, 'mask', cmap='gray')
# plot_2d(por_tensor[0, :,:,22], dx, dy, 'porosity', cmap='viridis')
# plt.clim(0.05, 0.2)
plot_2d(por_tensor_coarse[0,:,:,11], dx*(nrow/20), dy*(ncol/20), 'porosity', cmap='viridis')
plt.clim(0.05, 0.2)

for col in range(40):
    por_tensor_coarse[0,:,:,col] = np.multiply(por_tensor_coarse[0,:,:,col], mask)
    
plot_2d(por_tensor_coarse[0,:,:,11], dx*(nrow/20), dy*(ncol/20), 'porosity', cmap='viridis')
plt.clim(0.05, 0.2)

plot_2d(por_tensor_coarse[0,:,10,:], dx*(nrow/20), dy*(ncol/20), 'porosity', cmap='viridis')
# plt.clim(0.15, 0.2)



def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def half_core(data):
    r,c,s = np.shape(data)
    data = data[:,:-round(c/2),:]
    ncol = round(c/2)
    return data, ncol

##### 3D
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Arial']})
fs = 14
plt.rcParams['font.size'] = fs

dz = dz*nslice/40 # voxel size in z direction (parallel to axis of core)
dy = dy*nrow/20 # voxel size in y direction
dx = dx*ncol/20 # voxel size in x direction

nslice = 40
nrow = 20
ncol = 20


porosity_frame, ncol = half_core(por_tensor_coarse[0,:,:,:].cpu().detach().numpy())

# swap axes
porosity_frame = np.flip(porosity_frame, 0)
porosity_frame = np.swapaxes(porosity_frame,0,2)

# generate grid    
X, Y, Z = np.meshgrid(np.linspace(dy/2, (ncol-2)*dy+dy/2, num=(ncol+1)), \
                      np.linspace(dz/2, (nslice-2)*dz+dz/2, num=(nslice+1)), \
                      np.linspace(dx/2, (nrow-2)*dx+dx/2, num=(nrow+1)))


angle = -30
fig = plt.figure(figsize=(12, 9), dpi=300)
ax = fig.gca(projection='3d')
ax.view_init(30, angle)
# ax.set_aspect('equal') 

# if n==0: 
# norm = matplotlib.colors.Normalize(vmin=porosity_frame.min().min(), vmax=porosity_frame.max().max())
norm = matplotlib.colors.Normalize(vmin=0.1, vmax=0.22)

ax.voxels(X, Y, Z, porosity_frame, facecolors=plt.cm.BrBG(norm(porosity_frame)), \
          edgecolors='grey', linewidth=0.2, shade=False, alpha=0.7)

m = cm.ScalarMappable(cmap=plt.cm.BrBG, norm=norm)
m.set_array([])

divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", pad=0.05)
# cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(m, shrink=0.5)
set_axes_equal(ax)
# ax.set_xlim3d([0, 4])
ax.set_axis_off()
# PV = (i*tstep/60*q)/total_PV
# plt.title('PV = ' + str(PV))
# invert z axis for matrix coordinates
ax.invert_zaxis()
# Set background color to white (grey is default)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


# plt.savefig('berea_real_porosity_fs14.svg', format="svg")
plt.show()



