# -*- coding: utf-8 -*-
"""
threeD_data_plots

Created on Thu Apr  8 17:15:04 2021

@author: Christopher Zahasky (czahasky@wisc.edu)
"""

# import necessary packages
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


# Import arrival time data
# arrival_data_filename = 'Berea_C1_1ml_2_3mm_at_norm' 
# arrival_data_filename = 'Indiana_2ml_2_3mm_at_norm' 
# arrival_data_filename = 'Ketton_3ml_2_3mm_at_norm'
# arrival_data_filename = 'Bentheimer_2ml_2_3mm_at_norm'
arrival_data_filename = 'Bentheimer_2ml_2_3mm_at_norm_nodiff'

data_dir_arrival = os.path.join('.', 'pet_arrival_time_data')
# Import data
arrival_data = np.loadtxt(data_dir_arrival + '\\' + arrival_data_filename + '.csv', delimiter=',')

arrival_data = arrival_data[0:-1]
nslice = 40
nrow = 20
ncol = 20
arrival_data = arrival_data.reshape(nrow, ncol, nslice)
dz = 0.2329 # voxel size in z direction (parallel to axis of core)
dy = 0.2329 # voxel size in y direction
dx = 0.2388 # voxel size in x direction

# plot_2d(arrival_data[:,11,:], dz, dy, 'arrival time', cmap='bwr')


# crop core
arrival_data, ncol = half_core(arrival_data)
# swap axes
arrival_data = np.swapaxes(arrival_data,0,2)

# generate grid    
X, Y, Z = np.meshgrid(np.linspace(dy/2, (ncol-2)*dy+dy/2, num=(ncol+1)), \
                      np.linspace(dz/2, (nslice-2)*dz+dz/2, num=(nslice+1)), \
                      np.linspace(dx/2, (nrow-2)*dx+dx/2, num=(nrow+1)))


angle = -25
fig = plt.figure(figsize=(12, 9), dpi=300)
ax = fig.gca(projection='3d')
ax.view_init(25, angle)
# ax.set_aspect('equal') 

    
norm = matplotlib.colors.Normalize(vmin=arrival_data.min().min(), vmax=arrival_data.max().max())
    
# ax.voxels(filled, facecolors=facecolors, edgecolors='gray', shade=False)
ax.voxels(X, Y, Z, arrival_data, facecolors=plt.cm.PiYG(norm(arrival_data)), \
          edgecolors='grey', linewidth=0.2, shade=False, alpha=0.7)

m = cm.ScalarMappable(cmap=plt.cm.PiYG, norm=norm)
m.set_array([])
# format colorbar
divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", pad=0.05)
# cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(m, shrink=0.5)
set_axes_equal(ax)
# ax.set_xlim3d([0, 4])
ax.set_axis_off()

# invert z axis for matrix coordinates
ax.invert_zaxis()
# Set background color to white (grey is default)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.grid(False)

plt.show()