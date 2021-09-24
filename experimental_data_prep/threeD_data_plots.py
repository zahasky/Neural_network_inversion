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

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Arial']})
fs = 14
plt.rcParams['font.size'] = fs


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
# arrival_data_filename = 'Berea_C1_2ml_2_3mm_at_norm' 
data_filename = 'Berea_C1_2ml_2_3mm_at'
# data_filename = 'Berea_C1_2ml_2_3mm_at_norm_nodiff'

# arrival_data_filename = 'Berea_C1_1ml_2_3mm_at_norm' 
# arrival_data_filename = 'Indiana_2ml_2_3mm_at_norm' 
# arrival_data_filename = 'Ketton_3ml_2_3mm_at_norm'
# arrival_data_filename = 'Bentheimer_2ml_2_3mm_at_norm'
# arrival_data_filename = 'Bentheimer_2ml_2_3mm_at_norm_nodiff'

data_dir_arrival = os.path.join('.', 'pet_arrival_time_data')
# Import data
arrival_data = np.loadtxt(data_dir_arrival + '\\' + data_filename + '.csv', delimiter=',')

arrival_data = arrival_data[0:-1]
nslice = 40
nrow = 20
ncol = 20
arrival_data = arrival_data.reshape(nrow, ncol, nslice)
dz = 0.2329 # voxel size in z direction (parallel to axis of core)
dy = 0.2329 # voxel size in y direction
dx = 0.2388 # voxel size in x direction

###### Plot PET data
# data_filename = 'Berea_C1_2ml_2_3mm_cropped_nan'
# total_PV = 42

# data_filename = 'Edwards_2ml_2_3mm_cropped_nan'
# total_PV = 85.6

# data_filename = 'Indiana_2ml_2_3mm_cropped_nan'
# total_PV = 34.9

# data_filename = 'Ketton_2ml_2_3mm_cropped_nan'
# total_PV = 48.7

# data_dir = os.path.join('.', 'pet_data')
# # Import data
# all_data = np.loadtxt(data_dir + '\\' + data_filename + '.csv', delimiter=',')

# dz = all_data[-1] # voxel size in z direction (parallel to axis of core)
# dy = all_data[-2] # voxel size in y direction
# dx = all_data[-3] # voxel size in x direction
# tracer_volume = all_data[-4] # tracer injected (ml)
# q = all_data[-5] # flow rate (ml/min)
# tstep = all_data[-6] # timstep length (sec)
# ntime = int(all_data[-7])
# nslice = int(all_data[-8])
# print(nslice)
# nrow = int(all_data[-10])
# ncol = int(all_data[-9])

# all_data = all_data[0:-10]

# all_data = all_data.reshape(nrow, ncol, nslice, ntime)

# all_data = all_data/np.nanmax(all_data)

# timestep = [3, 6]
# n=0

# for i in timestep:
#     print(i)

    # data_frame = all_data[:,:,:, i]
    
    # data_frame[np.isnan(data_frame)]=0

    # crop core
    # arrival_data, ncol = half_core(data_frame)
    
# crop core
arrival_data, ncol = half_core(arrival_data)

# swap axes
arrival_data = np.flip(arrival_data, 0)
arrival_data = np.swapaxes(arrival_data,0,2)

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
norm = matplotlib.colors.Normalize(vmin=arrival_data.min().min(), vmax=arrival_data.max().max())
# norm = matplotlib.colors.Normalize(vmin=np.percentile(arrival_data[arrival_data != 0],2.1), vmax=np.percentile(arrival_data[arrival_data != 0],99.1))
# norm = matplotlib.colors.Normalize(vmin=-0.19, vmax=0.19)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=0.5)
    
# ax.voxels(filled, facecolors=facecolors, edgecolors='gray', shade=False)
ax.voxels(X, Y, Z, arrival_data, facecolors=plt.cm.YlGn(norm(arrival_data)), \
          edgecolors='grey', linewidth=0.2, shade=False, alpha=0.7)

m = cm.ScalarMappable(cmap=plt.cm.YlGn, norm=norm)
m.set_array([])
# format colorbar
# format colorbar
# divider = make_axes_locatable(ax)
# cbar = plt.colorbar(m,shrink=0.3,pad=-0.148,ticks=None)
# # cbar.outline.set_linewidth(0.5)
# # for t in cbar.ax.get_yticklabels():
# #      t.set_fontsize(fs-1.5)
# # cbar.ax.yaxis.get_offset_text().set(size=fs-1.5)
# cbar.set_label('Pore Volumes', fontsize=fs, **hfont)
# tick_locator = ticker.MaxNLocator(nbins=6)
# cbar.locator = tick_locator
# cbar.update_ticks()

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
# ax.grid(False)
i=1
plt.savefig(data_filename[:5] + str(i) + 'at.svg', format="svg")
plt.show()
# n+=1
