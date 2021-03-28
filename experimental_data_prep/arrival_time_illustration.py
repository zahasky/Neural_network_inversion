# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:44:19 2021

@author: zahas
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def quantile_function(btc_1d, timearray, quantile, t_increment):
    # first moment calculation
    m0 = np.trapz(btc_1d, timearray)
    m1 = np.trapz(btc_1d*timearray, timearray)
    M1 = m1/m0
        
    for i in range(2, ntime):
        M0i = np.trapz(btc_1d[:i], timearray[:i])
    
        if M0i/m0 > quantile:
            M0im = np.trapz(btc_1d[:i-1], timearray[:i-1])
            # linear interpolation
            m = (btc_1d[i-1] - btc_1d[i-2])/(timearray[i-1] - timearray[i-2])
            b = btc_1d[i-2] - m*timearray[i-2]
        
            for xt in np.arange(timearray[i-2], timearray[i-1]+t_increment, t_increment):
                M0int = M0im + np.trapz([btc_1d[i-2], m*xt+b], [timearray[i-2], xt])
            
                if M0int/m0 > quantile:
                    tau = xt
                    print(tau)
                    break
            
            break
        # output the line that is being used for linear interpolation of quantile
        x = np.arange(timearray[i-2], timearray[i-1], t_increment)
        
    return tau, M1, x, m, b




# ketton
data_filename = 'Ketton_4ml_1_2_3mm_cropped_nan'
phi = 48.7/((3.1415*2.54**2)*10)
km2 = 1920*9.869233E-13/1000

timestep = 4
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
tracer_volume = all_data[-4] # tracer injected (ml)
q = all_data[-5] # flow rate (ml/min)
tstep = all_data[-6] # timstep length (sec)
ntime = int(all_data[-7])
nslice = int(all_data[-8])
nrow = int(all_data[-10])
ncol = int(all_data[-9])
# calculate tracer injection duration in seconds
tracer_inj_duration = tracer_volume/q*60 

# crop off dataset information
pet_data = all_data[0:-10]
# reshape from imported column vector to 4D matrix
pet_data = pet_data.reshape(nrow, ncol, nslice, ntime)
# crop edges
pet_data = pet_data[1:-1, 1:-1, :, :]

C1d = np.nansum(np.nansum(pet_data, 0), 0)
timearray = np.arange(tstep/2, tstep*ntime, tstep)
# # BTC at a given location
# btc_1d = C1d[-1,:]

btc_1d = pet_data[0,7, 3,:]
    
    
quantile = 0.5
t_increment = 1

tau, M1, x, m, b = quantile_function(btc_1d, timearray, quantile, t_increment)
    
plt.figure(figsize=(8, 4), dpi=200)
plt.plot(timearray, btc_1d, '-ok')
plt.plot(x, m*x+b, '--r')

plt.plot([tau, tau], [0, np.max(btc_1d)], label = '0.5 quantile')
plt.plot([M1, M1], [0, np.max(btc_1d)], 'g', label = 'normalized first moment')
plt.legend()
plt.xlabel('Time [min]')


# Play around with plotting volumes
import plotly.graph_objects as go

pet_at_dir = 'C:\\Users\\zahas\\Dropbox\\Matlab\\Deep_learning\\Neural_network_inversion\\experimental_data_prep\\pet_arrival_time_data'
# Import arrival time data
arrival_data = np.loadtxt(pet_at_dir + '\\Berea_C1_1ml_2_3mm_at_norm' + '.csv', delimiter=',')
arrival_data = arrival_data[0:-1]
arrival_data = arrival_data.reshape(nrow-2, ncol-2, nslice-1)
    
X, Y, Z = np.meshgrid(np.linspace(dx/2, (nrow-2)*dx+dx/2, num=(nrow-2)), \
                      np.linspace(dy/2, (ncol-2)*dy+dy/2, num=(ncol-2)), \
                      np.linspace(dz/2, (nslice-2)*dz+dz/2, num=(nslice-1)))
# values = np.sin(X*Y*Z) / (X*Y*Z)

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=arrival_data.flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
fig.show()