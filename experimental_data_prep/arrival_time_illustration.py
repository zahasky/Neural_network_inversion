# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:44:19 2021

@author: zahas
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Arial']})
fs = 16
plt.rcParams['font.size'] = fs

def quantile_function(btc_1d, timearray, quantile, t_increment):
    # first moment calculation
    m0 = np.trapz(btc_1d, timearray)
    m1 = np.trapz(btc_1d*timearray, timearray)
    M1 = m1/m0
        
    for i in range(1, ntime):
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
        x = np.arange(timearray[i-1], timearray[i]+t_increment, t_increment)
        
    return tau, M1, x, m, b




# ketton
data_filename = 'Ketton_2ml_2_3mm_cropped_nan'
# phi = 48.7/((3.1415*2.54**2)*10)
pv = 48.7
# Berea
# data_filename = 'Berea_C1_2ml_2_3mm_cropped_nan

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


timearray = np.arange(tstep/2, tstep*ntime, tstep)/60
pv_array = timearray*q/pv

btc_1d = pet_data[10,10, 5,:]
btc_1d2 = pet_data[10,10, 30,:]
    
quantile = 0.5
t_increment = 1

tau, M1, x, m, b = quantile_function(btc_1d, timearray, quantile, t_increment)
tau2, M12, x2, m2, b2 = quantile_function(btc_1d2, timearray, quantile, t_increment)

# Berea
data_filename = 'Berea_C1_2ml_2_3mm_cropped_nan'
pv_ber = 42.0
# Import data
all_data = np.loadtxt(data_dir + '\\' + data_filename + '.csv', delimiter=',')

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

# define time array
timearray_ber = np.arange(tstep/2, tstep*ntime, tstep)/60
pv_array_ber = timearray_ber*q/pv_ber

btc_1d_ber = pet_data[10,10, 5,:]
btc_1d2_ber = pet_data[10,10, 30,:]

tau_ber, M1_ber, x, m, b = quantile_function(btc_1d_ber, timearray_ber, quantile, t_increment)
tau2_ber, M12_ber, x2, m2, b2 = quantile_function(btc_1d2_ber, timearray_ber, quantile, t_increment)
    

n = 10
colors = np.flipud(plt.cm.Greys(np.linspace(0,1,n)))
bcolors = np.flipud(plt.cm.Blues(np.linspace(0,1,n)))
rcolors = np.flipud(plt.cm.Reds(np.linspace(0,1,n)))

fig0, (ax01, ax02) =  plt.subplots(1, 2, figsize=(14, 4), dpi=400)

ax01.plot(pv_array_ber, btc_1d_ber, '.-', color=colors[1], label = 'voxel BTC (10,10,5)')
# ax01.plot(x, m*x+b, '--r')
ax01.plot([tau_ber*q/pv_ber, tau_ber*q/pv_ber], [0, 1], color=bcolors[2], label = '0.5 quantile (10,10,5)')
ax01.plot([M1_ber*q/pv_ber, M1_ber*q/pv_ber], [0, 1],  color=rcolors[2], label = 'normalized first moment (10,10,5)')

ax01.plot(pv_array_ber, btc_1d2_ber, '.-', color=colors[4], label = 'voxel BTC (10,10,30)')
# ax01.plot(x2, m2*x2+b2, '--r')
ax01.plot([tau2_ber*q/pv_ber, tau2_ber*q/pv_ber], [0, 1],  color=bcolors[5], label = '0.5 quantile (10,10,30)')
ax01.plot([M12_ber*q/pv_ber, M12_ber*q/pv_ber], [0, 1],  color=rcolors[5], label = 'normalized first moment (10,10,30)')

ax01.legend(prop={"size":fs-2}, bbox_to_anchor=(1.0, -0.55), loc='lower center', ncol = 2)
ax01.set_xlabel('Pore Volumes [-]')
ax01.set_ylabel('PET Units [nCi/cm$^2$]')
ax01.set_title('Berea', fontsize=fs+2)
ax01.set_ylim([0, 0.05])
ax01.set_xlim([0, 2])
ax01.tick_params(axis='both', which='major', labelsize=fs-2)


ax02.plot(pv_array, btc_1d, '.-', color=colors[1])
ax02.plot([tau*q/pv, tau*q/pv], [0, 1], color=bcolors[2])
ax02.plot([M1*q/pv, M1*q/pv], [0, 1],  color=rcolors[2])

ax02.plot(pv_array, btc_1d2, '.-', color=colors[4])
ax02.plot([tau2*q/pv, tau2*q/pv], [0, 1],  color=bcolors[5])
ax02.plot([M12*q/pv, M12*q/pv], [0, 1],  color=rcolors[5])

# ax02.legend(prop={"size":fs-2})
ax02.set_xlabel('Pore Volumes [-]')
ax02.set_ylabel('PET Units [nCi/cm$^2$]')
ax02.set_title('Ketton', fontsize=fs+2)
ax02.set_ylim([0, 0.05])
ax02.set_xlim([0, 2])
ax02.tick_params(axis='both', which='major', labelsize=fs-2)
    
