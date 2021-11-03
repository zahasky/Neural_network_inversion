# -*- coding: utf-8 -*-
"""
Standard deviation calc
Created on Tue Aug  18 08:23:38 2020

@author: Christopher Zahasky

This script is used to move around files
"""

import os
import shutil
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Arial']})
fs = 15
plt.rcParams['font.size'] = fs


## INPUT
            
td_file_prefix = 'corenm_k_3d_m2_'
path2files = 'D:\\Training_data_generation_3D\\tdata_python26k\\June_21_new_mask\\syn_core_perm_maps_new_mask'


# Ps = np.loadtxt('D:\\Training_data_generation_3D\\tdata_python26k\\parameter_space_26k_with_porosity.csv', delimiter=',', dtype= np.float32)

# load experiment data
Ps = np.loadtxt('parameter_space_26k_6_21.csv', delimiter=',')

# load rmse data
rmse_data = np.loadtxt('rmse_Test.csv', delimiter=',')
rsme_data = np.flipud(rmse_data)

# load stats data
stats_array = np.loadtxt('real_perm_stats_mD2.csv', delimiter=',')

# verify that original realization perm aligns with measured perm
plt.plot(np.log10(stats_array[0:26000,1]), Ps[0:26000,1], 'ok')
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, sharey= True, dpi=200, figsize=(9, 3.5))
ax1.plot(np.log10(stats_array[25500:26000,1]), np.log10(np.exp(rmse_data)), '.k')
ax1.set(xlabel = 'log$_{10}$('+ r'$\bar k$' +') [mD]', ylabel='RMSE log$_{10}$(k) [mD]')

ax2.plot(np.log10(stats_array[25500:26000,0]), np.log10(np.exp(rmse_data)), '.k')
# ax1.set_title('Sharing Y axis')
ax2.set(xlabel = r'log$_{10}( \sigma(k))$ [mD]')
plt.tight_layout()

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey= True, dpi=200, figsize=(12, 3.5))
ax1.plot(np.log10(stats_array[25500:26000,1]), np.log10(np.exp(rmse_data)), '.k')
ax1.set(xlabel = 'log$_{10}$('+ r'$\bar k$' +') [mD]', ylabel='RMSE log$_{10}$(k) [mD]')

ax2.plot(np.log10(stats_array[25500:26000,0]), np.log10(np.exp(rmse_data)), '.k')
# ax1.set_title('Sharing Y axis')
ax2.set(xlabel = r'log$_{10}( \sigma(k))$ [mD]')

ax3.plot(np.mean(Ps[25500:26000,2:5], axis=1)*0.25, np.log10(np.exp(rmse_data)), '.k')
# ax1.set_title('Sharing Y axis')
ax3.set(xlabel = 'Mean correlation length [cm]')

plt.tight_layout()



# crossplot to see if anything is correlated
# plt.plot(np.log10(stats_array[25500:26000,0]), rmse_data, 'ok')
# plt.show()

# plt.plot(np.log10(stats_array[25500:26000,1]), rmse_data, 'ok')
# plt.show()

# # number of training data used
# td_range = [1, 26000]

# # Turn on timer (optional)
# start_td = time.time() 

# stats_array = np.zeros([26000, 3])

# max_std = 0
# min_std = 1000

### Double check load
# i = 25501
# model_lab_filename_sp = path2files + '\\' + td_file_prefix + str(i) + '.csv'
# pdata_ex = np.loadtxt(model_lab_filename_sp, delimiter=',', dtype= np.float32)

# for i in range(1,26001):
#     model_lab_filename_sp = path2files + '\\' + td_file_prefix + str(i) + '.csv'
#     pdata_ex = np.loadtxt(model_lab_filename_sp, delimiter=',', dtype= np.float32)
#     pdata_ex = pdata_ex[0:-3]/np.float32(9.8692e-16)
    
#     # calculate std while ignoring zeros
#     stats_array[i-1, 0] = pdata_ex[pdata_ex!=0].std()
#     # cstd = pdata_ex.var()
#     stats_array[i-1, 1] = pdata_ex[pdata_ex!=0].mean()
#     # calculate the coefficient of variation
#     stats_array[i-1, 2] = stats_array[i-1, 0]/ (10**Ps[i-1, 1])
 
#     # print('  variance? = ' + str(10**(np.exp(10**Ps[i-1,0]) + Ps[i-1,1])))
#     # print('actual variance = ' + str(cstd))
    
#     # print('defined perm = ' + str(10**Ps[i-1,1]))
#     # print('actual mean perm = ' + str(mean_perm))
    
#     # if cstd > max_std:
#     #     max_std = cstd
#     #     # print('new max var = ' + str(max_std))
    
#     # if cstd < min_std:
#     #     min_std = cstd
#         # print('new min var = ' + str(min_std))



# end_td = time.time() # end timer
# print('Seconds to run ', (end_td - start_td)) # show run time

# np.savetxt('real_perm_stats_mD2', stats_array , delimiter=',', fmt='%.6e')

# UNCOMMENT THIS SECTION TO DELETE CREATED FOLDERS
# start_td = time.time() # start a timer
# delete_batchfiles(batch_size, td_range, batch_folder_prefix, path2batch_folders)

# end_td = time.time() # end timer
# print('Seconds to delete ', (end_td - start_td)) # show run time

### generate porosity- permeability example
plt.figure(dpi=200)

ndata= 500

color = iter(cm.viridis(np.linspace(0, 1, ndata)))

for i in range(25000,25000+ndata):
    # extract realization propertie information
    p = Ps[i-1]
    # Import permeability map    
    model_lab_filename_sp = path2files + '\\' + td_file_prefix + str(i) + '.csv'
    field_km2 = np.loadtxt(model_lab_filename_sp, delimiter=',', dtype= np.float32)
    # crop off last three values that give model dimensions
    field_km2 = field_km2[:-3]/(9.869233E-13/1000)
    # plot_2d(field_km2[:,:,1], grid_size[0], grid_size[1], 'perm', cmap='gray')
    
    # Calculatie heterogeneous porosity field
    prsity_field = ((np.log(field_km2)/p[9]) + p[10])/100
    # replace infinity values with zeros
    prsity_field[field_km2<1e-25]=0
    # set upper limit of 80 percent porosity
    prsity_field[prsity_field >0.8]=0.8
    
    c = next(color)
    plt.plot(np.log10(field_km2), prsity_field, '-', c=c, linewidth=0.5)
    
plt.xlabel(r'log$_{10}(k)$ [mD]')
plt.ylabel('porosity [-]')