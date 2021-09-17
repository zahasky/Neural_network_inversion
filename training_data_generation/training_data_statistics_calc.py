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



## INPUT
            
td_file_prefix = 'corenm_k_3d_m2_'
path2files = 'D:\\Training_data_generation_3D\\tdata_python26k\\June_21_new_mask\\syn_core_perm_maps_new_mask'


Ps = np.loadtxt('D:\\Training_data_generation_3D\\tdata_python26k\\parameter_space_26k_with_porosity.csv', delimiter=',', dtype= np.float32)

# number of training data used
td_range = [1, 26000]

# Turn on timer (optional)
start_td = time.time() 

stats_array = np.zeros([26000, 3])

max_std = 0
min_std = 1000

for i in range(1,26001):
    model_lab_filename_sp = path2files + '\\' + td_file_prefix + str(i) + '.csv'
    pdata_ex = np.loadtxt(model_lab_filename_sp, delimiter=',', dtype= np.float32)
    pdata_ex = pdata_ex[0:-3]/np.float32(9.8692e-16)
    
    # calculate std while ignoring zeros
    stats_array[i-1, 0] = pdata_ex[pdata_ex!=0].std()
    # cstd = pdata_ex.var()
    stats_array[i-1, 1] = pdata_ex[pdata_ex!=0].mean()
    # calculate the coefficient of variation
    stats_array[i-1, 2] = stats_array[i-1, 0]/ (10**Ps[i-1, 1])
 
    # print('  variance? = ' + str(10**(np.exp(10**Ps[i-1,0]) + Ps[i-1,1])))
    # print('actual variance = ' + str(cstd))
    
    # print('defined perm = ' + str(10**Ps[i-1,1]))
    # print('actual mean perm = ' + str(mean_perm))
    
    # if cstd > max_std:
    #     max_std = cstd
    #     # print('new max var = ' + str(max_std))
    
    # if cstd < min_std:
    #     min_std = cstd
        # print('new min var = ' + str(min_std))



end_td = time.time() # end timer
print('Seconds to run ', (end_td - start_td)) # show run time

np.savetxt('real_perm_stats_mD2', stats_array , delimiter=',', fmt='%.6e')

# UNCOMMENT THIS SECTION TO DELETE CREATED FOLDERS
# start_td = time.time() # start a timer
# delete_batchfiles(batch_size, td_range, batch_folder_prefix, path2batch_folders)

# end_td = time.time() # end timer
# print('Seconds to delete ', (end_td - start_td)) # show run time

plt.hist(stats_array[:, 2], bins='auto')