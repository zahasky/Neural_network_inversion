# -*- coding: utf-8 -*-
"""
batching_unbatching_td.py
Created on Tue Aug  18 08:23:38 2020

@author: Christopher Zahasky

This script is used to move around files
"""

import os
import shutil
import numpy as np
import time


#define the function
def batchfiles(batch_size, td_range, td_file_prefix, batch_folder_prefix, path2files, path2batch_folders):
    # define range of training data sorted
    batch_folder_inc = np.arange(td_range[0], td_range[-1]+1, batch_size) 
    # start training data file tracker
    td_counter = td_range[0]
    # loop through each folder to be created
    for k in range(1, len(batch_folder_inc)):
        batch_foldername = batch_folder_prefix + str(batch_folder_inc[k-1]) + '_to_' + str(batch_folder_inc[k]-1) 
        
        destination_folder = os.path.join(path2batch_folders, batch_foldername)
        # If destination folder doesn't exist then create it
        if not os.path.isdir(destination_folder):
            os.mkdir(destination_folder)
    
        while td_counter < batch_folder_inc[k]:
            # define path to files
            filepath = path2files + td_file_prefix + str(td_counter) + '.csv'
            # copy training data to new folder
            shutil.copy(filepath, destination_folder) # (source, destination) 
            td_counter += 1
            
def delete_batchfiles(batch_size, td_range, batch_folder_prefix, path2batch_folders):
    # define range of training data sorted
    batch_folder_inc = np.arange(td_range[0], td_range[-1]+1, batch_size) 

    # loop through each folder to be created
    for k in range(1, len(batch_folder_inc)):
        batch_foldername = path2batch_folders + batch_folder_prefix + str(batch_folder_inc[k-1]) + '_to_' + str(batch_folder_inc[k]-1) 
        # if folder exists then delete it
        if os.path.isdir(batch_foldername):
            shutil.rmtree(batch_foldername)



## INPUT
            
td_file_prefix = 'tdg_km2_'
batch_folder_prefix = 'td_'
path2files = 'D:\\Training_data_generation_2D\\gauss_fields\\no_rotation\\'
path2batch_folders = 'D:\\Training_data_generation_2D\\gauss_fields\\batch_test\\'
# number of files per folder
batch_size = 100
# number of training data used
td_range = [1, 1001]


# Turn on timer (optional)
start_td = time.time() 
# Call batch sorting function
batchfiles(batch_size, td_range, td_file_prefix, batch_folder_prefix, path2files, path2batch_folders)

end_td = time.time() # end timer
print('Seconds to sort ', (end_td - start_td)) # show run time

# UNCOMMENT THIS SECTION TO DELETE CREATED FOLDERS
# start_td = time.time() # start a timer
# delete_batchfiles(batch_size, td_range, batch_folder_prefix, path2batch_folders)

# end_td = time.time() # end timer
# print('Seconds to delete ', (end_td - start_td)) # show run time