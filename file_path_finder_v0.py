# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:56:26 2025

@author: imrul
"""

import os
import pandas as pd

def find_subfolders_with_extension(root_path, search_string, file_extension):
    matching_folders = []
    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Check if the folder name contains the search string
        if any(search_string in dirname for dirname in dirnames) or search_string in os.path.basename(dirpath):
            # Check if any file inside the folder has the given extension
            # if any(filename.endswith(file_extension) for filename in filenames):
            for filename in filenames:
                if filename.endswith(file_extension) is True and filename.startswith('.') is False:
                    matching_folders.append(dirpath)                
                    matching_files.append(os.path.join(dirpath,filename))
    return matching_files




root_directory = r"W:\raspberrypi\photos\PV_2025\B010"
search_string = "pl_bf"  #"pl_bead_offset"
file_extension = "stats.csv"  # Change this to the desired file extension

folders_with_files = find_subfolders_with_extension(root_directory, search_string, file_extension)
print(folders_with_files)


#%%

data_master = []

for csv_file in folders_with_files:
    print(csv_file)
    df = pd.read_csv(csv_file)
    data = [df['plt_count'].values[0],df['wbc_count_from_plt'].values[0],csv_file]
    # data = [df['total_rbc'].values[0],csv_file]
    data_master.append(data)
    
    

df_master = pd.DataFrame(data_master, columns = ['plt_count','wbc_count_from_plt','path'])
# df_master = pd.DataFrame(data_master, columns = ['bead_count','path'])

#%%

analysis_path = os.path.join(root_directory,search_string+'_bead_count_from_plt_compilation.csv')
df_master.to_csv(analysis_path)
    