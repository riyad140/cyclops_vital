# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:17:20 2025

@author: imrul
"""
import sys
sys.path.append(r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from file_path_finder_v0 import find_subfolders_with_extension



root_directory = r"W:\raspberrypi\photos\Beta\B020"
search_string = "pl_bf"  #"pl_bead_offset"
file_extension = "perfov_qc_checks.csv"  # Change this to the desired file extension

folders_with_files = find_subfolders_with_extension(root_directory, search_string, file_extension)
print(folders_with_files)


#%%
# csv_path = r"W:\raspberrypi\photos\Beta\B020\2025-03-28\s036_2\pl_bf-results-20250328_170944\S036_2-plt_perfov_qc_checks.csv"
random_number = np.random.randint(1, 11)

fig_num = random_number
for csv_path in folders_with_files:



    df = pd.read_csv(csv_path)
    
    
    sharpness_arr = df['plt_sharpness'][:-4] # skipping last two fovs for fiducials presence
    
    sharpness_norm = sharpness_arr/np.max(sharpness_arr)
    
    plt.figure(fig_num )
    plt.plot(sharpness_norm,label=os.path.split(csv_path)[-1])
    
plt.legend()