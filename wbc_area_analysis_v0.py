# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:05:38 2025

@author: imrul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def min_max_scaling(data):
    """
    Perform Min-Max Scaling on the given data.
    
    Parameters:
    data (numpy.ndarray): A 2D numpy array where rows are observations and columns are features.
    
    Returns:
    numpy.ndarray: The scaled data where each feature is scaled to the [0, 1] range.
    """
    # Convert the data to a numpy array if it's not already
    data = np.array(data)
    
    # Calculate the minimum and maximum values for each feature (column)
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    
    # Apply the Min-Max scaling formula: (X - min) / (max - min)
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    
    return scaled_data
#%%

csv_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R\WBC_Area_Analysis\master_stats.csv"
df = pd.read_csv(csv_path)

fig,ax=plt.subplots(1,3, sharey = True,figsize=(19.2, 10.8))

ax[0].plot(df['green_intensities'],df['red_intensities'],'*')
ax[0].set_title('RFL vs GFL')
ax[0].set_xlabel('Green FL')
ax[0].set_ylabel('Red FL')

ax[1].plot(df['nucleus_areas'],df['red_intensities'],'*')
ax[1].set_title('RFL vs Nucleus Area')
ax[1].set_xlabel('Nucleus Area')
ax[1].set_ylabel('Red FL')

ax[2].plot(df['cytoplasm_areas'],df['red_intensities'],'*')
ax[2].set_title('RFL vs Cytoplasm Area')
ax[2].set_xlabel('Cytoplasm Area')
ax[2].set_ylabel('Red FL')



#%%
fov = 1


filtered_df = df[df['fov']==fov]

print(filtered_df)

normalized_df = pd.DataFrame(columns=df.columns)
#%%
fovs = 10
fovs_to_skip = []
for fov in range(1,fovs+1):
    if fov in fovs_to_skip:
        continue  # Skip the current FOV
    print(f"Processing FOV {fov}")
    filtered_df = df[df['fov']==fov]
    filtered_df_copy = filtered_df.copy()
    
    cytoplasm_area_norm = min_max_scaling(filtered_df['cytoplasm_areas'])
    nuclus_area_norm = min_max_scaling(filtered_df['nucleus_areas'])
    
    green_intensities_norm = min_max_scaling(filtered_df['green_intensities'])
    red_intensities_norm = min_max_scaling(filtered_df['red_intensities'])
    
    
    filtered_df_copy['cytoplasm_areas'] = cytoplasm_area_norm
    filtered_df_copy['nucleus_areas'] = nuclus_area_norm
    filtered_df_copy['green_intensities'] = green_intensities_norm
    filtered_df_copy['red_intensities'] = red_intensities_norm
    
    normalized_df = pd.concat([normalized_df, filtered_df_copy], ignore_index=True)
    
    
fig,ax=plt.subplots(1,3, sharey = True,figsize=(19.2, 10.8))

ax[0].plot(normalized_df['green_intensities'],normalized_df['red_intensities'],'*')
ax[0].set_title('RFL vs GFL')
ax[0].set_xlabel('Green FL')
ax[0].set_ylabel('Red FL')

ax[1].plot(normalized_df['nucleus_areas'],normalized_df['red_intensities'],'*')
ax[1].set_title('RFL vs Nucleus Area')
ax[1].set_xlabel('Nucleus Area')
ax[1].set_ylabel('Red FL')

ax[2].plot(normalized_df['cytoplasm_areas'],normalized_df['red_intensities'],'*')
ax[2].set_title('RFL vs Cytoplasm Area')
ax[2].set_xlabel('Cytoplasm Area')
ax[2].set_ylabel('Red FL')
    
