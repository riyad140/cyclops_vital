# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:16:20 2025

@author: imrul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# def calculate_cn_ratios_and_areas(green_channel_path, red_channel_path):
    
green_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R\img_WBC_green_FLG_fov1.tiff"  # Path to green channel image
red_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R\img_WBC_red_FLR_fov1.tiff"     # Path to red channel image

    # Load the green and red channel images
green_image = cv2.imread(green_channel_path, cv2.IMREAD_UNCHANGED)
red_image = cv2.imread(red_channel_path, cv2.IMREAD_UNCHANGED)

if green_image is None or red_image is None:
    raise ValueError("Error loading images. Check file paths.")

# Apply thresholding to create binary masks for green and red channels
_, green_mask = cv2.threshold(green_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, red_mask = cv2.threshold(red_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Combine the binary masks to create a combined mask
combined_mask = (green_mask | red_mask).astype(np.uint8)

nand_mask = 1 - (green_mask & red_mask).astype(np.uint8)


nucleus_mask = nand_mask*green_mask
cytoplasm_mask = combined_mask - nucleus_mask

fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
ax[0].imshow(nucleus_mask)
ax[1].imshow(cytoplasm_mask)
ax[2].imshow(red_mask*red_image)


#%% area filtering

labeled_combined = label(combined_mask)
regions = regionprops(labeled_combined)

area_threshold = 500

# Step 3: Create a mask for large regions
filtered_mask = np.zeros_like(labeled_combined, dtype=bool)

# Iterate over the regions and keep only the ones with area above the threshold
for region in regions:
    if region.area > area_threshold:
        filtered_mask[labeled_combined == region.label] = True

#%%


#%%

fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(combined_mask)
ax[1].imshow(filtered_mask)


#%%


label_filtered = label(filtered_mask)
filtered_regions = regionprops(label_filtered)


#%%
cn_ratios = []

cytoplasm_areas = []
nucleus_areas = []

for region in filtered_regions:
    
    if region.label > 0 :
    # Extract the bounding box for each region
        min_row, min_col, max_row, max_col = region.bbox
    
        # Extract the nucleus and combined regions for this cell
        # nucleus_region = green_mask[min_row:max_row, min_col:max_col]
        
        cytoplasm_region = cytoplasm_mask[min_row:max_row, min_col:max_col]
        
        nucleus_region = nucleus_mask[min_row:max_row, min_col:max_col]
    
        # Calculate the area for this nucleus
        # cell_nucleus_area = np.sum(nucleus_region)
        
        cell_cytoplasm_area = np.sum(cytoplasm_region)
        
        cell_nucleus_area = np.sum(nucleus_region)
        
            
        
        print(f' cytoplasm : {cell_cytoplasm_area}\n nucleus : {cell_nucleus_area}')
        
        if cell_nucleus_area > 20 and cell_cytoplasm_area>20 and cell_nucleus_area<5000 and cell_cytoplasm_area<5000:
            cn_ratio = cell_cytoplasm_area / cell_nucleus_area
            cn_ratios.append(cn_ratio)
            cytoplasm_areas.append(cell_cytoplasm_area)
            nucleus_areas.append(cell_nucleus_area)
        
plt.figure()
plt.plot(cn_ratios,'+')        


plt.figure()
# bin_edges = np.arange(0, max(cn_ratios) + 0.1, 0.1)  # Bins of width 0.1

# Plot histogram of C:N ratios
plt.hist(np.array(cn_ratios), bins=200, color='blue', edgecolor='black')
plt.title("Cytoplasm-to-Nucleus Ratio Histogram")
plt.xlabel("C:N Ratio")
plt.ylabel("Frequency")


#%%

plt.figure()
# bin_edges = np.arange(0, max(cn_ratios) + 0.1, 0.1)  # Bins of width 0.1

# Plot histogram of C:N ratios
plt.hist(np.array(cytoplasm_areas), bins=100, color='blue', edgecolor='black')
plt.title("Cytoplasm area Histogram")
plt.xlabel("Cell area")
plt.ylabel("Frequency")



plt.figure()
# bin_edges = np.arange(0, max(cn_ratios) + 0.1, 0.1)  # Bins of width 0.1

# Plot histogram of C:N ratios
plt.hist(np.array(nucleus_areas), bins=100, color='red', edgecolor='black')
plt.title("Nucleus area Histogram")
plt.xlabel("Cell area")
plt.ylabel("Frequency")



plt.figure()
# bin_edges = np.arange(0, max(cn_ratios) + 0.1, 0.1)  # Bins of width 0.1

# Plot histogram of C:N ratios
plt.hist(np.array(nucleus_areas)*0.5 + np.array(cytoplasm_areas)*0.5, bins=100, color='red', edgecolor='black')
plt.title("Combined area Histogram")
plt.xlabel("Cell area")
plt.ylabel("Frequency")

#%%

# masked_red = combined_mask*red_image
# masked_green = combined_mask*green_image


# difference_image = masked_red-masked_green


# fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
# ax[0].imshow(masked_green)
# ax[1].imshow(masked_red)
# ax[2].imshow(difference_image,vmax = 1023)


# plt.figure()
# plt.imshow(difference_image)