# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:39:32 2025

@author: imrul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def calculate_nc_ratios(green_channel_path, red_channel_path):
    # Load the green and red channel images
    green_image = cv2.imread(green_channel_path, cv2.IMREAD_UNCHANGED)
    red_image = cv2.imread(red_channel_path, cv2.IMREAD_UNCHANGED)

    if green_image is None or red_image is None:
        raise ValueError("Error loading images. Check file paths.")

    # Threshold the green channel to segment the nucleus
    _, nucleus_mask = cv2.threshold(green_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    plt.figure()
    plt.imshow(nucleus_mask)
    plt.title('GREEN NUCLEUS')

    # Threshold the red channel to segment the cytoplasm
    _, cytoplasm_mask = cv2.threshold(red_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure()
    plt.imshow(cytoplasm_mask)
    plt.title('RED CYTOPLASM')

    # Label connected components in the nucleus mask
    labeled_nucleus = label(nucleus_mask)
    regions = regionprops(labeled_nucleus)

    nc_ratios = []

    for region in regions:
        # Extract the bounding box for each nucleus
        min_row, min_col, max_row, max_col = region.bbox

        # Extract the nucleus and cytoplasm regions for this cell
        nucleus_region = nucleus_mask[min_row:max_row, min_col:max_col]
        cytoplasm_region = cytoplasm_mask[min_row:max_row, min_col:max_col]

        # Calculate area for nucleus and cytoplasm
        nucleus_area = np.sum(nucleus_region)
        cytoplasm_area = np.sum(cytoplasm_region)

        # Skip if the cytoplasm area is zero
        if nucleus_area == 0:
            continue

        # Calculate the Nucleus-to-Cytoplasm ratio for this cell
        nc_ratio = cytoplasm_area / nucleus_area
        nc_ratios.append(nc_ratio)

    return nc_ratios

# Example usage
green_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R\img_WBC_green_FLG_fov1.tiff"  # Path to green channel image
red_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R\img_WBC_red_FLR_fov1.tiff"     # Path to red channel image

try:
    nc_ratios = calculate_nc_ratios(green_channel_path, red_channel_path)

    # Plot histogram of N:C ratios
    plt.figure()
    plt.hist(np.array(nc_ratios), bins=75, color='blue', edgecolor='black')
    plt.title("Cytoplasm-to-Nucleus Ratio Histogram")
    plt.xlabel("N:C Ratio")
    plt.ylabel("Frequency")
    plt.show()

except ValueError as e:
    print(e)


#%%
plt.figure()
plt.plot(nc_ratios,'o')