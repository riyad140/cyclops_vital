# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:50:14 2025

@author: imrul

this is doing SAG without info from other filter wheel position
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import os
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d


def calculate_weber_contrast(image,mask):
    """
    Compute Weber contrast for a sparse WBC image.
    
    Args:
        image: Grayscale fluorescence image.
    
    Returns:
        Weber contrast value.
    """
    # Step 1: Segment WBCs using Otsu's thresholding
    # _, wbc_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Compute foreground (WBC) and background intensities
    wbc_pixels = image[mask > 0]  # Pixels inside WBCs
    bg_pixels = image[mask == 0]  # Pixels outside WBCs

    if len(wbc_pixels) == 0 or len(bg_pixels) == 0:
        return 0  # Avoid errors if no cells are detected

    foreground = np.mean(wbc_pixels)
    background = np.mean(bg_pixels)

    # Step 3: Compute Weber contrast
    return (foreground - background) / (background + 1e-8)  # Avoid division by zero

# def find_optimum_interpolated_gain(contrast_values, gains, gain_range):
#     """
#     Find the best interpolated gain that maximizes Weber contrast.
    
#     Args:
#         frames: List of images taken at given gain settings.
#         gains: List of actual gain values for the frames.
#         gain_range: Tuple (min_gain, max_gain, step_size).
    
#     Returns:
#         Best interpolated gain value.
#     """
#     # contrast_values = np.array([calculate_weber_contrast(frame) for frame in frames])
    
#     # Interpolate contrast vs. gain using a quadratic function
#     interp_func = interp1d(gains, contrast_values, kind='quadratic', fill_value="extrapolate")
    
#     # Define optimization function (negative contrast for minimization)
#     def neg_contrast(gain):
#         return -interp_func(gain)

#     # Perform optimization within the gain range
#     result = minimize_scalar(neg_contrast, bounds=gain_range[:2], method='bounded')

#     return result.x, interp_func(result.x)


def find_optimum_interpolated_gain(contrast_values, gains, gain_range):
    """
    Find the best interpolated gain that maximizes Weber contrast with step size constraints.
    
    Args:
        contrast_values (list or np.array): List of contrast values corresponding to each gain.
        gains (list or np.array): List of actual gain values for the frames.
        gain_range (tuple): Tuple (min_gain, max_gain, step_size).
    
    Returns:
        tuple: (Best interpolated gain value, corresponding contrast value)
    """
    # Ensure input is numpy array
    contrast_values = np.array(contrast_values)
    gains = np.array(gains)

    # Extract gain range values
    min_gain, max_gain, step_size = gain_range

    # Interpolation function (quadratic if possible, otherwise linear)
    if len(gains) >= 3:
        interp_func = interp1d(gains, contrast_values, kind='quadratic', fill_value="extrapolate")
    else:
        interp_func = interp1d(gains, contrast_values, kind='linear', fill_value="extrapolate")

    # Generate discrete gain values within the given range
    gain_values = np.arange(min_gain, max_gain + step_size, step_size)

    # Compute interpolated contrast values at these discrete gain points
    interpolated_contrasts = interp_func(gain_values)

    # Find the best gain with maximum contrast
    best_index = np.argmax(interpolated_contrasts)
    best_gain = gain_values[best_index]
    best_contrast = interpolated_contrasts[best_index]
    
    plt.figure()
    plt.plot(gain_values, interpolated_contrasts)

    return best_gain, best_contrast


def get_foreground_background(image,mask):
    wbc_pixels = image[mask > 0]  # Pixels inside WBCs
    bg_pixels = image[mask == 0]  # Pixels outside WBCs

    if len(wbc_pixels) == 0 or len(bg_pixels) == 0:
        return 0  # Avoid errors if no cells are detected

    foreground = np.mean(wbc_pixels)
    background = np.mean(bg_pixels)
    
    return foreground, background



#%%

run_path = r"W:\raspberrypi\photos\Beta\B010\2025-02-20\S032_2"

sag_image_names = [
    
    "img_WBC_SAG_redFLR_fov0.tiff-2.tiff",
    "img_WBC_SAG_redFLR_fov0.tiff-4.tiff",
    "img_WBC_SAG_redFLR_fov0.tiff-6.tiff",    
    ]

sag_images = []
sag_masks = []
contrast_values = []
image_stats = []

for image_name in sag_image_names:
    sag_image = cv2.imread(os.path.join(run_path,image_name), cv2.IMREAD_UNCHANGED)
    sag_images.append(sag_image)   
    _, sag_mask = cv2.threshold(sag_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sag_masks.append(sag_mask)
    contrast_values.append(calculate_weber_contrast(sag_image,sag_mask))
    image_stats.append(get_foreground_background(sag_image,sag_mask))
    

contrast_values = np.array(contrast_values)
gains = np.array([2, 4, 6])  # Given gain values
gain_range = (1.0, 7.5, 0.25)  # Min, max, step size

best_gain, best_contrast = find_optimum_interpolated_gain(contrast_values, gains, gain_range)

print(f"Best Interpolated Gain: {best_gain:.2f}")
print(f"Estimated Weber Contrast at Best Gain: {best_contrast:.4f}")
    
fig,ax = plt.subplots(1,2,sharex=True, sharey=True)
ax[0].imshow(sag_images[0])
ax[1].imshow(sag_masks[0])    

#%%
image_stats = np.array(image_stats)
plt.figure()
plt.plot(gains,image_stats[:,0],label = 'cells')
plt.plot(gains,image_stats[:,1],label = 'bg')
# assayName = 'WBC'


# fovs = 30
# fovs_to_skip = []

# analysis_folder = os.path.join(run_path,f'{assayName}_Area_Analysis')

# df= pd.DataFrame(columns = ['green_intensities', 'red_intensities', 'nucleus_areas', 'cytoplasm_areas', 'fov' ])

# data=[]


# from datetime import datetime

# # Get the current timestamp with high precision
# unique_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")


# try:
#     os.mkdir(os.path.join(run_path,analysis_folder))
    
# except:
#     print('Analysis Folder Already Exists. Overwritting Files')    

# for fov in range(1,fovs+1):
    
#     if fov in fovs_to_skip:
#         continue  # Skip the current FOV
#     print(f"Processing FOV {fov}")
    
#     green_channel_path = os.path.join(run_path ,f"img_WBC_green_FLG_fov{fov}.tiff" ) # Path to green channel image
#     red_channel_path =os.path.join(run_path ,f"img_WBC_red_FLR_fov{fov}.tiff")
