# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:50:14 2025

@author: imrul

this is doing SAG with info from other filter wheel position
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



import numpy as np
import skimage.morphology as sm
import skimage.filters as sf
import scipy.ndimage as ndi
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2


from skimage.measure import regionprops

def detect_cells_cyclops_cir(img, th_block=41, th_offset=-10, th_size=1000, circularity_threshold=0.5):
    """
    Generates a binary mask for cell regions while reducing spurious detections
    and applying a circularity filter.
    
    Parameters:
        img (ndarray): Grayscale input image.
        th_block (int): Local thresholding block size (must be odd).
        th_offset (float): Offset for threshold adjustment.
        th_size (int): Minimum cell area size to keep.
        circularity_threshold (float): Minimum circularity to keep regions.
    
    Returns:
        mask (ndarray): Binary mask where cells = 1, background = 0.
    """
    t = sf.threshold_local(img, th_block, offset=th_offset)
    img_t = img > t  # Apply threshold
    img_tf = ndi.binary_fill_holes(img_t)  # Fill holes inside detected cells
    img_ts = sm.remove_small_objects(img_tf.astype(bool), th_size)  # Remove small noise

    # Convert mask to uint8 for OpenCV processing
    img_ts = img_ts.astype(np.uint8)

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Kernel size can be adjusted
    img_cleaned = cv2.morphologyEx(img_ts, cv2.MORPH_OPEN, kernel)  # Remove small objects
    img_cleaned = cv2.morphologyEx(img_cleaned, cv2.MORPH_CLOSE, kernel)  # Close small gaps

    # Label regions and apply circularity filter
    labeled_regions, num_labels = ndi.label(img_cleaned)  # Label connected regions
    properties = regionprops(labeled_regions)  # Get properties of labeled regions

    # Create a mask with only regions that meet the circularity threshold
    filtered_mask = np.zeros_like(img_cleaned)
    for prop in properties:
        area = prop.area
        perimeter = prop.perimeter
        if perimeter > 0:  # Prevent division by zero
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity >= circularity_threshold:  # Keep cells with high circularity
                filtered_mask[labeled_regions == prop.label] = 1  # Mark cell in mask

    return filtered_mask  # Return filtered mask with circularity filter applied


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

    foreground = np.median(wbc_pixels)
    background = np.median(bg_pixels)
    
    return foreground, background



#%%
run_path = r"\\10.106.0.71\cyclops\raspberrypi\photos\Beta\B011\2025-02-27\S042_3"
#r"W:\raspberrypi\photos\Beta\B006\2025-02-26\S003"#
#r"\\10.106.0.71\cyclops\raspberrypi\photos\Beta\B011\2025-02-27\S042_3"


image_list_BF = ["img_WBC_SAG_blankBF_fov0.tiff-1.tiff","img_WBC_SAG_blankBF_fov0.tiff-2.tiff","img_WBC_SAG_blankBF_fov0.tiff-3.tiff"]
image_list_DF = ["img_WBC_SAG_blueDF_fov0.tiff-2.tiff","img_WBC_SAG_blueDF_fov0.tiff-4.tiff","img_WBC_SAG_blueDF_fov0.tiff-6.tiff"]
image_list_FLR = ["img_WBC_SAG_redFLR_fov0.tiff-2.tiff","img_WBC_SAG_redFLR_fov0.tiff-4.tiff","img_WBC_SAG_redFLR_fov0.tiff-6.tiff"]
image_list_FLG = ["img_WBC_SAG_greenFLG_fov0.tiff-2.tiff","img_WBC_SAG_greenFLG_fov0.tiff-4.tiff","img_WBC_SAG_greenFLG_fov0.tiff-6.tiff"]



image_array = np.vstack([image_list_FLG,image_list_FLR,image_list_DF,image_list_BF]).T


df_image = pd.DataFrame(image_array, columns = ['FLG','FLR','DF','BF'])



#%%



df_cells_list=[]
df_bg_list = []
for index in df_image.index:
    print(index)
    iter0 = df_image.loc[index]
    sag_gfl = cv2.imread(os.path.join(run_path,iter0['FLG']), cv2.IMREAD_UNCHANGED)
    sag_rfl = cv2.imread(os.path.join(run_path,iter0['FLR']), cv2.IMREAD_UNCHANGED)
    sag_df = cv2.imread(os.path.join(run_path,iter0['DF']), cv2.IMREAD_UNCHANGED)
    sag_bf = cv2.imread(os.path.join(run_path,iter0['BF']), cv2.IMREAD_UNCHANGED)
    
    
    # _, sag_mask = cv2.threshold(sag_gfl, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # only using green for mask  
    
    plt.figure()
    plt.imshow(sag_gfl)
    
    # sag_mask_raw = cv2.adaptiveThreshold(sag_gfl.astype(np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 2)
    
    # plt.figure()
    # plt.imshow(sag_mask_raw)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # Adjust size as needed
    
    # # Apply morphological opening
    # sag_mask = cv2.morphologyEx(sag_mask_raw, cv2.MORPH_OPEN, kernel)
    
    sag_mask = detect_cells_cyclops_cir(sag_gfl)
    
    plt.figure()
    plt.imshow(sag_mask)
    
    try:
    
        #% read cell intensity and background for each channel
        signal_gfl, bg_gfl = get_foreground_background(sag_gfl,sag_mask)
        signal_rfl, bg_rfl = get_foreground_background(sag_rfl, sag_mask)
        signal_df, bg_df = get_foreground_background(sag_df, sag_mask)
        signal_bf, bg_bf = get_foreground_background(sag_bf,sag_mask)
        
        
        foreground_data = np.array([signal_gfl,signal_rfl,signal_df,signal_bf])
        background_data = np.array([bg_gfl,bg_rfl,bg_df,bg_bf])
        
        df_foreground = pd.DataFrame(foreground_data.reshape(1,-1),columns = ['FLG','FLR','DF','BF'])
        df_background = pd.DataFrame(background_data.reshape(1,-1),columns = ['FLG','FLR','DF','BF'])
        
        df_cells_list.append(df_foreground)
        df_bg_list.append(df_background)
    except:
        print('Warning: Cell detection failed')
        continue

df_cells = pd.concat(df_cells_list,ignore_index = True)
df_bg = pd.concat(df_bg_list,ignore_index = True)



#%%
sag_target_intensity = 512
gain_range = np.arange(1.0, 7.5, 0.25)  # Min, max, step size

gain_final_list =[]

for key in df_cells.columns:
    print(key)
    if key == 'BF':
        gains = np.array([1, 2, 3])  # Given gain values
    else:
        gains = np.array([2, 4, 6])  # Given gain values
    
    m,c  = np.polyfit(gains,df_cells[key],deg=1)


    gain_opt = (sag_target_intensity-c)/m
    print(f'Target intensity {sag_target_intensity} optimum gain {gain_opt} calculated with slope {m} and intercept {c}' )
    
    if gain_opt < gain_range[0] or gain_opt > gain_range[-1]:
        print(f'WARNING!!!!!Optimum gain {gain_opt} is out of range')
    
    gain_opt_final = gain_range[np.argmin(np.abs(gain_range-gain_opt))]
    gain_final_list.append(gain_opt_final)
    
    print(f'valid optimum gain {gain_opt_final}')
    
    
    # r,c = df_cells.shape
    
    plt.figure()
    plt.plot(gains,df_cells[key],'o-')
    plt.plot(gain_range, m*gain_range+c)
    plt.axvline(x = gain_opt_final, color = 'k', linestyle = '--')
    plt.title(key)    


print(pd.DataFrame(np.array(gain_final_list).reshape(1,-1), columns = df_cells.columns))





#%%

# sag_mask_raw = cv2.adaptiveThreshold(
#     sag_gfl.astype(np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY, 51, 2
# )

# # Visualization: Apply Gaussian blur instead of cv2.blur
# blurred_mask = cv2.GaussianBlur(sag_mask_raw, (1, 1), 0)

# plt.figure()
# plt.imshow(blurred_mask, cmap='gray')

# # Morphological Opening (removes noise)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
# sag_mask = cv2.morphologyEx(blurred_mask, cv2.MORPH_OPEN, kernel)

# plt.figure()
# plt.imshow(sag_mask, cmap='gray')

# plt.show()











