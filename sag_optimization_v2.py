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

    foreground = np.mean(wbc_pixels)
    background = np.mean(bg_pixels)
    
    return foreground, background



#%%
run_path = r"W:\raspberrypi\photos\Beta\B006\2025-02-26\S003"


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
    
    sag_mask_raw = cv2.adaptiveThreshold(sag_gfl.astype(np.uint8), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # 11,2
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # Adjust size as needed # 11,11
    
    # Apply morphological opening
    sag_mask = cv2.morphologyEx(sag_mask_raw, cv2.MORPH_OPEN, kernel)
    
    fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
    ax[0].imshow(sag_gfl)
    ax[1].imshow(sag_mask)
    
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
    print(gain_opt)
    
    if gain_opt < gain_range[0] or gain_opt > gain_range[-1]:
        print(f'Optimum gain {gain_opt} is out of range')
    
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












