# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:24:22 2025

@author: imrul
"""
import numpy as np
import skimage.morphology as sm
import skimage.filters as sf
import scipy.ndimage as ndi
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2


from skimage.measure import regionprops

def detect_cells_cyclops_cir(img, th_block=41, th_offset=-10, th_size=500, circularity_threshold=0.5):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Kernel size can be adjusted
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


def create_cell_mask(img, th_block=41, th_offset=-0.8, th_size=1000):
    """
    Generates a binary mask for cell regions.
    
    Parameters:
        img (ndarray): Grayscale input image.
        th_block (int): Local thresholding block size (must be odd).
        th_offset (float): Offset for threshold adjustment.
        th_size (int): Minimum cell area size to keep.

    Returns:
        mask (ndarray): Binary mask where cells = 1, background = 0.
    """
    # Adaptive thresholding (Sauvola for better contrast)
    t = sf.threshold_sauvola(img, window_size=th_block, k=th_offset)
    mask = img > t  # Apply thresholding
    
    # Fill holes in detected cell regions
    mask_filled = ndi.binary_fill_holes(mask)
    
    # Remove small objects (noise filtering)
    mask_cleaned = sm.remove_small_objects(mask_filled.astype(bool), th_size)
    
    return mask_cleaned.astype(np.uint8)

def detect_cells_cyclops5(img,th_block=41,th_offset=-0.8,th_size=1000):
    t=sf.threshold_local(img,th_block,offset=th_offset)
    img_t=img>t
    img_tf=ndi.binary_fill_holes(img_t)
    img_ts=sm.remove_small_objects(img_tf,th_size)
    labels_cells=sm.label(img_ts)
    return(img_ts)

file_path = r"W:\raspberrypi\photos\Beta\B006\2025-02-26\S003\img_WBC_SAG_greenFLG_fov0.tiff-6.tiff"

sag_gfl = imread(file_path)
#%%
# sag_mask = create_cell_mask(sag_gfl,41,-5,100)

label_cells = detect_cells_cyclops_cir(sag_gfl,41,-10,500)

fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(sag_gfl)
ax[1].imshow(label_cells)