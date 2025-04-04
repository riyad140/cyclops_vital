# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:43:35 2025

@author: imrul
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def detect_nucleus_by_intensity(green_channel_path, red_channel_path, threshold_value):
    # Load the green and red channel images
    green_image = cv2.imread(green_channel_path, cv2.IMREAD_UNCHANGED)
    red_image = cv2.imread(red_channel_path, cv2.IMREAD_UNCHANGED)

    if green_image is None or red_image is None:
        raise ValueError("Error loading images. Check file paths.")

    # Compute the difference image
    difference_image = green_image.astype(np.int32) - red_image.astype(np.int32)

    # Normalize the difference image to the range [0, 255]
    difference_image_normalized = cv2.normalize(difference_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply threshold to isolate nucleus regions
    _, nucleus_mask = cv2.threshold(difference_image_normalized, threshold_value, 1, cv2.THRESH_BINARY)

    # Label connected components in the nucleus mask
    labeled_nucleus = label(nucleus_mask)
    regions = regionprops(labeled_nucleus)

    # Visualize the results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Green Channel")
    plt.imshow(green_image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title("Difference Image (Normalized)")
    plt.imshow(difference_image_normalized, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Nucleus Mask")
    plt.imshow(nucleus_mask, cmap='gray')
    plt.show()

    return regions

# Example usage
green_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R\img_WBC_green_FLG_fov1.tiff"  # Path to green channel image
red_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R\img_WBC_red_FLR_fov1.tiff"     # Path to red channel image

try:
    plt.figure()
    threshold_value = 50  # Adjust based on your data
    detected_nuclei = detect_nucleus_by_intensity(green_channel_path, red_channel_path, threshold_value)
    print(f"Detected {len(detected_nuclei)} nuclei.")
except ValueError as e:
    print(e)
