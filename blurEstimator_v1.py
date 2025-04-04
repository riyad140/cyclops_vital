# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:25:03 2024

@author: imrul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image and convert to grayscale

imagePath = r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PLT\2024-08-07\PS_beads_fiducials_74p15_0p25_run01\img_PLT_AF-Fine-fiducials_fov1_offset_0.tiff\af-1-358.png"
image = cv2.imread(imagePath)
gray = image

# Step 2: Apply Gaussian Blur to simulate a blurry edge (optional)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Detect edges using Sobel operator (or use Canny edge detection)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
edges = np.hypot(sobelx, sobely)  # Combine both directions

# Step 4: Extract a row/column across the edge to analyze the blur
row_index = gray.shape[0] // 2  # Example: middle row
profile = edges[row_index, :]  # Intensity profile across the row

# Step 5: Plot the profile to visualize the blur
plt.plot(profile)
plt.title("Intensity Profile Across Edge")
plt.xlabel("Pixel Position")
plt.ylabel("Edge Strength")
plt.show()

# Step 6: Estimate the blur size
# The blur size can be estimated as the width of the transition region
# (between high and low intensity) in the profile.

# Example method: Count the pixels between threshold crossings
threshold_high = np.max(profile) * 0.8
threshold_low = np.max(profile) * 0.2
high_indices = np.where(profile > threshold_high)[0]
low_indices = np.where(profile < threshold_low)[0]

# Blur size estimation
if len(high_indices) > 0 and len(low_indices) > 0:
    blur_size = np.abs(high_indices[0] - low_indices[-1])
else:
    blur_size = None

print(f"Estimated Blur Size: {blur_size} pixels")
