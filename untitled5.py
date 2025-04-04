# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:55:09 2025

@author: imrul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread  # Handle TIFF images

# Load TIFF image
image = imread(r"W:\raspberrypi\photos\Vibration_Study\2025-03-04\run20\img_FAT_beads_blank_BF_ledPWR_200_fov59.tiff")

# Convert to grayscale if necessary
if len(image.shape) == 3:  # Check if it's RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
if image.dtype != np.uint8:
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

plt.figure()
plt.imshow(image, cmap="gray")
plt.title("Loaded TIFF Image")
plt.show()

#%%
# Apply adaptive thresholding to handle varying intensities
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 2)

# Find contours (beads)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Visualize contours
contour_image = image #cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

plt.figure()
plt.imshow(contour_image)
plt.title("Detected Beads")
plt.show()


#%%
aspect_ratios = []
circularities = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = h / w  # Height-to-width ratio
    aspect_ratios.append(aspect_ratio)

    # Compute circularity
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    circularities.append(circularity)

# Display results
print(f"Mean Aspect Ratio: {np.mean(aspect_ratios):.2f}")
print(f"Mean Circularity: {np.mean(circularities):.2f}")

#%%
laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
print(f"Laplacian Variance (Sharpness): {laplacian_var:.2f}")

#%%
# Compute FFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift) + 1)

# Plot FFT spectrum
plt.figure(figsize=(6, 6))
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Fourier Transform of Image")
plt.colorbar()
plt.show()

