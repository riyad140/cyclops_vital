# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:02:41 2024

@author: imrul
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def read_image(tiffPath,extension = 'tiff'): # to read an image and return a numpy array
    keyTiff=os.path.split(tiffPath)[-1][:-5]
    binPath=os.path.split(tiffPath)[0]
    ims=[]
    for file in os.listdir(binPath):
        if file.find(keyTiff)>-1 and file.endswith(extension):
            print(file)
            im=plt.imread(os.path.join(binPath,file))
            ims.append(im)          
    
    return ims[0]




def adaptive_threshold(image, max_value=255, method='mean', block_size=201, C=2):
    """
    Apply adaptive thresholding to an image.

    Parameters:
    - image: Input grayscale image.
    - max_value: Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
    - method: Adaptive method to use ('mean' or 'gaussian').
    - block_size: Size of a pixel neighborhood used to calculate a threshold value for the pixel.
    - C: Constant subtracted from the mean or weighted mean.

    Returns:
    - thresholded_image: The thresholded image.
    """
    # Convert image to grayscale if it is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Select the adaptive thresholding method
    if method == 'mean':
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError("Method must be 'mean' or 'gaussian'")

    # Apply adaptive thresholding
    thresholded_image = cv2.adaptiveThreshold(
        gray_image,
        max_value,
        adaptive_method,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    return thresholded_image

#%%
# Read the image
tiffPath = r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A10\PV\2024-06-04\MC-S1022\img_PLT_blank_BF_fov1_offset_0.tiff"

im = read_image(tiffPath)

# Normalize the pixel values to the range [0, 255]
image_8bit = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)

# Convert the normalized image to 8-bit
image = np.uint8(image_8bit)
# image = cv2.blur(image, (3,3))

# imTh = adaptive_threshold(image)


# plt.figure()
# plt.imshow(imTh, cmap= 'gray')
#%%

# Convert to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image to reduce noise
# blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply automatic threshold using Otsu's method

# th = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)



imHist = np.histogram(image,bins= 100)

diffHist = abs(np.diff(imHist[0]))

histTh = 10000

plt.figure()
plt.plot(diffHist)

valIndex = np.argwhere(diffHist>histTh)[0][0]

pixelTh = imHist[1][valIndex]


_, thresholded_image = cv2.threshold(image, pixelTh, 255, cv2.THRESH_BINARY_INV)


plt.figure()
plt.imshow(thresholded_image)


#%%

# Find contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on size
min_area = 0  # Minimum area of the particle (10 pixels wide)
max_area = 30**2  # Maximum area of the particle (30 pixels wide)

filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        filtered_contours.append(contour)

# Draw the filtered contours on the original image
result_image = image.copy()
final_image= cv2.drawContours(result_image, filtered_contours, -1, (0, 0, 255), 2)

# Display the result
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(final_image,cmap = 'gray')
# plt.subplot(1,2,2)
# plt.imshow(image,cmap = 'gray')



fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(image,cmap = 'gray')
ax[1].imshow(final_image,cmap='gray')