# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:09:45 2024

@author: imrul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_images(tiffPath,keyTiff='tiff',extension = 'tiff'): # to read a stack of images and return a numpy array

    binPath=tiffPath
    ims=[]
    files=[]
    for file in os.listdir(binPath):
        if file.find(keyTiff)>-1 and file.endswith(extension):
            print(file)
            im=plt.imread(os.path.join(binPath,file))
            ims.append(im) 
            files.append(file)
    
    return ims,files



# Function to compute the histogram with fixed bins
def compute_histogram(image, bins):
    # Flatten the image to 1D for grayscale images
    hist, _ = np.histogram(image.ravel(), bins=bins, range=(0, 1023))
    plt.figure(100)
    plt.plot(_[:-1],hist)
    return hist

# Function to compare histograms between images
def compare_histograms(image1, image2, bins=256):
    # Compute histograms for both images
    hist1 = compute_histogram(image1, bins)
    hist2 = compute_histogram(image2, bins)

    # Normalize histograms for better comparison
    # hist1 = hist1 / np.sum(hist1)
    # hist2 = hist2 / np.sum(hist2)

    # Compare histograms using correlation
    correlation = np.corrcoef(hist1, hist2)[0, 1]

    return hist1, hist2, correlation

# Function to compare a base image with multiple images
def compare_with_multiple(base_image, images, bins=16):
    results = []
    for idx, image in enumerate(images):
        _, hist, correlation = compare_histograms(base_image, image, bins)
        results.append((idx, correlation))

    return results

# Load example images (replace with your image paths)

#%%
tiffPath = r"W:\raspberrypi\photos\FAT_Captures\Beta\Unit-8\2024-12-18\ps_beads_benchmark_3m_tp_ctrl\subset"

ims,files = read_images(tiffPath)

#%%

base_image = ims[-1]  # Base image (monochrome)
images = ims[0:-1]


# Define number of bins for histogram
bins = 256

# Compare histograms of the base image with the 10 similar images
results = compare_with_multiple(base_image, images, bins)

# Print results
for idx, correlation in results:
    print(f"Image {idx + 1} Correlation: {correlation:.4f}")
