# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:55:23 2024

@author: imrul
"""

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def read_image(tiffPath,keyTiff='tiff',extension = 'tiff'): # to read an image and return a numpy array

    
    ims=[]
    # files=[]
    # for file in os.listdir(binPath):
    if tiffPath.find(keyTiff)>-1 and tiffPath.endswith(extension):
        print(tiffPath)
        im=plt.imread(tiffPath)
        image_8bit = (im / 1023.0 * 255).astype(np.uint8)
        ims.append(image_8bit) 
            # files.append(file)
    
    return ims[0]

def calculate_image_overlap(image1, image2):
    """
    Calculate the degree of overlap between two images of the same size.
    The result is a normalized score between 0 and 1.
    """
    # Convert to grayscale if images are in color
    if image1.ndim == 3:
        image1 = rgb2gray(image1)
    if image2.ndim == 3:
        image2 = rgb2gray(image2)

    # Normalize pixel values to range [0, 1] if not already
    image1 = image1 / np.max(image1)
    image2 = image2 / np.max(image2)

    # Compute pixel-wise absolute difference
    difference = np.abs(image1 - image2)

    # Calculate similarity score: 1 - mean absolute difference
    similarity = 1 - np.mean(difference)

    return similarity
import numpy as np
from scipy.signal import correlate2d
def estimate_shift_cross_correlation(image1, image2):
    """
    Estimate pixel shift between two images using cross-correlation.
    Assumes both images are of the same size.
    """
    # Convert to grayscale if needed
    if image1.ndim == 3:
        image1 = rgb2gray(image1)
    if image2.ndim == 3:
        image2 = rgb2gray(image2)

    # Compute the cross-correlation
    correlation = correlate2d(image1, image2, mode='full')

    # Find the peak of the cross-correlation
    max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift = (max_idx[0] - image1.shape[0], max_idx[1] - image1.shape[1])

    return shift

import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def estimate_shift_phase_correlation(image1, image2):
    """
    Estimate pixel shift between two images using phase correlation.
    Assumes both images are of the same size.
    """
    # Convert to grayscale if needed
    if image1.ndim == 3:
        image1 = rgb2gray(image1)
    if image2.ndim == 3:
        image2 = rgb2gray(image2)

    # Compute the Fourier transforms
    f1 = fft2(image1)
    f2 = fft2(image2)

    # Compute the cross-power spectrum
    cross_power_spectrum = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
    shift = np.abs(ifft2(cross_power_spectrum))

    # Find the peak of the cross-power spectrum
    max_idx = np.unravel_index(np.argmax(fftshift(shift)), shift.shape)
    center = np.array(shift.shape) // 2
    pixel_shift = max_idx - center

    return pixel_shift

# Example usage
# image1 = np.random.random((200, 300))  # Placeholder: Replace with imread('image1_path')
# image2 = np.roll(image1, shift=(5, 10), axis=(0, 1))  # Simulated shifted version of image1


from skimage.registration import phase_cross_correlation

def estimate_shift_mutual_information(image1, image2):
    """
    Estimate shift using mutual information or phase cross-correlation.
    """
    shift, error, diffphase = phase_cross_correlation(image1, image2)
    return shift

# Usage


# Example usage
# Replace these with real image data


tiffPaths = [r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-12-02\S009-16min-A\img_PLT_blank_BF_fov5_offset_0.tiff",
             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-12-02\S009-16min-A\img_PLT_blank_BF_fov4_offset_0.tiff"]


# image1 = np.random.random((200, 300))  # Placeholder: Replace with imread('image1_path')
# image2 = np.random.random((200, 300))  # Placeholder: Replace with imread('image2_path')

image1 = read_image(tiffPaths[0])
image2 = read_image(tiffPaths[1])

#%%

overlap_score = calculate_image_overlap(image1, image2)
print(f"Degree of overlap: {overlap_score:.2f}")
shift = estimate_shift_phase_correlation(image1, image2)
print(f"Estimated pixel shift: {shift}")
# shift = estimate_shift_cross_correlation(image1, image2)
# print(f"Estimated pixel shift: {shift}")

shift = estimate_shift_mutual_information(image1, image2)
print(f"Estimated shift using mutual information: {shift}")


#%%
from skimage.metrics import structural_similarity as ssim

def calculate_overlap_ssim(image1, image2):
    """
    Calculate the structural similarity (SSIM) between two images.
    Output: A value between 0 (no overlap) and 1 (perfect overlap).
    """
    if image1.ndim == 3:
        image1 = rgb2gray(image1)
    if image2.ndim == 3:
        image2 = rgb2gray(image2)

    # Compute SSIM
    score, _ = ssim(image1, image2, full=True)
    return score

# Example Usage
overlap_score = calculate_overlap_ssim(image1, image2)
print(f"Degree of overlap (SSIM): {overlap_score:.2f}")

#%%
import numpy as np
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.color import rgb2gray

def calculate_overlap_iou(image1, image2):
    """
    Calculate the degree of overlap between two images using IoU.
    Output: A value between 0 (no overlap) and 1 (perfect overlap).
    """
    # Convert to grayscale if needed
    if image1.ndim == 3:
        image1 = rgb2gray(image1)
    if image2.ndim == 3:
        image2 = rgb2gray(image2)

    # Binarize the images using Otsu's threshold
    thresh1 = threshold_otsu(image1)
    thresh2 = threshold_otsu(image2)

    binary1 = image1 > thresh1
    binary2 = image2 > thresh2

    # Compute Intersection and Union
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()

    # Avoid division by zero
    if union == 0:
        return 0.0

    iou = intersection / union
    return iou

# Example Usage
# image1 = np.random.random((100, 100))  # Placeholder: Replace with imread('image1_path')
# image2 = np.roll(image1, shift=(5, 5), axis=(0, 1))  # Simulate a shifted image

overlap_score = calculate_overlap_iou(image1, image2)
print(f"Degree of overlap (IoU): {overlap_score:.2f}")


#%%
from skimage.morphology import skeletonize
from skimage.metrics import adapted_rand_error

def calculate_dice_coefficient(image1, image2):
    """
    Calculate overlap using Dice Coefficient after skeletonization.
    """
    # Convert to binary
    image1 = image1 > np.mean(image1)
    image2 = image2 > np.mean(image2)

    # Skeletonize the images
    skeleton1 = skeletonize(image1)
    skeleton2 = skeletonize(image2)

    # Compute Dice Coefficient
    intersection = np.logical_and(skeleton1, skeleton2).sum()
    total_pixels = skeleton1.sum() + skeleton2.sum()

    if total_pixels == 0:
        return 0.0  # No meaningful data

    dice_coefficient = (2 * intersection) / total_pixels
    return dice_coefficient

# Example usage
overlap_score = calculate_dice_coefficient(image1, image2)
print(f"Degree of overlap (Dice Coefficient): {overlap_score:.2f}")


#%%



thresh1 = threshold_otsu(image1)
thresh2 = threshold_otsu(image2)

binary1 = image1 < thresh1
binary2 = image2 < thresh2


img1 = image1*binary1
img2 = image1*binary2

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(img1)
ax[1].imshow(img2)


overlap_score = calculate_dice_coefficient(img1, img2)
print(f"Degree of overlap (Dice Coefficient): {overlap_score:.2f}")

overlap_score = calculate_image_overlap(img1, img2)
print(f"Degree of overlap: {overlap_score:.2f}")

shift = estimate_shift_mutual_information(img1, img2)
print(f"Estimated shift using mutual information: {shift}")

#%%
import cv2
import matplotlib.pyplot as plt

def feature_matching(image1, image2):
    """
    Perform feature matching between two images using ORB and BFMatcher.
    """
    # Convert images to grayscale
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1

    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Compute match score (e.g., ratio of good matches to total keypoints)
    match_score = len(matches) / min(len(keypoints1), len(keypoints2)) if len(keypoints1) and len(keypoints2) else 0

    return matched_image, match_score

# Example usage
# Load two images
# image1 = cv2.imread("path_to_image1.jpg")
# image2 = cv2.imread("path_to_image2.jpg")

# Perform feature matching
matched_image, match_score = feature_matching(img1, img2)

# Display results
print(f"Match Score: {match_score:.2f}")
plt.figure(figsize=(10, 5))
plt.imshow(matched_image)
plt.axis('off')
plt.title("Feature Matches")
plt.show()
