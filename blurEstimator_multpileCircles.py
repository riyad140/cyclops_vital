# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:43:19 2024

@author: imrul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Step 1: Load the grayscale image
imagePath = r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PLT\2024-08-07\PS_beads_fiducials_74p15_0p25_run01\img_PLT_AF-Fine-fiducials_fov1_offset_0.tiff\af-1-378.png"
image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

# Step 2: Apply Canny edge detection to find edges
edges = cv2.Canny(image, 100, 200)

# Step 3: Detect all circular edges using HoughCircles
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))

    # Step 4: Loop through each detected circle
    for idx, circle in enumerate(circles[0, :]):
        center_x, center_y, radius = circle[0], circle[1], circle[2]

        # Extract a radial intensity profile from the center of the circle
        theta = np.linspace(0, 2 * np.pi, 360)
        radii = np.linspace(radius - 20, radius + 20, 100)  # Adjust range as needed
        profile = []

        for r in radii:
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)
            profile.append(np.mean(image[np.round(y).astype(int), np.round(x).astype(int)]))

        profile = np.array(profile)

        # Step 5: Plot the radial intensity profile for each circle
        plt.plot(radii, profile)
        plt.title(f"Radial Intensity Profile Across Circular Edge {idx+1}")
        plt.xlabel("Radius")
        plt.ylabel("Intensity")
        plt.show()

        # Step 6: Define a Gaussian function for fitting
        def gaussian(r, a, r0, sigma, offset):
            return a * np.exp(-(r - r0) ** 2 / (2 * sigma ** 2)) + offset

        # Step 7: Fit the Gaussian to the radial profile
        initial_guess = [np.max(profile) - np.min(profile), radius, 5, np.min(profile)]
        popt, _ = curve_fit(gaussian, radii, profile, p0=initial_guess)

        # Extract the fitted sigma value (related to blur size)
        sigma = popt[2]
        blur_size = sigma * 2.355  # Convert sigma to FWHM

        print(f"Estimated Blur Size for Circle {idx+1}: {blur_size} pixels")

        # Step 8: Plot the fitted Gaussian on top of the radial intensity profile
        plt.plot(radii, profile, label='Radial Intensity Profile')
        plt.plot(radii, gaussian(radii, *popt), label='Fitted Gaussian', linestyle='--')
        plt.legend()
        plt.show()
else:
    print("No circles found.")
