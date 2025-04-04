# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 20:01:11 2024

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt

# Example of truncated data (you can replace this with your actual data)
np.random.seed(0)
x_full = np.random.normal(5, 2, 1000)  # Original Gaussian distribution
x_truncated = x_full[x_full > 5]  # Truncated data, keeping only values greater than 5

# Plot histogram of the truncated data
plt.figure()
plt.hist(x_truncated, bins=30, density=True, alpha=0.6, color='gray')
plt.xlabel('X-axis')
plt.ylabel('Density')
plt.title('Histogram of Truncated Data')
# plt.show()

from scipy.optimize import curve_fit

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Generate a histogram and use the bin centers as x values
counts, bin_edges = np.histogram(x_truncated, bins=30, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Initial guesses for the parameters: amplitude, mean, and standard deviation
initial_guess = [max(counts), np.mean(x_truncated), np.std(x_truncated)]

# Fit the Gaussian curve
popt, _ = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
a_fit, mu_fit, sigma_fit = popt
print(f"Fitted parameters: a = {a_fit}, mu = {mu_fit}, sigma = {sigma_fit}")


# Plot the histogram and the fitted Gaussian curve
x_fit = np.linspace(min(x_truncated), max(x_truncated), 1000)
y_fit = gaussian(x_fit, *popt)


plt.figure()
plt.hist(x_truncated, bins=30, density=True, alpha=0.6, color='gray', label='Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Gaussian')
plt.xlabel('X-axis')
plt.ylabel('Density')
plt.title('Fitted Gaussian on Truncated Data')
plt.legend()
# plt.show()
