# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:51:10 2024

@author: imrul
"""
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from functools import partial
from scipy.stats import norm
from scipy.special import erf

def read_pickle_file(file_path):
    """
    Reads a pickle file and returns the content.

    Parameters:
        file_path (str): Path to the pickle file.
    
    Returns:
        data: The data loaded from the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("Pickle file read successfully!")
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except pickle.UnpicklingError:
        print("Error: The file could not be unpickled. It may not be a valid pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        

def gaussian(x, A, sigma, mu):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))




def skewed_gaussian(x, A, sigma, mu, alpha):
    """
    Skewed Gaussian function.
    """
    gaussian = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)  # Standard Gaussian
    skew = 1 + erf(alpha * (x - mu) / (np.sqrt(2) * sigma))  # Skew term
    return gaussian * skew


def find_peak_histogram(x,y, window_length = 21):

    # x = bin_centers
    # y = counts
    
    # Smooth data using Savitzky-Golay filter
    y_smooth = savgol_filter(y, window_length, polyorder=3)
    
    # Detect peaks
    peaks, properties = find_peaks(y_smooth, height=10)  # Adjust height or prominence if needed
    
    
    peak_positions = x[peaks]
    
    max_peak_index = np.argmax(properties['peak_heights'])
    max_peak_position = x[peaks[max_peak_index]]
    
    # Plot
    plt.figure()
    plt.plot(x, y, 'b.', alpha=0.6, label='Noisy Data')
    plt.plot(x, y_smooth, 'r-', label='Smoothed Data')
    plt.plot(peak_positions, y_smooth[peaks], 'go', label='Detected Peaks')
    plt.plot(max_peak_position, y_smooth[peaks[max_peak_index]], 'rx', label='Maximum Peak')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Gaussian Peak Detection via Smoothing')
    # plt.show()
    
    print(f"Detected Peak Positions: {peak_positions} with maximum prominence {max_peak_position}")
    
    return np.round(max_peak_position,1)






def plt_gaussian_fit(pickle_file,bins =100, x_limit = [-200,600], x_min = -200):


    df = read_pickle_file(pickle_file)
    
    filtered_df = df[df['include_event']==True]
    
    
    data = filtered_df['area']
    lower_threshold, upper_threshold = 6, x_limit[-1]  # Example thresholds
    trimmed_data = data[(data > lower_threshold) & (data < upper_threshold)]
    approx_mean = np.mean(trimmed_data)
    approx_std = np.std(trimmed_data)
    
    filtered_data = trimmed_data[(trimmed_data > approx_mean - 3 * approx_std) & (trimmed_data < approx_mean + 3 * approx_std)]
    
    # np.random.seed(0)
    
    data= filtered_data
    
    
    counts, bin_edges = np.histogram(data, bins=bins)  # Histogram
    
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Gaussian function
    
    
    # Fix the mean (mu)
    fixed_mean = find_peak_histogram (bin_centers,counts)  # Replace with your desired fixed mean
    # gaussian_fixed_mean = partial(gaussian, mu=fixed_mean)
    # skewed_gaussian_fixed_mean = partial(skewed_gaussian, mu=fixed_mean)
    
    # Fit the Gaussian to the histogram data
    # initial_guess = [max(counts), 3, 0.5]  # Adjust alpha (e.g., 0.5 for right-skew)
    popt, pcov = curve_fit(
        skewed_gaussian,  # Function with fixed mu
        bin_centers,
        counts,
        p0=[max(counts),  100, fixed_mean, 5]  # Initial guesses for A, sigma, mu, alpha
        )
    
    # Extract parameters
    A_fit, sigma_fit, mu_fit, alpha_fit = popt
    
    # Generate fitted curve
    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 500)
    y_fit = skewed_gaussian(x_fit, A_fit, sigma_fit,mu_fit, alpha_fit)
    
    
    
    # x_full = np.linspace(x_limit[0],x_limit[-1],1000)
    x_full = np.linspace(x_limit[0],x_limit[-1],1000)
    y_full = skewed_gaussian(x_full, A_fit, sigma_fit,mu_fit, alpha_fit)
    
    plt.figure()
    plt.hist(data, bins=bins, alpha=0.6, label='Histogram', color='blue', edgecolor='black')
    plt.plot(x_full, y_full, label=f'Gaussian Fit: A={A_fit:.2f}, sigma={sigma_fit:.2f}, mu={mu_fit:.2f}, skew={alpha_fit:.2f}', color='red')
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.title('Gaussian Fit on Histogram Data')
    plt.legend()
    
    
    
    #%
    y = counts
    x = bin_centers
    
    # finding location where the amplitude is 1/4th of the peak amplitude of the gaussian distribution
    
    # amplitude = np.max(y_full)

    # # Step 2: Calculate the target value (25% of amplitude)
    # target_value = 0.33 * amplitude
    
    # # Step 3: Find the index where y is closest to the target value
    # differences = np.abs(y_full - target_value)
    
    # # Step 4: Find the two indices with the smallest differences
    # closest_indices = np.argsort(differences)[:2]  # Sort and take the first two
    
    # # Step 5: Sort the indices to preserve left-to-right order
    # closest_indices = np.sort(closest_indices)
    
    # print(f"Two closest indices where y ≈ 25% of the amplitude: {closest_indices}")
    
    # print(x_full[closest_indices[0]])
    # print(f"Closest index: {closest_index}")
    
    
    
    
    area0 = np.trapz(y, x)
    area1 = np.trapz(y_fit,x_fit)
    # area2 = np.trapz(y_full[closest_indices[0]:],x_full[closest_indices[0]:])
    area2 = np.trapz(y_full,x_full)
    
    fit_goodness = 100 - np.abs( (area1-area0)/area0*100) # determine how good the fit is based on area difference
    

    
    diff_area = (area2-area0)/area0*100
    
    print(f'Curve fit score: {fit_goodness} %')
    print(f'Area difference {diff_area}  %')
    
    return diff_area,[area0,area1,area2],[x_fit,y_fit]



def plt_gaussian_fit_min_intensity(pickle_file,bins =100, x_limit = [-300,600], intensity_cut_off = 33):


    df = read_pickle_file(pickle_file)
    
    filtered_df = df[df['include_event']==True]
    
    
    data = filtered_df['area']
    lower_threshold, upper_threshold = 6, x_limit[-1]  # Example thresholds
    trimmed_data = data[(data > lower_threshold) & (data < upper_threshold)]
    approx_mean = np.mean(trimmed_data)
    approx_std = np.std(trimmed_data)
    
    filtered_data = trimmed_data[(trimmed_data > approx_mean - 3 * approx_std) & (trimmed_data < approx_mean + 3 * approx_std)]
    
    # np.random.seed(0)
    
    data= filtered_data
    
    
    counts, bin_edges = np.histogram(data, bins=bins)  # Histogram
    
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Gaussian function
    
    
    # Fix the mean (mu)
    fixed_mean = find_peak_histogram (bin_centers,counts)  # Replace with your desired fixed mean
    gaussian_fixed_mean = partial(gaussian, mu=fixed_mean)
    
    # Fit the Gaussian to the histogram data
    popt, pcov = curve_fit(gaussian_fixed_mean, bin_centers, counts, p0=[max(counts), 3])
    
    # Extract parameters
    A_fit, sigma_fit = popt
    
    # Generate fitted curve
    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 500)
    y_fit = gaussian_fixed_mean(x_fit, A_fit, sigma_fit)
    
    
    
    # x_full = np.linspace(x_limit[0],x_limit[-1],1000)
    x_full = np.linspace(x_limit[0],x_limit[-1],1000)
    y_full = gaussian_fixed_mean(x_full, A_fit, sigma_fit)
    

    
    
    
    #%
    y = counts
    x = bin_centers
    
    # finding location where the amplitude is 1/4th of the peak amplitude of the gaussian distribution
    
    amplitude = np.max(y_full)

    # Step 2: Calculate the target value (25% of amplitude)
    target_value = intensity_cut_off/100 * amplitude
    
    # Step 3: Find the index where y is closest to the target value
    differences = np.abs(y_full - target_value)
    
    # Step 4: Find the two indices with the smallest differences
    closest_indices = np.argsort(differences)[:2]  # Sort and take the first two
    
    # Step 5: Sort the indices to preserve left-to-right order
    closest_indices = np.sort(closest_indices)
    
    print(f"Two closest indices where y ≈ {intensity_cut_off}% of the amplitude: {closest_indices}")
    
    print(x_full[closest_indices[0]])
    # print(f"Closest index: {closest_index}")
    
    
    
    
    area0 = np.trapz(y, x)
    area1 = np.trapz(y_fit,x_fit)
    area2 = np.trapz(y_full[closest_indices[0]:],x_full[closest_indices[0]:])
    # area2 = np.trapz(y_full,x_full)
    
    fit_goodness = 100 - np.abs( (area1-area0)/area0*100) # determine how good the fit is based on area difference
    

    
    diff_area = (area2-area0)/area0*100
    
    print(f'Curve fit score: {fit_goodness} %')
    print(f'Area difference {diff_area}  %')
    
    
    plt.figure()
    plt.hist(data, bins=bins, alpha=0.6, label='Histogram', color='blue', edgecolor='black')
    plt.plot(x_full[closest_indices[0]:], y_full[closest_indices[0]:], label=f'Gaussian Fit: A={A_fit:.2f}, sigma={sigma_fit:.2f}, mu={fixed_mean}', color='red')
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.title('Gaussian Fit on Histogram Data')
    plt.legend()
    
    return diff_area,[area0,area1,area2],[x_fit,y_fit]



    
#%%

pickle_files =[
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-12-12\s16_HHCT_MAR1174_d03_16MINS\plt_test-results-20241212_145225\df_features.pickle"
    
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-19\S033_PREC_RUN1\pl_bf-results-20241120_092817\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-19\S033_PREC_RUN2\pl_bf-results-20241120_093631\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-19\S033_PREC_RUN3\pl_bf-results-20241120_094419\df_features.pickle",
    
    r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S019_PREC_RUN1\pl_bf-results-20241120_114641\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S019_PREC_RUN2\pl_bf-results-20241120_125425\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S019_PREC_RUN3\pl_bf-results-20241120_134723\df_features.pickle"
    
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S018_PREC_RUN1\pl_bf-results-20241121_095351\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S018_PREC_RUN2\pl_bf-results-20241121_100107\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S018_PREC_RUN3_RFL275\pl_bf-results-20241121_100823\df_features.pickle"
    
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-25\S046_3XACO_DABCO_AF_PC300_0.05%_RUN1\pl_bf-results-20241125_164923\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-25\S046_3XACO_DABCO_AF_PC300_0.05%_RUN2\pl_bf-results-20241125_164042\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-25\S046_3XACO_DABCO_AF_PC300_0.05%_RUN3\pl_bf-results-20241125_201810\df_features.pickle"
    ]

bins = 100

area_list = []

for pickle_file in pickle_files:
    
    diff_area,areas,z = plt_gaussian_fit(pickle_file,bins =100, x_limit = [-200,600], x_min = -200)

    # diff_area, areas, z = plt_gaussian_fit_min_intensity(pickle_file,bins,intensity_cut_off = 10)
    area_list.append(areas)
    
    
area_arr = np.array(area_list)

cv_before = np.std(area_arr[:,0])/np.mean(area_arr[:,0])*100
cv_after = np.std(area_arr[:,2])/np.mean(area_arr[:,2])*100

print(area_arr)
print(f'cv before {cv_before}')
print(f'cv after {cv_after}')
#%%
# plt.figure()
# plt.hist(data, bins=1000, density=True, alpha=0.6, color='g')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title('Histogram of Data')


# lower_threshold, upper_threshold = 6, 600  # Example thresholds
# trimmed_data = data[(data > lower_threshold) & (data < upper_threshold)]

# plt.figure()
# plt.hist(trimmed_data,bins=1000, density=True, alpha=0.6, color='g')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title('Histogram of Data')


#%%

x_full = np.linspace(-500,500,1000)
y_full = gaussian(x_full, 71, 116 ,92)
y_full1 = skewed_gaussian(x_full, 71, 116 ,92,-5)

plt.figure()
plt.plot(x_full,y_full)
plt.plot(x_full,y_full1)



# approx_mean = np.mean(trimmed_data)
# approx_std = np.std(trimmed_data)

# filtered_data = trimmed_data[(trimmed_data > approx_mean - 3 * approx_std) & (trimmed_data < approx_mean + 3 * approx_std)]

# Plot filtered data
# plt.figure()
# plt.hist(filtered_data, bins=100, density=True, alpha=0.6, color='orange')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.title('Filtered Data')



#%%



#%%


# Example histogram data
# np.random.seed(0)

# data= filtered_data
# bins = 100

# counts, bin_edges = np.histogram(data, bins=bins)  # Histogram

# # Bin centers
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# # Gaussian function


# # Fix the mean (mu)
# fixed_mean = find_peak_histogram (bin_centers,counts)  # Replace with your desired fixed mean
# gaussian_fixed_mean = partial(gaussian, mu=fixed_mean)

# # Fit the Gaussian to the histogram data
# popt, pcov = curve_fit(gaussian_fixed_mean, bin_centers, counts, p0=[max(counts), 3])

# # Extract parameters
# A_fit, sigma_fit = popt

# # Generate fitted curve
# x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 500)
# y_fit = gaussian_fixed_mean(x_fit, A_fit, sigma_fit)

# Plot the histogram and the fit
# plt.figure()
# plt.hist(data, bins=bins, alpha=0.6, label='Histogram', color='blue', edgecolor='black')
# plt.plot(x_fit, y_fit, label=f'Gaussian Fit: A={A_fit:.2f}, sigma={sigma_fit:.2f}, mu={fixed_mean}', color='red')
# plt.xlabel('X')
# plt.ylabel('Frequency')
# plt.title('Gaussian Fit on Histogram Data')
# plt.legend()




#%% FInd peak to get the mean of the gaussian distribution

#%%
# x_full = np.linspace(-25,600,1000)
# y_full = gaussian_fixed_mean(x_full, A_fit, sigma_fit)

# plt.figure()
# plt.hist(data, bins=bins, alpha=0.6, label='Histogram', color='blue', edgecolor='black')
# plt.plot(x_full, y_full, label=f'Gaussian Fit: A={A_fit:.2f}, sigma={sigma_fit:.2f}, mu={fixed_mean}', color='red')
# plt.xlabel('X')
# plt.ylabel('Frequency')
# plt.title('Gaussian Fit on Histogram Data')
# plt.legend()



# #%%
# y = counts
# x = bin_centers

# area0 = np.trapz(y, x)
# area1 = np.trapz(y_fit,x_fit)
# area2 = np.trapz(y_full,x_full)

# diff_area = (area2-area0)/area0*100


# print(f'Area difference {diff_area}  %')


#%%
# plt.figure()
# for i in range(4):
#     plt.plot(mpv_list[i],bias_list[i],'o',label=sample_list[i])
# plt.xlabel('Measured MPV by HT [a.u.]')
# plt.ylabel('PLT count Bias against sysmex [%]')
# plt.grid(True)
# plt.legend()

# bias_corrected = np.array(bias_list)+np.array(bias_correction_list)
# #%%
# plt.figure()
# for i in range(4):
#     plt.plot(bias_list[i],'o' ,label = sample_list[i])
#     plt.plot(bias_corrected[i,:],'x',label = sample_list[i])
# #%%
# bias_corrected_list = list(bias_corrected)
# plt.figure()
# for i in range(4):

#     plt.plot([i]*3,bias_list[i],'ro',label=sample_list[i])
#     plt.plot([i]*3,bias_corrected_list[i],'gx',label=sample_list[i])