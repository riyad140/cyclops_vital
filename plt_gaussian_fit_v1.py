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
import os

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
    try:
        max_peak_index = np.argmax(properties['peak_heights'])
        max_peak_position = x[peaks[max_peak_index]]
    except:
        print('Peak detection failed')
        max_peak_index = np.argmax(y_smooth)
        max_peak_position = x[max_peak_index]
    
    # Plot
    # plt.figure()
    # plt.plot(x, y, 'b.', alpha=0.6, label='Noisy Data')
    # plt.plot(x, y_smooth, 'r-', label='Smoothed Data')
    # plt.plot(peak_positions, y_smooth[peaks], 'go', label='Detected Peaks')
    # plt.plot(max_peak_position, y_smooth[peaks[max_peak_index]], 'rx', label='Maximum Peak')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.title('Gaussian Peak Detection via Smoothing')
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
    
    plt.figure()
    plt.hist(data, bins=bins, alpha=0.6, label='Histogram', color='blue', edgecolor='black')
    plt.plot(x_full, y_full, label=f'Gaussian Fit: A={A_fit:.2f}, sigma={sigma_fit:.2f}, mu={fixed_mean}', color='red')
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
    
    return diff_area,[x_fit,y_fit]



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
    plt.savefig(os.path.join(os.path.split(pickle_file)[0],'Area_compensation.png'))
    return diff_area,[area0,area1,area2],[x_fit,y_fit]
    
#%%

# pickle_files =[
    
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-08-29\1163_IM5.1_PBS_AS1\pl_bf-results-20240829_132852\df_features.pickle",
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-08-29\1164_IM5.1_PBS_AS1\pl_bf-results-20240829_133109\df_features.pickle",
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-03\S1165_IM5.1_PBS_AS1\pl_bf-results-20240903_160741\df_features.pickle",
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-03\S1166_IM5.1_PBS_AS1\pl_bf-results-20240903_161721\df_features.pickle",
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-04\S1167_IM5.1_PBS_AS1\pl_bf-results-20240905_093731\df_features.pickle",
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-04\S1168_IM5.1_PBS_AS1\pl_bf-results-20240905_094147\df_features.pickle",
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-04\S1169_IM5.1_PBS_AS1\pl_bf-results-20240905_094605\df_features.pickle",
            
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-05\S1170_IM5.1_PBS_AS1\pl_bf-results-20240905_162400\df_features.pickle",
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-05\S1171_IM5.1_PBS_AS1\pl_bf-results-20240905_163037\df_features.pickle",
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-06\S1172_IM5.1_PBS_AS1\pl_bf-results-20240909_094401\df_features.pickle",
#             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-06\S1173_IM5.1_PBS_AS1\pl_bf-results-20240909_095321\df_features.pickle",
            
#     ]


pickle_files = ['W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S08-02\\pl_bf-results-20250121_113637\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S09\\pl_bf-results-20250121_114121\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S10\\pl_bf-results-20250121_114608\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S11\\pl_bf-results-20250121_115045\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S12\\pl_bf-results-20250117_104246\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S13\\pl_bf-results-20250121_115909\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S13-2_S16R\\pl_bf-results-20250121_120327\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S14\\pl_bf-results-20250121_120748\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S14-2_S16R\\pl_bf-results-20250121_121222\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S15\\pl_bf-results-20250121_121644\\df_features.pickle',
 'W:\\raspberrypi\\photos\\Juravinski\\2025-01-16\\01-16-S16\\pl_bf-results-20250117_111250\\df_features.pickle']



result_folder = os.path.join(os.path.split(os.path.split(pickle_files[0])[0])[0],"Area_correction")
try:
    os.mkdir(result_folder)
    print(f"Dataframe save location {result_folder}")
except:
    print(f'folder exists: {result_folder}')

bins = 100

plt_counts = []

area_list = []

diff_areas = []

for pickle_file in pickle_files:
    
    base_folder = os.path.split(pickle_file)[0]
    

    
    for file in os.listdir(base_folder):
        if file.find('plt_stats.csv')>0 and file.find('.')>1:
            df_stat = pd.read_csv(os.path.join(base_folder,file))
            plt_count = df_stat['plt_count'].values[0]
            plt_counts.append(plt_count)
    
    
    

    diff_area, areas, z = plt_gaussian_fit_min_intensity(pickle_file,bins,intensity_cut_off = 32.47)
    diff_areas.append(diff_area)
    area_list.append(areas)
    
    
area_arr = np.array(area_list)



#%%
plt_recounts = []

for n,diff_area in enumerate(diff_areas):
    plt_recounts.append(plt_counts[n]*(100+diff_area)/100)
    
    
cv_before = np.std(plt_counts)/np.mean(plt_counts)*100
cv_after = np.std(plt_recounts)/np.mean(plt_recounts)*100

print('Count Before Correction')
print(plt_counts)
print('Count After Correction')
print(plt_recounts)
print(f'cv before {cv_before}')
print(f'cv after {cv_after}')

df_data = list(zip(pickle_files, area_arr[:,0], diff_areas, plt_counts, plt_recounts))

df = pd.DataFrame(df_data, columns = ['pickle file','Original Area','Correction_Area','Original_Count','Corrected_Count'])

df.to_csv(os.path.join(result_folder,'area_correction_stats.csv'))

    
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