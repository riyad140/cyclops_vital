# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:58:43 2025

@author: imrul
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
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
        
def calculate_fwhm(hist, peak_idx):
    peak_val = hist[peak_idx]
    half_max = peak_val / 2
    
    # Find where the signal crosses the half-maximum level
    try:
        left_idx = np.where(hist[:peak_idx] < half_max)[0][-1]
        right_idx = peak_idx + np.where(hist[peak_idx:] < half_max)[0][0]
    
        fwhm = right_idx - left_idx
    except:
        print('FWHM calculation failed')
        fwhm = np.nan
    return fwhm        

#%%

if __name__ == "__main__":
    

    pickle_folder_path = r"W:\raspberrypi\photos\PV_2025\B009\2025-03-10\s049_MC\wbc-results-20250310_172146-3part_flagging_nogating"
    
    
    key = ".pickle"
    
    
    analysis_folder_path = os.path.join(pickle_folder_path,'Dynamic_range_analysis')
    
    if os.path.isdir(analysis_folder_path) is False:
        os.mkdir(analysis_folder_path)
    
    for file in os.listdir(pickle_folder_path):
        if file.endswith(key):
            print(file)
            file_path = os.path.join(pickle_folder_path,file)
        
    
    df = read_pickle_file(file_path)
    
    
    raw_histogram, bin_edges = np.histogram(df[df['include_event_differential']]['mean_intensity_red'],100)
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # plt.figure()
    # plt.plot(bin_centers,raw_histogram)
    
    
    smoothed_histogram = gaussian_filter1d(raw_histogram, sigma=5)
    peaks, properties = find_peaks(smoothed_histogram, height=20, distance = 35 )
    
    
    
    
    
    
    if len(peaks) >= 2:
        peaks = np.array([peaks[0],peaks[-1]])
        peak_distance = np.round(abs(bin_centers[peaks[-1]] - bin_centers[peaks[0]]),1)  # Adjust for histogram scale if needed
        print(f"Distance between peaks: {peak_distance}")
    else:
        print("Not enough peaks detected.")
    fwhms = [calculate_fwhm(smoothed_histogram, idx) for idx in peaks]
    normalized_peak_distance = np.round(peak_distance/bin_centers[peaks[0]],2)
    peak_distances = [peak_distance, normalized_peak_distance] #  [Absolute distance,  Distance relative to Lymph population]
    
    
    results = pd.DataFrame({
        'Peak Position': bin_centers[peaks],
        'FWHM': np.array(fwhms)*np.mean(np.diff(bin_centers)), # taking bin centers spacing into account
        'Peak Distances': peak_distances
        }   
        )
    
    print(results)
    
    plt.figure()
    plt.plot(bin_centers,raw_histogram, label='Raw Histogram')
    plt.plot(bin_centers,smoothed_histogram, label='Smoothed Histogram')
    plt.scatter(bin_centers[peaks], smoothed_histogram[peaks], color='red', label='Peaks')
    plt.legend()
    plt.xlabel('Mean Cell Intensity [a.u.]')
    plt.ylabel('Count')
    plt.xlim([100,1000])
    for peak in peaks:
        plt.axvline(x=bin_centers[peak], color='green', linestyle='--', label=f'Peak at {bin_centers[peak]:.2f}')
    
    plt.title(os.path.split(pickle_folder_path)[0][-40:]+f"\n Dynamic Range (Abs, Rel) : ({peak_distance},{normalized_peak_distance}) ")
    
    plt.savefig(os.path.join(analysis_folder_path,'dynamic_range.png'))
    
    results.to_csv(os.path.join(analysis_folder_path,'dynamic_range_analysis.csv'))


