# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:58:43 2025

@author: imrul
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        
        

file_path = r"W:\raspberrypi\photos\Beta\B009\2025-01-24\s055\wbc-results-20250127_113210-3part_flagging_nogating\s055-df_features.pickle"

df = read_pickle_file(file_path)


raw_histogram, bin_edges = np.histogram(df[df['include_event_differential']]['mean_intensity_red'],100)
# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.figure()
plt.plot(bin_centers,raw_histogram)