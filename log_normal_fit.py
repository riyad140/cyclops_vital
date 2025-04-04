# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:56:57 2024

@author: imrul
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import minimize

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
        
        
def negative_log_likelihood(floc, data):
    shape, loc, scale = lognorm.fit(data, floc=floc)
    return -np.sum(np.log(lognorm.pdf(data, shape, loc, scale)))

#%%
pickle_files =[
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-12-12\s16_HHCT_MAR1174_d03_16MINS\plt_test-results-20241212_145225\df_features.pickle"
    
    r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-19\S033_PREC_RUN1\pl_bf-results-20241120_092817\df_features.pickle",
    r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-19\S033_PREC_RUN2\pl_bf-results-20241120_093631\df_features.pickle",
    r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-19\S033_PREC_RUN3\pl_bf-results-20241120_094419\df_features.pickle",
    
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S019_PREC_RUN1\pl_bf-results-20241120_114641\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S019_PREC_RUN2\pl_bf-results-20241120_125425\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S019_PREC_RUN3\pl_bf-results-20241120_134723\df_features.pickle"
    
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S018_PREC_RUN1\pl_bf-results-20241121_095351\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S018_PREC_RUN2\pl_bf-results-20241121_100107\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-20\S018_PREC_RUN3_RFL275\pl_bf-results-20241121_100823\df_features.pickle"
    
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-25\S046_3XACO_DABCO_AF_PC300_0.05%_RUN1\pl_bf-results-20241125_164923\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-25\S046_3XACO_DABCO_AF_PC300_0.05%_RUN2\pl_bf-results-20241125_164042\df_features.pickle",
    # r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2024-11-25\S046_3XACO_DABCO_AF_PC300_0.05%_RUN3\pl_bf-results-20241125_201810\df_features.pickle"
    ]

#%%
df = read_pickle_file(pickle_files[2])

filtered_df = df[df['include_event']==True]

bins = 100
data = filtered_df['area']
lower_threshold, upper_threshold = 6, 600  # Example thresholds
trimmed_data = data[(data > lower_threshold) & (data < upper_threshold)]
approx_mean = np.mean(trimmed_data)
approx_std = np.std(trimmed_data)

filtered_data = trimmed_data[(trimmed_data > approx_mean - 3 * approx_std) & (trimmed_data < approx_mean + 3 * approx_std)]

# np.random.seed(0)

data= filtered_data


# counts, bin_edges = np.histogram(data_master, bins=bins)  # Histogram

# # Bin centers
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


# data = counts    

# Example: Generate some data (you can replace this with your own data)
# data = np.random.lognormal(mean=0, sigma=1, size=1000)

# Fit the log-normal distribution to the data
# shape, loc, scale = lognorm.fit(data, floc = -200)  # Set floc=0 to fix the location to 0


floc_range = np.linspace(-500, 0, 100)  # Searching for floc between -500 and 0
nll_values = []

for floc_value in floc_range:
    nll = negative_log_likelihood(floc_value, data)
    nll_values.append(nll)

# Find the optimal floc value
optimal_floc = floc_range[np.argmin(nll_values)]

# Fit the log-normal distribution with the optimal floc
shape, loc, scale = lognorm.fit(data, floc=optimal_floc)

# Generate points for the fitted log-normal distribution
x = np.linspace(-200, 800, 1000)
pdf_fitted = lognorm.pdf(x, shape, loc, scale)


plt.figure()
# Plot the histogram of your data
plt.hist(data, bins=bins, density=True, alpha=0.6, color='g', label="Data")

# Plot the fitted log-normal distribution
plt.plot(x, pdf_fitted, 'r-', lw=2, label="Fitted log-normal")

plt.title("Fit Log-normal Distribution to Data")
plt.xlabel("Data values")
plt.ylabel("Density")
plt.legend()


# Print fitted parameters
print(f"Fitted parameters: shape={shape}, loc={loc}, scale={scale}")


# x_full = np.linspace(-500,600,1000)
# y_full = lognorm.pdf(x_full, shape, loc, scale)
