# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:36:28 2024

@author: imrul
"""
#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def calculate_r2_from_data(x, y):
    """
    Calculate the coefficient of determination (R^2) between two data sets.

    Parameters:
        x (numpy.ndarray): The first data set (independent variable).
        y (numpy.ndarray): The second data set (dependent variable).

    Returns:
        float: The R^2 value, representing the proportion of variance in y explained by x.
    """
    # Ensure input is numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate covariance
    covariance = np.sum((x - x_mean) * (y - y_mean))

    # Calculate standard deviations
    std_x = np.sqrt(np.sum((x - x_mean) ** 2))
    std_y = np.sqrt(np.sum((y - y_mean) ** 2))

    # Calculate Pearson correlation coefficient
    correlation_coefficient = covariance / (std_x * std_y)

    # R^2 is the square of the correlation coefficient
    r2 = correlation_coefficient ** 2
    return r2

def normalize(data):
    
    
    data = np.array(data,dtype=float)
    data[data<np.nanmedian(data)/2] = np.nan
    avg =np.nanmedian(data)
    
    print(f'Median: {avg}')
    return data/avg


def average(data):
    
    
    data = np.array(data,dtype=float)
    data[data<np.nanmean(data)/2] = np.nan
    avg =np.nanmean(data)
    std = np.nanstd(data)
    
    print(f'Mean: {avg}')
    return avg,np.round(std/avg*100,1)

csvPath = r"C:\Users\imrul\Downloads\PLT_performance\Beta0006 - Sheet3.csv"
df = pd.read_csv(csvPath,index_col="Folder Name")

# Reset the index to include it as a column
data_reset = df

data_reset['Bias'] = np.round((data_reset['Plt count TB'] - data_reset['Plt count sysmex'])/data_reset['Plt count sysmex']*100,1)
# Transpose the DataFrame
transposed = data_reset.T


# plt.figure()
# plt.plot(normalize(transposed[transposed.columns[9]][-18:-2]),label = transposed[transposed.columns[9]]['Bias'])
# plt.plot(normalize(transposed[transposed.columns[10]][-18:-2]),label = transposed[transposed.columns[10]]['Bias'])
# plt.plot(normalize(transposed[transposed.columns[11]][-18:-2]),label = transposed[transposed.columns[11]]['Bias'])

# # plt.plot(normalize(transposed[transposed.columns[4]][0:16]),label = transposed[transposed.columns[4]]['% Bias'])
# # plt.plot(normalize(transposed[transposed.columns[5]][0:16]),label = transposed[transposed.columns[5]]['% Bias'])
# # plt.plot(normalize(transposed[transposed.columns[6]][0:16]),label = transposed[transposed.columns[6]]['% Bias'])
# plt.xlabel('FOV')
# plt.ylabel('Norm. Count')
# plt.legend()
# plt.ylim([0.5,1.5])


#%%

# index = 5
from matplotlib import cm

num_colors = 100
colors = plt.cm.hsv(np.linspace(0, 1, num_colors)) 

def normalize_and_fit_data(data, deg = 1, label = '', color = 'r'):

    # data =np.array(normalize(transposed[transposed.columns[index]][0:16]),dtype = float)
    data = np.array(normalize(data), dtype= float)
    fovs = np.arange(1,len(data)+1)
    
    nan_indices = np.isnan(data)
    
    data1 = data[~nan_indices]
    fovs1 = fovs[~nan_indices]
    
    coefficients = np.polyfit(fovs1,data1, deg = deg)
    fitted_curve = np.polyval(coefficients, fovs1)
    plt.figure(200)
    plt.plot(fovs1,data1,'--',label = label)
    # plt.plot(fovs1,fitted_curve, color = color)
    plt.xlabel('FOV')
    plt.ylabel('Normalized cell count')
    plt.legend()
    plt.ylim([0.5,1.5])
    plt.xlim([-1,17])    
    
    errors = fitted_curve - data1
    print('---------')
    # print(errors)
    
    print(coefficients)
    return coefficients,errors


bias_arr = []
mean_arr = []
std_arr = []
coeff_arr = []
fit_errors_arr = []

for i in range(0,15):
    print(i)
    data = transposed[transposed.columns[i]][-18:-2]
    mean,std = average(data)
    label = transposed[transposed.columns[i]]['Bias']
    bias_arr.append(label)
    mean_arr.append(mean)
    std_arr.append(std)
    
    stats,errors = normalize_and_fit_data(data, deg = 1, label = label, color = colors[i*7])
    fit_errors_arr.append(np.mean(np.abs(errors)))
    coeff_arr.append(stats)
    
    
    
#%%    




coeff_arr = np.array(coeff_arr)


plt.figure()
plt.plot(coeff_arr[:,1],coeff_arr[:,0],'o')    
plt.xlabel('Intercept')
plt.ylabel('slope')


plt.figure()
plt.plot(bias_arr,coeff_arr[:,0],'*')
plt.ylabel('Coeff_0')
plt.xlabel('Bias')


plt.figure()
plt.plot(bias_arr,coeff_arr[:,1],'*')
plt.ylabel('Coeff_1')
plt.xlabel('Bias')


plt.figure()
plt.plot(bias_arr,std_arr,'*')
plt.ylabel('per fov normalized count std%')
plt.xlabel('Bias')

plt.figure()
plt.plot(bias_arr,fit_errors_arr,'*')
plt.ylabel('mean abs error from fit')
plt.xlabel('Bias')


#%%
r2 = calculate_r2_from_data(bias_arr,std_arr)
print(f'R2 value from Pearson Coefficent {r2}')    

x = std_arr
y= bias_arr
coefficients = np.polyfit(x,y, deg = 1)
fitted_curve = np.polyval(coefficients, x)
# def fit_data(x,y,deg=1):
#     coefficients = np.polyfit(x,y, deg = deg)
#     # fitted_curve = np.polyval(coefficients, x)
    
#     return coefficients


plt.figure()
plt.plot(x,y,'x',label='raw data')
plt.plot(x,fitted_curve, label= 'fit')
plt.plot(x,y-fitted_curve,'o',label='recovery')
plt.ylabel('per fov normalized count std%')
plt.xlabel('Bias')
plt.title(f'R2: {r2}')
plt.legend()
plt.grid()
plt.ylim([-20,20])

