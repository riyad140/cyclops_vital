# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:29:11 2024

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle, Polygon, Circle

# from ROI_manager_user import userROI
import os
#import sys
import cv2
import math
from scipy import ndimage, signal
#from ROI_manager_user import userROI
import pandas as pd
import time
from tqdm import tqdm




pathLog=r"W:\raspberrypi\photos\FAT_Captures\Beta\Unit-3\motor_FAT_20240917\run01\mechanical_B03_pos_precision_20240917_run1.csv"
df = pd.read_csv(pathLog)



def stairizatoin(peaks,bufferSize = 10):
    
    
    stairs = []
    
    stairs.append([0,peaks[0]-bufferSize])
    for i in range(len(peaks)-1):
        startIndex = peaks[i] + bufferSize
        stopIndex = peaks[i+1] - bufferSize    
        stairs.append([startIndex,stopIndex])
        
    return stairs
#%%
plt.figure()
plt.plot(df['Time'],df['Position Measure'])

plt.figure()
plt.plot(df['Time'][1:],np.diff(df['Position Measure']))


#%%
kernel_size = 1  # Size of the kernel for median filtering

pos = df['Position Measure']
posClean = ndimage.median_filter(pos, size=kernel_size)

diffSignal = np.diff(posClean)

# plt.figure()
# plt.plot(pos)
# plt.plot(posClean)


plt.figure()
plt.plot(diffSignal)

#%

#%% Sector identification

sectorBuffer = 50
peaks, _ = signal.find_peaks(diffSignal,prominence=1,distance=50)

plt.figure()
plt.plot(posClean)
plt.plot(peaks,posClean[peaks],'X')


sectors = []

sectors.append([0,peaks[0]-sectorBuffer])
for i in range(0,2):
    startIndex = peaks[i] + sectorBuffer
    stopIndex = peaks[i+1] - sectorBuffer    
    sectors.append([startIndex,stopIndex])
    
#%%
localPeakList =[]
localSignals=[]

stairCases=[]

for sector in sectors:
    
    localSignal = posClean[sector[0]:sector[1]]
    
    localPeaks,_ = signal.find_peaks(np.diff(localSignal),prominence=0.1,distance=50)
    
    localSignals.append(localSignal)
    localPeakList.append(localPeaks)
    stairCases.append(stairizatoin(localPeaks))
    

globalMeans=[]
globalStds=[]    
 
for n,localSignal in enumerate(localSignals):
    means=[]
    stds=[]  
    plt.figure()
    plt.plot(localSignal)
    for stair in stairCases[n]:
        plt.plot(stair, localSignal[stair],'.',color='r')
        sliceMean = np.mean(localSignal[stair[0]:stair[1]])
        sliceStd = np.std(localSignal[stair[0]:stair[1]])
        means.append(sliceMean)
        stds.append(sliceStd)
    globalMeans.append(means)
    globalStds.append(stds)
    

