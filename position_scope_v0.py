# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:42:46 2024

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

# from scipy.signal import medfilt


def detect_stairs(signal, window_size=1):
    # Apply median filtering to reduce noise
    filtered_signal = signal.medfilt(signal, kernel_size=window_size)
    # Find positive transitions
    positive_transitions = np.where(np.diff(filtered_signal) > 1)[0]
    # Define start and end indices for each stair
    stairs = []
    for i in range(len(positive_transitions) - 1):
        start_index = positive_transitions[i]
        end_index = positive_transitions[i + 1]
        stairs.append((start_index, end_index))
    return stairs
#%%


pathLog=r"Z:\raspberrypi\photos\FAT_Captures\Alpha_sharp\Cyc7_AS_3\BLDC_Precision-Stepper\20240424\worm-gear_A#03_precision_20240424_run01.csv"
df = pd.read_csv(pathLog)

#%%


pos = np.array(df['Position Measure'])

diffSignal = np.diff(pos)

peaks, _ = signal.find_peaks(diffSignal,prominence=0.1,distance=25)


plt.figure()
plt.plot(pos)
plt.plot(peaks,pos[peaks],'o')


stairs = []

bufferSize = 3

# for peak in peaks:
#     startIndex = peak - bufferSize
#     stopIndex = peak + bufferSize
    
#     stairs.append([startIndex,stopIndex])
    
for i in range(len(peaks)-1):
    startIndex = peaks[i] + bufferSize
    stopIndex = peaks[i+1] - bufferSize    
    stairs.append([startIndex,stopIndex])


plt.figure()
plt.plot(pos)

means=[]
stds = []

for stair in stairs:
    plt.plot(stair, pos[stair],'o')
    sliceMean = np.mean(pos[stair[0]:stair[1]])
    sliceStd = np.std(pos[stair[0]:stair[1]])/sliceMean*100
    means.append(sliceMean)
    stds.append(sliceStd)
# positive_transitions = np.where(np.diff(signal) > 0.1)[0]


# refined_index = np.where(np.diff(positive_transitions)>50)[0]

# edges=[]

# for i in refined_index:
#     edges.append(positive_transitions[i])
    

# plt.figure()
# plt.plot(signal)
# plt.plot(edges,signal[edges],'o')




# stairs = detect_stairs(posAngular)

# plt.figure()
# plt.plot(df['Position Measure'])

# print(stairs)