# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:03:08 2024

@author: imrul
"""

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


pathLog=r"W:\raspberrypi\photos\FAT_Captures\Alpha_sharp\Autofocus_flexure_tests\Microstep testing_20240905\AS03_focus_20240906_1036_microstep64_vel250_acc500_tightscrew.csv"
df = pd.read_csv(pathLog, names = ['a','b','c','d'])

# plt.figure()
# plt.plot(df['c'])
# # plt.ylim([1300,1800])
# plt.title(os.path.split(pathLog)[-1])
# plt.xlabel('Data points')
# plt.ylabel('Z height um')

#%%
pos = np.array(df['c'])[5000:50000]


kernel_size = 10  # Size of the kernel for median filtering


posClean = ndimage.median_filter(pos, size=kernel_size)

diffSignal = np.diff(posClean)

# plt.figure()
# plt.plot(pos)
# plt.plot(posClean)


# plt.figure()
# plt.plot(diffSignal)

#%


peaks, _ = signal.find_peaks(diffSignal,prominence=0.3,distance=100)

bufferSize = 10
stairs = []

for i in range(len(peaks)-1):
    startIndex = peaks[i] + bufferSize
    stopIndex = peaks[i+1] - bufferSize    
    stairs.append([startIndex,stopIndex])

# plt.figure()
# plt.plot(posClean)
# plt.plot(peaks,posClean[peaks],'X')

means=[]
stds = []

plt.figure()
plt.plot(pos)
for stair in stairs:
    plt.plot(stair, pos[stair],'.',color='r')
    sliceMean = np.mean(pos[stair[0]:stair[1]])
    sliceStd = np.std(pos[stair[0]:stair[1]])
    means.append(sliceMean)
    stds.append(sliceStd)
plt.xlabel('Data Points')
plt.ylabel('Height (um)')
plt.grid(True)
# plt.figure()
# plt.plot(diffSignal)
# plt.plot(peaks,diffSignal[peaks],'X')


# plt.figure()
# plt.title('Step Size Uniformity')
# plt.plot(np.diff(means),'.')
# plt.xlabel('Step Count')
# plt.ylabel('Step Size (um)')

# plt.figure()
# plt.title('Step Size Stability')
# plt.plot((stds),'.')
# plt.xlabel('Step Count')
# plt.ylabel('Step Size (um)')

avgStepSize = np.round(np.mean(np.diff(means)),2)

avgStepSizeVariation = np.round(np.std(np.diff(means))/avgStepSize*100,2)


fig,ax = plt.subplots(2,1,sharex=True)
ax[0].set_title('Average Step Size Uniformity')
ax[0].plot(np.diff(means),'.')
ax[0].set_xlabel('Stair Count')
ax[0].set_ylabel('Step Size (um)')
ax[0].set_ylim([0.5,3.0])
ax[0].text(0,2.5,f'average step size {avgStepSize} with variation of {avgStepSizeVariation}%')

ax[1].set_title('Stability at each step')
ax[1].plot(stds,'+')
ax[1].set_xlabel('Stair Count')
ax[1].set_ylabel('Step Size (um)')
ax[1].set_ylim([0,1])
ax[0].text(0,5,f'average step size {avgStepSize} with variation of {avgStepSizeVariation}')
fig.suptitle(os.path.split(pathLog)[-1])

pngName = (pathLog)[:-3]+'png'
plt.savefig(pngName,dpi=400)

#%%


