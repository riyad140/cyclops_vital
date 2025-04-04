# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:41:37 2024

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
from datetime import date

def read_image(tiffPath,extension = 'tiff'): # to read an image and return a numpy array
    keyTiff=os.path.split(tiffPath)[-1][:-5]
    binPath=os.path.split(tiffPath)[0]
    ims=[]
    for file in os.listdir(binPath):
        if file.find(keyTiff)>-1 and file.endswith(extension):
            print(file)
            im=plt.imread(os.path.join(binPath,file))
            ims.append(im)          
    
    return ims[0]


tiffPath = r"W:\raspberrypi\photos\FAT_Captures\Beta\Unit-1\2024-08-09\betaFATv2A_run00\img_FAT_readout_4_blank_BF_ledPWR_200.tiff"


im = read_image(tiffPath)

plt.figure()
plt.imshow(im)

#%%

invCrossSection = list(1023 -  np.mean(im,axis = 0))
pxToDistUm = 0.1375 # um per pixel
plt.figure()
plt.plot(invCrossSection)

paddingLength = 100


invCrossSectionPadded = np.array([invCrossSection[0]]*paddingLength + invCrossSection + [invCrossSection[-1]]*paddingLength)

plt.figure()
plt.plot(invCrossSectionPadded)


peaks, _ = signal.find_peaks(invCrossSectionPadded,prominence= 50,distance=600)


plt.figure()
plt.plot(invCrossSectionPadded)
plt.plot(peaks,invCrossSectionPadded[peaks],'ro')


peakVals = invCrossSectionPadded[peaks] - paddingLength

maxIdx = np.argwhere(peakVals == max(peakVals))[0][0]

maxPeak = peaks[maxIdx]

idealPeak = len(invCrossSection) // 2

readOutDeviationPx = idealPeak - maxPeak
readOutDeviationUm = readOutDeviationPx * pxToDistUm


if len(peaks) < 3 :
    print("not all three peaks are found")
# print(f'55 readout radius line is found at vertical line drawn at pixel position of {maxPeak}')

print(f'readout radius deviation {readOutDeviationPx} pixels or {readOutDeviationUm} um')

    
