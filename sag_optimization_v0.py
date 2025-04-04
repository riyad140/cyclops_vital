# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:34:44 2024

@author: imrul
"""

import sys
import numpy as np
import os

import matplotlib.pyplot as plt
from scipy import ndimage, signal
import pandas as pd
import cv2


def read_images(tiffPath,keyTiff='tiff',extension = 'tiff'): # to read an image and return a numpy array

    
    ims=[]
    # files=[]
    # for file in os.listdir(binPath):
    if tiffPath.find(keyTiff)>-1 and tiffPath.endswith(extension):
        print(tiffPath)
        im=plt.imread(tiffPath)
        ims.append(im) 
            # files.append(file)
    
    return ims[0]


def read_images(tiffPath,keyTiff='tiff',extension = 'tiff'): # to read a stack of images and return a numpy array

    binPath=tiffPath
    ims=[]
    files=[]
    for file in os.listdir(binPath):
        if file.find(keyTiff)>-1 and file.endswith(extension):
            # logging.info(file)
            print(file)
            im=plt.imread(os.path.join(binPath,file))
            ims.append(im) 
            files.append(file)
    
    return ims,files


#%%

# dataset with low and high wbc W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-10-30\

tiffPath = r'W:\raspberrypi\photos\Beta\B006\2024-12-02\S009_PREC_RUN2'
keyTiff = 'img_WBC_SAG_redFLR_fov0.tiff'


ims,files = read_images(tiffPath,keyTiff)



#%%


def get_percentile_values(percent = 50):

    # percent = 63
    gains = [2,4,6]
    values = []
    for im in ims:
        values.append(np.percentile(im,percent))
    
    
    print(values)
    
    m,c = np.polyfit(gains,values,deg=1)
    m=np.round(m,1)
    c=np.round(c,1)
    print(m,c)
    
    return values, [m,c]


percentileArr = np.linspace(1,100,20)

values2d = []
coeffs2d = []

for percent in percentileArr:
    values, coeffs = get_percentile_values (percent)
    values2d.append(values)
    coeffs2d.append(coeffs)



valueArr = np.array(values2d)
coeffsArr = np.array(coeffs2d)
plt.figure()
plt.plot(percentileArr,valueArr)
plt.title(os.path.split(tiffPath)[-1])
plt.xlabel('Percentile')
plt.ylabel('Intensity')
plt.ylim([50,600])

plt.figure()
plt.plot(percentileArr,coeffsArr[:,1])
plt.title(os.path.split(tiffPath)[-1]+'_slope')
plt.xlabel('Percentile')
plt.ylabel('Sag slope')
plt.ylim([0,120])

# plt.figure(50)
# plt.plot(gains,values,'o-',label = f'{percent}:{m}x+{c}')
# plt.legend()
    

#%%

# im0=np.copy(ims[2])

# im0[im0<np.median(im0)] = 0

# im0_nz = im0[im0 !=0]


# vals = np.nanmean(im0_nz)
# print(vals)

# plt.figure()
# plt.imshow(im0,cmap='gray')


# from scipy import stats

# mode_values =stats.mode(ims[2].flatten())[0][0]
# print(mode_values)