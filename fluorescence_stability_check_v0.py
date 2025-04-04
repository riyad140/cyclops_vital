# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:53:00 2024

@author: imrul
"""
import logging
import sys
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

#%%

def read_images(tiffPath,keyTiff='tiff',extension = 'tiff'): # to read a stack of images and return a numpy array

    binPath=tiffPath
    ims=[]
    files=[]
    for file in os.listdir(binPath):
        if file.find(keyTiff)>-1 and file.endswith(extension):
            print(file)
            im=plt.imread(os.path.join(binPath,file))
            ims.append(im) 
            files.append(file)
    
    return ims,files



#%%
tiffPath = r"W:\raspberrypi\photos\Stability_FL\2024-09-05\Run11_LumilassExposureSeries_Set1"


keyTiff = 'img_STABILITYLumilassSet1_glassFL_green_FLG_fov1_illuminationCount'

ims,files = read_images(tiffPath,keyTiff)


#%%
key1='_'
key2='.tiff'

illuminationCount = []
meanIm = []
for n,file in enumerate(files):
    
    count = int(file[file.rfind(key1)+len(key1):file.find(key2)]) # parsing illumination count
    illuminationCount.append(count)
    meanIm.append(np.mean(ims[n]))
    
plt.figure()
plt.plot(illuminationCount,meanIm,'o')
plt.xlabel('Illumination Count')
plt.ylabel('Mean Intensity')
plt.title(keyTiff)
    
    
    
    
