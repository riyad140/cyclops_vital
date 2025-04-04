# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:59:55 2025

@author: imrul


"""

import sys
import numpy as np
import os

import matplotlib.pyplot as plt
from scipy import ndimage, signal
import pandas as pd
import cv2
from datetime import datetime
import time

def create_folder(tiffPath, timeStamp = False, prefix = "fstack_" ): # to create a folder to store analysis result
    # binPath=tiffPath #os.path.split(tiffPath)[0]
    # keyTiff=prefix #os.path.split(tiffPath)[-1][:-5]
    resultPath=os.path.join(tiffPath,f'analysis_final_{prefix}')
    if timeStamp is True:
        ts=str(int(np.round(time.time(),0)))
        resultPath=resultPath+'_'+ts
    try:
        os.mkdir(resultPath)
    except:
        print("Folder Already Exists")
        pass
    return resultPath

def fm_helm(image,WSIZE=21): # Algorithm to calculate sharpness of an image FP_sharpness
    u=cv2.blur(image,(WSIZE,WSIZE))
    r1=u/image
    r1[image==0]=1
    index = (u>image)
    fm=1/r1
    fm[index]=r1[index]
    fm=np.mean(fm)
    return(fm)


def read_images(tiffPath,keyTiff='tiff',extension = 'tiff'): # to read an image and return a numpy array

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

def crop_image(im):
    nr,nc = im.shape
    imLeft = im[:,0:nc//2]
    imRight = im[:,nc//2:-1]
    # imTop = im[0:nr*1//3,nc*1//3:nc*2//3]
    # imBottom = im[nr*2//3:-1,nc*1//3:nc*2//3]
    
    return imLeft,imRight


def fov_parser(files):
    key1='fov'
    key2='.tiff'
    fovs=[]
    for file in files:
        idx1,idx2=file.find(key1),file.find(key2)
        fov = np.int32(file[idx1+len(key1):idx2])
        fovs.append(fov)
    return fovs
#%%

tiffPath = r"W:\raspberrypi\photos\Vibration_Study\2025-03-06\run22"
keyTiff = "FLR"
ims,files = read_images(tiffPath, keyTiff=keyTiff)

#%%

fovs = fov_parser(files)

#%%
sharpness_left = []
sharpness_right = []
for n,im in enumerate(ims):
    print(n)
    imL, imR = crop_image(im)
    sharpness_left.append(fm_helm(imL))
    sharpness_right.append(fm_helm(imR))
#%%
# plt.figure()
# plt.plot(fovs,sharpness_left,'o')
# plt.xlabel('FOV count')
# plt.ylabel('Sharpness')
# plt.title(tiffPath)


# plt.figure()
# plt.plot(fovs,sharpness_left-np.max(sharpness_left),'o')
# plt.xlabel('FOV count')
# plt.ylabel('Normalized Sharpness')
# plt.ylim([-0.04,0])
# plt.title(os.path.join(tiffPath,keyTiff))

#%%
fovs = np.array(fovs)
sharpness_left = np.array(sharpness_left)
sorted_index = np.argsort(fovs)
sorted_fovs = fovs[sorted_index]
sorted_sharpness_left = sharpness_left[sorted_index]



#%%
analysis_folder_name = os.path.join(tiffPath,'sharpness_analysis')

if os.path.exists(analysis_folder_name) is False:
    os.mkdir(analysis_folder_name)

#%%
plt.figure()
plt.plot(sorted_fovs, sorted_sharpness_left-np.max(sorted_sharpness_left),'.-')
plt.xlabel('FOV count')
plt.ylabel('Normalized Sharpness')
plt.ylim([-0.04,0])
plt.title(os.path.join(tiffPath,keyTiff))
plt.savefig(os.path.join(analysis_folder_name,keyTiff +'_plot.png'))


data = np.vstack((sorted_fovs,sorted_sharpness_left)).T
df = pd.DataFrame(data, columns = ['FOV','Sharpness'])
df.to_csv(os.path.join(analysis_folder_name,keyTiff+'_sharpness.csv'))
