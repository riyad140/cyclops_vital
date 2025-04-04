# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:37:44 2023

@author: imrul
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import os


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.filters import laplace
import time

def fm_lape(im):
    fm=laplace(im)
    fm=np.mean(abs(fm))
    return(fm)

def fm_helm(image,WSIZE=21):
    
    im=cv2.blur(image,(5,5)) 
    u=cv2.blur(im,(WSIZE,WSIZE))
    r1=u/image
    r1[image==0]=1
    index = (u>image)
    fm=1/r1
    fm[index]=r1[index]
    fm=np.mean(fm)
    return(fm)

def evaluateSharpness(im):
    im1=np.copy(im)
    # im1=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3]
    # plt.figure()
    # plt.imshow(im_resized)
    fm=fm_lape(im1)
    fm_=fm_helm(im1)
    return fm,fm_


def getSharpness(focalPlane, zippedSharpness): # this shall be replaced by the sharpnessCalculator function
    try:
        idx=list(zippedSharpness[:,0]).index(focalPlane)
        return zippedSharpness[idx,1]
    except:
        print(f'WARNING: af-{focalPlane} not found. Discarding from calculation')
        return 0  # for non existing vcms, should return warning

    

def smartAF(midPlane, stepSize, maxIter =3):
    
    vcm_step=stepSize
    vcm0=midPlane
    print (f'mid plane {midPlane}')
    
    v_s=[] # to store vcm and sharpness value side by side
    nFrames = 0
    s=getSharpness(vcm0,zippedSharpness) # capture and calculate sharpness of vcm0
    nFrames = nFrames + 1
    v_s.append([vcm0,s])
    
    for i in range(0,maxIter):
        print(f'iteration no {i}')
        
        for v in [vcm0+vcm_step//2**i,vcm0-vcm_step//2**i]: 
            print(f'measuring vcm of {v}')
            s=getSharpness(v,zippedSharpness)  #capture and calculate sharpness of vcm0-vcm_step and vcm0+vcm_step
            nFrames = nFrames + 1
            v_s.append([v,s])
        
        af_arr=np.array(v_s)  # vcm,sharpness [vcm0, vcm0-vcm_step, vcm0+vcm_step, vcm0-vcm_step/2, vcm0 + vcm_step/2 ...]
        
        maxInd=np.argmax(af_arr[:,1])
        vcm0=int(af_arr[maxInd,0])
    
        print(f'sharpestVCM: {vcm0} calculated with {nFrames} number of frames')
    print('Focal planes considered for all the iterations')
    print(af_arr[:,0])

binPath=r'W:\raspberrypi\photos\Alpha_plus\CYC7_A10\2024-09-25\S1206_Si_3beads_zstack\img_PLT_AF-Fine_fov1_offset_0.tiff'
# binPath = r'Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\Cyc7_A11\20230727\Channel1 Tiffs\subset'
key='af-0-'
# key = 'run00'
extension = 'png'
# extension='tiff'
files=[]
focusPlanes=[]
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith(extension):
        print(file)
        files.append(os.path.join(binPath,file))
        focusPlanes.append(np.int32(file[file.find(key)+0+len(key):file.find(extension)-1]))
#         im=plt.imread(os.path.join(binPath,file))
#         ims.append(im)
#   reading the images after fov sorting
# files.sort(key=os.path.getctime)

sharpnessVals=[]
for file in files:
    im=plt.imread(file)
    sharpnessVals.append(evaluateSharpness(im))
    
sharpnessHelm=np.array(sharpnessVals)[:,1]

maxIdx = list(sharpnessHelm).index(max(sharpnessHelm))
minIdx = list(sharpnessHelm).index(min(sharpnessHelm))

sharpestPlane = focusPlanes[maxIdx]

print(f'sharpest focal plane {sharpestPlane}')

plt.figure()
im=plt.imread(files[maxIdx])
plt.imshow(im,cmap='gray')

plt.figure()
plt.plot(focusPlanes,sharpnessHelm,'o')

zippedSharpness = np.array(list(zip(focusPlanes,sharpnessHelm)))
    
#%%



#%%

smartAF(sharpestPlane+12,16,4)
        
#%%smart AF algo
# vcm_step=20
# vcm0=330
# maxIter=3

# v_s=[] # to store vcm and sharpness value side by side
# s=get_sharpness(vcm0)
# v_s.append([vcm0,s])

# for i in range(0,maxIter):
    
#     for v in [vcm0+vcm_step//2**i,vcm0-vcm_step//2**i]:
#         s=get_sharpness(v)
#         v_s.append([v,s])
    
#     af_arr=np.array(v_s)  # vcm,sharpness
    
#     maxInd=np.argmax(af_arr[:,1])
#     vcm0=int(af_arr[maxInd,0])
#     print(f'sharpestVCM: {vcm0}')