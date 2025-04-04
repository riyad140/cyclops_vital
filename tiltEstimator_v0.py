# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:04:12 2023

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
    imLeft = im[nr*1//3:nr*2//3,0:nc*1//3]
    imRight = im[nr*1//3:nr*2//3,nc*2//3:-1]
    imTop = im[0:nr*1//3,nc*1//3:nc*2//3]
    imBottom = im[nr*2//3:-1,nc*1//3:nc*2//3]
    
    return imLeft,imRight,imTop,imBottom

def focus_offset_parser(files):
    key1='offset_'
    key2='.tiff'
    focusOffsets=[]
    for file in files:
        idx1,idx2=file.find(key1),file.find(key2)
        focusOffset = np.int32(file[idx1+len(key1):idx2])
        focusOffsets.append(focusOffset)
    return focusOffsets
        
        

#%%
tiffPath = r"Y:\Cyclops Backups\FAT_Captures\BetaFATv2\AS01\2024-06-21\run00"
keyTiff = "FAT_fstack"

ims,files = read_images(tiffPath, keyTiff=keyTiff)



#%%
imL,imR,imT,imB = crop_image(ims[0])
#%%
plt.figure()
plt.imshow(imL)

#%%
focusOffsets = focus_offset_parser(files)

#%%

leftSharpness=[]
rightSharpness=[]
midSharpness=[]
topSharpness=[]
bottomSharpness=[]

for n,im in enumerate(ims):
    print(n)
    imL,imR,imT,imB = crop_image(im)
    leftSharpness.append(fm_helm(imL))
    rightSharpness.append(fm_helm(imR))
    topSharpness.append(fm_helm(imT))
    bottomSharpness.append(fm_helm(imB))
    
#%%
resultPath = create_folder(tiffPath, prefix = keyTiff)



plt.figure()
plt.plot(focusOffsets,leftSharpness,'bo',label='left')
plt.plot(focusOffsets,rightSharpness,'ko',label='right')
plt.legend()
plt.title('Tilt in Radial direction')
plt.xlabel("Focus Offset [steps]")
plt.ylabel("Sharpness [a.u.]")

# plt.text(0, 1.06, "gg", bbox=dict(fill=False, edgecolor='red', linewidth=2))

figName=f'{keyTiff}_tilt_radial.png'
plt.savefig(os.path.join(resultPath,figName))

plt.figure()
plt.plot(focusOffsets,topSharpness,'bo',label='top')
plt.plot(focusOffsets,bottomSharpness,'ko',label='bottom')
plt.legend()
plt.title('Tilt in Tangential direction')
plt.xlabel("Focus Offset [steps]")
plt.ylabel("Sharpness [a.u.]")

figName=f'{keyTiff}_tilt_tangential.png'
plt.savefig(os.path.join(resultPath,figName))

#%%
import math

nr,nc = im.shape
pixelToDistanceRatio = 0.1375 # um per pixel # pixel to distance ratio

radialDistancePx = nc*2//3 # Left to right pixel distance from respective crop centers
tangentialDistancePx = nr*2//3 # top to bottom pixel distance from respective crop centers

xOffsetL2R = radialDistancePx*pixelToDistanceRatio
yOffsetT2B = tangentialDistancePx*pixelToDistanceRatio



stepToum = 0.8 # 1 step equals ~0.7 um

leftSharpestOffset = focusOffsets[np.argmax(leftSharpness)]
rightSharpestOffset = focusOffsets[np.argmax(rightSharpness)]

topSharpestOffset = focusOffsets[np.argmax(topSharpness)]
bottomSharpestOffset = focusOffsets[np.argmax(bottomSharpness)]


offsetDifferenceL2R = leftSharpestOffset - rightSharpestOffset 
offsetDifferenceT2B = topSharpestOffset - bottomSharpestOffset

zOffsetL2R = offsetDifferenceL2R*stepToum
zOffsetT2B = offsetDifferenceT2B*stepToum

tiltRadial = np.round(np.degrees(math.atan(zOffsetL2R/xOffsetL2R)),2)
tiltTangential = np.round(np.degrees(math.atan(zOffsetT2B/yOffsetT2B)),2)

print(f'Tilt in Radial Direction (Left to Right) {tiltRadial} degrees')
print(f'Tilt in Tangential Direction (Top to Bottom) {tiltTangential} degrees')



fStackDict = {'Pixel-to-distance': pixelToDistanceRatio,
              'step-size-um': stepToum,
              'tilt-radial': tiltRadial,
              'tilt-tangential' : tiltTangential   
    
    }

df = pd.DataFrame.from_dict(fStackDict,orient='index')

filename='tiltEstimation.xlsx'
excelFileName=os.path.join(resultPath,filename)

with pd.ExcelWriter(excelFileName) as writer:  
    df.to_excel(writer,sheet_name='tilt_estimation')

#%%





