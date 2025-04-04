# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:08:42 2021

@author: imrul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import tqdm

#%%
binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Team Camera Files\20211027\NoDiscImages\subset'
imRawArr=[]
filenameArr=[]
expArr=[]
keys=['exp','_analog']



# filepaths=[]
# for file in os.listdir(src_path):
#     if file.endswith('raw'):
#         filepaths.append(os.path.join(src_path,file))
# filepaths.sort(key=os.path.getctime)  

#%%

for file in (os.listdir(binPath)):
    if file.endswith('raw'):  
        print(file)
        exposure_ms=file[file.find(keys[0])+len(keys[0]):file.find(keys[1])]
        npimg = np.fromfile(os.path.join(binPath,file), dtype=np.uint16)
        imageSize = (3118, 4208)
        imRaw = npimg.reshape(imageSize)
        imRawArr.append(imRaw)
        filenameArr.append(file)
        expArr.append(float(exposure_ms))
plt.figure()
plt.imshow(imRaw)

#%%

roi={'x0':100,'x1':150,'y0':100,'y1':150}
mArr=[]
sArr=[]


for i in range(len(imRawArr)):
    im=imRawArr[i]
    m=np.nanmean(im[roi['y0']:roi['y1'],roi['x0']:roi['x1']])
    s=np.nanstd(im[roi['y0']:roi['y1'],roi['x0']:roi['x1']])
    mArr.append(m)
    sArr.append(s)
    
expArr=np.array(expArr) 
mArr=np.array(mArr)  
sArr=np.array(sArr) 
#%%
inds=expArr.argsort()

mArr=mArr[inds]
sArr=sArr[inds]
expArr=expArr[inds]

slope,intercept=np.polyfit(expArr[:-5],mArr[:-5],1)
fit=slope*np.array(expArr)+intercept

plt.figure()
plt.errorbar(expArr, mArr, sArr,label='data',fmt='o-')
plt.plot(expArr,fit,label='fit')
plt.xlabel('exposure (ms)') 
plt.ylabel('Pixel Intensity (a.u.)')   
plt.grid(True)
plt.legend()