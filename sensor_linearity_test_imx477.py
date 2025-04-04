# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:13:41 2021

@author: imrul
"""

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
binPath=r'Z:\raspberrypi\photos\Misc\2021-12-09\run00-80percent_sample_s601-1to40diluted_cyc4Metal\TIFFs_8bit'
imRawArr=[]
filenameArr=[]
expArr=[]
keys=['exp_','_ms']



# filepaths=[]
# for file in os.listdir(src_path):
#     if file.endswith('raw'):
#         filepaths.append(os.path.join(src_path,file))
# filepaths.sort(key=os.path.getctime)  

#%%

for file in (os.listdir(binPath)):
    if file.endswith('tiff'):  
        print(file)
        exposure_ms=file[file.find(keys[0])+len(keys[0]):file.find(keys[1])]
        # npimg = np.fromfile(os.path.join(binPath,file), dtype=np.uint16)
        # imageSize = (3118, 4208)
        imRaw=cv2.imread(os.path.join(binPath,file),1)
        # imRaw = npimg.reshape(imageSize)
        imRawArr.append(imRaw[:,:,1])
        filenameArr.append(file)
        expArr.append(float(exposure_ms))
plt.figure()
plt.imshow(imRaw)

#%%

roi={'x0':2000,'x1':2100,'y0':1500,'y1':1600}
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

slope,intercept=np.polyfit(expArr[:-1],mArr[:-1],1)
fit=slope*np.array(expArr)+intercept

plt.figure()
plt.errorbar(expArr, mArr, sArr,label='data',fmt='o-')
plt.plot(expArr,fit,label='fit')
plt.xlabel('exposure (ms)') 
plt.ylabel('Pixel Intensity (a.u.)')   
plt.grid(True)
plt.legend()