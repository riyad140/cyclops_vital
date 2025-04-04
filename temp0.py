# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import skimage.io
import numpy as np
import os
import sys
import cv2
from skimage.filters import laplace
import matplotlib.pyplot as plt

def fm_lape(im):
    fm=laplace(im)
    fm=np.mean(abs(fm))
    return(fm)

#%%
binPath=r'G:\Shared drives\Experimental Results\CBC\Data\20200731-MARSDEN-USAF1951'

fNames=[]
eS=[]

for filename in os.listdir(binPath):
    if filename.endswith('.png'):
        if filename.find('184')>-1 or filename.find('193')>-1:
            print(filename)
            im=cv2.imread(os.path.join(binPath,filename))
            im_crop=im[im.shape[0]*1//3:im.shape[0]*2//3,im.shape[1]*1//3:im.shape[1]*2//3]
            edge_strength=fm_lape(im_crop)
            fNames.append(filename)
            eS.append(edge_strength)
            
ind=np.argmax(eS)
print(f'Sharpest image plane is: {fNames[ind]} ')
#%%
imS=cv2.imread(os.path.join(binPath,fNames[ind]))            
plt.figure()
plt.imshow(imS)
            
#%%            
im=cv2.imread(r"G:\Shared drives\Experimental Results\CBC\Data\20200731-MARSDEN-USAF1951\20200731-184725-oneshot-image-19500.png")   
im_crop=im[im.shape[0]*1//3:im.shape[0]*2//3,im.shape[1]*1//3:im.shape[1]*2//3]     
edge_strength=fm_lape(im_crop)