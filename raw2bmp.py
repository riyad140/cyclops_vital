# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:00:30 2021

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters
import os
import tqdm
#%%
def raw2bmp(rawPath,imageSize = (3120, 4200)):   #imageSize = (3118, 4208)
    
    dstPath=os.path.join(rawPath,'bmps')
    if os.path.isdir(dstPath)==False:
        os.mkdir(dstPath)
    
    for file in (os.listdir(rawPath)):
        if file.endswith('raw'):
            print(file)
            npimg = np.fromfile(os.path.join(rawPath,file), dtype=np.uint16)
            imRaw = npimg.reshape(imageSize)
            imBmp=(imRaw//4).astype(np.uint8)
            filename=file[:-3]+'.bmp'
            cv2.imwrite(os.path.join(dstPath,filename),imBmp)

rawPath=r'\\files.vital.company\cyclops\raspberrypi\photos\Erics Sandbox\20220309_ARTesting\Red Channel Images\subset'
raw2bmp(rawPath)            
        