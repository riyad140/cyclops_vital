# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:32:17 2021

@author: imrul
"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os
#%%

# binPath=r"Z:\raspberrypi\photos\Method_comp_Hb\2021-11-17\14-16-53_run00_sample_BC_low_depth_1_Hemoglobin\cyclops-G_BF-FOV_0.png"
# img=cv2.imread(binPath)
# plt.figure()
# plt.imshow(img)


#%%

filename_im='cyclops-G_BF-FOV_0.png'

binPath=r'Z:\raspberrypi\photos\Method_comp_Hb\2021-11-17'
key='depth_1'
meanIntArr=[]
stdIntArr=[]
filename_arr=[]

for folder in os.listdir(binPath):
    if folder.find(key) >=0:
        
        newFolder=os.path.join(binPath,folder)
        print(newFolder)
        img=cv2.imread(os.path.join(newFolder,filename_im))
        img_green=img[:,:,1]
        h,w=img_green.shape
        imCrop=img_green[h//3:2*h//3,w//3:2*w//3]# central ROI
        
        mean_int=np.nanmean(imCrop)
        std_int=np.nanstd(imCrop)
        
        meanIntArr.append(mean_int)
        stdIntArr.append(std_int)
        filename_arr.append(newFolder)
#%%        

plt.figure(11)
plt.errorbar(np.arange(len(meanIntArr)),meanIntArr,stdIntArr)        
        
        