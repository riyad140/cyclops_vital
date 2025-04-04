# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:14:36 2022

@author: imrul
"""

import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt
#%%
binPath=r'Z:\raspberrypi\photos\Misc\2022-03-08_led_optical_power_variance_test\00_sample_Flowcell_Empty_Cyc4Metal'
key='R_FL'
fov_key='FOV_'
w,h=4056,3040

means=[]
stds=[]
fovs=[]

for file in os.listdir(binPath):
    if file.endswith('png') and file.find(key)!=-1:
        print(file)
        fov=int(file[file.find(fov_key)+len(fov_key):file.find('.png')])
        print(fov)
        im=cv2.imread(os.path.join(binPath,file),1)
        im_crop=np.copy(im[2*h//5:3*h//5,2*w//5:3*w//5,0])
        
        means.append(np.nanmean(im_crop))
        stds.append(np.nanmean(im_crop))
        fovs.append(fov)
        
#%%        
plt.figure()
plt.plot(fovs,means,'o')
