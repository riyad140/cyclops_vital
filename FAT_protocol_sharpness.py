# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:59:05 2022

@author: Jeff
"""

import numpy as np
import skimage.filters as sf
import os
import cv2
import matplotlib.pyplot as plt

def fm_lape(im):
    im=sf.gaussian(im,(5,5))
    fm=sf.laplace(im)
    fm=np.mean(abs(fm))
    return(fm)

def measure_sharpness_ACMO(image,hist_bins=100,hist_range=[0,4096]):
    hist, edges=np.histogram(image,bins=hist_bins,range=hist_range,density=True)
    hist = np.log(hist+1)
    hist = hist / np.sum(hist)
    centers = (edges[1:]+edges[:-1])/2
    mean = np.sum(hist * centers)
    phist=np.abs(centers-mean)*hist
    return(np.sum(phist))



#%%

#binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2022-05-13_Vibration_test_rpm_sweep\run00_rpm_my_6000-8000_600ms_mg_1500-3500_continuous_motion_bloodsample_sample_beads_Cyc4Vesta'
binPath=r'\\files.vital.company\cyclops\raspberrypi\photos\Vibration test\2022-05-20_VCM_Hysteresis\run00_rpm_my_3700_Hys_sample_beads_Cyc4Vesta'
channel=1 # BGR
key='FL-FOV_'

ims=[]
sharpness_laplace=[]
sharpness_acmo=[]
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith('png'):
        print(file)
        im=cv2.imread(os.path.join(binPath,file),1)
        imCrop=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3]
        ims.append(imCrop)
        
#%
for imCrop in ims:
    sharpness_acmo.append(measure_sharpness_ACMO(imCrop[:,:,channel],hist_range=[0,255]))
    sharpness_laplace.append(fm_lape(imCrop[:,:,channel]))
        
fig,ax=plt.subplots(2,1)
ax[0].plot(sharpness_acmo,'o-',label='ACMO')
ax[0].set_title('ACMO_abs_value')
# ax[0].set_xlabel('Frame Count')
ax[0].set_ylabel('Sharpness [a.u.]')
ax[1].plot(sharpness_laplace,'*-',label='Laplace')
ax[1].set_title('Laplace_abs_value')
ax[1].set_xlabel('Frame Count')
ax[1].set_ylabel('Sharpness [a.u.]')
ax[0].grid()
ax[1].grid()
#%%
s0_diff=np.diff(sharpness_acmo)
s1_diff=np.diff(sharpness_laplace)
#%
sharpness_acmo_pt=[]
sharpness_laplace_pt=[]
for i,s in enumerate(s0_diff):
    sharpness_acmo_pt.append(s0_diff[i]/sharpness_acmo[i]*100)
    sharpness_laplace_pt.append(s1_diff[i]/sharpness_laplace[i]*100)

fig,ax=plt.subplots(2,1)
ax[0].plot(sharpness_acmo_pt,'o-',label='ACMO')
ax[0].set_title('ACMO_rel_value')
ax[0].set_ylabel('Change in Sharpness [%]')
ax[1].plot(sharpness_laplace_pt,'*-',label='Laplace')
ax[1].set_ylabel('Change in Sharpness [%]')
ax[1].set_title('Laplace_rel_value')
ax[1].set_xlabel('Frame Count')
ax[0].grid()
ax[1].grid()

#%%

# import time

# t0=time.time()
# s=fm_lape(imCrop[:,:,channel])
# t1=time.time()
# s=measure_sharpness_ACMO(imCrop[:,:,channel],hist_range=[0,255])
# t2=time.time()

# print(t1-t0)
# print(t2-t1)

#%%
nr=int(np.sqrt(len(ims)))
nc=int(np.sqrt(len(ims)))
fig,ax=plt.subplots(nr,nc,sharex=True,sharey=True)

for i in range(nr**2):#range(len(ims)-1):
    # print(i)
    r,c=i//nr,i%nc
    print(f'r:{r} c:{c}')
    ax[r,c].imshow(ims[i][:,:,channel],cmap='gray')
    ax[r,c].set_title(f'Frame_{i}')
