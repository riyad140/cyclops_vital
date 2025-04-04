# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:14:33 2022

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

from ROI_manager_user import userROI
import os
import sys
import cv2
import math
from scipy import ndimage, signal
from ROI_manager_user import userROI
#%%

# binPath=r'Z:\raspberrypi\photos\FAT_Captures\run03_sample_FAT_v3_usaf_Cyc4Juno\stepping'
binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\FAT_Captures\run03_sample_FAT_v3_usaf_Cyc4Juno\stepping'
key='R_BF-FOV_0'
channel=2 # BGR

ims=[]
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith('bmp'):
        print(file)
        im=cv2.imread(os.path.join(binPath,file),1)
        ims.append(im[:,:,channel])
        

#%%

plt.figure()
plt.imshow(ims[0])

#%%

#rotation angle in degree
im_rotated = ndimage.rotate(ims[0], -5,reshape=False)

plt.figure()
plt.imshow(im_rotated)

#%%
HH = userROI(im_rotated, no_of_corners=2, sort=False, rectangle=True)
#%
buffer=10
imCrop=im_rotated[HH.coords[0][1]:HH.coords[1][1],HH.coords[0][0]:HH.coords[1][0]]
imCrop=imCrop[buffer:-buffer,buffer:-buffer]

plt.figure()
plt.imshow(imCrop)

#%
xSection=np.nanmean(imCrop,axis=0)
plt.figure()
plt.plot(xSection)
#%
peaks, _ = signal.find_peaks(xSection,prominence=20,distance=5)
troughs,_=signal.find_peaks(-xSection,prominence=20,distance=5)
plt.figure()
plt.plot(xSection)
plt.plot(peaks,xSection[peaks],'ro')
plt.plot(troughs,xSection[troughs],'bo')
plt.grid(True)


levels=[np.mean(xSection[peaks]),np.mean(xSection[troughs])]
contrast=abs(np.diff(levels)/np.sum(levels)*100)
print(f'Contrast Ratio : {contrast} %')

#% PDR
spatial_freq=128 #lp/mm
interpolation_factor=8
bayer_factor=2
px_size=1.55e3 #nm

mid_intensity=np.mean(levels)

y=xSection-mid_intensity


xx=range(len(y))
y_int=np.interp(np.linspace(xx[0],xx[-1],len(xx)*interpolation_factor),xx,y)

# x_int=np.interp(range(len(x)*interpolation_factor),range(len(x)),x)


zc=np.diff(np.sign(y_int))

zc_ind_p=np.where(zc>0)[0]
zc_ind_n=np.where(zc<0)[0]

pos_edge_transition=abs(np.diff(np.where(zc>0)[0])[0])
neg_edge_transition=abs(np.diff(np.where(zc<0)[0])[0])


plt.figure()
plt.plot(y_int)
plt.plot(zc_ind_p,y_int[zc_ind_p],'ro')
plt.plot(zc_ind_n,y_int[zc_ind_n],'bo')
plt.grid(True)


lp_px=np.nanmean([pos_edge_transition,neg_edge_transition])*bayer_factor/interpolation_factor
lp_nm=1/spatial_freq/1e-6


pixel_to_distance_ratio=lp_nm/lp_px  #nm/px
magnification=px_size/pixel_to_distance_ratio
hfov=pixel_to_distance_ratio*ims[0].shape[1]*bayer_factor/1e3  #um
vfov=pixel_to_distance_ratio*ims[0].shape[0]*bayer_factor/1e3  #um



print(f'pixel to distance ratio : {pixel_to_distance_ratio} nm/px')
print(f'FOV: {hfov} um * {vfov} um')
print(f'Magnification : {magnification} x')


# plt.figure()
# #plt.plot(x)
# plt.plot(np.diff(np.sign(x)))

