# -*- coding: utf-8 -*-
"""
Created on Thu May 12 08:58:03 2022

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
def normalize(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def get_mid_point_location(xSection,interpolation_factor=100):
    y=normalize(xSection)-0.5
    x=np.linspace(0,len(y)-1,len(y))
    xp=np.linspace(x[0],x[-1],len(x)*interpolation_factor)    
    yp=np.interp(xp,x,y)
    zc=np.diff(np.sign(yp))
    zc_ind=np.where(zc!=0)[0]    
    mid_point_location=xp[zc_ind]
    return mid_point_location


def get_imRadius(coords,imSize=[1520,2024]):
#%    
    mid_coords=[imSize[1]//2,imSize[0]//2]
    diff=np.array(coords)-np.array(mid_coords)
    
    distance=np.sqrt(diff[0]**2+diff[1]**2)
    max_distance=np.sqrt(mid_coords[0]**2+mid_coords[1]**2)
    imRadius_pt=distance/max_distance*100
    
    imAngle=np.degrees(math.atan(-diff[1]/diff[0]))
    
    return imRadius_pt,imAngle,diff
    
#%%

binPath=r'Z:\raspberrypi\photos\FAT_Captures\run03_sample_FAT_v3_usaf_Cyc4Juno\stepping'
key='G_BF-FOV_'
channel=2 # BGR

ims=[]
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith('bmp'):
        print(file)
        im=cv2.imread(os.path.join(binPath,file),1)
        ims.append(im[:,:])
        

#%%
index=3

# plt.figure()
# plt.imshow(ims[index])

#%

#rotation angle in degree
im_rotated = ndimage.rotate(ims[index], -5.6,reshape=False)

plt.figure()
plt.imshow(im_rotated)


#%
HH = userROI(im_rotated, no_of_corners=1, sort=False)
window_size=50
coords=HH.coords[0]
rad,ang,diff=get_imRadius(coords)


roi={'x0':coords[0]-window_size//2,
     'x1':coords[0]+window_size//2,
     'y0':coords[1]-window_size//2,
     'y1':coords[1]+window_size//2     
     }
#%
imCrop=im_rotated[roi['y0']:roi['y1'],roi['x0']:roi['x1']]

xSection=np.nanmean(imCrop,axis=0)




plt.figure()
plt.plot(normalize(xSection[:,0]),'b')
plt.plot(normalize(xSection[:,1]),'g')
plt.plot(normalize(xSection[:,2]),'r')
plt.grid(True)



loc_g=get_mid_point_location(xSection[:,1]) # g
loc_r=get_mid_point_location(xSection[:,2]) # r
loc_b=get_mid_point_location(xSection[:,0]) # b

LCA_r=loc_g-loc_r
LCA_b=loc_g-loc_b

#print(f'imHeight [{rad},{ang}]')
print(f'imHeight [dx,dy]: {diff}')
print(f'LCA_R {LCA_r}\nLCA_B {LCA_b}')

#%


    
dxdy.append(diff)
LCA.append([LCA_r,LCA_b])    
#%%
dxdy_arr=np.array(dxdy)
LCA_arr=np.array(LCA)


plt.figure()
plt.plot(dxdy_arr[:,0],LCA_arr[:,1],'bo-')
plt.grid(True)
plt.xlabel('dx from image center [px]')
plt.ylabel('chromatic aberration (g-b) [px]')
    