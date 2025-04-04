# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:02:01 2021

@author: imrul
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:23:11 2021

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters
import os
from skimage.feature import register_translation
#%%
# def raw2bmp(rawPath,imageSize = (3118, 4208)):
    
#     dstPath=os.path.join(rawPath,'bmps')
#     if os.path.isdir(dstPath)==False:
#         os.mkdir(dstPath)
    
#     for file in os.listdir(rawPath):
#         if file.endswith('raw'):
#             print(file)
#             npimg = np.fromfile(binPath, dtype=np.uint16)
#             imRaw = npimg.reshape(imageSize)
#             imBmp=(imRaw//4).astype(np.uint8)
#             filename=file[:-3]+'.bmp'
#             cv2.imwrite(os.path.join(dstPath,filename),imBmp)

# rawPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\20210916_TargetImages_IMX258\subset'
# raw2bmp(rawPath)            
        
    
    
        
    
    

#%%
binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\20210927_IMX258_IRF\run2\20210927_IMX258_RotationTest_Round2'
angle_arr=[]
p0=[]
p1=[]
pdaf_blocks=[]
for file in os.listdir(binPath):
    print(file)
    npimg = np.fromfile(os.path.join(binPath,file), dtype=np.uint16)
    imageSize = (3118, 4208)
    imRaw = npimg.reshape(imageSize)
    imCrop=imRaw[24:-22,24:-24]
    imCrop=cv2.medianBlur((imCrop//4).astype(np.uint8), ksize=3)
    
    key0='ledon_'
    key1='_calibration'
    file_lower=file.lower()
    
    angle=int(file[file_lower.find(key0)+len(key0):file_lower.find(key1)])
    print(f'angle: {angle}')
    
    maskPDAF_L=np.zeros(imCrop.shape)
    maskPDAF_R=np.zeros(imCrop.shape)
    
    left_pixel_coords=[(5,2),(5,18),(24,9),(24,25)]
    right_pixel_coords=[(8,1),(8,17),(21,10),(21,26)]
    
    # left_pixel_coords=[(2,5),(18,5),(9,24),(25,24)]
    # right_pixel_coords=[(1,8),(17,8),(10,21),(26,21)]
    
    
    # im_binned=[]
    # pdaf_grads=[]
    left_pixel_values=[]
    right_pixel_values=[]

    for row in [40*32]:#np.arange(0,imCrop0.shape[0],32):
        for col in [60*32]:#np.arange(0,imCrop0.shape[1],32): 
            pdaf_arr=imCrop[row:row+32,col:col+32]
            lpv=[]
            rpv=[]
            for i in range(4):
                ref=np.array([row,col])
                lpos=np.array(left_pixel_coords[i])
                rpos=np.array(right_pixel_coords[i])
                temp_index=tuple(ref+lpos)
                maskPDAF_L[temp_index]=1
                temp_index_=tuple(ref+rpos)
                maskPDAF_R[temp_index_]=1
                lpv.append(imCrop[temp_index])
                rpv.append(imCrop[temp_index_])
            left_pixel_values.append(lpv)
            right_pixel_values.append(rpv)
    pdaf_blocks.append(pdaf_arr)
    angle_arr.append(angle)
    
    p0.append(np.mean(left_pixel_values[0]))
    p1.append(np.mean(right_pixel_values[0]))        
    # fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
    # ax[0].imshow(maskPDAF_L,interpolation='none')      
    # ax[1].imshow(maskPDAF_R,interpolation='none')        
    # plt.figure(202)
    # plt.plot(angle_arr,p0,'k-')
    # plt.plot(angle_arr,p1,'g-')
    # plt.figure(202)
    # plt.plot(angle,np.mean(left_pixel_values[1]),'o-',color='k')
    # plt.plot(angle,np.mean(right_pixel_values[1]),'+-',color='g')
#%
irf=(np.array(p0)-np.array(p1))/(np.array(p0)+np.array(p1))
plt.figure(200)
plt.plot(angle_arr,irf)

#%%
pdaf_blocks=np.array(pdaf_blocks)

plt.figure()
plt.plot(angle_arr,pdaf_blocks[:,1,6],'o--')

#%%
    

# binPath=r"Z:\raspberrypi\photos\20210927_IMX258_RotationImages\1ms Exp\20210927_RotationImages_IMX258_LEDON_plus0_Calibration_Plane0_exp1_analogGain0.raw"
# npimg = np.fromfile(binPath, dtype=np.uint16)
# imageSize = (3118, 4208)
# imRaw = npimg.reshape(imageSize)
# imCrop=imRaw[24:-22,24:-24]

# key0='plus'
# key1='_Calibration'

# angle=int(binPath[binPath.find(key0)+len(key0):binPath.find(key1)])


# #%% applying the maskGrad
# imCrop0=imCrop
# # imGrad0=imGrad



# # #%%
# maskPDAF_L=np.zeros(imCrop0.shape)
# maskPDAF_R=np.zeros(imCrop0.shape)

# # for row in np.arange(imCrop0.shape[0]):
# #     for col in np.arange(imCrop0.shape[1]):
        


# #%% First PDAF block
# # imCrop=imRaw[24:-22,24:-24]
# # imGrad=edges_x

# left_pixel_coords=[(5,2),(5,18),(24,9),(24,25)]
# right_pixel_coords=[(8,1),(8,17),(21,10),(21,26)]

# pdaf_blocks=[]
# im_binned=[]
# pdaf_grads=[]
# left_pixel_values=[]
# right_pixel_values=[]
# for row in [0]:#np.arange(0,imCrop0.shape[0],32):
#     for col in [0]:#np.arange(0,imCrop0.shape[1],32):     
        
#         # print(f'(r,c)={row,col}')
#         # pdaf_arr=imCrop0[row:row+32,col:col+32]
#         # pdaf_grad=imGrad0[row:row+32,col:col+32]
#         # pdaf_blocks.append(pdaf_arr)
#         # # pdaf_grad[pdaf_grad==0]=np.nan
#         # pdaf_grads.append(pdaf_grad)
#         # im_binned.append(np.nanmean(pdaf_grad))
#         lpv=[]
#         rpv=[]
#         for i in range(4):
#             ref=np.array([row,col])
#             lpos=np.array(left_pixel_coords[i])
#             rpos=np.array(right_pixel_coords[i])
#             temp_index=tuple(ref+lpos)
#             maskPDAF_L[temp_index]=1
#             temp_index_=tuple(ref+rpos)
#             maskPDAF_R[temp_index_]=1
#             lpv.append(imCrop0[temp_index])
#             rpv.append(imCrop0[temp_index_])
#         left_pixel_values.append(lpv)
#         right_pixel_values.append(rpv)
            
        
# # fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
# # ax[0].imshow(maskPDAF_L,interpolation='none')      
# # ax[1].imshow(maskPDAF_R,interpolation='none')        
# plt.figure()
# plt.plot(angle,lpv[0],'o')
