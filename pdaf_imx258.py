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


binPath=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\20210916_TargetImages_IMX258\subset\20210916_IMX258_USAFTarget_400mA_BF_Plane475_exp80_analogGain0.raw"
npimg = np.fromfile(binPath, dtype=np.uint16)
imageSize = (3118, 4208)
imRaw = npimg.reshape(imageSize)
imCrop=imRaw[24:-22,24:-24]
# imCrop=cv2.medianBlur((imCrop//4).astype(np.uint8),ksize=3)


# kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
# edges_x = cv2.filter2D(imCrop,cv2.CV_16U,kernelx)
# 

edges_x = filters.sobel_v(cv2.medianBlur((imCrop//4).astype(np.uint8),ksize=11))
edges_x=edges_x/np.max(edges_x)


fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(imCrop,cmap='gray')
ax[1].imshow(edges_x)


#%%
th=0.05
imGrad=np.copy(edges_x)
maskGrad=np.zeros(imGrad.shape)*np.nan
maskGrad[abs(imGrad)>th]=1


# imGrad[abs(imGrad)<=th]=np.nan

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(imCrop*maskGrad,cmap='gray',interpolation='none')
ax[0].set_title('raw')
ax[1].imshow(imGrad*maskGrad,cmap='jet_r',interpolation='none')
ax[1].set_title('gradient')

#%% applying the maskGrad
imCrop0=imCrop*maskGrad
imGrad0=imGrad*maskGrad



#%%
maskPDAF_L=np.zeros(imCrop0.shape)
maskPDAF_R=np.zeros(imCrop0.shape)

# for row in np.arange(imCrop0.shape[0]):
#     for col in np.arange(imCrop0.shape[1]):
        


#%% First PDAF block
# imCrop=imRaw[24:-22,24:-24]
# imGrad=edges_x

left_pixel_coords=[(5,2),(5,18),(24,9),(24,25)]
right_pixel_coords=[(8,1),(8,17),(21,10),(21,26)]

pdaf_blocks=[]
im_binned=[]
pdaf_grads=[]
for row in np.arange(0,imCrop0.shape[0],32):
    for col in np.arange(0,imCrop0.shape[1],32):     
        
        # print(f'(r,c)={row,col}')
        pdaf_arr=imCrop0[row:row+32,col:col+32]
        pdaf_grad=imGrad0[row:row+32,col:col+32]
        pdaf_blocks.append(pdaf_arr)
        # pdaf_grad[pdaf_grad==0]=np.nan
        pdaf_grads.append(pdaf_grad)
        im_binned.append(np.nanmean(pdaf_grad))
        
        for i in range(4):
            ref=np.array([row,col])
            lpos=np.array(left_pixel_coords[i])
            rpos=np.array(right_pixel_coords[i])
            temp_index=tuple(ref+lpos)
            maskPDAF_L[temp_index]=1
            temp_index_=tuple(ref+rpos)
            maskPDAF_R[temp_index_]=1
            
        
fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(maskPDAF_L,interpolation='none')      
ax[1].imshow(maskPDAF_R,interpolation='none')        

#%%

imleft=imCrop0*maskPDAF_L
imright=imCrop0*maskPDAF_R

r,c=imright.shape
shift, error, diffphase = register_translation(imright[r//3:2*r//3,1*c//3:2*c//3], imleft[r//3:2*r//3,1*c//3:2*c//3],1000)
print(f'shift: {shift} ')

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(imCrop0*maskPDAF_L,interpolation='none')      
ax[1].imshow(imCrop0*maskPDAF_R,interpolation='none')     

#%% block by block shift calculation
for row in np.arange(0,imleft.shape[0],32):
    for col in np.arange(0,imleft.shape[1],32):
        shift,error,diffphase=register_translation(imleft[row:row+32,col:col+32],imleft[row:row+32,col:col+32],100)
        if shift[0]!=-0.75:
            print(f'({row,col} : {shift},{error})')



#%%
pdaf_arr_full=np.array(pdaf_blocks).reshape(96,130,32,32)   
pdaf_grad_arr=np.array(pdaf_grads).reshape(96,130,32,32) 

block_index=(45,50)

temp=pdaf_arr_full[block_index]

for ind in range(len(left_pixel_coords)):
    start_point=np.flip(np.array(left_pixel_coords[ind])-np.array([1,1])) # open cv flips the coordinate w.r.t numpy
    end_point=np.flip(np.array(left_pixel_coords[ind])+np.array([1,1]))
    
    start_point_=np.flip(np.array(right_pixel_coords[ind])-np.array([1,1]))
    end_point_=np.flip(np.array(right_pixel_coords[ind])+np.array([1,1]))
    
    imCanvas=cv2.rectangle(temp, start_point, end_point, color=255, thickness=0)
    imCanvas=cv2.rectangle(imCanvas, start_point_, end_point_, color=200, thickness=0)
    temp=np.copy(imCanvas)

plt.figure()
plt.imshow(imCanvas,cmap='gray')
plt.title(f"index: {block_index}")
#%%     
im_binned=np.array(im_binned).reshape(96,130) 
avg_disp=[]   
avg_grad=[]     
imL=[]
imR=[]
for count,pdaf_b in enumerate(pdaf_blocks):
    disp=[]
    grad=[]
    iml=[]
    imr=[]
    for i in range(4):
        lpx=(pdaf_b[left_pixel_coords[i]])
        rpx=(pdaf_b[right_pixel_coords[i]])
        grad_lpx=pdaf_grads[count][left_pixel_coords[i]]
        grad_rpx=pdaf_grads[count][right_pixel_coords[i]]
        
        iml.append(lpx)
        imr.append(rpx)
        disp.append(lpx-rpx)
        grad.append((grad_lpx+grad_rpx)/2.)
    avg_disp.append(np.nanmedian(disp))
    avg_grad.append(np.nanmedian(grad))
    imL.append(np.nanmean(iml))
    imR.append(np.nanmean(imr))
    
        
disparity_map=np.array(avg_disp).reshape(96,130)
grad_map=np.array(avg_grad).reshape(96,130)
imLeft=np.array(imL).reshape(96,130)
imRight=np.array(imR).reshape(96,130)
#%%
fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(grad_map)
ax[1].imshow(disparity_map,cmap='jet_r',vmin=-30,vmax=+30)

#%%
fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(imLeft)
ax[1].imshow(imRight)

shift, error, diffphase = register_translation(imRight, imLeft,1000)

print(f'Image Shift: {shift}')
#%% cross correlation 2D

r,c=imLeft.shape
imL_slice=imLeft[r//3:2*r//3,1*c//3:2*c//3]
imR_slice=imRight[r//3:2*r//3,1*c//3:2*c//3]

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(imL_slice)
ax[1].imshow(imR_slice)

plt.figure()
plt.plot(imL_slice[0,:],label='left')
plt.plot(imR_slice[0,:],label='Right')
shift, error, diffphase = register_translation(imR_slice, imL_slice,100)

print(f'Image Shift1: {shift}')
#%%
from scipy import signal
plt.figure()
for i in range(20):
    corr = signal.correlate(imL_slice[i,:], imR_slice[i,:])
    corr /= np.max(corr)
    lags = signal.correlation_lags(len(imL_slice[i,:]), len(imR_slice[i,:]))
    
    plt.plot(lags,corr)
    plt.grid(True)
    
#%%
imCorr=signal.correlate2d(imL_slice, imR_slice,mode='same')

plt.figure() 
plt.imshow(imCorr)
   
#%% diff/grad
imDisp=disparity_map/grad_map
mask1=np.zeros(grad_map.shape)*np.nan
gTh1=0.1
mask1[np.abs(grad_map)>gTh1]=1

imD=imDisp*mask1
imG=grad_map*mask1

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(imG,cmap='jet_r')
ax[1].imshow(imD,cmap='jet_r')
r,c=imD.shape
print('STAT')
print(np.nanmean(imD[r//3:r*2//3,c//3:c*2//3]))
#%% masking
# mask=np.zeros(im_binned.shape)*np.nan
# gTh=0.0001
# mask[abs(im_binned)>gTh]=1

# # disp=disparity_map*mask

# # plt.figure()
# # plt.imshow(disp,cmap='jet_r',vmin=-30,vmax=+30)

# # print('STATS')
# # print(np.nanmean(disp))
# # #%%
# imDisp_=imDisp*mask

# fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
# ax[0].imshow(im_binned*mask)
# ax[1].imshow(imDisp_,cmap='jet_r',vmin=-300,vmax=+300)

# print('STATS')
# print(np.nanmean(imDisp_))
#%% median blurring

imBad=(np.copy(imCrop)//4).astype(np.uint8)
imGood=cv2.medianBlur(imBad, ksize=9)

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(imBad,cmap='gray')
ax[0].set_title('Raw')
ax[1].imshow(imGood,cmap='gray')
ax[1].set_title('Median Blur')

#%%
