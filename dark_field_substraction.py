# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 08:17:20 2021

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle
import cv2
#%%
# filename=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\20210812_WBCsample454_exp_5000ms_iso_200_v00_DarkfieldAF_DARKIMAGES_Fluor__RED_0.bmp"
filename=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\20210826\15-30-37\15-30-37_cell_462a_s17___gg__DF__RED_0.bmp"
im = imread(filename)
# im=im.astype(np.int16)
# im=im[:,:,0]
# im[:,:,0]=np.ones(im.shape[:-1])
# im[:,:,2]=np.ones(im.shape[:-1])
# plt.figure()
# plt.imshow(im)


# filename_bcklvl=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\20210813_FFandDarkImages_exp_5000ms_iso_200_v00_DarkfieldAF_DARKIMAGES_DARK__DARKRED_0.bmp"

# imBfield=imread(filename_bcklvl)
# # imBfield=imBfield.astype(np.int16)
# imBfield=imBfield[:,:,0]
# # imBfield[:,:,1]=np.ones(imBfield.shape[:-1])
# # imBfield[:,:,2]=np.ones(imBfield.shape[:-1])

# filename_ufield=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\Spencer - CC\20210810_Imaging Settings Images\FLURO LIGHTING TEST\bcklvl\20210811_BF_FL_cells_sample453_exp_5s_bcklvl_Fluor__RED_0.bmp"
filename_ufield=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\20210826\FLATFIELD\10-30-20\10-30-20_cell_FLATFIELD___gg__DF__RED_0.bmp"
imUfield=imread(filename_ufield)
# imUfield=imUfield.astype(np.int16)
# imUfield=imUfield[:,:,0]
# imUfield[:,:,0]=np.ones(imUfield.shape[:-1])
# imUfield[:,:,2]=np.ones(imUfield.shape[:-1])
# plt.figure()
# plt.imshow(im-imDark)

# imFinal=(im-imBfield)/imUfield*np.mean(imUfield)
# imFinal=np.clip(imFinal,0,255)

fudge_factor=1.2
imUfield_n=imUfield*np.mean(im)/np.mean(imUfield)/fudge_factor # leveling the mean value between two capture
imUfield_n=imUfield_n.astype(int)
imFinal=im-imUfield_n

fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
ax[0].imshow(im)
ax[0].set_title('Fluro Image')
ax[1].imshow(imUfield_n)
ax[1].set_title('Flat field')
ax[2].imshow(imFinal.astype(int))
ax[2].set_title('Fluro-Flat field')
#%%
fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(im)
ax[0].set_title('Raw Image [R channel]')
ax[1].imshow(imFinal.astype(int).clip(0,255))
ax[1].set_title('Flat field corrected [R channel]')


#%% by division
im=im.astype(np.int16)
im_=im[:,:,0]+1
# im[:,:,1]=np.ones(im.shape[:-1])
# im[:,:,2]=np.ones(im.shape[:-1])


imUfield=imUfield.astype(np.int16)
imUfield_=imUfield[:,:,0]+1
# imUfield[:,:,1]=np.ones(imUfield.shape[:-1])
# imUfield[:,:,2]=np.ones(imUfield.shape[:-1])

if imUfield_.shape[-1]!=3:
    imFinal_d=im_/imUfield_*np.nanmean(imUfield_)
    imFinal_d=imFinal_d.clip(0,255)
else:
    imFinal_t=im_/imUfield_
    imFinal_d=np.zeros(imFinal_t.shape,dtype=int)
    k0=(imFinal_t[:,:,0]*np.nanmean(imUfield_[:,:,0])).clip(0,255)
    imFinal_d[:,:,0]=k0.astype(np.uint16)
    k1=(imFinal_t[:,:,1]*np.nanmean(imUfield_[:,:,1])).clip(0,255)
    imFinal_d[:,:,1]=k1.astype(np.uint16)
    k2=(imFinal_t[:,:,2]*np.nanmean(imUfield_[:,:,2])).clip(0,255)
    imFinal_d[:,:,2]=k2.astype(np.uint16)
    # imFinal_d=imFinal_d.clip(0,255)
    
fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
ax[0].imshow(im_)
ax[0].set_title('Fluro Image')
ax[1].imshow(imUfield_)
ax[1].set_title('Flat field')
ax[2].imshow(imFinal_d)
ax[2].set_title('Fluro-Flat field')
#%%
# import cv2
# blurred=cv2.medianBlur(imFinal.astype(np.uint16), 5)
# _,img=cv2.threshold(blurred.astype(np.uint16),np.mean(imFinal),255,cv2.THRESH_TOZERO)
# # plt.figure()

# # th3 = cv2.adaptiveThreshold(imFinal.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,101,2)
# # plt.imshow(img)

# fig,ax=plt.subplots(2,1,sharex=True,sharey=True)
# ax[0].imshow(imFinal)
# ax[1].imshow(img)

# #%% blurring the image
# sigma=(901,901)
# blur = cv2.GaussianBlur(im,sigma,0)
# plt.figure()
# plt.imshow(blur)

# #%%
# imFinal=-im+blur
# plt.figure()
# plt.imshow(imFinal)
