# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:31:49 2021

@author: imrul.kayes
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle
import cv2


#%%
def percentile_whitebalance(image, percentile_value):
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    for channel, color in enumerate('rgb'):
            channel_values = image[:,:,channel]
            value = np.percentile(channel_values, percentile_value)
            ax.step(np.arange(256), 
                        np.bincount(channel_values.flatten(), 
                        minlength=256)*1.0 / channel_values.size, 
                        c=color)
            ax.set_xlim(0, 255)
            ax.axvline(value, ls='--', c=color)
            ax.text(value-70, .01+.012*channel, 
                        "{}_max_value = {}".format(color, value), 
                        weight='bold', fontsize=10)
            ax.set_xlabel('channel value')
            ax.set_ylabel('fraction of pixels');
            ax.set_title('Histogram of colors in RGB channels')  
            
    whitebalanced = img_as_ubyte((image*1.0 / np.percentile(image,percentile_value, axis=(0, 1))).clip(0, 1))
    
    fig, ax = plt.subplots(1,2, figsize=(12,6),sharex=True,sharey=True)
    ax[0].imshow(image)
    ax[0].set_title('original Image')
    ax[1].imshow(whitebalanced);
    ax[1].set_title('Whitebalanced Image')
    
    print(f'white balanced ')
    print(whitebalanced.shape)
    return ax,whitebalanced


def whitepatch_balancing(image, from_row, from_column, 
                         row_width, column_width):
    fig, ax = plt.subplots(1,2, figsize=(10,5),sharex=True,sharey=True)
    ax[0].imshow(image)
    ax[0].add_patch(Rectangle((from_column, from_row), 
                              column_width, 
                              row_width, 
                              linewidth=3,
                              edgecolor='r', facecolor='none'));
    ax[0].set_title('Original image')
    image_patch = image[from_row:from_row+row_width, 
                        from_column:from_column+column_width]
    image_max = (image*1.0 / 
                 image_patch.max(axis=(0, 1))).clip(0, 1)
    ax[1].imshow(image_max);
    ax[1].set_title('Whitebalanced Image')

#%%


filename=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\Spencer - CC\20210810_Imaging Settings Images\FLURO LIGHTING TEST\20210810_SpinzExposureTest_Fluor_1800mA_awb_off_semioptCH1_EXP3000000.bmp"
dinner = imread(filename)
plt.figure()
plt.imshow(dinner)

flatfield=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\20210719_FlatFieldCorrection_YellowandWhiteTest1\20210719_FFImage_White_30mm_FlatField_10msISO100_.bmp"

imUfield=imread(flatfield)


im_final=((dinner/imUfield).clip(0,1)*255).astype(np.uint8)

plt.figure()
plt.imshow(im_final)

#%%
dinner=np.copy(im_final)
dinner[np.isnan(dinner)]=0
#%%
# im_gray=cv2.cvtColor(dinner, cv2.COLOR_BGR2GRAY)
# dinner=np.copy(im_gray)
#%% white patch algorithm
fig, ax = plt.subplots(1,2, figsize=(10,6),sharex=True,sharey=True)
ax[0].imshow(dinner)
ax[0].set_title('Original Image')
dinner_max = (dinner*1.0 / dinner.max(axis=(0,1)))
dinner_max=(dinner_max*255).astype(np.uint8)

ax[1].imshow(dinner_max);
ax[1].set_title('Whitebalanced Image');
#%%
fig, ax = plt.subplots(1,2, figsize=(10,6),sharex=True,sharey=True)
ax[0].imshow(dinner)
ax[0].set_title('Original Image')
dinner_mean = (dinner*1.0 / dinner.mean(axis=(0,1)))
# dinner_mean=(dinner_mean*255).astype(np.uint8)
ax[1].imshow(dinner_mean.clip(0,1))
ax[1].set_title('Whitebalanced Image');

#%%
ax,im=percentile_whitebalance(dinner, 75)
#%%
# ax,im=percentile_whitebalance(im_gray, 75)

#%% grey patch algorithm
fig, ax = plt.subplots(1,2, figsize=(10,6),sharex=True,sharey=True)
ax[0].imshow(dinner)
ax[0].set_title('Original Image')
dinner_gw = ((dinner * (dinner.mean() / dinner.mean(axis=(0, 1))))
             .clip(0, 255).astype(int))
ax[1].imshow(dinner_gw);
ax[1].set_title('Whitebalanced Image');

#%%
whitepatch_balancing(dinner, 1408,2414, 150, 150)