# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 23:56:18 2021

@author: imrul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
#%%
def linearize_companded_image_data(buffer: np.ndarray,
pedestal: int = 240) -> np.ndarray:
    """
    24-bit companded input to linear float32
    """
    # Knee points raw -> companded:
    _knee_points_in = [0, 3904, 287152, 2238336, 16777200]
    # Matching companded knee points:
    _knee_points_out = [0, 3904, 23520, 54416, 65280]
    buffer = buffer.astype(np.float32)
    # In principle, we may get negative values in linear light
    # (keeping them is good for noise cancellation in the shadows):
    buffer -= pedestal
    decompanded = np.zeros_like(buffer).astype(np.float32)
    # Below lowest knee point (0), use same as just above:
    mask = (buffer <= _knee_points_out[0])
    decompanded[mask] = (_knee_points_in[0] +
    ((_knee_points_in[1] - _knee_points_in[0]) /
    (_knee_points_out[1] - _knee_points_out[0]))
    * (buffer[mask] - _knee_points_out[0]))
    for i in range(len(_knee_points_out) - 1):
        mask = ((_knee_points_out[i] <= buffer) &
        (buffer < _knee_points_out[i + 1]))
        decompanded[mask] = (_knee_points_in[i] +
        ((_knee_points_in[i + 1] - _knee_points_in[i]) /
        (_knee_points_out[i + 1] - _knee_points_out[i]))
        * (buffer[mask] - _knee_points_out[i]))
    decompanded[buffer >= _knee_points_out[-1]] = _knee_points_in[-1]
    return decompanded

def delinearize_companded_image_data(buffer: np.ndarray,
pedestal: int = 240) -> np.ndarray:
    """
    24-bit companded input to linear float32
    """
    # Knee points raw -> companded:
    _knee_points_out = [0, 3904, 287152, 2238336, 16777200]
    # Matching companded knee points:
    _knee_points_in = [0, 3904, 23520, 54416, 65280]
    buffer = buffer.astype(np.float32)
    # In principle, we may get negative values in linear light
    # (keeping them is good for noise cancellation in the shadows):
    buffer += pedestal
    decompanded = np.zeros_like(buffer).astype(np.float32)
    # Below lowest knee point (0), use same as just above:
    mask = (buffer <= _knee_points_out[0])
    decompanded[mask] = (_knee_points_in[0] +
    ((_knee_points_in[1] - _knee_points_in[0]) /
    (_knee_points_out[1] - _knee_points_out[0]))
    * (buffer[mask] - _knee_points_out[0]))
    for i in range(len(_knee_points_out) - 1):
        mask = ((_knee_points_out[i] <= buffer) &
        (buffer < _knee_points_out[i + 1]))
        decompanded[mask] = (_knee_points_in[i] +
        ((_knee_points_in[i + 1] - _knee_points_in[i]) /
        (_knee_points_out[i + 1] - _knee_points_out[i]))
        * (buffer[mask] - _knee_points_out[i]))
    decompanded[buffer >= _knee_points_out[-1]] = _knee_points_in[-1]
    return decompanded

    
#%% Read raw files
imPath=r"C:\Users\imrul\Downloads\ISP_Programming_Assessment\ISP_Programming_Assessment\images\002.raw"
npimg = np.fromfile(imPath, dtype=np.uint16)
imageSize = (1871, 2880)
imRaw_ = npimg.reshape(imageSize)
header_row=5
trailer_row=6

imRaw=imRaw_[header_row:-trailer_row,:]

plt.figure()
plt.imshow(imRaw)
#%% decompanding
imRaw_linear=linearize_companded_image_data(imRaw)
plt.figure()
plt.imshow(imRaw_linear)

imRaw_nonlinear=delinearize_companded_image_data(imRaw_linear)
imRaw=imRaw.astype(np.uint16)

# imRaw_norm=(imRaw_linear-np.min(imRaw_linear))/(np.max(imRaw_linear)-np.min(imRaw_linear))
# plt.figure()
# plt.imshow(imRaw_norm)
#%% demosaicing
imBGR=cv2.demosaicing(	imRaw,  cv2.COLOR_BayerRG2BGR_EA)
img_scaled = cv2.normalize(imBGR, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
img_scaled=img_scaled/65535
# plt.figure()
# cv2.imshow('imBGR',imBGR)

#%% fixing exposure
imGray=cv2.cvtColor(img_scaled.astype(np.float32), cv2.COLOR_BGR2GRAY)
imGray_8bit=(imGray*255).astype(np.uint8)

plt.figure()
plt.imshow(imGray_8bit,cmap='gray')
#%%
hist=np.histogram(imGray.flatten())
plt.figure()
plt.plot(hist[1][1:],hist[0])

ind=np.argmax(hist[0])
meanIntensity=hist[1][1+ind]
targetIntensity=0.5
digital_gain=(targetIntensity/meanIntensity)/1

imEx=(imGray_8bit.astype(float)*digital_gain).clip(0,2**8-1).astype(np.uint8)
# plt.figure()
# plt.imshow(imEx,cmap='gray')

img_exposed=(img_scaled.astype(float)*digital_gain).clip(0,1).astype(np.float32)
cv2.namedWindow('autoexposure', cv2.WINDOW_AUTOSIZE)
cv2.imshow('autoexposure',img_exposed)


# imBGR_exposed_8bit=(imBGR_exposed//2**8).astype(np.uint8)
plt.figure()
plt.imshow(img_exposed)

cv2.namedWindow('autoexposure8', cv2.WINDOW_AUTOSIZE)
cv2.imshow('autoexposure8',img_exposed)

##R and B getting switched somehow in this block

#%%
# cv2.imshow('B',imBGR_exposed[:,:,0])
# cv2.imshow('G',imBGR_exposed[:,:,1])
# cv2.imshow('R',imBGR_exposed[:,:,2])
#%% denoising
bilateral = cv2.bilateralFilter(img_exposed.astype(np.float32), 10, 0.05, 10).clip(0,1).astype(np.float32)


# bilateral_8bit = cv2.bilateralFilter(imBGR_exposed_8bit, 10, 10, 10)
plt.figure()
plt.imshow(bilateral)

#%% white balance
from skimage import img_as_ubyte
def percentile_whitebalance(image, percentile_value,bit_depth=8):
    # fig, ax = plt.subplots(1,1, figsize=(12,6))
    for channel, color in enumerate('rgb'):
            channel_values = image[:,:,channel]
            value = np.percentile(channel_values, percentile_value)
            # ax.step(np.arange(256), 
            #             np.bincount(channel_values.flatten(), 
            #             minlength=256)*1.0 / channel_values.size, 
            #             c=color)
            # ax.set_xlim(0, 255)
            # ax.axvline(value, ls='--', c=color)
            # ax.text(value-70, .01+.012*channel, 
            #             "{}_max_value = {}".format(color, value), 
            #             weight='bold', fontsize=10)
            # ax.set_xlabel('channel value')
            # ax.set_ylabel('fraction of pixels');
            # ax.set_title('Histogram of colors in RGB channels')  
    # if bit_depth==8:
    #     whitebalanced = (((image*1.0 / np.percentile(image,percentile_value, axis=(0, 1))).clip(0, 1))*(2**bit_depth-1)).astype('uint8')
    # elif bit_depth==16:
    #     whitebalanced = (((image*1.0 / np.percentile(image,percentile_value, axis=(0, 1))).clip(0, 1))*(2**bit_depth-1)).astype('uint16')
        
    whitebalanced =( ((image*1.0 / np.percentile(image,percentile_value, axis=(0, 1))).clip(0, 1)).astype(np.float32))
    
    fig, ax = plt.subplots(1,2, figsize=(12,6),sharex=True,sharey=True)
    ax[0].imshow(image)
    ax[0].set_title('original Image')
    ax[1].imshow(whitebalanced);
    ax[1].set_title('Whitebalanced Image')
    
    print(f'white balanced ')
    print(whitebalanced.shape)
    return whitebalanced
bilateral_wb=percentile_whitebalance(bilateral, percentile_value=99.0)
# bilateral_wb=percentile_whitebalance(bilateral, percentile_value=95,bit_depth=16)
# fig, ax = plt.subplots(1,2, figsize=(10,6),sharex=True,sharey=True)
# ax[0].imshow(bilateral_8bit)
# ax[0].set_title('Original Image')
# # bilateral_8bit_wb = ((bilateral_8bit * (bilateral_8bit.mean() / bilateral_8bit.mean(axis=(0, 1))))
# #              .clip(0, 255).astype(int))

# ax[1].imshow(bilateral_8bit_wb);
# ax[1].set_title('Whitebalanced Image');
#%% gamma correction
def adjust_gamma(image, gamma=1.0, bit_depth=8):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    image=(image*(2**bit_depth-1)).astype(np.uint8)
    invGamma = 1.0 / gamma
    if bit_depth==8:
    	table = np.array([((i / ((2**bit_depth-1))) ** invGamma) * (2**bit_depth-1)
    		for i in np.arange(0, 2**bit_depth)]).astype("uint8")
    elif bit_depth==16:
    	table = np.array([((i / ((2**bit_depth-1))) ** invGamma) * (2**bit_depth-1)
    		for i in np.arange(0, 2**bit_depth)]).astype("uint16")        
            
    	# apply gamma correction using the lookup table
    return cv2.LUT(image, table)

imGamma_8bit=adjust_gamma(bilateral_wb,gamma=1.2)
plt.figure()
plt.imshow(imGamma_8bit)

# cv2.imshow('',imGamma_8bit)