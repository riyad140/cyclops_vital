# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:01:05 2022

@author: imrul
"""
import rawpy
import imageio
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
from glob import glob
import cv2


def raw2bmp(imRaw_path,imSize=(3040,4048),filename='gg.png',channel='R',bit_depth=12):
    output_bit_depth=8
    bit_conversion_factor=int(2**bit_depth/2**output_bit_depth)
    if imRaw_path.find('cyclops-G')>-1:
        channel='G'
        
    # img=np.frombuffer(imRaw, dtype=np.uint16)
    img=np.fromfile(imRaw_path,dtype=np.uint16)
    print('raw image shape')
    print(img.shape)
    
    img=img.reshape(imSize)
    img_8bit=(((img+0)//bit_conversion_factor).astype(np.uint8)).clip(0,255)
    if channel=='R':
        im_rgb=cv2.cvtColor(img_8bit, cv2.COLOR_BAYER_BG2BGR)
    else:
        im_rgb=cv2.cvtColor(img_8bit, cv2.COLOR_BAYER_GR2BGR)
    
    cv2.imwrite(filename,im_rgb)
    return im_rgb


def raw2bmpNEW(filename,imageSize = (3120, 4224)):   #imageSize updated   
    # dstPath=os.path.join(rawPath,'bmps')
    new_filename=filename[:-4]+'.bmp'
    npimg = np.fromfile(filename, dtype=np.dtype('<u2'))
    imRaw = npimg.reshape(imageSize)
    imBmp=(imRaw >> 2).astype(np.uint8)
    cv2.imwrite(new_filename,imBmp)
    return new_filename
#%% for IMX477

folder = r'Z:\raspberrypi\photos\Method_Comp\2022-06-14_Juno\run00_beast_demo_sample_s858_WBC_Diff_Intdisc_17min_Cyc4Juno\subset'
folder= r'Z:\raspberrypi\photos\imrul_sandbox'
filenames = glob(os.path.join(folder, '*.raw'))


for filename in filenames:
    print(filename)
    fname=filename[:filename.find('.raw')]+'.png'
    print(fname)
    raw2bmp(imRaw_path=filename,filename=fname)
    
#%% for AR1335

folder = r'G:\Shared drives\Experimental Results\Cyclops_Server_Backup\WBC\2022-06-30_Diana\run0_3DLR0_shim0um_sample_s881_Cyc5Diana\subset'
filenames = glob(os.path.join(folder, '*.raw'))

for filename in filenames:
    print(filename)
    raw2bmpNEW(filename)
    