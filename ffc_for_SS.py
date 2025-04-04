# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 16:13:38 2022

@author: imrul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import os
#%%

im_path=r"Z:\raspberrypi\photos\trash\ss_troubleshoot\2\20220223_ts_DF_vcm_275.bmp"
ffc_path=r"Z:\raspberrypi\photos\trash\ss_troubleshoot\2\20220223_ts_DF_vcm_375.bmp"

im=cv2.imread(im_path)
im=im[:,:,2] # only red channel

im_ffc=cv2.imread(ffc_path)
im_ffc=im_ffc[:,:,2]
im_ffc=cv2.blur(im_ffc,(201,201))


fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(im)
ax[1].imshow(im_ffc)


#%%

multiplier=1

imCorrected=(im/im_ffc/multiplier*np.mean(im_ffc)).clip(0,255)
imCorrected=imCorrected.astype(np.uint8)


fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(im,cmap='gray')
ax[1].imshow(imCorrected,cmap='gray')