# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 15:53:24 2021

@author: imrul
"""
import matplotlib.pyplot as plt
from skimage.filters import laplace #for edge detection filters
import skimage.io #for image processing
from skimage.io import imread, imshow
import numpy as np #for processing images as numpy arrays
import sys 
import os 
import cv2
import time

#%%
def fm_lape(im):
    
    fm=laplace(im)
    fm=np.mean(abs(fm))
    return(fm)
def fm_helm(image,WSIZE):
    u=cv2.blur(image,(WSIZE,WSIZE))
    r1=u/image
    r1[image==0]=1
    index = (u>image)
    fm=1/r1
    fm[index]=r1[index]
    fm=np.mean(fm)
    return(fm)

binPath=r'Z:\raspberrypi\photos\2021-09-07\15-41-07\FL'

ksize=5
for file in os.listdir(binPath):
    print(file)
    im=imread(os.path.join(binPath,file))
    h,w,c=im.shape
    imFL=np.copy(im[:,:,0])
    imFL=cv2.blur(imFL, (ksize,ksize))
    imT=imFL[h//3:h*2//3,w//3:w*2//3]
    plt.figure()
    plt.imshow(imT)
    t0=time.time()
    sharpness=fm_lape(imT)
    t1=time.time()
    sharpness_=fm_helm(imT,ksize)
    t2=time.time()
    dt=t1-t0
    dt_=t2-t1
    print(f'sharpness value {sharpness} # time taken {dt}')
    print(f'sharpness value {sharpness_} # time taken {dt_}')
    



