# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:32:17 2023

@author: imrul
"""

import numpy as np
import skimage.filters as sf
import os
import cv2
import matplotlib.pyplot as plt
#import imutils
import time

def detect_blur_fft(image, size=60):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    magnitude = np.abs(fftShift)
    mean=np.nanmean(magnitude)
    
    # fftShift = np.fft.ifftshift(fftShift)
    # recon = np.fft.ifft2(fftShift)
    # magnitude = 20 * np.log(np.abs(recon))
    # mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean

def fm_lape(im,blur_size=5):
    im=cv2.blur(im,(blur_size,blur_size)) 
    fm=sf.laplace(im)
    fm=np.mean(abs(fm))
    return(fm)

def fm_helm(image,WSIZE=21):
    u=cv2.blur(image,(WSIZE,WSIZE))
    r1=u/image
    r1[image==0]=1
    index = (u>image)
    fm=1/r1
    fm[index]=r1[index]
    fm=np.mean(fm)
    return(fm-1)

def measure_sharpness_ACMO(image,hist_bins=100,hist_range=[0,256]):
    hist, edges=np.histogram(image,bins=hist_bins,range=hist_range,density=True)
    hist = np.log(hist+1)
    hist = hist / np.sum(hist)
    centers = (edges[1:]+edges[:-1])/2
    mean = np.sum(hist * centers)
    phist=np.abs(centers-mean)*hist
    return -(np.sum(phist))

def read_image(tiffPath,keyTiff='tiff',extension = 'tiff'): # to read an image and return a numpy array

    
    ims=[]
    # files=[]
    # for file in os.listdir(binPath):
    if tiffPath.find(keyTiff)>-1 and tiffPath.endswith(extension):
        print(tiffPath)
        im=plt.imread(tiffPath)
        ims.append(im) 
            # files.append(file)
    
    return ims[0]
#%%

# tiffPath=r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A10\2023-06-22\WBC_Cyc7-A10_s413_run00_FA_offset_17_14\img_red_FLR_fov30.tiff"
# # r'Z:\raspberrypi\photos\Alpha_plus\CYC7_A7\2023-06-16\WBC_Cyc7-A7_s404_run00_30cycle_Sdelivery_800_2p2s'
# # reference: r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A10\2023-06-22\WBC_Cyc7-A10_s413_run00_FA_offset_17_14\img_red_FLR_fov30.tiff"


# print(os.path.split(tiffPath)[-1])
# im=plt.imread(tiffPath)
# plt.figure()
# plt.imshow(im,cmap='gray',vmin=100,vmax=900)
# plt.title(os.path.split(tiffPath)[-1])
# #%%        reading the images after fov sorting

# meanIm=100 #np.nanmean(im)

# scoreHelm= fm_helm(im)*1000/meanIm
# scoreLape= fm_lape(im)*10000000/meanIm
# scoreFft= detect_blur_fft(im)/100/meanIm

# scoreDict = {'Helm': scoreHelm,
#              'Lape': scoreLape,
#              'FFT': scoreFft   
#     }

# print(scoreDict)


#%%

if __name__=='__main__':
    pngPath=r'Z:\raspberrypi\photos\Alpha_plus\CYC7_A2\2024-03-04\S829\img_PLT_AF-Coarse_fov1.tiff'
    sharpHelm = []
    sharpLape = []
    focusPlanes = []
    tag = 'af-0-'
    key = [tag,'.png']
    for file in os.listdir(pngPath):
        
        if file.find(tag)>=0 and file.endswith(key[1]):
            print(file)
            zPlane = int(file[file.find(key[0])+len(key[0]):file.find(key[1])])
            if key[1] == 'tiff':              
            
                im =  (read_image(os.path.join(pngPath,file),keyTiff=key[0],extension=key[1])*255).astype('int')
                
                im = im[im.shape[0]//2-512:im.shape[0]//2+511,im.shape[1]//2-512:im.shape[1]//2+511]
            else:
                
                im =  (read_image(os.path.join(pngPath,file),keyTiff=key[0],extension=key[1])*255).astype('int')
            
            sharpVal = fm_helm(im)
            sharpVal_= fm_lape(im,blur_size=5)
            
            focusPlanes.append(zPlane)
            sharpHelm.append(sharpVal)
            sharpLape.append(sharpVal_)
        
    plt.figure()
    plt.subplot(211)
    plt.plot(focusPlanes,sharpHelm,'o')
    plt.title('Helm')
    plt.ylabel('sharpness score')
    plt.subplot(212)
    plt.plot(focusPlanes,sharpLape,'*')
    plt.title('Laplace')
    plt.xlabel('Z position')
    plt.suptitle(tag)
    
        
        
        