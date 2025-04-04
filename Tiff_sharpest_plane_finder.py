# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:14:41 2024

@author: imrul
"""

import numpy as np
import skimage.filters as sf
import os
import cv2
import matplotlib.pyplot as plt


def fm_lape(im):
    im=cv2.blur(im,(5,5)) 
    fm=sf.laplace(im)
    fm=np.mean(abs(fm))
    return(fm)


def variance_of_laplacian(im):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    im=cv2.blur(im,(5,5))
    fm = cv2.Laplacian(im, cv2.CV_64F).var()
	
    return fm
    
    
    
    
    
def fm_helm(image,WSIZE=21):
    im=cv2.blur(image,(5,5)) 
    u=cv2.blur(im,(WSIZE,WSIZE))
    r1=u/image
    r1[image==0]=1
    index = (u>image)
    fm=1/r1
    fm[index]=r1[index]
    fm=np.mean(fm)
    return(fm)

def measure_sharpness_ACMO(image,hist_bins=25,hist_range=[0,1024]):
    hist, edges=np.histogram(image,bins=hist_bins,range=hist_range,density=True)
    hist = np.log(hist+1)
    hist = hist / np.sum(hist)
    centers = (edges[1:]+edges[:-1])/2
    mean = np.sum(hist * centers)
    phist=np.abs(centers-mean)*hist
    return(np.sum(phist))

def read_image(tiffPath,extension = 'tiff'): # to read an image and return a numpy array
    keyTiff=os.path.split(tiffPath)[-1][:-5]
    binPath=os.path.split(tiffPath)[0]
    ims=[]
    for file in os.listdir(binPath):
        if file.find(keyTiff)>-1 and file.endswith(extension):
            print(file)
            im=plt.imread(os.path.join(binPath,file))
            ims.append(im)          
    
    return ims[0]


binPath=r'W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-08-27\fstack\PS-PLT\S1161_PS-4um_0pt05pc_PBS_PLT_AS1\fov4'
channel=1 # BGR
key='offset_'
extension = '.tiff'

ims=[]
vcms=[]
sharpness_laplace=[]
sharpness_acmo=[]
sharpness_helm=[]
blur_estimator = []
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith(extension):
        print(file)
        vcm=int(file[file.rfind(key)+len(key):file.find(extension)])
        vcms.append(vcm)
        im=read_image(os.path.join(binPath,file))
        im=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3]
        # im=cv2.blur(im,(1,1))
        # ims.append(im)
        # sharpness_acmo.append(measure_sharpness_ACMO(im,hist_range=[0,1024]))
        sharpness_laplace.append(fm_lape(im))
        sharpness_helm.append(fm_helm(im))
        blur_estimator.append(variance_of_laplacian(im))
        
# fig,ax=plt.subplots(3,1)
# ax[0].plot(vcms,sharpness_acmo,'o',label='ACMO')
# ax[0].set_title('ACMO_abs_value')
# # ax[0].set_xlabel('Frame Count')
# ax[0].set_ylabel('Sharpness [a.u.]')
# ax[1].plot(vcms,sharpness_laplace,'*',label='Laplace')
# ax[1].set_title('Laplace_abs_value')
# # ax[1].set_xlabel('VCM')
# ax[1].set_ylabel('Sharpness [a.u.]')
# ax[0].grid()
# ax[1].grid()
# ax[2].plot(vcms,sharpness_helm,'+')
# ax[2].grid()
# ax[2].set_xlabel('VCM')
# ax[2].set_ylabel('Sharpness [a.u.]')
# ax[2].set_title('Helm_abs_value')
# fig.suptitle(key)


plt.figure()
plt.plot(vcms,sharpness_helm,'+')
plt.xlabel('Focus Plane')
plt.ylabel('Sharpness')
plt.title('HELM ALGO')


sharpestPlaneHelm = vcms[np.argmax(sharpness_helm)]
print(f'SHARPEST PLANE HELM : {sharpestPlaneHelm}')

plt.figure()
plt.plot(vcms,blur_estimator,'+')
plt.xlabel('Focus Plane')
plt.ylabel('Blur')
plt.title('Blur Estimator')