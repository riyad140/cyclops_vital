# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:34:35 2022

@author: imrul
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:59:05 2022

@author: Jeff
"""

import numpy as np
import skimage.filters as sf
import os
import cv2
import matplotlib.pyplot as plt


def fm_lape(im):
#     im=blur(im,(5,5)) 
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
    return(fm)

def measure_sharpness_ACMO(image,hist_bins=25,hist_range=[0,1024]):
    hist, edges=np.histogram(image,bins=hist_bins,range=hist_range,density=True)
    hist = np.log(hist+1)
    hist = hist / np.sum(hist)
    centers = (edges[1:]+edges[:-1])/2
    mean = np.sum(hist * centers)
    phist=np.abs(centers-mean)*hist
    return(np.sum(phist))

#%%

binPath=r'Z:\raspberrypi\photos\Method_comp_RBC_PLT\2022-07-14\run00_plt_sample_S900_RBC_Plt_FC_Cyc5Diana\AF'
channel=1 # BGR
key='AF_19'

ims=[]
vcms=[]
sharpness_laplace=[]
sharpness_acmo=[]
sharpness_helm=[]
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith('png'):
        print(file)
        vcm=int(file[file.find(key)+len(key)+1:file.find(key)+len(key)+4])
        vcms.append(vcm)
        im=cv2.imread(os.path.join(binPath,file),1)
        #imCrop=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3]
        im=cv2.blur(im,(1,1))
        ims.append(im)
        sharpness_acmo.append(measure_sharpness_ACMO(im,hist_range=[0,1024]))
        sharpness_laplace.append(fm_lape(im))
        sharpness_helm.append(fm_helm(im))
        
fig,ax=plt.subplots(3,1)
ax[0].plot(vcms,sharpness_acmo,'o',label='ACMO')
ax[0].set_title('ACMO_abs_value')
# ax[0].set_xlabel('Frame Count')
ax[0].set_ylabel('Sharpness [a.u.]')
ax[1].plot(vcms,sharpness_laplace,'*',label='Laplace')
ax[1].set_title('Laplace_abs_value')
# ax[1].set_xlabel('VCM')
ax[1].set_ylabel('Sharpness [a.u.]')
ax[0].grid()
ax[1].grid()
ax[2].plot(vcms,sharpness_helm,'+')
ax[2].grid()
ax[2].set_xlabel('VCM')
ax[2].set_ylabel('Sharpness [a.u.]')
ax[2].set_title('Helm_abs_value')
fig.suptitle(key)