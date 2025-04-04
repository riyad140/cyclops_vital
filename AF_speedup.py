# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:44:44 2022

@author: imrul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.filters import laplace
import time

def fm_lape(im):
    fm=laplace(im)
    fm=np.mean(abs(fm))
    return(fm)

def fm_helm(image,WSIZE=3):
    u=cv2.blur(image,(WSIZE,WSIZE))
    r1=u/image
    r1[image==0]=1
    index = (u>image)
    fm=1/r1
    fm[index]=r1[index]
    fm=np.mean(fm)
    return(fm)

def evaluateSharpness(im):
    im_resized=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3]
    # plt.figure()
    # plt.imshow(im_resized)
    fm=fm_lape(im_resized)
    fm_=fm_helm(im_resized)
    return fm,fm_
    
    
#     #process the image
#     # logging.debug("processing AF image for sharpness")
#     for i in range (0, len(fnames),1):
#         #for all images, if the sharpness for the index is not the sharpest
#         im = skimage.io.imread(fname=fnames[i]) #determine the file name or path for each image
#         im_resized=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3,0] # giving it only red channel image's central FOV
#         fm=fm_lape(im_resized) #call the lapace filter function to process the image
# #         fm=fm_helm(im_resized)
#         sharpnesses.append(fm) #append the sharpness result to the array
#%%

binPath=r'Z:\raspberrypi\photos\Method_Comp\2022-02-23\run00_sample_s678_WBC_diff_Meth_Comp_DF_1400_FCell_Cyc4Metal\AF\subset'
ims=[]
files=[]
sharpness=[]
sharpness_=[]
vcms=[]
channel=2 # for red
for file in os.listdir(binPath):
    im=cv2.imread(os.path.join(binPath,file))
    ims.append(im[:,:,channel])
    files.append(files)
    vcms.append(int(file[-7:-4]))
    s1,s2=evaluateSharpness(im[:,:,channel])
    sharpness.append(s1*100)
    sharpness_.append(s2*50)

fig,ax=plt.subplots(1,2,sharex=True)
ax[0].plot(vcms,sharpness,'o-')
ax[0].grid(True)
ax[1].plot(vcms,sharpness_,'*-')
ax[1].grid(True)
vcms=np.array(vcms)

#%%
# t0=time.time()
# gg=fm_lape(im[:,:,2])
# t1=time.time()
# print('Elapsed Time')
# print(t1-t0)
WSIZE=3
image=np.copy(im[:,:,2])
u=cv2.blur(image,(WSIZE,WSIZE))
r1=u/image
r1[image==0]=1
index = (u>image)
fm=1/r1
fm[index]=r1[index]
fm=np.mean(fm)


#%%

# vcm_0=610


# sharpness=np.array(sharpness)
# ind=np.where(vcms==vcm_0)[0][0]

# momentum=0

# take first 3 captures to calculate momentum

def get_sharpness(vcm,vcms=vcms):
    print(f'for VCM : {vcm}')
    ind=np.where(vcms==vcm)[0][0]
    return sharpness[ind]

def limit_momentum(momentum, limit=2):
    if momentum>=limit:
        momentum=limit
    elif momentum<=-limit:
        momentum=-limit
    else:
        momentum=momentum
    return momentum

#%%

# vcm0=630
# direction=1
# s_vals=[]
# momentum=0
# for vcm in [vcm0, vcm0+5*direction, vcm0+10*direction]:
#     s_val=get_sharpness(vcm)
#     s_vals.append(s_val)

# ds=np.diff(np.array(s_vals))    
# momentum=momentum+np.sum(np.sign(ds))
# momentum=limit_momentum(momentum)
# print(f'Momentum: {momentum}')
# print(f'Direction: {direction}')
# #%%
# s_vals=[]

# if momentum<0:
#     vcm0=vcm0
#     direction=-1*direction
#     momentum=momentum*direction
#     for vcm in [vcm0,vcm0+5*direction,vcm0+10*direction]:
#         s_vals.append(get_sharpness(vcm))
#     ds=np.diff(np.array(s_vals))
#     momentum=momentum+np.sum(np.sign(ds))
#     momentum=limit_momentum(momentum)
# print(f'Momentum: {momentum}')
# print(f'Direction: {direction}')
# #%    
# s_vals=[]    
# if momentum > 0:
#     direction=direction
#     vcm0=vcm0+10*direction
    
#     for vcm in [vcm0,vcm0+5*direction,vcm0+10*direction]:
#         s_vals.append(get_sharpness(vcm))
#     ds=np.diff(np.array(s_vals))
#     momentum=momentum+np.sum(np.sign(ds))
#     momentum=limit_momentum(momentum)
    
# print(f'Momentum: {momentum}')
# print(f'Direction: {direction}')

        
        
# if momentum==0:

#     final_vcm=vcm0
#     print(f'final_vcm: {final_vcm}')

    

#%%
vcm_step=20
vcm0=330
maxIter=3

v_s=[] # to store vcm and sharpness value side by side
s=get_sharpness(vcm0)
v_s.append([vcm0,s])

for i in range(0,maxIter):
    
    for v in [vcm0+vcm_step//2**i,vcm0-vcm_step//2**i]:
        s=get_sharpness(v)
        v_s.append([v,s])
    
    af_arr=np.array(v_s)  # vcm,sharpness
    
    maxInd=np.argmax(af_arr[:,1])
    vcm0=int(af_arr[maxInd,0])
    print(f'sharpestVCM: {vcm0}')

#%%
# for v in [vcm0,vcm0+vcm_step,vcm0-vcm_step]:
#     s=get_sharpness(v)
#     v_s.append([v,s])

# af_arr=np.array(v_s)  # vcm,sharpness

# maxInd=np.argmax(af_arr[:,1])
# vcm0=int(af_arr[maxInd,0])

# for v in [vcm0-vcm_step//2,vcm0+vcm_step//2]:
#     s=get_sharpness(v)
#     v_s.append([v,s])
# af_arr=np.array(v_s)

# maxInd=np.argmax(af_arr[:,1])
# vcm0=int(af_arr[maxInd,0])

