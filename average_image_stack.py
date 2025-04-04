# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:59:01 2021

@author: imrul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
#%%

def average_image(filenames,dst_filename='avg.png'):
    ims=[]
    for file in filenames:
        im=cv2.imread(file,1)
        ims.append(im)
    ims=np.array(ims)
    avgIm=np.nanmean(ims,axis=0).astype(np.uint8) 
    cv2.imwrite(dst_filename,avgIm)
    return avgIm,im
#%%

def selective_average_image(src_path,key):


    print(key)
    filepaths=[]
    for file in os.listdir(src_path):
        if file.endswith('bmp') and file.find(key)>=0:
            filepaths.append(os.path.join(src_path,file))
    filepaths.sort(key=os.path.getctime)        # sorting by date modified
    dst_path=os.path.join(src_path,'avg')
    if os.path.exists(dst_path) is False:
        os.mkdir(dst_path)
    dst_filename=os.path.join(dst_path,key+'AVG.bmp')    
    avgIm,im=average_image(filepaths,dst_filename=dst_filename)
    
    
    fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
    ax[0].imshow(avgIm)
    ax[0].set_title('Averaged Image')
    ax[1].imshow(im)
    ax[1].set_title('Single Image')

#%%
src_path=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2021-01-26_SensorCharacterization\ar1335\vcm360_exp_m4'

filepaths=[]
for filename in os.listdir(src_path):
    if filename.endswith('bmp'):
        filepaths.append(os.path.join(src_path,filename))
        
im=average_image(filepaths,os.path.join(src_path,'avg.bmp'))        
# key='IMX258BF_exp25_analogGain0_vcm330_'



# selective_average_image(src_path,key)
#%%
fname=[]
for file in os.listdir(src_path):
    if file.endswith('.bmp'):
        print(file[:-5])
        fname.append(file[:-5])
fname=np.array(fname)
fname_u=np.unique(fname)        

keys=list(fname_u)
 
for key in keys:
    selective_average_image(src_path,key)
    



# # #%%
# # ims=[]
# # for file in filepaths[-5:]:
# #     im=cv2.imread(file,1)
# #     ims.append(im)
# # ims=np.array(ims)
# # avgIm=np.nanmean(ims,axis=0).astype(np.uint8)


# # plt.figure()
# # plt.imshow(avgIm)
# #%%
# src_path=r'Z:\raspberrypi\photos\Misc\20211208_imx477_linearity_test\run00_sample_linearity_test_cyc4Stable'
# nAvg=10
# exposure_ms=[]
# dst_path=os.path.join(src_path,'Average')
# if os.path.isdir(dst_path) is False:
#     os.mkdir(dst_path)

# for i in range(len(filepaths)):
#     if i%nAvg==0:
#         print(i)
#         # print(filepaths[i])
#         filename=filepaths[i][filepaths[i].find('cyclops-'):]
#         exposure=np.float16(filename[filename.find('exp_')+4:filename.find('_ms')])
#         exposure_ms.append(exposure)
#         dst_filename=os.path.join(dst_path,'avg_'+filename)
#         avg=average_image(filepaths[i:i+nAvg],dst_filename)
# #%%
# h,w,c=3040,4056,3
# const=501 # number of chunks
# roi={'x0':(const//2)*w//const,'x1':(const//2+1)*w//const,'y0':(const//2)*h//const,'y1':(const//2+1)*h//const}
# expArr=[]
# mArr=[]
# sArr=[]

# for filename in os.listdir(dst_path):
#     print(filename)
#     im=cv2.imread(os.path.join(dst_path,filename))
#     exposure=np.float16(filename[filename.find('exp_')+4:filename.find('_ms')])
#     m=np.nanmean(im[roi['y0']:roi['y1'],roi['x0']:roi['x1'],1])
#     s=np.nanstd(im[roi['y0']:roi['y1'],roi['x0']:roi['x1'],1])
#     mArr.append(m)
#     sArr.append(s)
#     expArr.append(exposure)
# #%%
# slope,intercept=np.polyfit(expArr,mArr,1)
# fit=slope*np.array(expArr)+intercept

# plt.figure()
# plt.errorbar(expArr, mArr, sArr,label='data')
# plt.plot(expArr,fit,label='fit')
# plt.xlabel('exposure (ms)') 
# plt.ylabel('Pixel Intensity (a.u.)')   
# plt.grid(True)
# plt.legend()
# #%% draw bounding box
# imROI= cv2.rectangle(im, (roi['x0'],roi['y0']), (roi['x1'],roi['y1']), color=(255,0,0), thickness=2)
# plt.figure()
# plt.imshow(imROI)
