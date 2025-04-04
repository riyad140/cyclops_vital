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

def fm_lape(im,blur_size=1):
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
    return(fm)

def measure_sharpness_ACMO(image,hist_bins=100,hist_range=[0,256]):
    hist, edges=np.histogram(image,bins=hist_bins,range=hist_range,density=True)
    hist = np.log(hist+1)
    hist = hist / np.sum(hist)
    centers = (edges[1:]+edges[:-1])/2
    mean = np.sum(hist * centers)
    phist=np.abs(centers-mean)*hist
    return -(np.sum(phist))

#%%


def autofocus_algo_analyzer(binPath,key):
    blur_size=1
    
    ims=[]
    vcms=[]
    sharpness_laplace=[]
    sharpness_acmo=[]
    sharpness_helm=[]
    sharpness_fft=[]
    im_recon=[]
    im_orig=[]
    for file in os.listdir(binPath):
        if file.find(key)>-1 and file.endswith('png'):
            print(file)
            vcm=int(file[file.find(key)+len(key)+0:file.find(key)+len(key)+3])
            vcms.append(vcm)
            im=cv2.imread(os.path.join(binPath,file),0)
            #imCrop=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3]
            
            ims.append(im)
    
    #%%        
    for im in ims:
        im=cv2.blur(im,(blur_size,blur_size))
        t0=time.time()
        sharpness_acmo.append(measure_sharpness_ACMO(im,hist_range=[0,1024]))
        t1=time.time()
        sharpness_laplace.append(fm_lape(im))
        t2=time.time()
        sharpness_helm.append(fm_helm(im))
        t3=time.time()
        val=detect_blur_fft(im)
        sharpness_fft.append(val)
        # im_recon.append(imr)
        # im_orig.append(im)
        t4=time.time()
        
    # fig,ax=plt.subplots(4,1)
    # ax[0].plot(vcms,sharpness_fft,'o',label='fft')
    # ax[0].set_title('FFT_abs_value')
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
    # ax[3].plot(vcms,sharpness_acmo,'x',label='Acmo')
    # ax[3].set_title('ACMO_abs_value')
    # fig.suptitle(key+'_BS_'+str(blur_size))
    
    
    # fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
    # ax[0].imshow(im_orig[1],cmap='gray')
    # ax[1].imshow(np.real(im_recon[1]),cmap='gray')
    
    #%% FFT
    figPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\AutoFocus_Captures\analysis'
    param_arr=[60]
    blur_size=1
    
    result_arr=np.zeros(shape=(len(param_arr),len(vcms)),dtype=float)
    
    plt.figure()
    plt.title(f'algo_fft blur size {blur_size}')
    for count0,param in enumerate(param_arr):
        print(param)
        for count1,im in enumerate(ims):
            if blur_size>0:
                im=cv2.blur(im,(blur_size,blur_size))
            result_arr[count0,count1]=detect_blur_fft(im,size=param)
        plt.plot(vcms,result_arr[count0,:],'o',label=f'param: {param}')
    plt.legend()
    plt.grid(True)
    
    param_to_focus=60
    param_index=param_arr.index(param_to_focus)
    vcm_index=np.argmax(result_arr[param_index,:])
    print(f'Sharpest VCM for AlgoFFT with parameter {param_to_focus} is {vcms[vcm_index]}')
    
    plt.figure()
    plt.title(f'AlgoFFT with parameter {param_to_focus}\n{key} vcm: {vcms[vcm_index]}')
    plt.imshow(ims[vcm_index])
    
    figName=f'AlgoFFT_{key}vcm_{vcms[vcm_index]}_{param_to_focus}.png'
    plt.savefig(os.path.join(figPath,figName))
    
    #%% HELM
    
    param_arr=[11,31]
    blur_size=1
    
    result_arr1=np.zeros(shape=(len(param_arr),len(vcms)),dtype=float)
    
    plt.figure()
    plt.title(f'algo_helm blur size {blur_size}')
    for count0,param in enumerate(param_arr):
        print(param)
        for count1,im in enumerate(ims):
            if blur_size>0:
                im=cv2.blur(im,(blur_size,blur_size))
            result_arr1[count0,count1]=fm_helm(im,WSIZE=param)
        plt.plot(vcms,result_arr1[count0,:],'o',label=f'param: {param}')
    plt.legend()
    plt.grid(True)
    
    param_to_focus=31
    param_index=param_arr.index(param_to_focus)
    vcm_index=np.argmax(result_arr1[param_index,:])
    print(f'Sharpest VCM for AlgoHELM with parameter {param_to_focus} is {vcms[vcm_index]}')
    
    plt.figure()
    plt.title(f'AlgoHELM with parameter {param_to_focus}\n{key} vcm: {vcms[vcm_index]}')
    plt.imshow(ims[vcm_index])
    figName=f'AlgoHelm_{key}vcm_{vcms[vcm_index]}_{param_to_focus}.png'
    plt.savefig(os.path.join(figPath,figName))


#%% ACMO
# param_arr=[25,50,100]
# blur_size=31

# result_arr2=np.zeros(shape=(len(param_arr),len(vcms)),dtype=float)

# plt.figure()
# plt.title(f'algo_acmo blur size {blur_size}')
# for count0,param in enumerate(param_arr):
#     print(param)
#     for count1,im in enumerate(ims):
#         if blur_size>0:
#             im=cv2.blur(im,(blur_size,blur_size))
#         result_arr2[count0,count1]=measure_sharpness_ACMO(im,hist_bins=param)
#     plt.plot(vcms,result_arr2[count0,:],'o',label=f'param: {param}')
# plt.legend()
# plt.grid(True)

# #%% Lapl
# param_arr=[5,11,21,31]
# blur_size=0

# result_arr3=np.zeros(shape=(len(param_arr),len(vcms)),dtype=float)

# plt.figure()
# plt.title(f'algo_lapl blur size {blur_size}')
# for count0,param in enumerate(param_arr):
#     print(param)
#     for count1,im in enumerate(ims):
#         if blur_size>0:
#             im=cv2.blur(im,(blur_size,blur_size))
#         result_arr3[count0,count1]=fm_lape(im,blur_size=param)
#     plt.plot(vcms,result_arr3[count0,:],'o',label=f'param: {param}')
# plt.legend()
# plt.grid(True)
if __name__ == '__main__':
    
    binPath=r'Z:\raspberrypi\photos\imrul_sandbox\2022-07-28\run300_plt_bs1_focusStack_sample_S920_RBC_Plt_Con_Cyc5Diana\AF'
    channel=1 # BGR
    key='AF_1111'
    keys=['AF_0_','AF_1_','AF_2_','AF_3_','AF_4_','AF_5_','AF_6_','AF_7_','AF_8_','AF_9_']
    for key in keys:
        autofocus_algo_analyzer(binPath,key)