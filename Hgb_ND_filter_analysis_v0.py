# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:47:14 2022

@author: imrul
"""

import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from imageio import imread
import json

def load_img(im_path: str) -> (np.ndarray, dict):
    # XXX NOTE: Import imread from imageio. cv2 will cast single channel to 3
    # channels
    img = imread(im_path)

    # Extract custom metadata
    vital_meta = img.meta['description']
    config_meta = json.loads(vital_meta)
    meta = config_meta['config']

    return (np.asarray(img), meta)



def get_stats(imPath): # get mean and std from a crop area
    data=load_img(imPath)
    im=data[0]
    exp_ms=data[1]['ss']/1e3
    
    # plt.figure()
    # plt.title(f'Shutter_speed: {exp_ms} ms')
    # plt.imshow(im)
    
    h,w=im.shape
    
    roi={'x0':w//3,'x1':2*w//3,
         'y0':h//3,'y1':2*h//3     
         }
    
    imCrop=im[roi['y0']:roi['y1'],roi['x0']:roi['x1']]
    
    imMean,imStd,imStd_pt=np.nanmean(imCrop),np.nanstd(imCrop),np.nanstd(imCrop)/np.nanmean(imCrop)*100
    
    print(f' Mean {imMean} \n Std% {imStd_pt}')
    
    return np.array([imMean,imStd,imStd_pt,exp_ms])

def get_linear_fit(binPath,fig_num=100,key='G_BF-FOV_0'):
    stats=[]
    for file in os.listdir(binPath):
        if file.endswith('tiff') and file.find(key)>=0:
            imPath=os.path.join(binPath,file)
            stat=get_stats(imPath)
            stats.append(stat)
    
    #%%        
    stats_arr=np.array(stats)
    
    mean_arr=stats_arr[:,0]
    exp_arr=stats_arr[:,3]
    std_pt_arr=stats_arr[:,2]
    std_arr=stats_arr[:,1]
    
    # plt.figure()
    # plt.errorbar(exp_arr,mean_arr,std_arr)
    
    #%% line fit
    
    val_range=[100,1000]
    ar1=np.where(mean_arr>val_range[0])[0]
    ar2=np.where(mean_arr<val_range[1])[0]
    idx=np.intersect1d(ar1, ar2)
    
    
    
    
    x=exp_arr[idx]
    y=mean_arr[idx] 
    s=std_pt_arr[idx]       
    m, c = np.polyfit(x, y, 1,w=1/s)  
    
    plt.figure(fig_num)
    plt.title('ND Filter Analysis')
    plt.errorbar(exp_arr[idx],mean_arr[idx],std_arr[idx],fmt='o-',label=os.path.split(binPath)[-1][-30:])   
    plt.plot(exp_arr,m*exp_arr+c,'k--')
    plt.xlabel('Exposure [ms]')
    plt.ylabel('Intensity [a.u.]') 
    plt.ylim([0,2*1023])
    plt.xlim([0,5])
    plt.grid(True)
    plt.legend()
    
    return np.array([m,c])

#%%
# binPath=r"Z:\raspberrypi\photos\imrul_sandbox\Hgb\2022-08-25\run100_ND_filter_delay1s_sample_ND_201_Cyc5apollo"

# get_linear_fit(binPath)

#%%

#path=r'Z:\raspberrypi\photos\imrul_sandbox\Hgb\2022-09-02\If_50mA_gain2'
path=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\subset\550_60'
# path=r'Z:\raspberrypi\photos\imrul_sandbox\cyc5captures\2022-11-29'
params=[]

for folder in sorted(os.listdir(path)):
    print(folder)
    binPath=os.path.join(path,folder)
    param=get_linear_fit(binPath,fig_num=3000)    
    params.append(param)
#%%

target_exposure=50
OD_list=np.array([6.7,14.2,17.4])
values=[]

for param in params:
    values.append(param[0]*target_exposure+param[1])
    
plt.figure(900)
plt.title(f'Normalized to exposure {target_exposure} ms')
plt.plot(OD_list,np.array(values),'o--',label=path[-15:])
plt.xlabel('OD/Conc')
plt.ylabel('Intensity [a.u.]')
plt.legend()

#%% slope plot
params=np.array(params)
plt.figure(1500)
plt.plot(1/params[:,0],OD_list,'o--',label=path[-15:])
plt.legend()
plt.ylabel('Conc [g/dL] or OD')
plt.xlabel('1/Slope (ms)')
plt.title('Calibration Curve')
plt.grid(True)

# params=np.array(params)
plt.figure(1600)
plt.plot(params[:,0],OD_list,'o--',label=path[-15:])
plt.legend()
plt.ylabel('Conc [g/dL] or OD')
plt.xlabel('Slope (1/ms)')
plt.title('Calibration Curve')
plt.grid(True)

# plt.plot(1/params[:,0],OD_list,'o')

#%%
xx=1/params[:,0]
yy=OD_list

zz=np.polyfit(xx,yy,2)
pp=np.poly1d(zz)

xx_array=np.linspace(0.01,2.5,20)
# plt.figure()
# plt.plot(xx,yy,'o')
# plt.plot(xx_array,pp(xx_array))

#%% absorption theoretical expectations

# # I=I0*np.exp(-aL)


# plt.figure()
# plt.plot(params[:,0],10**OD_list,'o--',label=path[-15:])
# plt.legend()
# plt.ylabel('OD')
# plt.xlabel('1/Slope (ms)')
# plt.title('Calibration Curve')
# plt.grid(True)

# #%%
# plt.figure()
# plt.title(f'Normalized to exposure {target_exposure} ms')
# plt.plot(10**OD_list,np.array(values),'o--',label=f'exposure {target_exposure}')

# #%%
# a=np.arange(100)

# plt.figure()
# plt.plot(a,np.log10(a),label='log10')
# plt.plot(a,np.log(a),label='loge')
# plt.legend()

# #%% I=I0*np.exp(-Absorption) Formula
# OD_list_linear=10**OD_list
# OD_list_ln=np.log(OD_list_linear)
# plt.figure()
# plt.plot(OD_list,200*np.exp(-OD_list_ln))
# plt.plot(OD_list,400*np.exp(-OD_list_ln))
# plt.plot(OD_list,800*np.exp(-OD_list_ln))
