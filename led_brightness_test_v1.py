# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:59:53 2023

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle, Polygon, Circle

#from ROI_manager_user import userROI
import os
#import sys
#import cv2
import math
from scipy import ndimage, signal
#from ROI_manager_user import userROI
import pandas as pd
import time


#%%

binPath=r'Z:\raspberrypi\photos\Alpha_plus\CYC7_A4\2024-01-19\blankRun01'
#r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2023-03-27_Volpi_Visit\Cyc7_A2\BC_Ctrl_normal_run04'
#r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2023-03-27_Volpi_Visit\Cyc7_A2\PS_beads_run04_df_allblue'
#binPath=r'Z:\raspberrypi\photos\FAT_Captures\cyc6Dionysus\2023-02-09\run100_FAT5b_sample_Ronchi-1_Red_puck_dionysus'
# binPath=r'\\files.vital.company\cyclops\raspberrypi\photos\Erics Sandbox\2022-10-04_RONCHI_CollimatedvsDiffused\run05_FOV2_Diffused_exp10000_012A_sample_SNA_RONCHI__Cyc5Artemis'

keys=['fov1_','fov2_','fov3_','fov4_','fov5_','fov6_','fov7_','fov8_','fov9_','fov10_']
# zoom_window=250 # sets the window of the zoom for the ROI selectoin

# resultPath=os.path.join(binPath,f'analysis_final_{key}')

# ts=str(int(np.round(time.time(),0)))
# resultPath=resultPath+'_'+ts
# try:
#     os.mkdir(resultPath)
# except:
#     pass

slopeArr=[]
interceptArr=[]
maxVarArr = []

for key in keys:
    files=[]
    for file in os.listdir(binPath):
        if file.find(key)>-1 and file.endswith('tiff'):
            print(file)
            files.append(os.path.join(binPath,file))
    #         im=plt.imread(os.path.join(binPath,file))
    #         ims.append(im)
    #%%        reading the images after fov sorting
    files.sort(key=os.path.getctime)  # sort by date created
    # dfiles=files[-6:]
    
    # for file in files[:-6]:
    #     dfiles.append(file)
    # files=dfiles
    # im_main=ims[0]
    meanStats=[]
    ims=[]
    for count,file in enumerate(files):
        print(file)
        im=plt.imread(file)
        imCrop=im[im.shape[0]*1//3:im.shape[0]*2//3,im.shape[1]*1//3:im.shape[1]*2//3]
        avgInt=np.nanmedian(imCrop)
        stdInt=np.nanstd(imCrop)
        
        meanStats.append([avgInt,stdInt])
     #%% analysis
     
    stats=np.array(meanStats)
    
    # plt.figure()
    # plt.plot(stats[:,0],'o-')   
    # plt.xlabel('nFOV')
    # plt.ylabel('AvgInt')
    # plt.title(key)
    
    varInt = (np.max(stats[:,0])-np.min(stats[:,0]))/np.mean(stats[:,0])*100
    
    print(f"Maximum intensity variation for {key} : {varInt} %")
    
    
    #%% This is just for HGB FAT analysis
    expList = np.array([300, 500, 1000, 2000, 3000, 4000, 6000])
    intList = stats[:,0]
    
    
    
    validIndex = np.where(intList<1000)[0]
    
    xx = expList[validIndex]
    yy = intList[validIndex]
    
    plt.figure()
    plt.plot(xx,yy,'o-')
    plt.xlabel('Exposure')
    plt.ylabel('Intensity')
    plt.title(key)
    plt.savefig(os.path.join(binPath,key)+'.png')
    slope,intercept = np.polyfit(xx,yy,1)
    
    print(f'Slope : {slope} \nIntercept : {intercept}')
    
    slopeArr.append(slope)
    interceptArr.append(intercept)
    maxVarArr.append(varInt)
#%%

data={'slope':slopeArr,
      'intercept':interceptArr,
      'percentVariation': maxVarArr
      }

df_final = pd.DataFrame.from_dict(data)

df_final.to_csv(os.path.join(binPath,'all_fov_stat')+'.csv')



