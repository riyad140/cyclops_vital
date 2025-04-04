# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:31:17 2024

@author: imrul
"""

import sys
import numpy as np
import os

import matplotlib.pyplot as plt
from scipy import ndimage, signal
import pandas as pd
import cv2


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

def equalized_LED_DAC(slope,intercept,targetY):
    
    return (targetY-intercept)/slope


def fm_helm(image,WSIZE=21):
    u=cv2.blur(image,(WSIZE,WSIZE))
    r1=u/image
    r1[image==0]=1
    index = (u>image)
    fm=1/r1
    fm[index]=r1[index]
    fm=np.mean(fm)
    return(fm-1)

#%%

if __name__=='__main__':
    
    folderName = r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\Instrument_Calibration\2024-02-01\FAT_brightness_calibration_Cyc7-A02_ps_beads_WBC_run001_LRV2.13"
    date = "20240426"
    crop = True
    cropWindow = 1500
    
    resultPath = os.path.join(folderName,date)
    
    if os.path.exists(resultPath) is False:
        os.mkdir(resultPath)
    
    fovsToAnalyze = list(np.arange(1,11))
    
    sharpnessArr = []
    slopeArr=[]
    interceptArr=[]
    maxVarArr = []
    dacArr = []
    fovArr = []
    
    for fovCount in fovsToAnalyze:
        # probeFilenames=[f"img_green_FLG_fov{fovCount}_intensityB_250.tiff", f"img_green_FLG_fov{fovCount}_intensityB_500.tiff",f"img_green_FLG_fov{fovCount}_intensityB_750.tiff",f"img_green_FLG_fov{fovCount}_intensityB_1000.tiff"]
        # signalFilenames =[f"img_blue_DF_fov{fovCount}_intensityB_250.tiff", f"img_blue_DF_fov{fovCount}_intensityB_500.tiff",f"img_blue_DF_fov{fovCount}_intensityB_750.tiff",f"img_blue_DF_fov{fovCount}_intensityB_1000.tiff"]
        
        signalFilenames = [f"img_blank_BF_fov{fovCount}_intensityG_100.tiff",f"img_blank_BF_fov{fovCount}_intensityG_150.tiff"]
        
        
        #probeFilenames
        #[f"img_blue_DF_fov{fovCount}_intensityB_250.tiff", f"img_blue_DF_fov{fovCount}_intensityB_500.tiff",f"img_blue_DF_fov{fovCount}_intensityB_750.tiff",f"img_blue_DF_fov{fovCount}_intensityB_1000.tiff"]

        #[f"img_green_FLG_fov{fovCount}_intensityB_250.tiff", f"img_green_FLG_fov{fovCount}_intensityB_500.tiff",f"img_green_FLG_fov{fovCount}_intensityB_750.tiff",f"img_green_FLG_fov{fovCount}_intensityB_1000.tiff"]
    
        
        meanInts=[]
        meanSharpness = []
        for i in range(len(signalFilenames)):
            # probeFile = os.path.join(folderName,probeFilenames[i])
            signalFile = os.path.join(folderName,signalFilenames[i])
            # print(probeFile)
            
            imSignal = read_image(signalFile)
            r,c = imSignal.shape
            
            # if probeFile == signalFile:
            #     print('signal and probe files are the same')
            #     imSignal = np.copy(imProbe)
            # else:
            #     imSignal = read_image(signalFile)
            
            if crop is True:
                print ('image cropping enabled')
                # imProbe = imProbe[r//2-cropWindow//2: r//2 + cropWindow//2, c//2-cropWindow//2: c//2+cropWindow//2]
                imSignal = imSignal[r//2-cropWindow//2: r//2 + cropWindow//2, c//2-cropWindow//2: c//2+cropWindow//2]
            
            meanSharpness.append(fm_helm(imSignal)) # calculating image sharpness    
            
            # xHist,yHist = display_raw_histogram(imProbe, imSignal)
            # # xHist,yHist = get_histogram(signalFile)
            # meanInt,meanFreq,peakWidth=histogram_analysis(xHist,yHist,figTitle=f'gDF_{i}')
            
            meanInt = np.median(imSignal)
            meanInts.append(meanInt)
        
        
        
            
        X=np.array([100,150],dtype=float)
        Y=meanInts   
        slope, intercept = np.polyfit(X, Y, deg=1)
        slope = np.round(slope,3)
        intercept = np.round(intercept,3)
        
        plt.figure()
        plt.plot(X,Y,'o-')
        plt.plot(X,slope*X+intercept,'k--')
        # plt.ylim([100,500])
        # plt.xlim([200,1050])
        plt.grid(True)
        plt.title(os.path.split(folderName)[-1]+'\n'+signalFilenames[-1][:-20]+'\n'+f" slope {slope} intercept {intercept}") 
        plt.xlabel("LED DAC Value")
        plt.ylabel("Mean Intensity [a.u.]")
        plt.savefig(os.path.join(resultPath,f"calibration_of_{fovCount}"+signalFilenames[-1][-30:-20])+'.png')
        
        print(f" slope {slope} \n intercept {intercept}")
    
        slopeArr.append(slope)
        interceptArr.append(intercept)
        maxVarArr.append(Y[1])
        dacArr.append(X[1])
        fovArr.append(fovCount)
        sharpnessArr.append(np.mean(meanSharpness))
#%

    data={'fov':fovArr,
          'slope':slopeArr,
          'intercept':interceptArr,
          'led_dac_value': dacArr,
          'baseline_intensity': maxVarArr,
          'mean_sharpness': sharpnessArr
          }
    
    df_final = pd.DataFrame.from_dict(data)
    
    df_final.to_csv(os.path.join(resultPath,date+'_all_fov_stat_of_'+signalFilenames[-1][:10])+'.csv')
    
    slopeMedian = np.median(slopeArr)
    interceptMedian = np.median(interceptArr)
    intensityMedian = np.median(maxVarArr)
    sharpnessMedian = np.median(sharpnessArr)
    
    
    slopeStd = np.std(slopeArr)/slopeMedian*100
    interceptStd = np.std(interceptArr)/interceptMedian*100
    intensityStd = np.std(maxVarArr)/intensityMedian*100
    sharpnessStd = np.std(sharpnessArr)/sharpnessMedian*100 
    
    
    dataSummary = {
        
        'type': ['Median', 'Std%'],
        'slope': [slopeMedian,slopeStd],
        'intercept': [interceptMedian,interceptStd],
        'baseline_intensity': [intensityMedian,intensityStd],
        'sharpness': [sharpnessMedian,sharpnessStd]       
        
        }
    
    df_final_summary = pd.DataFrame.from_dict(dataSummary)
    
    df_final_summary.to_csv(os.path.join(resultPath,date+'SUMMARY_stat_of_'+signalFilenames[-1][:10])+'.csv')
    print('SUMMARY')
    print(dataSummary)
    
    #%% Equalized DAC
    targetY = 800
    m = slopeMedian#0.342
    c = interceptMedian #140.1
    
    targetX = (targetY-c)/m
    
    print(targetX)