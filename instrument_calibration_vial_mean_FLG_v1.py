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
from datetime import datetime

# Get the current date and time
current_datetime = datetime.now()
date = current_datetime.strftime("%Y%m%d_%H%M%S")


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

def get_dac_value(targetY,m,c):
    targetX = (targetY-c)/m
    print(f'Calibratied LED DAC value with target Intensity of {targetY} : {targetX}')
    return targetX
    


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
    
    folderName = r"W:\raspberrypi\photos\FAT_Captures\Beta\Unit-9\2025-02-07\betaFATv2B_run1"
    analysisFolder = date + '_mean_FL_LED_Calibration'
    crop = True
    cropWindow = 1500
    targetY = 625  # target intensity for calibrated LED BF
    
    resultPath = os.path.join(folderName,analysisFolder)
    
    if os.path.exists(resultPath) is False:
        os.mkdir(resultPath)
    
    fovsToAnalyze = [1] #list(np.arange(1,11))
    
    sharpnessArr = []
    slopeArr=[]
    interceptArr=[]
    maxVarArr = []
    dacArr = []
    fovArr = []   
    
    
    
    for fovCount in fovsToAnalyze:

        signalFilenames = ["img_FAT_ronchi80_green_FLG_ledPWR_250.tiff","img_FAT_ronchi80_green_FLG_ledPWR_500.tiff","img_FAT_ronchi80_green_FLG_ledPWR_750.tiff"]
        
        

        
        meanInts=[]
        meanSharpness = []
        for i in range(len(signalFilenames)):

            signalFile = os.path.join(folderName,signalFilenames[i])

            
            imSignal = read_image(signalFile)
            r,c = imSignal.shape

            
            if crop is True:
                print ('image cropping enabled')

                imSignal = imSignal[r//2-cropWindow//2: r//2 + cropWindow//2, c//2-cropWindow//2: c//2+cropWindow//2]
            
            meanSharpness.append(fm_helm(imSignal)) # calculating image sharpness    

            
            meanInt = np.mean(imSignal)
            meanInts.append(meanInt)
        
        
        
            
        X=np.array([250,500,750],dtype=float)
        Y=meanInts   
        slope, intercept = np.polyfit(X, Y, deg=1)
        slope = np.round(slope,3)
        intercept = np.round(intercept,3)
        
        plt.figure()
        plt.plot(X,Y,'o-')
        plt.plot(X,slope*X+intercept,'k--')
        plt.ylim([100,700])
        plt.xlim([X[0]-100,X[-1]+100])
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
    
     # target intensity for the calibrated LED power
    
    calibrated_dac_value = get_dac_value(targetY,slopeMedian,interceptMedian)
    
    
    slopeStd = np.std(slopeArr)/slopeMedian*100
    interceptStd = np.std(interceptArr)/interceptMedian*100
    intensityStd = np.std(maxVarArr)/intensityMedian*100
    sharpnessStd = np.std(sharpnessArr)/sharpnessMedian*100 
    
    
    dataSummary = {
        
        'type': ['Median', 'Std%'],
        'slope': [slopeMedian,slopeStd],
        'intercept': [interceptMedian,interceptStd],
        'baseline_intensity': [intensityMedian,intensityStd],
        'sharpness': [sharpnessMedian,sharpnessStd],  
        'target_intensity': [targetY,0],
        'calibrated_led_dac':[calibrated_dac_value, 0]        
        }
    
    df_final_summary = pd.DataFrame.from_dict(dataSummary)
    
    df_final_summary.to_csv(os.path.join(resultPath,date+'SUMMARY_stat_of_'+signalFilenames[-1][:10])+'.csv')
    print('SUMMARY')
    print(dataSummary)
    

