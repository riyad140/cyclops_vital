# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:57:52 2024

@author: imrul

For Alpha Sharps with beads
"""

import sys
import numpy as np
import os

import skimage.segmentation as ss#
import skimage.filters as sf#
import skimage.morphology as sm#
import skimage.util as su#
import skimage.exposure as se#
from skimage import io#
import skimage.measure as sms#
import skimage.transform as st
import skimage.color as sc
import scipy.stats as stats
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
from scipy import ndimage, signal
import pandas as pd
import cv2
import datetime

def fm_helm(image,WSIZE=21):
    u=cv2.blur(image,(WSIZE,WSIZE))
    r1=u/image
    r1[image==0]=1
    index = (u>image)
    fm=1/r1
    fm[index]=r1[index]
    fm=np.mean(fm)
    return(fm-1)

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

def detect_cells_cyclops5(img,th_block=81,th_offset=-0.8,th_size=500):
    t=sf.threshold_local(img,th_block,offset=th_offset)
    img_t=img>t
    img_tf=ndi.binary_fill_holes(img_t)
    img_ts=sm.remove_small_objects(img_tf,th_size)
    labels_cells=sm.label(img_ts)
    # print(f"cell count")
    # plt.figure()
    # plt.imshow(labels_cells)
    return(labels_cells)

def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region], ddof=1)

def image_median(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.median(intensities[region])

def display_raw_histogram(imProbe,imSignal,showFigure=False):

    solidity_cutoff = 0.6
    
    
    meanint_bg = np.array([])
    meanint_ss = np.array([])
    stdint_ss = np.array([])
    medianint_ss = np.array([])
    
    im_g = imProbe          #read_image(probe_filename)
    labels_g = detect_cells_cyclops5(im_g,41,-5,100)
    labels_bg=ss.expand_labels(labels_g, distance=3)-labels_g
    
    im_ss = imSignal #read_image(signal_filename)
    
    regprops_ss = sms.regionprops(labels_g,im_ss,extra_properties=[image_stdev,image_median])
    
    solidity = np.array([r['solidity'] for r in regprops_ss])
    mean_intensity_ss = np.array([r['mean_intensity'] for r in regprops_ss])
    standard_dev = np.array([r['image_stdev'] for r in regprops_ss])
    median_intensity_ss = np.array([r['image_median'] for r in regprops_ss])
    
    
    
    if showFigure is True:
        fig, ax = plt.subplots(1)
        ax.imshow(im_ss)
        ax.contour(labels_g>0,1,colors='r',)
        fig, ax = plt.subplots(1)
        cellHistogram = ax.hist(mean_intensity_ss,100)
        ax.set_xlim([100,900])
    #ax[1,0].hist(median_intensity_ss,60)
    #ax[1,1].hist(standard_dev,60)
    else:
        plt.figure(36)
        cellHistogram = plt.hist(mean_intensity_ss,100)
    xHist = cellHistogram[1][1:]
    yHist = cellHistogram[0]   

    return xHist,yHist

def histogram_analysis(xHist,yHist,figTitle='gg'):
    yHist1= ndimage.gaussian_filter1d(yHist,1) # smoothing the histogram
    peaks, _ = signal.find_peaks(yHist1,prominence=5,distance=100)
    peakWidths = signal.peak_widths(yHist1, peaks, rel_height=0.5)[0]
    meanInt= np.round(xHist[peaks][0],1)
    meanFreq = np.round(yHist[peaks][0],0)
    peakWidth = np.round(peakWidths[0]*np.mean(np.diff(xHist)),0)
    # plt.figure()
    # plt.plot(xHist,yHist1)
    # plt.plot(xHist[peaks],yHist1[peaks],'ro')
    # plt.xlim([20,900])
    # plt.title(f"{figTitle}")
    print(f'Mean Intensity {meanInt} with Frequency of {meanFreq} and peak width of {peakWidth}')
    
    return meanInt,meanFreq,peakWidth


# targetY = 240
# m = 0.17 #0.342
# c = 84.6 #140.1

# targetX = (targetY-c)/m

# print(targetX)

def equalized_LED_DAC(slope,intercept,targetY):
    
    return (targetY-intercept)/slope



# def get_calibration(filenames_FLG, filenames_DF,fovsToAnalyze = [1,2,3,4,5]):
#     sharpnessArr = []
#     slopeArr=[]
#     interceptArr=[]
#     maxVarArr = []
#     dacArr = []
#     fovArr = []
    
    

#%%
if __name__=='__main__':
    
    module_id = 'B009'
    folderName = r"W:\raspberrypi\photos\FAT_Captures\Beta\Unit-9\2025-04-02\ps_beads_0"
    date = datetime.date.today().strftime("%Y-%m-%d")
    crop = True
    cropWindow = 1500
    
    resultPath = os.path.join(folderName,date)
    
    if os.path.exists(resultPath) is False:
        os.mkdir(resultPath)
    
    fovsToAnalyze = list(np.arange(1,11))
    
    sharpnessArr_DF = []
    slopeArr_DF=[]
    interceptArr_DF=[]
    maxVarArr_DF = []
    dacArr_DF = []
    fovArr_DF = []
    
    
    sharpnessArr_FLG = []
    slopeArr_FLG =[]
    interceptArr_FLG =[]
    maxVarArr_FLG = []
    dacArr_FLG = []
    fovArr_FLG = []
    
    for fovCount in fovsToAnalyze:
        probeFilenames=[f"img_FAT_beads_green_FLG_ledPWR_250_fov{fovCount}.tiff", f"img_FAT_beads_green_FLG_ledPWR_500_fov{fovCount}.tiff",f"img_FAT_beads_green_FLG_ledPWR_750_fov{fovCount}.tiff",f"img_FAT_beads_green_FLG_ledPWR_1000_fov{fovCount}.tiff"]
        signalFilenames = [f"img_FAT_beads_blue_DF_ledPWR_250_fov{fovCount}.tiff", f"img_FAT_beads_blue_DF_ledPWR_500_fov{fovCount}.tiff",f"img_FAT_beads_blue_DF_ledPWR_750_fov{fovCount}.tiff",f"img_FAT_beads_blue_DF_ledPWR_1000_fov{fovCount}.tiff"]
        #probeFilenames
        #[f"img_blue_DF_fov{fovCount}_intensityB_250.tiff", f"img_blue_DF_fov{fovCount}_intensityB_500.tiff",f"img_blue_DF_fov{fovCount}_intensityB_750.tiff",f"img_blue_DF_fov{fovCount}_intensityB_1000.tiff"]

        #[f"img_green_FLG_fov{fovCount}_intensityB_250.tiff", f"img_green_FLG_fov{fovCount}_intensityB_500.tiff",f"img_green_FLG_fov{fovCount}_intensityB_750.tiff",f"img_green_FLG_fov{fovCount}_intensityB_1000.tiff"]
    
        
        meanInts_FLG=[]
        meanSharpness_FLG = []
        
        meanInts_DF = []
        meanSharpness_DF = []
        
        for i in range(len(signalFilenames)):
            probeFile = os.path.join(folderName,probeFilenames[i])
            signalFile = os.path.join(folderName,signalFilenames[i])
            # print(probeFile)
            
            imProbe = read_image(probeFile)
            imSignal = read_image(signalFile)
            r,c = imProbe.shape
            
            # if probeFile == signalFile:
            #     print('signal and probe files are the same')
            #     imSignal = np.copy(imProbe)
            # else:
            #     imSignal = read_image(signalFile)
            
            if crop is True:
                print ('image cropping enabled')
                imProbe = imProbe[r//2-cropWindow//2: r//2 + cropWindow//2, c//2-cropWindow//2: c//2+cropWindow//2]
                imSignal = imSignal[r//2-cropWindow//2: r//2 + cropWindow//2, c//2-cropWindow//2: c//2+cropWindow//2]
            
            meanSharpness_FLG.append(fm_helm(imProbe)) # calculating image sharpness   
            meanSharpness_DF.append(fm_helm(imSignal))
            
            xHist,yHist = display_raw_histogram(imProbe, imSignal)  # DF
            # xHist,yHist = get_histogram(signalFile)
            meanInt,meanFreq,peakWidth=histogram_analysis(xHist,yHist,figTitle=f'gDF_{i}')
            meanInts_DF.append(meanInt)
            
            
            xHist,yHist = display_raw_histogram(imProbe, imProbe)  # FLG
            # xHist,yHist = get_histogram(signalFile)
            meanInt,meanFreq,peakWidth=histogram_analysis(xHist,yHist,figTitle=f'gFL_{i}')
            meanInts_FLG.append(meanInt)
        
        
        
            
        X=np.array([250,500,750,1000],dtype=float)
        Y_DF = meanInts_DF   
        slope_DF, intercept_DF = np.polyfit(X, Y_DF, deg=1)
        slope_DF = np.round(slope_DF,3)
        intercept_DF = np.round(intercept_DF,3)
        
        plt.figure()
        plt.plot(X,Y_DF,'o-')
        plt.plot(X,slope_DF*X+intercept_DF,'k--')
        plt.ylim([100,1000])
        plt.xlim([200,1050])
        plt.grid(True)
        plt.title(os.path.split(folderName)[-1]+'\n'+signalFilenames[-1][0:-10]+'\n'+f" slope {slope_DF} intercept {intercept_DF}") 
        plt.xlabel("LED DAC Value")
        plt.ylabel("Mean Intensity [a.u.]")
        plt.savefig(os.path.join(resultPath,f"DF_calibration_of_{fovCount}"+signalFilenames[-1][0:-10])+'.png')
        
        print(f" slope {slope_DF} intercept {intercept_DF}")
        

        Y_FLG = meanInts_FLG   
        slope_FLG, intercept_FLG = np.polyfit(X, Y_FLG, deg=1)
        slope_FLG = np.round(slope_FLG,3)
        intercept_FLG = np.round(intercept_FLG,3)
        
        plt.figure()
        plt.plot(X,Y_FLG,'o-')
        plt.plot(X,slope_FLG*X+intercept_FLG,'k--')
        plt.ylim([100,1000])
        plt.xlim([200,1050])
        plt.grid(True)
        plt.title(os.path.split(folderName)[-1]+'\n'+signalFilenames[-1][0:-10]+'\n'+f" slope {slope_FLG} intercept {intercept_FLG}") 
        plt.xlabel("LED DAC Value")
        plt.ylabel("Mean Intensity [a.u.]")
        plt.savefig(os.path.join(resultPath,f"FL_calibration_of_{fovCount}"+probeFilenames[-1][0:-10])+'.png')
        
        print(f" slope {slope_FLG} intercept {intercept_FLG}")
        
############################# WIP ######################################
        slopeArr_DF.append(slope_DF)
        interceptArr_DF.append(intercept_DF)
        maxVarArr_DF.append(Y_DF[1])
        dacArr_DF.append(X[1])
        fovArr_DF.append(fovCount)
        sharpnessArr_DF.append(np.mean(meanSharpness_DF))
        
        slopeArr_FLG.append(slope_FLG)
        interceptArr_FLG.append(intercept_FLG)
        maxVarArr_FLG.append(Y_FLG[1])
        dacArr_FLG.append(X[1])
        fovArr_FLG.append(fovCount)
        sharpnessArr_FLG.append(np.mean(meanSharpness_FLG))
#%

    data={'fov':fovArr_DF,
          'slope_df':slopeArr_DF,
          'intercept_df':interceptArr_DF,
          'led_dac_value_df': dacArr_DF,
          'baseline_intensity_df': maxVarArr_DF,
          'mean_sharpness_df': sharpnessArr_DF,
          'slope_flg':slopeArr_FLG,
          'intercept_flg':interceptArr_FLG,
          'led_dac_value_flg': dacArr_FLG,
          'baseline_intensity_flg': maxVarArr_FLG,
          'mean_sharpness_flg': sharpnessArr_FLG,
          
          }
    
    df_final = pd.DataFrame.from_dict(data)
    
    df_final.to_csv(os.path.join(resultPath,date+'_'+ module_id  +'_all_fov_stat.csv'))
    
    slopeMedian_DF = np.median(slopeArr_DF)
    interceptMedian_DF = np.median(interceptArr_DF)
    intensityMedian_DF = np.median(maxVarArr_DF)
    sharpnessMedian_DF = np.median(sharpnessArr_DF)
    
    
    slopeStd_DF = np.std(slopeArr_DF)/slopeMedian_DF*100
    interceptStd_DF = np.std(interceptArr_DF)/interceptMedian_DF*100
    intensityStd_DF = np.std(maxVarArr_DF)/intensityMedian_DF*100
    sharpnessStd_DF = np.std(sharpnessArr_DF)/sharpnessMedian_DF*100 
    
    
    target_DF = 400
    calibrated_DF = (target_DF- interceptMedian_DF)/ slopeMedian_DF
    
    
    
    
    
    slopeMedian_FLG = np.median(slopeArr_FLG)
    interceptMedian_FLG = np.median(interceptArr_FLG)
    intensityMedian_FLG = np.median(maxVarArr_FLG)
    sharpnessMedian_FLG = np.median(sharpnessArr_FLG)
    
    
    slopeStd_FLG = np.std(slopeArr_FLG)/slopeMedian_FLG*100
    interceptStd_FLG = np.std(interceptArr_FLG)/interceptMedian_FLG*100
    intensityStd_FLG = np.std(maxVarArr_FLG)/intensityMedian_FLG*100
    sharpnessStd_FLG = np.std(sharpnessArr_FLG)/sharpnessMedian_FLG*100 
    
    
    target_FLG = 800
    calibrated_FLG = (target_FLG- interceptMedian_FLG)/ slopeMedian_FLG
    
    
    
    
    
    
    
    
    dataSummary = {
        
        'type': ['Median', 'Std%'],
        'slope_df': [slopeMedian_DF,slopeStd_DF],
        'intercept_df': [interceptMedian_DF,interceptStd_DF],
        'baseline_intensity_df': [intensityMedian_DF,intensityStd_DF],
        'sharpness_df': [sharpnessMedian_DF,sharpnessStd_DF],
        'target_intensity_df': [target_DF,0],
        'calibrated_dac_df': [calibrated_DF,0],
        
        'slope_flg': [slopeMedian_FLG,slopeStd_FLG],
        'intercept_flg': [interceptMedian_FLG,interceptStd_FLG],
        'baseline_intensity_flg': [intensityMedian_FLG,intensityStd_FLG],
        'sharpness_flg': [sharpnessMedian_FLG,sharpnessStd_FLG],
        'target_intensity_flg': [target_FLG,0],
        'calibrated_dac_flg': [calibrated_FLG,0],        
        
        }
    
    df_final_summary = pd.DataFrame.from_dict(dataSummary)
    
    df_final_summary.to_csv(os.path.join(resultPath,date+'_'+ module_id  +'_MASTER_SUMMARY_stat.csv'))
    print('SUMMARY')
    print(dataSummary)
    
    #%%
        #%% 800 for FLG and 400 for DF
    
    # if probeFilenames == signalFilenames:    
    #     targetY = 800
    # else:
    #     targetY = 400
    # m = slopeMedian #0.342
    # c = interceptMedian #140.1
    
    # targetX = (targetY-c)/m
    
    # print(targetX)
    