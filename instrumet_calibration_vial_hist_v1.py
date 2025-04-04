# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:57:52 2024

@author: imrul

with betaFATV2B ronchi80 target
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

def display_raw_histogram(imProbe,imSignal,showFigure=True):

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
    peaks, _ = signal.find_peaks(yHist1,prominence=0.004,distance=100)
    
    if len(peaks) !=2:
        print('Two peaks not found')
    
    peakWidths = signal.peak_widths(yHist1, peaks, rel_height=0.5)[0]
    meanIntLo= np.round(xHist[peaks][0],1)
    meanFreqLo = np.round(yHist[peaks][0],0)
    peakWidthLo = np.round(peakWidths[0]*np.mean(np.diff(xHist)),0)
    
    try:
        meanIntHi= np.round(xHist[peaks][1],1)
        meanFreqHi = np.round(yHist[peaks][1],0)
        peakWidthHi = np.round(peakWidths[1]*np.mean(np.diff(xHist)),0)
    except:
        print('Image is too bright')
        meanIntHi = np.round(xHist[np.argmax(yHist)],1)
    # plt.figure()
    # plt.plot(xHist,yHist1)
    # plt.plot(xHist[peaks],yHist1[peaks],'ro')
    # plt.xlim([20,900])
    # plt.title(f"{figTitle}")
    # print(f'Mean Intensity {meanIntHi} with Frequency of {meanFreqHi} and peak width of {peakWidthHi}')
    
    return meanIntHi,meanIntLo


# targetY = 240
# m = 0.17 #0.342
# c = 84.6 #140.1

# targetX = (targetY-c)/m

# print(targetX)

def equalized_LED_DAC(slope,intercept,targetY):
    
    return (targetY-intercept)/slope

#%%
if __name__=='__main__':
    
    folderName = r"W:\raspberrypi\photos\FAT_Captures\BetaFAT_v2\AS3\2024-09-19\betaFATv2A_run00"
    date = "20240710_hist"
    crop = True
    cropWindow = 1500
    
    resultPath = os.path.join(folderName,date)
    
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
        probeFilenames=["img_FAT_ronchi80_green_FLG_ledPWR_250.tiff","img_FAT_ronchi80_green_FLG_ledPWR_500.tiff","img_FAT_ronchi80_green_FLG_ledPWR_750.tiff"]
        signalFilenames = probeFilenames
        #[f"img_FAT_beads_blue_DF_ledPWR_250.tiff", f"img_FAT_beads_blue_DF_ledPWR_500.tiff",f"img_FAT_beads_blue_DF_ledPWR_750.tiff",f"img_FAT_beads_blue_DF_ledPWR_1000.tiff"]
        #probeFilenames
        #[f"img_blue_DF_fov{fovCount}_intensityB_250.tiff", f"img_blue_DF_fov{fovCount}_intensityB_500.tiff",f"img_blue_DF_fov{fovCount}_intensityB_750.tiff",f"img_blue_DF_fov{fovCount}_intensityB_1000.tiff"]

        #[f"img_green_FLG_fov{fovCount}_intensityB_250.tiff", f"img_green_FLG_fov{fovCount}_intensityB_500.tiff",f"img_green_FLG_fov{fovCount}_intensityB_750.tiff",f"img_green_FLG_fov{fovCount}_intensityB_1000.tiff"]
    
        
        meanInts=[]
        meanSharpness = []
        for i in range(len(signalFilenames)):
            probeFile = os.path.join(folderName,probeFilenames[i])
            signalFile = os.path.join(folderName,signalFilenames[i])
            # print(probeFile)
            
            imProbe = read_image(probeFile)
            r,c = imProbe.shape
            
            if probeFile == signalFile:
                print('signal and probe files are the same')
                imSignal = np.copy(imProbe)
            else:
                imSignal = read_image(signalFile)
            
            if crop is True:
                print ('image cropping enabled')
                imProbe = imProbe[r//2-cropWindow//2: r//2 + cropWindow//2, c//2-cropWindow//2: c//2+cropWindow//2]
                imSignal = imSignal[r//2-cropWindow//2: r//2 + cropWindow//2, c//2-cropWindow//2: c//2+cropWindow//2]
            
            meanSharpness.append(fm_helm(imSignal)) # calculating image sharpness  
            
            cellHistogram, bin_edges = np.histogram(imSignal, bins=1023, range=(0, 1023), density=True)
            xHist = bin_edges[0:-1]
            yHist = cellHistogram  
            
            plt.figure(50)
            plt.plot(xHist,yHist)
            
            # xHist,yHist = display_raw_histogram(imProbe, imSignal)
            # xHist,yHist = get_histogram(signalFile)
            meanIntHi,meanIntLo=histogram_analysis(xHist,yHist,figTitle=f'gDF_{i}')
            meanInts.append(meanIntHi)
        
        
        
            
        X=np.array([250,500,750],dtype=float)
        Y=meanInts   
        slope, intercept = np.polyfit(X, Y, deg=1)
        slope = np.round(slope,3)
        intercept = np.round(intercept,3)
        
        plt.figure()
        plt.plot(X,Y,'o-')
        plt.plot(X,slope*X+intercept,'k--')
        plt.ylim([0,1050])
        plt.xlim([0,1050])
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
    
    #%%
        #%% 400 for FLG and 250 for DF
    
    if probeFilenames == signalFilenames:    
        targetY = 400
    else:
        targetY = 250
    m = slopeMedian #0.342
    c = interceptMedian #140.1
    
    targetX = (targetY-c)/m
    
    print(targetX)
    