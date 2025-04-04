# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:39:46 2022

@author: Jeff
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

#plt.rcParams['figure.figsize'] = [15, 15]

def detect_cells_cyclops5(img,th_block=81,th_offset=-0.8,th_size=500):
    t=sf.threshold_local(img,th_block,offset=th_offset)
    img_t=img>t
    img_tf=ndi.binary_fill_holes(img_t)
    img_ts=sm.remove_small_objects(img_tf,th_size)
    labels_cells=sm.label(img_ts)
    return(labels_cells)

def imread_raw(filename):
    # Read in the whole binary image tail of the
    # .jpg file with appended raw image data
    try:
        with open(filename, mode='rb') as file: # b is important -> binary
            filraw = file.read()
    except FileNotFoundError:
        print("File does not exist.")
        sys.exit(1)
    start=filraw.find(b"BRCM")
    if start==-1:
        print("RAW image data not found.")
        sys.exit(1)

    bin=filraw[start+2**15:]

    # Image data proper starts after 2^15 bytes = 32768
    imdata = np.frombuffer(bin, dtype=np.uint8)

    # Reshape the data to 3056 rows of 6112 bytes each and crop to 3040 rows of 6084 bytes
    imdata = imdata.reshape((3056, 6112))[:3040, :6084]

    # Convert to 16 bit data
    imdata = imdata.astype(np.uint16)

    # Make an output 16 bit image
    im = np.zeros((3040, 4056), dtype=np.uint16)
    # Unpack the low-order bits from every 3rd byte in each row
    for byte in range(2):
        im[:, byte::2] = ( (imdata[:, byte::3] << 4) | ((imdata[:, 2::3] >> (byte * 4)) & 0b1111) )

    B=im[0::2,0::2]
    G1=im[1::2,0::2]
    G2=im[0::2,1::2]
    R=im[1::2,1::2]
    G=(G1+G2)//2

    rgb=np.dstack((R,G,B))
    return rgb

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

def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region], ddof=1)

def image_median(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.median(intensities[region])

#if len(sys.argv)==3:
#    G_FL_filename=sys.argv[1]
#    SS_filename=sys.argv[2]
#else:
#    print("""Usage: display_raw_histogram.py G_FL_IMAGE SS_IMAGE
#        G_FL_IMAGE - path to G_FL jpg image with raw data
#        SS_IMAGE - path to SS jpg image with raw data
#    """)
#    sys.exit(0)

def display_raw_histogram(probe_filename,signal_filename):

    solidity_cutoff = 0.6
    
    
    meanint_bg = np.array([])
    meanint_ss = np.array([])
    stdint_ss = np.array([])
    medianint_ss = np.array([])
    
    im_g = read_image(probe_filename)
    labels_g = detect_cells_cyclops5(im_g,41,-5,100)
    labels_bg=ss.expand_labels(labels_g, distance=3)-labels_g
    
    im_ss = read_image(signal_filename)
    
    regprops_ss = sms.regionprops(labels_g,im_ss,extra_properties=[image_stdev,image_median])
    
    solidity = np.array([r['solidity'] for r in regprops_ss])
    mean_intensity_ss = np.array([r['mean_intensity'] for r in regprops_ss])
    standard_dev = np.array([r['image_stdev'] for r in regprops_ss])
    median_intensity_ss = np.array([r['image_median'] for r in regprops_ss])
    
    
    
    
    fig, ax = plt.subplots(1)
    ax.imshow(im_ss)
    ax.contour(labels_g>0,1,colors='r',)
    fig, ax = plt.subplots(1)
    cellHistogram = ax.hist(mean_intensity_ss,100)
    ax.set_xlim([100,900])
    #ax[1,0].hist(median_intensity_ss,60)
    #ax[1,1].hist(standard_dev,60)
    
    xHist = cellHistogram[1][1:]
    yHist = cellHistogram[0]   
    
    return xHist,yHist


def get_histogram(probe_filename,background = 50):
    im_g = read_image(probe_filename)
    r,c=im_g.shape
    imCrop = im_g
    
    imCropFlat = imCrop[imCrop>=background]

    fig, ax = plt.subplots(1)
    cellHistogram = ax.hist(imCropFlat,256)
    ax.set_xlim([50,900])
    #ax[1,0].hist(median_intensity_ss,60)
    #ax[1,1].hist(standard_dev,60)
    
    xHist = cellHistogram[1][1:]
    yHist = cellHistogram[0]  
    
    return xHist,yHist
    
def get_median(probe_filename, background = 50):
    im_g = read_image(probe_filename)
    r,c=im_g.shape
    imCrop = im_g
    
    imCropFlat = imCrop[imCrop>=background]

    medInt= np.median(imCropFlat)
    stdInt = np.std(imCropFlat) 
    
    return medInt,stdInt    

def get_robust_median(probe_filename, backgroundPercentile = 25):
    im_g = read_image(probe_filename)
    r,c=im_g.shape
    imCrop = im_g
    
    imCropFlat = imCrop[imCrop>=10]
    imCropFlat = imCropFlat[imCropFlat>=np.percentile(imCropFlat,backgroundPercentile)]
    
    medInt= np.median(imCropFlat)
    stdInt = np.std(imCropFlat) 
    print("25 th percentiles")
    print(np.percentile(imCropFlat,25))
    return medInt,stdInt     
    
    
    

def histogram_analysis(xHist,yHist,figTitle='gg'):
    yHist1= ndimage.gaussian_filter1d(yHist,1) # smoothing the histogram
    peaks, _ = signal.find_peaks(yHist1,prominence=30,distance=100)
    peakWidths = signal.peak_widths(yHist1, peaks, rel_height=0.5)[0]
    meanInt= np.round(xHist[peaks][0],1)
    meanFreq = np.round(yHist[peaks][0],0)
    peakWidth = np.round(peakWidths[0]*np.mean(np.diff(xHist)),0)
    plt.figure()
    plt.plot(xHist,yHist1)
    plt.plot(xHist[peaks],yHist1[peaks],'ro')
    plt.xlim([20,900])
    plt.title(f"{figTitle}")
    print(f'Mean Intensity {meanInt} with Frequency of {meanFreq} and peak width of {peakWidth}')
    
    return meanInt,meanFreq,peakWidth

def get_percentile_from_image(tiffPath,percentiles=[1,25,50,75,99]):
    im=read_image(tiffPath)
    r,c=im.shape
    im= im[r//3:2*r//3,c//3:2*c//3]
    data = im[im>=0]
    vals =[]
    for percentile in percentiles:
        val = np.percentile(data,percentile)
        vals.append(val)
        print(f"{percentile} % : {val}")
    return vals
    
    
#%%    

# if __name__ == "__main__":
if __name__=='__main__':
    
    # file_gfl=r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\CYC7_A04\2023-12-11\FAT_exposure_calibration_Cyc7-A4_GLASS_FATv8_run02_LRV2_e\img_green_FLG_fov1_exposure_1.tiff"
    # file_gdf=r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\CYC7_A04\2023-12-11\FAT_exposure_calibration_Cyc7-A4_GLASS_FATv8_run02_LRV2_e\img_green_FLG_fov1_exposure_3.tiff"
    # # file_gdf=r"Z:\raspberrypi\photos\Method_Comp\2022-02-18\run01_sample_s675_WBC_diff_Inj_Mold_Cyc4Metal\cyclops-G_DF-FOV_6.png"
    # # file_gfl=r"Z:\raspberrypi\photos\Method_Comp\2022-02-18\run01_sample_s675_WBC_diff_Inj_Mold_Cyc4Metal\cyclops-G_FL-FOV_6.png"
    
    # xHist,yHist=display_raw_histogram(file_gfl,file_gdf)
    
    # #%%
    # plt.figure()
    # plt.plot(xHist,yHist)
    
    # meanInt,meanFreq,peakWidth=histogram_analysis(xHist,yHist,figTitle='gfl')
    
    
    # yHist1= ndimage.gaussian_filter1d(yHist,1)
    
    # peaks, _ = signal.find_peaks(yHist1,prominence=30,distance=100)
    # peakWidths = signal.peak_widths(yHist1, peaks, rel_height=0.5)[0]
    # plt.figure()
    # plt.plot(xHist,yHist1)
    # plt.plot(xHist[peaks],yHist1[peaks],'ro')
    # plt.xlim([100,900])
    
    # meanInt= np.round(xHist[peaks][0],1)
    # meanFreq = np.round(yHist[peaks][0],0)
    # peakWidth = np.round(peakWidths[0],0)*np.mean(np.diff(xHist))
    
    # print(f'Mean Intensity {meanInt} with Frequency of {meanFreq} and peak width of {peakWidth}')
    #%%
    folderName = r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\CYC7_A04\2023-12-11\FAT_brightness_calibration_Cyc7-A4_GLASS_FATv8_ps_LRV2_e"
    probeFilenames=["img_green_FLG_fov1_intensityB_500.tiff","img_green_FLG_fov1_intensityB_750.tiff","img_green_FLG_fov1_intensityB_1000.tiff"]
    signalFilenames = ["img_blue_DF_fov1_intensityB_500.tiff","img_blue_DF_fov1_intensityB_750.tiff","img_blue_DF_fov1_intensityB_1000.tiff"]
    
    meanInts=[]
    for i in range(len(signalFilenames)):
        probeFile = os.path.join(folderName,probeFilenames[i])
        signalFile = os.path.join(folderName,signalFilenames[i])
        print(probeFile)
        
        xHist,yHist = display_raw_histogram(probeFile, signalFile)
        # xHist,yHist = get_histogram(signalFile)
        meanInt,meanFreq,peakWidth=histogram_analysis(xHist,yHist,figTitle=f'gDF_{i}')
        meanInts.append(meanInt)
    
    plt.figure()
    plt.plot([500,750,1000],meanInts)
    
    
     #%%
    
    # tiffPath = r'Z:\\raspberrypi\\photos\\FAT_Captures\\Alpha_Plus\\CYC7_A04\\2023-12-11\\FAT_brightness_calibration_Cyc7-A4_GLASS_FATv8_ps_LRV2_e\\img_blue_DF_fov1_intensityB_500.tiff'
    
    # im= read_image(tiffPath)
    
    # # fig, ax = plt.subplots(1)
    # # cellHistogram = ax.hist(im,100)
    # # ax.set_xlim([100,900])
    # # #ax[1,0].hist(median_intensity_ss,60)
    # # #ax[1,1].hist(standard_dev,60)
    
    # # xHist = cellHistogram[1][1:]
    # # yHist = cellHistogram[0]  
    
    # x,y=get_histogram(tiffPath)
    # histogram_analysis(x,y)
    
    # signalFilenames_=["img_blue_DF-BG_fov1_intensityB_500.tiff",
    #                   "img_blue_DF-BG_fov1_intensityB_750.tiff",
    #                   "img_blue_DF-BG_fov1_intensityB_1000.tiff"
    #                   ]
    
    signalFilenames_=[
        r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\Instrument_Calibration\2024-01-26\FAT_brightness_calibration_Cyc7-A04_ps_beads_run00_LRV2.4\img_blue_DF_fov1_intensityB_500.tiff",
        r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\Instrument_Calibration\2024-01-26\FAT_brightness_calibration_Cyc7-A04_ps_beads_run00_LRV2.4\img_blue_DF_fov1_intensityB_750.tiff"
        r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\Instrument_Calibration\2024-01-26\FAT_brightness_calibration_Cyc7-A04_ps_beads_run00_LRV2.4\img_blue_DF_fov1_intensityB_1000.tiff"
        #r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\CYC7_A10\2023-12-21\FAT_wbc_Cyc7-A10_FATv8_s757_run01_LRV2_LED_1000\img_blue_DF_fov1.tiff"
        
        ]
    meanInts_=[]
    for i in range(len(signalFilenames_)):

        signalFile = signalFilenames_[i]

        meanInt,stdInt = get_robust_median(signalFile,backgroundPercentile=25)
        meanInts_.append(meanInt)
        
        
    X=np.array([500,750,1000],dtype=float)
    Y=meanInts_
    
    
    slope, intercept = np.polyfit(X, Y, deg=1)
    
    plt.figure()
    plt.plot(X,Y)
    plt.plot(X,slope*X+intercept,'k--')
    plt.title(signalFilenames_[0])
    plt.xlabel("LED CURRENT [a.u.]")
    plt.ylabel("Median Intensity [a.u.]")
    
    print(f" slope {slope} \n intercept {intercept}")
    
    
    targetIntensity = 300  # 400 for DF-BG 550 for FL [300,150,120]
    
    ledPwr = (targetIntensity - intercept)/slope
    
    print(f"To achieve target intensity of {targetIntensity} required led power {ledPwr}")
    
    
    
    
    #%%
    def get_percentile(data,percentiles=[1,25,50,75,99]):
        vals=[]
        for percentile in percentiles:
            val = np.percentile(data,percentile)
            vals.append(val)
            print(f"{percentile} % : {val}")
        return vals
    
        
    
    
    
    tiff1=r"Z:\raspberrypi\photos\FAT_Captures\Alpha_Plus\CYC7_A10\2023-12-20\FAT_wbc_Cyc7-A10_FATv8_s116-6_run00_LRV2_Dyna_LED_1000\img_green_FLG_fov1.tiff"
    # tiff2=r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A10\2023-12-15\WBC_Cyc7-A10_500_S116_4_LRV2.3_Dynacare\img_blue_DF_bg_fov2.tiff"
    
    imDf=read_image(tiff1)
    # imDfBg=read_image(tiff2)
                
    plt.figure()
    plt.imshow(imDf)
    
    
    #%
    imFlat = imDf[imDf>10]
    # val= np.percentile(imFlat,25)
    # print(val)
    vals = get_percentile(imFlat)
        
#%%

    signalFilenames_=[
        r"Z:\raspberrypi\photos\FAT_Captures\Alpha_sharp\Cyc7_As2\20240118\FAT_ps_beads_Cyc7-AS-2_GLASS_FATv8b_run00_LRV3_RUBIX_LEDPWR_750\img_blue_DF_fov1.tiff",
        r"Z:\raspberrypi\photos\FAT_Captures\Alpha_sharp\Cyc7_As2\20240118\FAT_ps_beads_Cyc7-AS-2_GLASS_FATv8b_run00_LRV3_LUXEONCZ_LEDPWR_750\img_blue_DF_fov1.tiff",
        # r"Z:\raspberrypi\photos\FAT_Captures\Alpha_sharp\Cyc7_As2\20240118\FAT_ps_beads_Cyc7-AS-2_GLASS_FATv8b_run00_LRV3_LUXEONCZ_LEDPWR_1000\img_red_FLR_fov1.tiff"
        
        ]
    meanInts_=[]
    for i in range(len(signalFilenames_)):

        signalFile = signalFilenames_[i]

        meanInt,stdInt = get_robust_median(signalFile,backgroundPercentile=25)
        meanInts_.append(meanInt)
        
    plt.figure()
    plt.plot(['Rubix','LuxeonCz'],meanInts_)
    plt.title(signalFilenames_[0])
    plt.ylabel('Intensity [a.u.]')
    
    
#%%

    signalFilenames_=[
        r"Z:\raspberrypi\photos\FAT_Captures\Alpha_sharp\Cyc7_As2\20240118\FAT_ps_beads_Cyc7-AS-2_GLASS_FATv8b_reRun00_LRV3_RUBIX_LEDPWR_250\img_blue_DF_fov1.tiff",
        r"Z:\raspberrypi\photos\FAT_Captures\Alpha_sharp\Cyc7_As2\20240118\FAT_ps_beads_Cyc7-AS-2_GLASS_FATv8b_reRun00_LRV3_RUBIX_LEDPWR_500\img_blue_DF_fov1.tiff",
        r"Z:\raspberrypi\photos\FAT_Captures\Alpha_sharp\Cyc7_As2\20240118\FAT_ps_beads_Cyc7-AS-2_GLASS_FATv8b_reRun00_LRV3_RUBIX_LEDPWR_750\img_blue_DF_fov1.tiff",
        r"Z:\raspberrypi\photos\FAT_Captures\Alpha_sharp\Cyc7_As2\20240118\FAT_ps_beads_Cyc7-AS-2_GLASS_FATv8b_reRun00_LRV3_RUBIX_LEDPWR_1000\img_blue_DF_fov1.tiff"
        
        ]
    meanInts_=[]
    for i in range(len(signalFilenames_)):

        signalFile = signalFilenames_[i]

        vals = get_percentile_from_image(signalFile)
        meanInts_.append(vals[3])
#%       
    plt.figure()
    plt.plot([250,500,750,1000],meanInts_)
    plt.title(signalFilenames_[0])
    plt.ylabel('Mean Intensity [a.u.]')    
    plt.xlabel('LED current')
    plt.title('75percentile_cropRUBIX')