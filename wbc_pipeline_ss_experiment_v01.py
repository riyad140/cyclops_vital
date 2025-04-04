# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:29:12 2022

@author: imrul
"""
# try:
#     get_ipython().run_line_magic('reset', '-f')
# except:
#     pass

#path="/home/imrul/cyclops/raspberrypi/photos/imrul_sandbox/DF_LED_NUMBER_optimization/2022-10-14/run_000_SS_LED_SC03_sample_s000_Cyc5Artemis"
# path=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\DF_optimization\2022-10-14\run_000_SS_LED_SC03_sample_s000_Cyc5Artemis\subset'
path=r'G:\Shared drives\Experimental Results\Cyclops_Server_Backup\WBC\2022-06-30_Diana\run0_3DLR0_shim0um_sample_s881_Cyc5Diana\subset'

#%%
import sys

# try:
#     get_ipython
# except:
#     path=sys.argv[1]
#     if len(sys.argv)>2:
#         config_file=sys.argv[2]


import logging
import shutil 

import os
from skimage import io
from scipy import ndimage as ndi
import datetime
import string
import glob
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import skimage.measure as sms#
from skimage.morphology import disk
import importlib
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['figure.figsize'] = [15, 15]

from imageio import imread
import json
import skimage.filters as sf
import skimage.morphology as sm
import skimage.measure as sms
import skimage.segmentation as ss#

def detect_cells(img,th_block=81,th_offset=-0.8,th_size=500,sigma=0,erosion=0):
    if sigma>0:
        img = sf.gaussian(img,sigma,preserve_range=True)
    t=sf.threshold_local(img,th_block,offset=th_offset)
    img_t=img>t
    img_tf=ndi.binary_fill_holes(img_t)
    img_ts=sm.remove_small_objects(img_tf,th_size)
    if erosion>0:
        img_ts = sm.erosion(img_ts,sm.disk(erosion))
    labels_cells=sm.label(img_ts)

    return labels_cells

def load_tiff(img_path,extension='tiff'):
    """
    Loads a TIFF image and metadata

    Parameters
    ----------
    img_path : str
        Path to img

    Returns
    -------
    tuple[np.ndarray, dict]
        Img loaded as a numpy array and dict containing metadata
    """

    
    if extension == 'tiff':
        img = imread(img_path)
        # Extract custom metadata
        vital_meta = img.meta['description']
        config_meta = json.loads(vital_meta)
        meta = config_meta['config']    
        return (np.asarray(img), meta)
    
    elif extension == 'bmp':
        img = imread(img_path)
        return (np.asarray(img),0)
    
    elif extension == 'raw':
        imageSize = (3120, 4224)   #imageSize updated 
        npimg = np.fromfile(img_path, dtype=np.dtype('<u2'))
        imRaw = npimg.reshape(imageSize)        
        return (imRaw,0)


def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    if len(region)>0:
        return np.std(intensities[region], ddof=1)
    else:
        return 0

def image_median(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    if len(region)>0:
        return np.median(intensities[region])
    else:
        return 0
    
def getMask(imMask,coords,radius=30):
    r,c=np.shape(imMask)
    imCrop=np.zeros((radius*2,radius*2),dtype=int)
    r_mid,c_mid=radius,radius
    
    for i in range(2*radius):
        for j in range(2*radius):
            if (i-r_mid)**2+(j-c_mid)**2<=radius**2:
                # print(f'{i},{j}')
                imCrop[i,j]=1
    try:
        imMask[coords[1]-radius:coords[1]+radius,coords[0]-radius:coords[0]+radius]=imCrop
    except:
        imMask=imMask
    
    return imMask

def normalize_data(df,quantiles=[0.7,0.1,0.4]):
    # Find upper, mid and lower quantiles
    fov_uq_SS = df.groupby('fov')['image_median_ss-bg'].quantile(quantiles[0])
    fov_lq_SS = df.groupby('fov')['image_median_ss-bg'].quantile(quantiles[1])
    fov_mq_SS = df.groupby('fov')['image_median_ss-bg'].quantile(quantiles[2])

    #Use the median of quantiles as the normalization targets
    uq_SS = np.median(fov_uq_SS)
    lq_SS = np.median(fov_lq_SS)
    mq_SS = np.median(fov_mq_SS)

    #Append the per FOV upper and lower quantiles to each row
    df = df.join(fov_uq_SS, on='fov', rsuffix='_uq').join(fov_lq_SS, on='fov', rsuffix='_lq')

    #Normalize channels to match the median upper and lower quantiles
    df['image_median_ss-bg_corr'] = (df['image_median_ss-bg'] - df['image_median_ss-bg_lq']) * (uq_SS - lq_SS)/(df['image_median_ss-bg_uq'] - df['image_median_ss-bg_lq']) + lq_SS

    # Return updated dataframe
    return df
    
#%%
# FL = sorted(glob.glob(os.path.join(path,'*-G_FL-*')))
# BGDF = sorted(glob.glob(os.path.join(path,'*-G_bg_DF-*')))
# DF = sorted(glob.glob(os.path.join(path,'*-G_DF-*')))

# df = pd.DataFrame()
# extension='raw'
# key0='FOV_'
# key1='.'+extension    #'_0.tiff'
# for i, img in enumerate(FL):
#     fov_count=int(img[img.find(key0)+len(key0):img.find(key1)])
#     print(f'fov count {fov_count}')
#     img_FL, metadata_FL = load_tiff(img,extension)
#     img_FL = img_FL[:,:3500]  # [:,1500:]

#     img_DF, metadata_DF = load_tiff(DF[i],extension)
#     img_DF = img_DF[:,:3500]

#     img_BGDF, metadata_BGDF = load_tiff(BGDF[i],extension)
#     img_BGDF = img_BGDF[:,:3500]
#     img_BGDF = sf.gaussian(img_BGDF,25,preserve_range=True)
    
#     #Detect cells from BF
#     labels_green = detect_cells(img_FL,th_block=81,th_offset=-10,th_size=100,sigma=3,erosion=0)
#     labels_expand = ss.expand_labels(labels_green, distance=3)
#     img_diff = img_DF - img_BGDF
#     #img_diff = np.maximum(img_DF-img_BGDF,0)
    
#     #Populate regionprops dataframe
#     regprops_ssbg_df = pd.DataFrame(sms.regionprops_table(labels_expand,
#                                                           img_diff,
#                                                           properties=['label', 'mean_intensity','area','perimeter','eccentricity','solidity','centroid'],
#                                                           extra_properties=[image_stdev,image_median])).set_index("label").rename(columns={"mean_intensity": "mean_intensity_ss-bg", 
#                                                                                                                                            "image_stdev": "image_stdev_ss-bg", 
#                                                                                                                                            "image_median": "image_median_ss-bg"})
    
#     #Some cell filtering using hardcoded parameters
#     regprops_ssbg_df[f"area_filter"]=(regprops_ssbg_df["area"]<3500) & (regprops_ssbg_df["area"]>100)
#     regprops_ssbg_df["circularity"]=4*np.pi*(regprops_ssbg_df["area"]/regprops_ssbg_df["perimeter"]**2) #Measure of circularity. 1 = circle.  Perimeter can be zero, so circularity [0,inf)
#     regprops_ssbg_df["solidity_filter"]=(regprops_ssbg_df["solidity"]>0) & (regprops_ssbg_df["eccentricity"]<0.9) & (np.abs(regprops_ssbg_df["circularity"]-1)<0.4) 
#     regprops_ssbg_df['include_event']=(regprops_ssbg_df["area_filter"]) & (regprops_ssbg_df["solidity_filter"])
#     regprops_ssbg_df['fov']=fov_count
    
#     df=df.append(regprops_ssbg_df,ignore_index=True)
# print(df)


#%%

FL = sorted(glob.glob(os.path.join(path,'*-G_FL-*')))
BGDF = sorted(glob.glob(os.path.join(path,'*-G_bg_DF-*')))
DF = sorted(glob.glob(os.path.join(path,'*-G_DF-*')))

fovs_to_ignore = []
extension='raw'
key0='FOV_'
key1='.'+extension

df = pd.DataFrame()

for i, img in enumerate(FL):
    fov_count=int(img[img.find(key0)+len(key0):img.find(key1)])
    print(f'fov count {fov_count}')
    if fov_count not in fovs_to_ignore:
        img_FL, metadata_FL = load_tiff(img,extension)
        img_FL = img_FL[:,:3000]

        img_DF, metadata_DF = load_tiff(DF[i],extension)
        img_DF = img_DF[:,:3000]

        img_BGDF, metadata_BGDF = load_tiff(BGDF[i],extension)
        img_BGDF = img_BGDF[:,:3000]
        img_BGDF = sf.gaussian(img_BGDF,25,preserve_range=True)

        #Detect cells from BF
        labels_green = detect_cells(img_FL,th_block=81,th_offset=-10,th_size=100,sigma=3,erosion=0)
        labels_expand = ss.expand_labels(labels_green, distance=3)
        img_diff = img_DF - img_BGDF
        #img_diff = np.maximum(img_DF-img_BGDF,0)

        #Populate regionprops dataframe
        regprops_ssbg_df = pd.DataFrame(sms.regionprops_table(labels_expand,
                                                              img_diff,
                                                              properties=['label', 'mean_intensity','area','perimeter','eccentricity','solidity','centroid'],
                                                              extra_properties=[image_stdev,image_median])).set_index("label").rename(columns={"mean_intensity": "mean_intensity_ss-bg", 
                                                                                                                                               "image_stdev": "image_stdev_ss-bg", 
                                                                                                                                               "image_median": "image_median_ss-bg"})

        #Some cell filtering using hardcoded parameters
        regprops_ssbg_df['fov'] = fov_count
        regprops_ssbg_df[f"area_filter"]=(regprops_ssbg_df["area"]<3500) & (regprops_ssbg_df["area"]>300)
        regprops_ssbg_df["circularity"]=4*np.pi*(regprops_ssbg_df["area"]/regprops_ssbg_df["perimeter"]**2) #Measure of circularity. 1 = circle.  Perimeter can be zero, so circularity [0,inf)
        regprops_ssbg_df["solidity_filter"]=(regprops_ssbg_df["solidity"]>0) & (regprops_ssbg_df["eccentricity"]<0.9) & (np.abs(regprops_ssbg_df["circularity"]-1)<0.4) 
        regprops_ssbg_df['include_event']=(regprops_ssbg_df["area_filter"]) & (regprops_ssbg_df["solidity_filter"])

        df=df.append(regprops_ssbg_df,ignore_index=True)

df = normalize_data(df,quantiles=[0.7,0.1,0.4])

#%%
#Plot histogram 
plt.figure()
plt.hist(df[df['include_event']]['image_median_ss-bg'],bins=200,range=(-50,200))
plt.title(os.path.split(path)[1])
plt.xlabel('SS')
plt.ylabel('Frequency')

plt.figure()
plt.hist(df[df['include_event']]['image_median_ss-bg_corr'],bins=200,range=(-50,200))
plt.title(os.path.split(path)[1]+'_corr')
plt.xlabel('SS')
plt.ylabel('Frequency')
#%%

bins=150
fig,ax=plt.subplots(6,5,sharex=True,sharey=True)

hist_list=[]

for i in range(30):
    
    r,c=i//5,i%5
    hist_list.append(ax[r,c].hist(df[df['include_event']][df['fov']==i]['image_median_ss-bg'],bins=bins,range=(-100,500)))
    ax[r,c].set_title(f'fov-{i}')
fig.suptitle(f'Hist_Bins : {bins}')

#%%
plt.figure()

for y in hist_list:
    if np.nanmean(y[0])>1:
        plt.plot(y[0])
        
#%%

from scipy.signal import find_peaks

x=hist_list[0][0]
peaks, _ = find_peaks(x, prominence=10)
plt.figure()
plt.plot(x)
plt.plot(peaks, x[peaks], "x")

lym_idx=peaks[0]

#%
hist_shifted=[]
plt.figure()
for y in hist_list:
    if np.nanmean(y[0])>1:
        x=y[0]
        peaks, _ = find_peaks(x, prominence=7)
        peak_shift=lym_idx-peaks[0]
        
        x_shifted=np.roll(x,peak_shift)
        hist_shifted.append(x_shifted)
        plt.plot(x_shifted,label=f'{peak_shift}')
plt.legend()
#%
hist_final=np.mean(np.array(hist_shifted),axis=0)
plt.figure()
plt.bar(hist_list[0][1][:-1],hist_final)
#%%#Plot the last FOV analyzed
plt.figure()
plt.imshow(img_DF-img_BGDF)
plt.contour(labels_green>0)

#%%
import cv2
roi_x=np.array(df[df['fov']==9]['centroid-1'])
roi_y=np.array(df[df['fov']==9]['centroid-0'])

# roi_xx=df[df['fov']==9]['centroid-0']

rois=np.vstack([roi_x,roi_y]).T
rois=np.round(rois).astype(int)
#%% Mask creation with circular ROI
overlay=np.copy(img_FL)
imMask=(np.copy(overlay)*0).astype(int)

for roi in rois:
    # print(roi)
    cv2.circle(overlay,roi,30,(800),5)
    imMask=getMask(imMask,roi)
    
fig,ax=plt.subplots(2,1,sharex=True,sharey=True)
ax[0].imshow(overlay)
ax[1].imshow(imMask)


#%% 



    
    
    
