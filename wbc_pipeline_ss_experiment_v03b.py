# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:10:14 2022

@author: imrul
"""

# path=r"G:\Shared drives\Experimental Results\Cyclops_Server_Backup\WBC\2022-06-30_Diana\run0_3DLR0_shim0um_sample_s881_Cyc5Diana\subset" # baseline
# path=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\DF_optimization\2022-10-14\run_000_SS_LED_SC03_sample_s000_Cyc5Artemis\subset'
# path=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\DF_optimization\2022-11-04\run_002_IBF500mA_IDF700mA_RED_sample_wbc_assay_Cyc5Artemis'
path=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\Baseline_capture_for_Volpi_Visit\Captures_at_Volpi\Capture_from_device_1_(flowcell)\WBC\Combined_Captures'
# path=r'Z:\raspberrypi\photos\Erics Sandbox\Scatter_Tests\2022-11-01\GoodRun_ScatterAngle_GRN_635nm_Tangential_40deg_WBC_Cyc6Poseidon'
figure_location=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\results_folder'
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
#imageio.v2.imread

import json
import skimage.filters as sf
import skimage.morphology as sm
import skimage.measure as sms
import skimage.segmentation as ss#
import skimage.transform as st
from skimage.feature import match_template

#Functions to read in image
def imread_raw_monochrome(filename: str, 
                          horizontal_flip: bool=False,
                          vertical_flip: bool=False, 
                          downscale: int=2,
                          multiply_intensity: float=4.) -> np.ndarray:
    """
    Function to read in raw images taken with the monochrome camera

    Parameters
    ----------
    filename : string
        Path to image
    horizontal_flip : bool, optional
        If True, flip image horizontally, by default False
    vertical_flip : bool, optional
        If True, flip image vertically, by default False
    downscale : int, optional
        Factor to downscale image to maintain compatibility, by default 2
    multiply_intensity : int, optional
        Factor to multiply image intensities by to maintain compatibility, by default 4

    Returns
    -------
    np.ndarray
        Loaded image as a numpy array with type float32
    """
    # Read in the whole binary raw image
    #
    # Image sizes and bitdepth are hardcoded
    # Values are being multiplied by 4 to go from 10 to 12 bit and image is downsampled by factor of 2
    # This is to allow algorithms developed for Bayer images to be used with no changes
    #
    img_flat = np.fromfile(filename, dtype='int16', sep="")
    img = img_flat.reshape((3120,4224))
    #im = im_flat.reshape((3040,4048))
    if horizontal_flip:
        img = np.fliplr(img)
    if vertical_flip:
        img = np.flipud(img)
    if downscale > 1:
        img = st.downscale_local_mean(img,(downscale,downscale))*multiply_intensity
    else:
        img = img*multiply_intensity

    return img.astype('float32')

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
FL = sorted(glob.glob(os.path.join(path,'*G_FL_RT-FLGRN_550-60*')))  # #-G_FL-
BGDF = sorted(glob.glob(os.path.join(path,'*-G_bg_DF-*'))) #'*-G_bg_DF-*'
DF = sorted(glob.glob(os.path.join(path,'*G_FL_T-DF_480-40*')))  #*-G_DF-*
extension='bmp'
disable_BGDF=True
key0='FOV_'
key1='.'+extension
fovs_to_ignore =[]# [12,15] #This corresponds to FOV 2 and 22 in sorted array


img_FL_list=[]
img_DF_list=[]
labels_expand_list=[]


df = pd.DataFrame()



for i in range(len(FL)) :
    print(i)
    if i not in fovs_to_ignore:
        img_FL,metadata_FL =load_tiff(FL[i],extension)# imread_raw_monochrome(img)
        img_FL_list.append(img_FL)
        #img_FL = img_FL[:,1500:]

        img_DF,metadata_DF =load_tiff(DF[i],extension)# imread_raw_monochrome(DF[i])

        #img_DF = img_DF[:,1500:]
        if disable_BGDF is False:
            img_BGDF,_ = load_tiff(BGDF[i],extension)#imread_raw_monochrome(BGDF[i])
        # print(img_BGDF)
        #img_BGDF = img_BGDF[:,1500:]
            img_BGDF = sf.gaussian(img_BGDF,25,preserve_range=True)
        else:
            img_BGDF=0


        #Detect cells from BF
        label_erosion = 0
        expand = 0
        labels_green = detect_cells(img_FL,th_block=30,th_offset=-10,th_size=100,sigma=3,erosion=label_erosion) #default th_block 
        labels_expand = ss.expand_labels(labels_green, distance=expand+label_erosion)
        labels_expand_list.append(labels_expand)
        img_diff = img_DF - img_BGDF
        img_DF_list.append(img_diff)

        #Populate regionprops dataframe
        regprops_ssbg_df = pd.DataFrame(sms.regionprops_table(labels_expand,
                                                              img_diff,
                                                              properties=['label', 'centroid','mean_intensity','area','perimeter','eccentricity','solidity'],
                                                              extra_properties=[image_stdev,image_median])).set_index("label").rename(columns={"mean_intensity": "mean_intensity_ss-bg", 
                                                                                                                                               "image_stdev": "image_stdev_ss-bg", 
                                                                                                                                               "image_median": "image_median_ss-bg"})

        #Some cell filtering using hardcoded parameters
        regprops_ssbg_df['fov'] = i
        
        regprops_ssbg_df[f"area_filter"]=(regprops_ssbg_df["area"]<3500) & (regprops_ssbg_df["area"]>300)
        regprops_ssbg_df["circularity"]=4*np.pi*(regprops_ssbg_df["area"]/regprops_ssbg_df["perimeter"]**2) #Measure of circularity. 1 = circle.  Perimeter can be zero, so circularity [0,inf)
        regprops_ssbg_df["solidity_filter"]=(regprops_ssbg_df["solidity"]>0) & (regprops_ssbg_df["eccentricity"]<0.9) & (np.abs(regprops_ssbg_df["circularity"]-1)<0.4) 
        regprops_ssbg_df['include_event']=(regprops_ssbg_df["area_filter"]) & (regprops_ssbg_df["solidity_filter"])

        coords=regprops_ssbg_df[['centroid-1','centroid-0']].to_numpy()
        coords_oc=coords-[2100,1500]
        radius=np.sqrt(np.sum(coords_oc**2,1))
        regprops_ssbg_df['radius']=radius
        regprops_ssbg_df['radius_filter']=regprops_ssbg_df['radius']<1000
        
        df=df.append(regprops_ssbg_df)

        print(df)
        
#%%        
df_include=df[df['include_event'] & df['radius_filter']]


#%

# for ind in df_include.index:
#     print(df_include['fov'][ind])

df_new=df_include.copy()
df_new['fix_area_mean']=0
df_new['fix_area_std']=0
roi_size=50

df_new_list=[]
for index,row in df_new.iterrows():
    print(row)
    pt_x=int(row['centroid-1'])
    pt_y=int(row['centroid-0'])
    fov=row['fov']
    
    imCrop=img_DF_list[fov]
    imCrop0=imCrop[pt_y-roi_size//2:pt_y+roi_size//2,pt_x-roi_size//2:pt_x+roi_size//2]
    # imCrop1=imCrop[pt_y-roi_size//4:pt_y+roi_size//4,pt_x-roi_size//4:pt_x+roi_size//4]
    row['fix_area_mean']=np.nanmean(imCrop0)
    row['fix_area_std']=np.nanstd(imCrop0)
    df_new_list.append(row)
    
    
df_new=pd.DataFrame(df_new_list)    




df_include = normalize_data(df_include,quantiles=[0.7,0.1,0.4])
#Plot histogram 
# plt.figure()
#%%
data=[]

fig,ax=plt.subplots(1,2)
data.append(ax[0].hist(df_include['image_median_ss-bg'],bins=100, range=(-100,400)))
ax[0].set_title('Contour_Mean')
ax[0].set_xlabel('SS')
ax[0].set_ylabel('Frequency')
#Plot histogram 
# plt.figure()
data.append(ax[1].hist(df_include['image_median_ss-bg_corr'],bins=100,range=(-100,400)))
ax[1].set_title('Contour_Mean_Corr')
ax[1].set_xlabel('SS')
ax[1].set_ylabel('Frequency')

figName='contour_'+os.path.split(path)[-1]+'.png'
fig.savefig(os.path.join(figure_location,figName))

# data.append(np.histogram(df_include['image_median_ss-bg'],bins=100, range=(-100,400)))
# data.append(np.histogram(df_include['image_median_ss-bg_corr'],bins=100,range=(-100,400)))


#%%

fig,ax=plt.subplots(1,2)
data.append(ax[0].hist(df_new['fix_area_mean'],bins=100,range=(-100,400)))
ax[0].set_title('Fixed_area_Mean')
ax[0].set_xlabel('SS')
ax[0].set_ylabel('Frequency')
#Plot histogram 
# plt.figure()
data.append(ax[1].hist(df_new['fix_area_std'],bins=100,range=(-10,100)))
ax[1].set_title('Fixed_area_Std')
ax[1].set_xlabel('SS')
ax[1].set_ylabel('Frequency')




figName='fixArea_'+os.path.split(path)[-1]+'.png'
fig.savefig(os.path.join(figure_location,figName))
#%%
import pickle
dataFileName=os.path.join(figure_location,os.path.split(path)[-1])+'.pickle'
# file = open(dataName, 'w')
# pickle.dump(data, file)
# file.close()



with open(dataFileName,'wb') as fp:
       # logging.info(f'Writing {filename_dict} for analysis Script to {masterfilename}')
       pickle.dump(data,fp)
fp.close()
#%%
# plt.figure()
# plt.hist(df_new['fix_area_mean'],bins=100,range=(-100,400))
# plt.title('Fixed_area_Mean')
# plt.xlabel('SS')
# plt.ylabel('Frequency')

# plt.figure()
# plt.hist(df_new['fix_area_std'],bins=100,range=(-10,100))
# plt.title('Fixed_area_Std')
# plt.xlabel('SS')
# plt.ylabel('Frequency')

#%%
for i in [0]:
    fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
    ax[0].imshow(img_FL_list[i])
    ax[1].contour(np.isin(labels_expand_list[i],df[df['include_event'] & df['radius_filter'] & (df['fov']==i)].index.to_numpy()))
    ax[1].imshow(img_DF_list[i],cmap='gray')   

# #%%
# fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
# ax[0].imshow(img_FL)
# ax[1].contour(np.isin(labels_expand,df[df['include_event'] & df['radius_filter'] & (df['fov']==13)].index.to_numpy()))
# ax[1].imshow(img_DF-img_BGDF)
# # plt.contour(labels_expand>0)




