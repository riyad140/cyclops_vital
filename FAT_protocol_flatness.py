# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:40:13 2023

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

from ROI_manager_user import userROI
import os
import sys
import cv2
import math
from scipy import ndimage, signal
from ROI_manager_user import userROI
import pandas as pd

def normalize(data): # normalize a 1D data with min,max
    return (data-np.min(data))/(np.max(data)-np.min(data))

# def get_mid_point_location(xSection,interpolation_factor=100):
#     y=normalize(xSection)-0.5
#     x=np.linspace(0,len(y)-1,len(y))
#     xp=np.linspace(x[0],x[-1],len(x)*interpolation_factor)    
#     yp=np.interp(xp,x,y)
#     zc=np.diff(np.sign(yp))
#     zc_ind=np.where(zc!=0)[0]    
#     mid_point_location=xp[zc_ind]
#     return mid_point_location


def get_imRadius(coords,imSize=[1520,2024]): # get the ROI height and angle w.r.t to image center
#%    
    mid_coords=[imSize[1]//2,imSize[0]//2]
    diff=np.array(coords)-np.array(mid_coords)
    
    distance=np.sqrt(diff[0]**2+diff[1]**2)
    max_distance=np.sqrt(mid_coords[0]**2+mid_coords[1]**2)
    imRadius_pt=distance/max_distance*100
    
    imAngle=np.degrees(math.atan(-diff[1]/diff[0]))
    
    return imRadius_pt,imAngle,diff


def get_roi(imShape,n=5):
    r,c=imShape[0],imShape[1]
    coords=[]
    x0,y0=c//(2*n),r//(2*n)
    # coords.append([x0,y0])
    dx=c//n
    dy=r//n
    for i in range(n):
        for j in range(n):
            coords.append([x0+i*dx,y0+j*dy])
    return coords

def get_cross_sections(imCrop_ctr,orientation=['000','090','045','135']):
    midPoint=imCrop_ctr.shape[0]//2
    rot_angle=45
    imCrop_ctr_rot = ndimage.rotate(imCrop_ctr, rot_angle,reshape=False,order=1)
    section_090=imCrop_ctr[:,midPoint]
    section_000=imCrop_ctr[midPoint,:]
    section_045=imCrop_ctr_rot[:,midPoint]
    section_135=imCrop_ctr_rot[midPoint,:]
    
    
    cross_sections={'000':section_000,
                   '090':section_090,
                   '045':section_045,
                   '135':section_135
        
        
        }
    return cross_sections

def get_peaks_troughs(crossSection,trimSize=5,prominence=200,distance=10):  # finds the peaks and troughs 
    xSection2=(crossSection[trimSize:-trimSize]).astype(int) # removing the leading and trailing part of the cross section for cleaner signal processing
    
    peaks, _ = signal.find_peaks(xSection2,prominence=200,distance=10)
    troughs,_=signal.find_peaks(-xSection2,prominence=200,distance=10)
    
    if len(peaks)>1 or len(troughs)>2:
        print('Check cross section manually. More than 1 peak or 2 troughs')
    
    return xSection2,peaks, troughs


def get_peaks_analysis(crossSections,keys=['ctr_ctr'],key_angle='000'):
    colors=['r','g','b','k']
    marker=['X','s','P','8']
    
    # df=pd.DataFrame(columns=['Area','Slice_angle','Central_Contrast','Trough_Contrast'])# init dataframe
    
    dataList=[]
    plt.figure()
    for count,key in enumerate(keys):
        xSection0=crossSections[key][key_angle]
        xSection,peaks,troughs=get_peaks_troughs(xSection0)
        peakContrast=(np.mean(xSection[peaks])-np.mean(xSection[troughs]))/(np.mean(xSection[peaks])+np.mean(xSection[troughs]))*100
        troughContrast=np.diff(xSection[troughs])/np.sum(xSection[troughs])*100
        
        linestyle=colors[count]+'--'
        plt.plot(xSection,linestyle,label=key)
        plt.plot(peaks,xSection[peaks],colors[count]+marker[count])
        plt.plot(troughs,xSection[troughs],colors[count]+marker[count])
        plt.grid(True)
        plt.legend()
        plt.title(f'cross_section_angle: {key_angle}')
        
        # print(f' area: {key} \n direction: {key_angle} deg')
        # print(f' Central Contrast: {peakContrast} \n Trough Contrast {troughContrast}')
        
        data={'Area': key,
              'Slice_angle': key_angle,
              'Central_Contrast': np.round(peakContrast,1),
              'Trough_Contrast':np.round(troughContrast[0],1)            
            }
        
        dataList.append(data)
        # df0=pd.DataFrame(data)
        # print(df0)
    df=pd.DataFrame(dataList)
    
    print(df)
    return df
        
#%%
binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2023-03-27_Volpi_Visit\Cyc7_A2\PS_beads_run04_df_allblue'
#binPath=r'Z:\raspberrypi\photos\FAT_Captures\cyc6Dionysus\2023-02-09\run100_FAT5b_sample_Ronchi-1_Red_puck_dionysus'
# binPath=r'\\files.vital.company\cyclops\raspberrypi\photos\Erics Sandbox\2022-10-04_RONCHI_CollimatedvsDiffused\run05_FOV2_Diffused_exp10000_012A_sample_SNA_RONCHI__Cyc5Artemis'

key='img_blank'

resultPath=os.path.join(binPath,f'analysis_final_{key}')

ims=[]
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith('tiff'):
        print(file)
        im=plt.imread(os.path.join(binPath,file))
        ims.append(im)
        

im_main=ims[0]

coord_list=get_roi(im_main.shape,n=5)


## finding the ROI center of different area in the FOV
coords_dict={'top_left':coord_list[0],
        'top_ctr':coord_list[10],
        'top_right':coord_list[20],
        'ctr_left':coord_list[2],
        'ctr_ctr':coord_list[12],
        'ctr_right':coord_list[22],
        'bot_left':coord_list[4],
        'bot_ctr':coord_list[14],
        'bot_right':coord_list[24]
        }

#%%  


## keys for specific areas
keys=coords_dict.keys() 
imCrop_dict={}
crossSections={}

for key in keys:
    zoom_ctr=coords_dict[key]
    print(f'Area of interest: {key}')
    print(f'zoom_center : {zoom_ctr}')
    print('select a bead at the center by click and enter. Press spacebar if you want to redo selection')
    HH=userROI(im_main, no_of_corners=1, sort=False, zoom_ctr=zoom_ctr,figure_title=key)
    window_size=51
    coords=HH.coords[0]
    rad,ang,diff=get_imRadius(coords)
    roi={'x0':coords[0]-window_size//2,
      'x1':coords[0]+window_size//2+1,
      'y0':coords[1]-window_size//2,
      'y1':coords[1]+window_size//2+1     
      }
    imCrop_ctr=im_main[roi['y0']:roi['y1'],roi['x0']:roi['x1']]
    imCrop_dict[key]=imCrop_ctr
    crossSections[key]=get_cross_sections(imCrop_ctr)
    # plt.close('all')
    
# plt.figure()
# plt.plot(crossSections[keys[0]]['000'])    
# plt.plot(crossSections[keys[1]]['000'])    
# plt.plot(crossSections[keys[2]]['000'])


#%% for horizontal cross section at the middle, angle key= '000'

key_horz=['ctr_left','ctr_ctr','ctr_right']
df_table_000=get_peaks_analysis(crossSections,key_horz,'000')   

#%% for vertical cross section at the middle, angle key= '090'
key_vert=['top_ctr','ctr_ctr','bot_ctr']
df_table_090=get_peaks_analysis(crossSections,key_vert,'090')


#%% for 45 deg cross section at the middle, angle_key = '045'
key_45=['bot_left','ctr_ctr','top_right']
df_table_045=get_peaks_analysis(crossSections,key_45,'045')

#%% for 135 deg cross sectoin at the middle, angle_key = '135'
key_135=['bot_right','ctr_ctr','top_left']
df_table_135=get_peaks_analysis(crossSections,key_135,'135')

#%% 
n=len(keys)
nr,nc=n//3,n//3

fig,ax=plt.subplots(nr,nc,sharex=True,sharey=True)

for count,key in enumerate(keys):
    # print(key)
    # print(count)
    r,c=count//nr,count%nc
    # print(f'r:{r} c:{c}')
    ax[r,c].imshow(imCrop_dict[key],cmap='gray')
    ax[r,c].set_title(f'{key}')


#%% finding peaks and troughs of cross sections
# xSection=crossSections[keys[1]]['000']

# trim_size=5

# xSection2=(xSection[trim_size:-trim_size]).astype(int) # removing the leading and trailing part of the cross section for cleaner signal processing

# plt.figure()
# plt.plot(xSection2)

# peaks, _ = signal.find_peaks(xSection2,prominence=200,distance=10)
# troughs,_=signal.find_peaks(-xSection2,prominence=200,distance=10)
# plt.figure()
# plt.plot(xSection2,'r--')
# plt.plot(peaks,xSection2[peaks],'rs')
# plt.plot(troughs,xSection2[troughs],'rs')
# plt.grid(True)

# peakContrast=(np.mean(xSection2[peaks])-np.mean(xSection2[troughs]))/(np.mean(xSection2[peaks])+np.mean(xSection2[troughs]))*100
# troughContrast=np.diff(xSection2[troughs])/np.sum(xSection2[troughs])*100

# print(peakContrast)
# print(troughContrast[0])



#%%


        
    
     


#%%






#%%

# print('select a bead at the center by click and enter. Press spacebar if you want to redo selection')







# HH = userROI(im_main, no_of_corners=1, sort=False, zoom_ctr=[2112,1560])
# window_size=50
# coords=HH.coords[0]
# rad,ang,diff=get_imRadius(coords)


# roi={'x0':coords[0]-window_size//2,
#      'x1':coords[0]+window_size//2+1,
#      'y0':coords[1]-window_size//2,
#      'y1':coords[1]+window_size//2+1     
#      }
# #%
# imCrop_ctr=im_main[roi['y0']:roi['y1'],roi['x0']:roi['x1']]


#%% get cross section


    
# x_sections=  get_cross_sections(imCrop_ctr)  

# mid_point=window_size//2

# section_090=imCrop_ctr[:,mid_point]
# section_000=imCrop_ctr[mid_point,:]

# #%
# rot_angle=45
# imCrop_ctr_rot=im_rotated = ndimage.rotate(imCrop_ctr, rot_angle,reshape=False,order=1)

# plt.figure()
# plt.imshow(imCrop_ctr_rot)

# section_045=imCrop_ctr_rot[:,mid_point]
# section_135=imCrop_ctr_rot[mid_point,:]

# plt.figure()
# plt.plot(section_000)
# plt.plot(section_090)
# plt.plot(section_045)
# plt.plot(section_135)



