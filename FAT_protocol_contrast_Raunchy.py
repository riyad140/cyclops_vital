# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:32:39 2022

@author: imrul
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:14:33 2022

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

# from ROI_manager_user import userROI
import os
import sys
import cv2
import math
from scipy import ndimage, signal
from ROI_manager_user import userROI

def rot(image, xy, angle):
    im_rot = ndimage.rotate(image,angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot, new+rot_center

def get_imRadius(coords,imSize=[1520,2024]):
#%    
    mid_coords=[imSize[1]//2,imSize[0]//2]
    diff=np.array(coords)-np.array(mid_coords)

    distance=np.sqrt(diff[0]**2+diff[1]**2)
    max_distance=np.sqrt(mid_coords[0]**2+mid_coords[1]**2)
    imRadius_pt=distance/max_distance*100
    
    imAngle=np.degrees(math.atan(-diff[1]/diff[0]))
    
    return imRadius_pt,imAngle,diff

def get_roi(imShape,n=3):
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
            
            
    
#%%



#%%
def get_contrast(im_main,initial_coords,binPath,resultPath,name='gg',closeFigure=False):
    # im_main=ims[0]
    
    # GG=userROI(im_main,no_of_corners=1,sort=False) 
    # resultPath=os.path.join(binPath,'analysis')
    
    if os.path.exists(resultPath) is False:
        os.mkdir(resultPath)
    coords=initial_coords
    window_size=200
    
    rad,ang,diff=get_imRadius(coords,np.array(im_main.shape)) # getting ROI position in the image
    
    
    
    im_snip=im_main[coords[1]-window_size:coords[1]+window_size,coords[0]-window_size:coords[0]+window_size]

    
    
    
    #%%
    
    #rotation angle in degree
    im_rotated = ndimage.rotate(im_snip, 60.2,reshape=False,order=1)
    
    # plt.figure()
    # plt.imshow(im_rotated)
    

    
    #%%
    # HH = userROI(im_rotated, no_of_corners=2, sort=False, rectangle=True)
    HH_coords=[[window_size//2,window_size//2],[3*window_size//2,3*window_size//2]]
    #%
    buffer=1
    imCrop=im_rotated[HH_coords[0][1]:HH_coords[1][1],HH_coords[0][0]:HH_coords[1][0]]
    imCrop=imCrop[buffer:-buffer,buffer:-buffer]
    
    # offset_px=40 # slice width to check for a positive edge at the beginning
    # if np.nanmean(imCrop[:,offset_px])<
    
    plt.figure()
    plt.imshow(imCrop,cmap='gray')
    figName=f'{name}_coords_{initial_coords}_crop.png'
    plt.savefig(os.path.join(resultPath,figName))
    
    #%
    xSection=np.nanmean(imCrop,axis=0)
    # plt.figure()
    # plt.plot(xSection)
    #%
    peaks, _ = signal.find_peaks(xSection,prominence=20,distance=5)
    troughs,_=signal.find_peaks(-xSection,prominence=20,distance=5)
    plt.figure()
    plt.plot(xSection)
    plt.plot(peaks,xSection[peaks],'ro')
    plt.plot(troughs,xSection[troughs],'bo')
    plt.grid(True)
    plt.title(f'coords: {initial_coords} Cross Section')
    plt.xlabel('Pixels [px]')
    plt.ylabel('Intensity [a.u.]')
    figName=f'{name}_coords_{initial_coords}_cross_section_0.png'
    plt.savefig(os.path.join(resultPath,figName))
    
    
    levels=[np.mean(xSection[peaks]),np.mean(xSection[troughs])]
    contrast=abs(np.diff(levels)/np.sum(levels)*100)
    print(f'Contrast Ratio : {contrast} %')
    
    #% PDR
    spatial_freq=80 #lp/mm
    interpolation_factor=8
    bayer_factor=1
    px_size=1.1e3 #nm
    
    mid_intensity=np.mean(levels)
    
    y=xSection-mid_intensity
    
    
    xx=range(len(y))
    y_int=np.interp(np.linspace(xx[0],xx[-1],len(xx)*interpolation_factor),xx,y)
    
    # x_int=np.interp(range(len(x)*interpolation_factor),range(len(x)),x)
    
    
    zc=np.diff(np.sign(y_int))
    
    zc_ind_p=np.where(zc>0)[0]
    zc_ind_n=np.where(zc<0)[0]
    
    pos_edge_transition=abs(np.diff(np.where(zc>0)[0])[0])
    neg_edge_transition=abs(np.diff(np.where(zc<0)[0])[0])
    
    
    plt.figure()
    
    plt.plot(y_int)
    plt.plot(zc_ind_p,y_int[zc_ind_p],'ro')
    plt.plot(zc_ind_n,y_int[zc_ind_n],'bo')
    plt.grid(True)
    plt.title(f'coords: {initial_coords} Cross Section')
    plt.xlabel('Pixels [px]')
    plt.ylabel('Intensity [a.u.]')
    figName=f'{name}_coords_{initial_coords}_cross_section_1.png'
    plt.savefig(os.path.join(resultPath,figName))
    
    lp_px=np.nanmean([pos_edge_transition,neg_edge_transition])*bayer_factor/interpolation_factor
    lp_nm=1/spatial_freq/1e-6
    
    
    pixel_to_distance_ratio=lp_nm/lp_px  #nm/px
    magnification=px_size/pixel_to_distance_ratio
    hfov=pixel_to_distance_ratio*im_main.shape[1]*bayer_factor/1e3  #um
    vfov=pixel_to_distance_ratio*im_main.shape[0]*bayer_factor/1e3  #um
    
    
    print(f'ROI Radius {rad} %\nROI Angle {ang}')
    print(f'ROI [X,Y]: [{coords[0]},{coords[1]}]')
    print(f'pixel to distance ratio : {pixel_to_distance_ratio} nm/px')
    print(f'FOV: {hfov} um * {vfov} um')
    print(f'Magnification : {magnification} x')
    
    
    # plt.figure()
    # #plt.plot(x)
    # plt.plot(np.diff(np.sign(x)))
    if closeFigure is True:
        plt.close('all')
    return contrast,pixel_to_distance_ratio,magnification
#%%
# binPath=r'Z:\raspberrypi\photos\FAT_Captures\run03_sample_FAT_v3_usaf_Cyc4Juno\stepping'
binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\FAT_Captures\cyc5apollo\2022-08-19'

key='Rn2_BF-FOV_0'
n=5
resultPath=os.path.join(binPath,f'analysis_final_n{n}_{key}')
channel=2 # BGR

ims=[]
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith('bmp'):
        print(file)
        im=cv2.imread(os.path.join(binPath,file))
        ims.append(im[:,:,channel])

im_main=ims[0]


coords_list=get_roi(im_main.shape,n)
result=[]

for i,coords in enumerate(coords_list):
    con,pdr,mag=get_contrast(im_main,coords,binPath,resultPath,name='sector_'+str(i),closeFigure=True)
    result.append([con,pdr,mag])
    
#%%
result=np.array(result,dtype=float)
con_map=result[:,0].reshape(n,n)
pdr_map=result[:,1].reshape(n,n)

plt.figure()
plt.imshow(con_map,cmap='jet_r')
plt.colorbar()
plt.title('Contrast heat map')
figName=f'Contrast_heat_map_{key}.png'
plt.savefig(os.path.join(resultPath,figName))


plt.figure()
plt.imshow(pdr_map,cmap='jet_r')
plt.colorbar()
plt.title('PDR heat map')
figName=f'PDR_heat_map_{key}.png'
plt.savefig(os.path.join(resultPath,figName))
# fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
# ax[0].imshow(a)
# ax[0].set_colorbar()
# ax[1].imshow(a)
# ax[1].colorbar()