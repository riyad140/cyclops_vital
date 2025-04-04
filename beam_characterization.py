# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:23:17 2022

@author: imrul
"""
#%%
import rawpy
import imageio
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
from glob import glob
import cv2
from ROI_manager_user import userROI

def normalize(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))
def get_mid_point_location(xSection,interpolation_factor=100):
    y=normalize(xSection)-0.5
    x=np.linspace(0,len(y)-1,len(y))
    xp=np.linspace(x[0],x[-1],len(x)*interpolation_factor)    
    yp=np.interp(xp,x,y)
    zc=np.diff(np.sign(yp))
    zc_ind=np.where(zc!=0)[0]    
    mid_point_location=xp[zc_ind]
    return mid_point_location




#r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2022-06-24_FL_beam_characterization\baseline_beam_run00_0001.ascii.csv"

#r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2022-06-24_FL_beam_characterization\thorlabs_secnario_7_run2_0008.ascii.csv"
def get_beam_profile(imPath, buffer= [500,50],gain_db=3.99, exp_ms=0.13, attenuation_filter_db=0, label='baseline' ):
    
    im=np.genfromtxt(imPath, delimiter=',')
    im=im[:,:-1]   
    HH = userROI(im, no_of_corners=1, sort=False, rectangle=False)
    
    #%%
    # buffer=[500,50]
    roi_x={'x0':HH.coords[0][0]-buffer[0],'x1':HH.coords[0][0]+buffer[0],
           'y0':HH.coords[0][1]-buffer[1],'y1':HH.coords[0][1]+buffer[1]}
    
    roi_y={'x0':HH.coords[0][0]-buffer[1],'x1':HH.coords[0][0]+buffer[1],
           'y0':HH.coords[0][1]-buffer[0],'y1':HH.coords[0][1]+buffer[0]}
    
    im_x=im[roi_x['y0']:roi_x['y1'],roi_x['x0']:roi_x['x1']]
    im_y=im[roi_y['y0']:roi_y['y1'],roi_y['x0']:roi_y['x1']]
    
    beam_x=np.nanmean(im_x,axis=0)
    beam_y=np.nanmean(im_y,axis=1)
    fig,ax=plt.subplots(2,1)
    ax[0].plot(beam_x)
    ax[0].set_title('beam x cross section')
    ax[1].plot(beam_y)
    ax[1].set_title('beam y cross section')
    
    
    #%%
    # gain_db=3.99
    # exp_ms=0.13
    gain=10**(gain_db/10)
    
    # attenuation_filter_db=0
    att=10**(attenuation_filter_db/10)
    
    sensor_size_um=[7015,5278]
    px_to_um_ratio=np.mean([sensor_size_um])/np.mean(im.shape)
    
    beam_x_norm=beam_x/exp_ms/gain*att
    beam_y_norm=beam_y/exp_ms/gain*att
    
    mid_points=get_mid_point_location(beam_x_norm,interpolation_factor=1)
    fwhm_x=(mid_points[1]-mid_points[0])*px_to_um_ratio
    mid_points=get_mid_point_location(beam_y_norm,interpolation_factor=1)
    fwhm_y=(mid_points[1]-mid_points[0])*px_to_um_ratio
    
    peak_power=np.max(beam_x_norm)
    blk_lvl=np.min(beam_x_norm)
    ext_ratio=peak_power//blk_lvl
    
    spatial_axis=np.linspace(-buffer[0],buffer[0],2*buffer[0])*px_to_um_ratio
    
    plt.figure(100)
    plt.plot(np.linspace(-buffer[0],buffer[0],2*buffer[0])*px_to_um_ratio,beam_x_norm,label=label)
    plt.xlabel('Beam Size X [um]')
    plt.legend()
    plt.grid(True)
    
    plt.figure(101)
    plt.plot(np.linspace(-len(beam_y_norm)//2,len(beam_y_norm)//2,len(beam_y_norm))*px_to_um_ratio,beam_y_norm,label=label)
    plt.xlabel('Beam Size Y [um]')
    plt.legend()
    plt.grid(True)
    print(imPath[-38:-10])
    print(f'FWHM_x = {fwhm_x} um \nFWHM_y = {fwhm_y} um')
    print(f'Max Power = {peak_power} a.u. \nBlk Level = {blk_lvl} a.u. \nExtinction Ratio = {ext_ratio}')
    
    return beam_x_norm,beam_y_norm, spatial_axis
#%% 

imPath=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2022-06-24_FL_beam_characterization\edmund_secnario2_run0_0009.ascii.csv"
get_beam_profile(imPath, buffer= [500,50],gain_db=2.10, exp_ms=0.32, attenuation_filter_db=10, label='edmund_scenario_2' )