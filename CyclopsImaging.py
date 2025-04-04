# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 19:12:36 2022

@author: imrul
"""
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import laplace #for edge detection filters
import skimage.io #for image processing
import numpy as np #for processing images as numpy arrays
import sys 
import os #for using BASH commands
import subprocess #for using BASH commands
import re
import RPi.GPIO as gpio #For controlling accessories by GPIO outputs
from smbus import SMBus #For I2C controls - AD5821A - VCM
from time import sleep  #For delays
import time
from datetime import datetime
import json
from picamera import PiCamera #for camera controls
import os
from cv2 import blur
from tqdm import tqdm
import logging
import pickle
from vital.phoenix import *
import asyncio
from vital.phoenix import *
import asyncio
import RPi.GPIO as gp
import os
import subprocess
import re
from smbus import SMBus
from time import sleep
from time import time
import sys

#%%
class CyclopsImaging:
    
    SSPin = 35 #digital output for the Red Side Scatter LEDs
    DFPin = 36 #Darkfield White LEDs 
    BFPin = 37 #Brightfield LED 
    FLPin = 38 #Blue Laser Diode
    
    
    i2cbus = SMBus(3) #create a new I2C bus
    ADDRESS = 0x0c #address of the AD5821A
    REGISTER = 0x00 #null register
    defaultStep = 300 #motor position (0-1024), middle assumed to be 512, or ((1/6 x 1024)/2)+(1/6 x 1024)=597

    gp.setwarnings(False)
    gp.setmode(gp.BOARD)
    
    gp.setup(SSPin, gp.OUT)
    gp.setup(DFPin, gp.OUT)
    gp.setup(BFPin, gp.OUT)
    gp.setup(FLPin, gp.OUT)       
    
    gp.setup(7, gp.OUT)
    gp.setup(11, gp.OUT)
    gp.setup(12, gp.OUT)
    
    gp.setup(15, gp.OUT)
    gp.setup(16, gp.OUT)
    gp.setup(21, gp.OUT)
    gp.setup(22, gp.OUT)
    
    gp.output(11, True)
    gp.output(12, True)
    gp.output(15, True)
    gp.output(16, True)
    gp.output(21, True)
    gp.output(22, True)
    
    # defaultFileName = ' -o testimage_000' #filepath and name
    # defaultAWB = ' -awb off -awbg 1.0,1.0' #AWB settings
    # defaultWidth = ' -w 4056'      #sensor width in pixels
    # defaultHeight = ' -h 3040'     #sensor height in pixels
    # defaultShutterSpeed = ' -ss 20000' #shutter speed in us
    # defaultISO = ' -ag 1.0 -dg 1.0'            #ISO setting
    # defaultExposure = ' -ex auto'   #exposure mode
    # defaultDelay = ' -t 1'
    # defaultPreview = ' -n'
    # defaultFileExtention='.png'
    afWidth = ' -w 1014'
    afHeight = ' -h 760'
    defaultWidth = ' -w 4056'      #sensor width in pixels
    defaultHeight = ' -h 3040'     #sensor height in pixels
    defaultAWB = ' -awb off -awbg 1.0,1.0'#' -awb auto'#' -awb off -awbg 1.0,1.0' #AWB settings
    defaultShutterSpeed = ' -ss 500000' #shutter speed in us
    defaultISO = ' -ag 1.0 -dg 1.0'            #ISO setting
    defaultExposure = ' -ex auto'   #exposure mode
    defaultDelay = ' -t 1'
    defaultPreview = ' -n'
    defaultFileExtention='.png'
    
    
    def __init__(self):
        print('INIT') 
        cell_no='s667'#'s647'#'468_s18'#'464b_s18'
        suffix='Cyc4Metal_new-lightring'#'Cyc4Stable_Met_comp_WBC_diff'
        Folder_name='run000'
        self.testbench='testbench-8.vital.company'
        self.disable_progress_bar=False
        self.log_level_stderr=True
        self.defaultFOVrange = 16 #number of FOVs to examine for the test, minimum value 1
        self.defaultFOVresolution = 5 #Angular Resolution= defaultFOVresolution/10 (degree)
        self.home_deg=0# Initial Starting position of the BLDC motor, original_home_deg=home_deg/10
        self.fov_zigzag=True
        self.raw_capture=True # to enable raw capture        
        self.HDR=False  # to enable 3 frames at different exposure instead of 1 frame
        self.disable_motor=False
        self.AFpin='BF'  # DF BF SS # Specify which light source to be used for autofocus
        self.AF_camera='G' # 'R'
        
        # create necessary directories
        now=datetime.now()
        current_date=now.strftime('%Y-%m-%d')
        current_time=now.strftime('%H-%M-%S')
        current_folder=Folder_name+'_sample_'+cell_no+'_'+suffix
        
        if os.path.isdir(current_date)==False:
            os.mkdir(current_date)
        self.defaultFolderName=os.path.join(current_date,current_folder)
        if os.path.isdir(defaultFolderName)==False:
            os.mkdir(defaultFolderName)
        self.default_AF_folder=os.path.join(defaultFolderName,'AF')
        if os.path.isdir(default_AF_folder)==False:
            os.mkdir(default_AF_folder)
            
        #% logging
        logName='cyclops.log'
        # f=open(os.path.join(current_folder,logName),'w')
        # f.close()
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="[#%(asctime)s][%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(defaultFolderName,logName))
            ] + ([logging.StreamHandler()] if log_level_stderr else [])
        )
        
    def set_capture_settings(self):

        #%######################### BF DF SS FL FL_G#########################
        exif_data={'shutter_speed':[1.0,400,400,700, 300], 
                   'analog_gain':[1.0,2.0,1.5,2.0, 1.0]        } # in miliseconds
        self.exif_df=pd.DataFrame.from_dict(exif_data,orient='index',columns=['BF','DF','SS','FL','FL_G'])
        ################################################################
        logging.info(exif_df)
        self.BFShutterSpeed =' -ss '+str(exif_df['BF']['shutter_speed']*1e3) #' -ss 20000'
        self.DFShutterSpeed =' -ss '+str(exif_df['DF']['shutter_speed']*1e3)  #' -ss 300000'
        self.SSShutterSpeed =' -ss '+str(exif_df['SS']['shutter_speed']*1e3)  #' -ss 300000'
        self.FluorShutterSpeed =' -ss '+str(exif_df['FL']['shutter_speed']*1e3)  #' -ss 300000' #us
        self.FluorShutterSpeed_G =' -ss '+str(exif_df['FL_G']['shutter_speed']*1e3) 
        
        self.defaultISO = ' -ISO 200'            #ISO setting
        self.BFISO =' -ag '+str(exif_df['BF']['analog_gain'])+' -dg 1.0'      #' -ISO '+str(exif_df['BF']['iso']) #' -ISO 100'
        self.DFISO =' -ag '+str(exif_df['DF']['analog_gain'])+' -dg 1.0'  #' -ag 4.0 -dg 1.0'     #' -ISO '+str(exif_df['DF']['iso']) #' -ISO 100'
        self.SSISO =' -ag '+str(exif_df['SS']['analog_gain'])+' -dg 1.0'  #' -ag 1.0 -dg 1.0'     #' -ISO '+str(exif_df['SS']['iso']) #' -ISO 100'
        self.FluorISO =' -ag '+str(exif_df['FL']['analog_gain'])+' -dg 1.0'  #' -ag 1.0 -dg 1.0'   #' -ISO '+str(exif_df['FL']['iso']) #' -ISO 100'       
        self.FluorISO_G =' -ag '+str(exif_df['FL_G']['analog_gain'])+' -dg 1.0'

        if self.raw_capture==True:
            logging.info('RAW captures enabled')
            defaultRaw=' -r '
        else:
            defaultRaw=''
            
        if self.HDR is True:
            self.lo_factor=0.5 # factor for exposure time multiplication for HDR
            self.hi_factor=2
            self.DFShutterSpeed_lo=' -ss '+str(exif_df['DF']['shutter_speed']*1e3*lo_factor)  
            self.DFShutterSpeed_hi=' -ss '+str(exif_df['DF']['shutter_speed']*1e3*hi_factor)
            
            self.FluorShutterSpeed_lo =' -ss '+str(exif_df['FL']['shutter_speed']*1e3*lo_factor)
            self.FluorShutterSpeed_hi =' -ss '+str(exif_df['FL']['shutter_speed']*1e3*hi_factor)
            self.FluorShutterSpeed_G_lo =' -ss '+str(exif_df['FL_G']['shutter_speed']*1e3*lo_factor) 
            self.FluorShutterSpeed_G_hi =' -ss '+str(exif_df['FL_G']['shutter_speed']*1e3*hi_factor) 
        