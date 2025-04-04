# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:58:38 2021

@author: imrul
"""


#Import statements
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

# base_dir=r'/home/pi/Desktop/Fileserver/Cyclops_repo/'
# os.chdir(os.path.dirname(base_dir))
# camera=PiCamera()
#%% USER INPUT 
cell_no='beads'#'s678_WBC_diff_Meth_Comp_DF_1400_FCell_+60PBS'#'s647'#'468_s18'#'464b_s18'
suffix='Cyc4Metal'#'Cyc4Stable_Met_comp_WBC_diff'
Folder_name='run00'
testbench='testbench-8.vital.company'
disable_progress_bar=False
log_level_stderr=True
defaultFOVrange = 1 #number of FOVs to examine for the test, minimum value 1
defaultFOVresolution = 5 #Angular Resolution= defaultFOVresolution/10 (degree)
home_deg=0# Initial Starting position of the BLDC motor, original_home_deg=home_deg/10
fov_zigzag=True
raw_capture=True # to enable raw capture

HDR=False  # to enable 3 frames at different exposure instead of 1 frame
   
init_accuation=True
init_range=32

disable_motor= True
AFpin='BF'  # DF BF SS # Specify which light source to be used for autofocus
AF_camera='G' # 'R'
disable_AF=False

#%% Making directory with specific folder structure with timestamps
now=datetime.now()
current_date=now.strftime('%Y-%m-%d')
current_time=now.strftime('%H-%M-%S')
current_folder=Folder_name+'_sample_'+cell_no+'_'+suffix

if os.path.isdir(current_date)==False:
    os.mkdir(current_date)
defaultFolderName=os.path.join(current_date,current_folder)
if os.path.isdir(defaultFolderName)==False:
    os.mkdir(defaultFolderName)
default_AF_folder=os.path.join(defaultFolderName,'AF')
if os.path.isdir(default_AF_folder)==False:
    os.mkdir(default_AF_folder)

#%% logging
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
    

#%%########################### BF DF SS FL FL_G#########################
exif_data={'shutter_speed':[1.0,1400,2000,600, 150], 
           'analog_gain':[1.0,2.0,1.5,2.0, 1.0]        } # in miliseconds
exif_df=pd.DataFrame.from_dict(exif_data,orient='index',columns=['BF','DF','SS','FL','FL_G'])
################################################################
logging.info(exif_df)


####################################################################################3
#declare arrays and variables // Could be local to the zstack capture loop
sharpnesses = [] #create null array to contain sharpness values
fnames = [] #create null array to contain filenames
afIndex = [] #create null array to contain autofocus indices

# camera = PiCamera() #create a camera object
filename='cyclops-'
defaultFname = os.path.join(defaultFolderName,filename)
defaultAFname= os.path.join(default_AF_folder,filename)
afWidth = ' -w 1014'
afHeight = ' -h 760'
defaultWidth = ' -w 4056'      #sensor width in pixels
defaultHeight = ' -h 3040'     #sensor height in pixels
defaultAWB = ' -awb off -awbg 1.0,1.0'#' -awb auto'#' -awb off -awbg 1.0,1.0' #AWB settings
defaultShutterSpeed = ' -ss 500000' #shutter speed in us
if raw_capture==True:
    logging.info('RAW captures enabled')
    defaultRaw=' -r '
else:
    defaultRaw=''

# AFShutterSpeed = ' -ss 500000'
BFShutterSpeed =' -ss '+str(exif_df['BF']['shutter_speed']*1e3) #' -ss 20000'
DFShutterSpeed =' -ss '+str(exif_df['DF']['shutter_speed']*1e3)  #' -ss 300000'
SSShutterSpeed =' -ss '+str(exif_df['SS']['shutter_speed']*1e3)  #' -ss 300000'
FluorShutterSpeed =' -ss '+str(exif_df['FL']['shutter_speed']*1e3)  #' -ss 300000' #us



defaultISO = ' -ISO 200'            #ISO setting
BFISO =' -ag '+str(exif_df['BF']['analog_gain'])+' -dg 1.0'      #' -ISO '+str(exif_df['BF']['iso']) #' -ISO 100'
DFISO =' -ag '+str(exif_df['DF']['analog_gain'])+' -dg 1.0'  #' -ag 4.0 -dg 1.0'     #' -ISO '+str(exif_df['DF']['iso']) #' -ISO 100'
SSISO =' -ag '+str(exif_df['SS']['analog_gain'])+' -dg 1.0'  #' -ag 1.0 -dg 1.0'     #' -ISO '+str(exif_df['SS']['iso']) #' -ISO 100'
FluorISO =' -ag '+str(exif_df['FL']['analog_gain'])+' -dg 1.0'  #' -ag 1.0 -dg 1.0'   #' -ISO '+str(exif_df['FL']['iso']) #' -ISO 100'

FluorShutterSpeed_G =' -ss '+str(exif_df['FL_G']['shutter_speed']*1e3) 
FluorISO_G =' -ag '+str(exif_df['FL_G']['analog_gain'])+' -dg 1.0'
FluorDRC=' -drc off'

lo_factor=0.5 # factor for exposure time multiplication for HDR
hi_factor=2

DFShutterSpeed_lo=' -ss '+str(exif_df['DF']['shutter_speed']*1e3*lo_factor)  
DFShutterSpeed_hi=' -ss '+str(exif_df['DF']['shutter_speed']*1e3*hi_factor)

FluorShutterSpeed_lo =' -ss '+str(exif_df['FL']['shutter_speed']*1e3*lo_factor)
FluorShutterSpeed_hi =' -ss '+str(exif_df['FL']['shutter_speed']*1e3*hi_factor)
FluorShutterSpeed_G_lo =' -ss '+str(exif_df['FL_G']['shutter_speed']*1e3*lo_factor) 
FluorShutterSpeed_G_hi =' -ss '+str(exif_df['FL_G']['shutter_speed']*1e3*hi_factor) 


defaultExposure = ' -ex off'   #exposure mode
defaultDelay = ' -t 1'
defaultPreview = ' -n'
defaultIFX = ' -set '
defaultGain=' -ag 8.0'


#define default motor position and scan ranges
fov = 0 #counter to remember the step index
defaultZstep = 600 #motor position (0-1024), middle assumed to be 512, or ((1/6 x 1024)/2)+(1/6 x 1024)=597
defaultZresolution = 5 # 1 vcm unit= 0.7 um. so 5 vcm unit = 3.5 um 
defaultZrange = 30 #(+- z range)number of z locations above and below the default Z to sweep


#define VCM address
i2cbus = SMBus(3) #create a new I2C bus (3 is used as the new SDA and SCL pins are used for bus 3 - bus 1 default is used by the MCA board now)
ADDRESS = 0x0c    #address of the AD5821A
REGISTER = 0x00   #null register

#####*** GPIO PIN DEFINITIONS FOR LED, LASER, and MOTOR CONTROL ***####
PULPin = 33 #GPIO 13 / PWM1 is Board pin 33
DIRPin = 32 #GPIO 12 / PWM0 is Board pin 32
PWRPin = 40 #pin 40 used as a positive supply pin to the motor driver
#NOTE the enable pin is grounded at the moment to leave enabled, this can be changed to an available GPIO pin if one is available

SSPin = 35 #digital output for the Red Side Scatter LEDs
DFPin = 36 #Darkfield White LEDs 
BFLEDPin = 37 #Brightfield LED 
FluorPin = 38 #Blue Laser Diode

#define and initialize GPIO pins and settings
gpio.setmode(gpio.BOARD) #we are using pin numbers, not GPIO numbers
gpio.setwarnings(False)
gpio.setup(PULPin, gpio.OUT) #set the pin as an output for trigger steps
gpio.setup(DIRPin, gpio.OUT) #set the pin as an output for supply to Direction
gpio.setup(PWRPin, gpio.OUT)
gpio.setup(BFLEDPin, gpio.OUT) #set the pin as an output for trigger
gpio.setup(FluorPin, gpio.OUT) #set the pin as an output for power supply to LED
gpio.setup(DFPin, gpio.OUT) 
gpio.setup(SSPin, gpio.OUT)
gpio.output(PULPin, 0) #disable the trigger pin
gpio.output(DIRPin, 0) #disable the direction pin
gpio.output(PWRPin, 1) #enable the power pin to power the motor driver
gpio.output(BFLEDPin, 0) #disable the LED Pin on startup
gpio.output(FluorPin, 0) #disable the LASERPin on startup
gpio.output(DFPin, 0)
gpio.output(SSPin, 0)

#####*** GPIO PIN DEFINITIONS FOR MULTICAMERA ADAPTER ***####
gpio.setup(7, gpio.OUT) #set camera selection pin as output for MCA
gpio.setup(11, gpio.OUT) #set camera enable pin as output for MCA
gpio.setup(12, gpio.OUT) #set camera enable pin as output for MCA
gpio.setup(15, gpio.OUT) #define and initialize remaining MCA pins
gpio.setup(16, gpio.OUT)
gpio.setup(21, gpio.OUT)
gpio.setup(22, gpio.OUT)
gpio.output(11, True)
gpio.output(12, True)
gpio.output(15, True)
gpio.output(16, True)
gpio.output(21, True)
gpio.output(22, True)

#Function to calculate sharpnesses based on laplace filter
def fm_lape(im):
    fm=laplace(im)
    fm=np.mean(abs(fm))
    return(fm)

def fm_helm(image,WSIZE=15):
    u=blur(image,(WSIZE,WSIZE))
    r1=u/image
    r1[image==0]=1
    index = (u>image)
    fm=1/r1
    fm[index]=r1[index]
    fm=np.mean(fm)
    return(fm)
#Function to drive VCM
def driveVCM(i):
#     print("setting VCM position to " + str(i))
    #generate the appropriate bytes based on the starting position
    if (str(bin(i))==str(bin(i<<4))):
        binposition = str(bin(i))+'0000'
    else:
        binposition = str(bin(i << 4)) #shift the bits for trailing 0s
    if (len(binposition)<=10):
        b1 ='0b0'
        b2 = binposition
    else:    
        b1 = '0b'+binposition[(2):(len(binposition)-8)]
        b2 = '0b'+binposition[(len(binposition)-8):len(binposition)]
    #generate the initial bytes for the middle position of the motor  
    byte1 = int(b1,2)
    byte2 = int(b2,2)
    #drive the motor to the middle position and request the user to align the sample to continue
    i2cbus.write_byte_data(0x0c, byte1, byte2)
#     print("VCM at position" + str(i))

def runVCM(final_vcm,init_vcm=0): #slowly drive the vcm
    
    for i in np.linspace(init_vcm,final_vcm,abs(final_vcm-init_vcm)):
        driveVCM(int(i))
        
        

def aligncameras():
    #turn on the white LED to align the channels
    logging.info("Enabling White BF LED")
    gpio.output(BFLEDPin, 1)

    #Enable One camera and call a function to enable the preview
    logging.warning("Start testing the camera A - Beginning Preview")
    i2c = "i2cset -y 1 0x70 0x00 0x04"
    os.system(i2c)
    gpio.output(7, False)
    gpio.output(11, False)
    gpio.output(12, True)
    cmd = "raspistill -t 0 -awb greyworld -o capture_RED" + ".bmp"
    os.system(cmd)
        
    filename = defaultFname + "Alignment_RED.bmp"    
    
    #after alignment, capture an image to compare with the AF and BF images
    cmd = "raspistill" + defaultWidth + defaultHeight + BFShutterSpeed + BFISO + defaultExposure + defaultAWB + " -o " + filename    
    os.system(cmd)
    
    #finally turn off the White LED
    logging.debug("Alignment Complete - Disabling the White BF LED")
    gpio.output(BFLEDPin,0)
    
def captureAFImage(stepcount, zindex):
    #capture the BF image for AF
    filename = defaultAFname + "_AF_" + str(stepcount) + "_" + str(zindex) + ".bmp" #generate the file name and call the raspistill function
    if AFpin=='DF':
        AFShutterSpeed=DFShutterSpeed
        AFISO=DFISO
    else:
        AFShutterSpeed=BFShutterSpeed
        AFISO=BFISO
            
        
#     print("capturing AF image")
    cmd = "raspistill" + afWidth + afHeight + AFShutterSpeed + AFISO + defaultExposure + defaultDelay + defaultPreview + " -o " + filename    
    # print(cmd)
    os.system(cmd) #run the command from terminal
    
    #update arrays
    fnames.append(filename)
    afIndex.append(zindex)
    return filename

def evaluateSharpness():
    #process the image
    logging.debug("processing AF image for sharpness")
    for i in range (0, len(fnames),1):
        #for all images, if the sharpness for the index is not the sharpest
        im = skimage.io.imread(fname=fnames[i]) #determine the file name or path for each image
        im_resized=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3,0] # giving it only red channel image's central FOV
        fm=fm_lape(im_resized) #call the lapace filter function to process the image
#         fm=fm_helm(im_resized)
        sharpnesses.append(fm) #append the sharpness result to the array

def getSharpness(filename):
    logging.debug(f"Calculating sharpness of {filename}")
    im=skimage.io.imread(filename)
    im_resized=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3,0] # giving it only red channel image's central FOV
    sharpness=fm_lape(im_resized)
    return sharpness

#     print("Files cleaned up")        

# vcm_step=20
# vcm0=330
# maxIter=3

# v_s=[] # to store vcm and sharpness value side by side
# s=get_sharpness(vcm0)
# v_s.append([vcm0,s])

# for i in range(0,maxIter):
    
#     for v in [vcm0+vcm_step//2**i,vcm0-vcm_step//2**i]:
#         s=get_sharpness(v)
#         v_s.append([v,s])
    
#     af_arr=np.array(v_s)  # vcm,sharpness
    
#     maxInd=np.argmax(af_arr[:,1])
#     vcm0=int(af_arr[maxInd,0])
#     print(f'sharpestVCM: {vcm0}')


def smartAFRoutine(fov,centerZ=500,resolutionZ=5,maxIter=3,channel='R'):
    vcm_step=resolutionZ*2**(maxIter-1)
    vcm0=centerZ
    v_s=[] # to store vcm and sharpness value side by side
    runVCM(vcm0)
    selectCamera(channel)
    filename=captureAFImage(fov,vcm0)    
    s=getSharpness(filename)
    v_s.append([vcm0,s])
    
    for i in range(0,maxIter):
        last_vcm=vcm0
        
        for v in [vcm0+vcm_step//2**i,vcm0-vcm_step//2**i]:
            # vcm step size is progressively getting halved
            runVCM(v,last_vcm)
            last_vcm=v
            filename=captureAFImage(fov,v) 
            s=getSharpness(filename)
            v_s.append([v,s])
        
        af_arr=np.array(v_s)  # vcm,sharpness
        
        maxInd=np.argmax(af_arr[:,1])
        vcm0=int(af_arr[maxInd,0])
        logging.info(f'sharpestVCM: {vcm0}')
    logging.info('Driving VCM to sharpest plane')
    runVCM(vcm)
    
    
    
        
def selectCamera(channel='R'):
    if channel =='R':
        logging.warning("Selecting camera A - RED")
        i2c = "i2cset -y 1 0x70 0x00 0x04"
        os.system(i2c)
        gpio.output(7, False)
        gpio.output(11, False)
        gpio.output(12, True)
    else:
        logging.info("Selecting camera B - GREEN")
        i2c = "i2cset -y 1 0x70 0x00 0x05"
        os.system(i2c)
        gpio.output(7, True)
        gpio.output(11, False)
        gpio.output(12, True)
        
        

def automateAFRoutine(step, centerZ, rangeZ, resolutionZ, channel='R'):
    global defaultZstep #reference the defaultZstep beyond the function scope for update
    #clear the arrays each time the AF Routine is called for a new FOV
    del fnames[:]
    del afIndex[:]
    del sharpnesses[:]
    if channel =='R':
        logging.warning("Selecting camera A - RED for AutoFocus")
        i2c = "i2cset -y 1 0x70 0x00 0x04"
        os.system(i2c)
        gpio.output(7, False)
        gpio.output(11, False)
        gpio.output(12, True)
    else:
        logging.info("Selecting camera B - GREEN for Autofocus")
        i2c = "i2cset -y 1 0x70 0x00 0x05"
        os.system(i2c)
        gpio.output(7, True)
        gpio.output(11, False)
        gpio.output(12, True)
    
    logging.debug("AF Routine called and arrays cleared - enabling LED")
    t0=time.time()
    #start by enabling the BF White LED
    if AFpin=='DF':
        # print('Using DF for autofocus')
        gpio.output(DFPin, 1)
    elif AFpin=='SS':
        # print('Using SS for autofocus')
        gpio.output(SSPin,1)
    else:               
        gpio.output(BFLEDPin, 1) # SS light
    
    #use try statement to allow the user to abort the routine
    logging.debug("Stepping through z planes")
    #Engage auto focus (capture z stack for range of z about center)
    zindex = (centerZ - rangeZ) #we begin at the lower analysis position
    logging.info('Driving VCM and Taking AF captures')
    for zindex in tqdm(range((centerZ - rangeZ),(centerZ + rangeZ + 1), resolutionZ),disable=disable_progress_bar):
        #Drive VCM to target z layer
#         if count==0: # only to print the first time
#             print("Driving VCM")
#             count=count+1
        driveVCM(zindex) #z index is the step 0-1024, translated to two bytes and send to VCM driver
        #capture image and determine sharpness
#         print("Calling AF image Capture")
        captureAFImage(step, zindex) #capture the BF Images for coarse AF investigation
        
        #once the BF Images for AF are captured, the arrays are updated from the capture function
    
    logging.info("AF Images captured! - checking sharpnesses")
    evaluateSharpness()
    dt=time.time()-t0
    #Drive to the zindex of greatest sharpness
    sharpestIndex = afIndex[sharpnesses.index(max(sharpnesses))] #returns the zindex list item, based on the index of the first instance of the image of greatest sharpness. 
    logging.info("The sharpest image was at:" + str(sharpestIndex) + "_" + str(max(sharpnesses)))
    sharpness_arr= np.round((np.array(sharpnesses))/max(sharpnesses)*100,2)
    table=zip(afIndex,sharpness_arr)
#     print(*table)        
    logging.debug("Driving VCM to Plane of Sharpest Image")
    #after capturing images, return slowly to the SHARPEST FOCAL PLANE
    for zindex in range(zindex, (sharpestIndex-1), -1):
        driveVCM(zindex)
    logging.info("VCM Driven to sharpest plane")
    defaultZstep = sharpestIndex        
    #disable the White BF LED at the end of the autofocus routine
    logging.debug("Disabling the LED for AF lighting")
#     gpio.output(BFLEDPin, 0)
    if AFpin=='DF':
#         print('Using DF for autofocus')
        gpio.output(DFPin, 0)
    elif AFpin=='SS':
#         print('Using SS for autofocus')
        gpio.output(SSPin,0)
    else:               
        gpio.output(BFLEDPin, 0)
    return np.round(dt,3),sharpestIndex

def captureBFImage(channel, step, flip=False): #capture BF Images once at the sharpest focal plane
    #Enable the appropriate light source
    gpio.output(BFLEDPin, 1)
#    sleep(0.1)
    
    #format the file name for the images
    filename = defaultFname + channel + "_BF-"  + 'FOV_'+ str(step) + ".png" #generate the file name and call the raspistill function
    
    logging.info("capturing BF Image")
    t0=time.time()
    #capture the image
    if flip==False:
        cmd = "raspistill" + defaultWidth + defaultHeight + BFISO + BFShutterSpeed + defaultExposure + defaultDelay+defaultAWB+defaultRaw + " -o " + filename
    else:
        cmd = "raspistill" + defaultWidth + defaultHeight + BFISO + BFShutterSpeed  + defaultExposure + defaultDelay+defaultAWB+defaultRaw + " -vf -o " + filename
    os.system(cmd) #run the command from terminal
#     print('Printing exposure time')
#     print(camera.shutter_speed())
    dt=time.time()-t0 # time taken for execution
    #disable the light source
    gpio.output(BFLEDPin, 0)
    return dt,filename
    
def captureFluorImage(channel, step,flip=False, vcm_shift=-2): #capture FLuorescence images once at the sharpest focal plane

    t0=time.time()

    
    logging.info(f"Capturing FL image vcm*{vcm_shift}")
    driveVCM(defaultZstep+defaultZresolution*vcm_shift)
    logging.debug('Driven to down VCM')
    gpio.output(FluorPin, 1)  
    filename = defaultFname + channel + "_FL-"  + 'FOV_'+ str(step) + ".png"
    logging.info(f'DRC setting: {FluorDRC}')
    if flip==False:
        cmd = "raspistill" + defaultWidth + defaultHeight + FluorShutterSpeed + FluorISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw + FluorDRC + " -o " + filename
        os.system(cmd)
        if HDR==True:
            logging.info('HDR FL captures')
            sleep(1)
            filename = defaultFname + channel + "_FL-"  + 'FOV_'+ str(step) + "_loExp.png"
            cmd = "raspistill" + defaultWidth + defaultHeight + FluorShutterSpeed_lo + FluorISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw + FluorDRC + " -o " + filename
            os.system(cmd)
            sleep(1)
            filename = defaultFname + channel + "_FL-"  + 'FOV_'+ str(step) + "_hiExp.png"
            cmd = "raspistill" + defaultWidth + defaultHeight + FluorShutterSpeed_hi + FluorISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw + FluorDRC + " -o " + filename
            os.system(cmd)
            
            
    else:
        logging.info('applying Flip')
        cmd = "raspistill" + defaultWidth + defaultHeight + FluorShutterSpeed_G + FluorISO_G + defaultExposure + defaultDelay+defaultAWB+defaultRaw + FluorDRC + " -vf -o " + filename
        os.system(cmd) #run the command from terminal
        if HDR==True:
            sleep(0.1)
            filename = defaultFname + channel + "_FL-"  + 'FOV_'+ str(step) + "_loExp.png"
            cmd = "raspistill" + defaultWidth + defaultHeight + FluorShutterSpeed_G_lo + FluorISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw + FluorDRC + " -vf -o " + filename
            os.system(cmd)
            sleep(0.1)
            filename = defaultFname + channel + "_FL-"  + 'FOV_'+ str(step) + "_hiExp.png"
            cmd = "raspistill" + defaultWidth + defaultHeight + FluorShutterSpeed_G_hi + FluorISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw + FluorDRC + " -vf -o " + filename
            os.system(cmd)
        
    gpio.output(FluorPin, 0)    
    dt=time.time()-t0    
    return dt,filename

    

def captureDFImage(channel, step, flip=False, vcm_shift=0): #capture DF images once at the sharpest focal plane
    logging.info(f"Capturing DF image vcm*{vcm_shift}")
    driveVCM(defaultZstep+defaultZresolution*vcm_shift)
    logging.debug('Driven to down VCM')
    #Enable the appropriate light source
    gpio.output(DFPin, 1)
#    sleep(0.1)
    
    #format the file name for the images
    filename = defaultFname + channel + "_DF-"  + 'FOV_'+ str(step) + ".png" #generate the file name and call the raspistill function
    
    logging.info("capturing DF image")
    t0=time.time()
    #capture the image
    if flip==False:
        cmd = "raspistill" + defaultWidth + defaultHeight + DFShutterSpeed + DFISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw+defaultIFX + " -o " + filename    
        os.system(cmd) #run the command from terminal
        
        if HDR==True:
            logging.info('HDR DF captures')
            filename = defaultFname + channel + "_DF-"  + 'FOV_'+ str(step) + "_loExp.png"
            cmd = "raspistill" + defaultWidth + defaultHeight + DFShutterSpeed_lo + DFISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw+defaultIFX + " -o " + filename    
            os.system(cmd) #run the command from terminal
            filename = defaultFname + channel + "_DF-"  + 'FOV_'+ str(step) + "_hiExp.png"
            cmd = "raspistill" + defaultWidth + defaultHeight + DFShutterSpeed_hi + DFISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw+defaultIFX + " -o " + filename    
            os.system(cmd) #run the command from terminal
    else:
        logging.info('Applying Flip')
        cmd = "raspistill" + defaultWidth + defaultHeight + DFShutterSpeed + DFISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw+defaultIFX + " -vf -o " + filename    
        os.system(cmd) #run the command from terminal
        
        if HDR==True:
            logging.info('HDR DF captures')
            filename = defaultFname + channel + "_DF-"  + 'FOV_'+ str(step) + "_loExp.png"
            cmd = "raspistill" + defaultWidth + defaultHeight + DFShutterSpeed_lo + DFISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw+defaultIFX + " -vf -o " + filename    
            os.system(cmd) #run the command from terminal
            filename = defaultFname + channel + "_DF-"  + 'FOV_'+ str(step) + "_hiExp.png"
            cmd = "raspistill" + defaultWidth + defaultHeight + DFShutterSpeed_hi + DFISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw+defaultIFX + " -vf -o " + filename    
            os.system(cmd) #run the command from terminal
        
        
    
    dt=time.time()-t0
    #disable the light source
    gpio.output(DFPin, 0)
    return dt,filename

def captureSSImage(channel, step): #capture SS images once at the sharpest focal plane
    #Enable the appropriate light source
    
    gpio.output(SSPin, 1)
#    sleep(0.1)
    
    #format the file name for the images
    filename = defaultFname + channel + "_SS-"  + 'FOV_'+ str(step) + ".png" #generate the file name and call the raspistill function
    
    logging.info("capturing SS image")
    t0=time.time()
    #capture the image
    cmd = "raspistill" + defaultWidth + defaultHeight + SSShutterSpeed + SSISO + defaultExposure + defaultDelay+defaultAWB+defaultRaw+defaultIFX + " -o " + filename    
    os.system(cmd) #run the command from terminal    
    dt=time.time()-t0
    #disable the light source
    gpio.output(SSPin, 0)
    return dt,filename
    



def automateAnalysisRoutine(step):
    #use try statement to allow the user to abort the routine
    logging.info("Capturing Analysis Images for current FOV")
    #Select Channel A
    logging.info("Selecting camera A")
    i2c = "i2cset -y 1 0x70 0x00 0x04"
    os.system(i2c)
    gpio.output(7, False)
    gpio.output(11, False)
    gpio.output(12, True)
    #engage channel A - RED for images
    logging.info("Capturing Red Channel images")
    logging.debug(f'VCM: {defaultZstep}')
    dt_BF,filename_BF=captureBFImage("R", step)
#     dt_DF,filename_DF=captureDFImage("R", step)
    #dt_SS,filename_SS=captureSSImage("R", step)
    dt_FL,filename_FL=captureFluorImage("R", step,vcm_shift=0)
#    dt_FL,filename_FL=captureFluorImage("Rp1", step,vcm_shift=1)
#    dt_FL,filename_FL=captureFluorImage("Rp2", step,vcm_shift=2)
#    dt_FL,filename_FL=captureFluorImage("Rn1", step,vcm_shift=-1)
#    dt_FL,filename_FL=captureFluorImage("Rn2", step,vcm_shift=-2)
#    dt_FL,filename_FL=captureFluorImage("Rn3", step,vcm_shift=-3)
    #Select Channel B
    logging.info("Selecting camera B")
    i2c = "i2cset -y 1 0x70 0x00 0x05"
    os.system(i2c)
    gpio.output(7, True)
    gpio.output(11, False)
    gpio.output(12, True)
    
#     print("Start testing the camera C")
#     i2c = "i2cset -y 1 0x70 0x00 0x06"
#     os.system(i2c)
#     gpio.output(7, False)
#     gpio.output(11, True)
#     gpio.output(12, False)
    #engage channel C - GREEN for images
    logging.info("Capturing Green Channel images")
    #dt,filename_BF_G=captureBFImage("G", step, flip=True)
#     captureDFImage("_GREEN_", step)
#     captureSSImage("_GREEN_",step)
    dt_DF,filename_DF=captureDFImage("G", step, flip=True)
    dt_DF,filename_DF=captureDFImage("G_bg", step, flip=True,vcm_shift=20)
    dt,filename_FL_G=captureFluorImage("G", step, flip=True)
#     dt,filename_FL_G=captureFluorImage("Gp2", step, flip=True, vcm_shift=2)
#     dt,filename_FL_G=captureFluorImage("Gn2", step, flip=True, vcm_shift=-2)
    
    #filename_list=[filename_BF,filename_DF,filename_SS,filename_FL,filename_BF_G,filename_FL_G]
    filename_dict = {
        #'rbf': filename_BF,
        'df': filename_DF,
        #'rsc': filename_SS,
        'red': filename_FL,
        #'gbf': filename_BF_G,
        'green': filename_FL_G
    }
    masterfilename=os.path.join(defaultFolderName,'filename_list')
    
    with open(masterfilename,'wb') as fp:
        logging.info(f'Writing {filename_dict} for analysis Script to {masterfilename}')
        pickle.dump(filename_dict,fp)
    print(masterfilename)
    # for name in filename_list:
    #     print(name)
    
    return np.round(dt_DF,3),np.round(dt_FL,3)

def move_motor(defaultFOVresolution=4,direction=0,initialization=False,init_range=32): # moving the stepper motor
    gpio.output(DIRPin, direction)
    if initialization==True:
        logging.debug("Init:Accuation ACTIVE")
        for pulse in range (0, init_range, 1):             
            gpio.output(PULPin,1) #send one pulse to the motor to move
            sleep(0.01) #wait a moment for motor driver to receive pulse
            gpio.output(PULPin,0)
            sleep(0.01) #wait a moment for the motor to settle
        sleep(0.25)
    else:
        sleep(0.25)
        logging.debug("Motor Moving 1 step")
        for pulse in range (0, defaultFOVresolution, 1):
            gpio.output(PULPin,1) #send one pulse to the motor to move
            sleep(0.04) #wait a moment for motor driver to receive pulse
            gpio.output(PULPin,0)
            sleep(0.04) #wait a moment for the motor to settle

async def setHome():
    on_alert = lambda c: ""

    async with PhoenixClient(testbench, on_alert=on_alert) as client:
        await client.run_custom_program(program=[set_home_to_here(duration=0.1)], title="motor_control_test")       

async def go(deg,direction=0): 
    # moves the BLDC motor through network
    reads = []
    temps = []
    motor_dir=1
    if direction !=0:
        motor_dir=-1
        
    vel=motor_dir*400
    acc=150
    # print("@@@ before move")
    # on_alert = eng_on_alert_handler(reads, temps)
    on_alert = lambda c: ""
    async with PhoenixClient(testbench, on_alert=on_alert) as client:
        programA = [
            #home_spindle(duration=0),
            position_spindle(duration=0.5, position=deg, velocity_limit=vel, acceleration=acc, deceleration=acc),
#             set_spindle_velocity(duration=10, velocity=50, acceleration=25, deceleration=25),
            power_off_spindle(duration=0.1)
        ]
        await client.run_custom_program(program=programA, title="motor_control_test")       
        # print("@@@ after move")
        
def disable_trigger():
    logging.info("Disabling light sources and triggers")
    driveVCM(0)
    gpio.output(PULPin, 0) #disable the trigger pin
    gpio.output(DIRPin, 0) #disable the direction pin
    gpio.output(PWRPin, 0) #disable the power pin to power down the motor driver
    gpio.output(BFLEDPin, 0) #disable the LED Pin 
    gpio.output(FluorPin, 0) #disable the LASERPin
    gpio.output(DFPin, 0) #disable the LED Pin 
    gpio.output(SSPin, 0) #disable the LASERPin
    gpio.cleanup() #cleanup and free the GPIO pins on exit
    logging.info("Triggers diabled - Exit Gracefully!") #inform the user of a graceful exit
    
def initialize():
    logging.info('Initializing the IMAGING SYSTEM')
    i = 0
    #slowly drive the VCM to the middle position
    logging.info("Driving VCM to default position for alignment")
#         for i in range (0,(defaultZstep + 1)):
    driveVCM(defaultZstep) #begin by driving the VCM to the default position defined above
    logging.debug("VCM Driven to Target position")
    gpio.output(DIRPin, 0)
    # accuating motor
    if disable_motor is False:
        logging.info(f"Driving Motor to Home position of {home_deg/10} degree ")
        asyncio.run(go(home_deg)) # driving the motor to home position
    # if init_accuation:
        # move_motor(direction=0,initialization=True,init_range=init_range)    
#     flatfieldCaptures() #call an alignment function to select a region for FF Image capture
    if disable_AF is False:
        dt_AF1,sharpest_VCM=automateAFRoutine(0, defaultZstep, defaultZrange*10, defaultZresolution*10,channel=AF_camera)
    else:
        logging.info('AF disabled')
        dt_AF1,sharpest_VCM=0,defaultZstep
#     
    logging.debug(f"VCM driven to sharpest_vcm: {sharpest_VCM} prior to alignment")
    logging.info(f"VCM driven to defaultZstep: {defaultZstep} prior to alignment")
#     
    driveVCM(sharpest_VCM)
    
#     aligncameras() #next call the alignment function to ensure Target is focused at the middle position
#     startTime = time.time()
    
    logging.info(f"writing vcm value: {defaultZstep} to a file")
    
    vcmfilename=os.path.join(defaultFolderName,'vcm_value')
    
    with open(vcmfilename,'wb') as fp:
        logging.info('Writing defaultZstep to a file')
        pickle.dump(defaultZstep,fp)    
    
    
    logging.info("alignment completed - driving VCM to lower AF range")
    
    
    #slowly drive the VCM to the lower analysis range
    for i in range (defaultZstep, (defaultZstep - defaultZrange), -1):
        driveVCM(i)
    logging.debug("VCM Driven to lower AF range")
    
#     if fov_zigzag==True:
    logging.info(f"writing home_deg value of {home_deg} to a file")
    home_deg_filename=os.path.join(defaultFolderName,'home_deg_value')
    with open(home_deg_filename,'wb') as fp:
        logging.info('writing home_deg to a file')
        pickle.dump(home_deg,fp)
    
    
    
def run_oneFOV(fov): # capturing images for one fov
    logging.info(f'Capturing ONE FOV of index #{fov}')
    vcmfilename=os.path.join(defaultFolderName,'vcm_value')
    logging.warning(f'reading vcm value from {vcmfilename}')
    try:
        with open (vcmfilename, 'rb') as fp:
            defaultZstep = pickle.load(fp)
    except:
        logging.info('Failed to read file [vcm_value]. Using defaultZstep')
        defaultZstep=defaultZstep         
    logging.warning(f'defaultZstep : {defaultZstep}')
    
    home_deg_filename=os.path.join(defaultFolderName,'home_deg_value')
    logging.warning(f'reading home_deg value from {home_deg_filename}')
    try:
        with open (home_deg_filename, 'rb') as fp:
            home_deg = pickle.load(fp)
    except:
        logging.info('Failed to read file [home_deg]. Using default home_deg value')
        home_deg=home_deg         
    logging.warning(f'home_deg : {home_deg}')
    
    
    
    
    if disable_AF is False:
        dt_AF,sharpest_VCM=automateAFRoutine(fov, defaultZstep, defaultZrange, defaultZresolution,channel=AF_camera)
    else:
        logging.info('AF disabled')
        dt_AF,sharpest_VCM=0,defaultZstep
        
    logging.info("Beginning Analysis Routine for current FOV")    
    dt_DF,dt_FL=automateAnalysisRoutine(fov) #automated function to capture analysis images from both channels is called
    temp=[fov,dt_DF,dt_FL,dt_AF,sharpest_VCM]    
    
    with open(vcmfilename,'wb') as fp:
        logging.warning(f'Writing sharpest VCM value : {sharpest_VCM} to the filename')
        pickle.dump(sharpest_VCM,fp)
    logging.info("Analysis images captured! - advancing FOV")
    #step forward to new FOV position
    logging.debug("Finding new FOV")
#    move_motor(defaultFOVresolution=defaultFOVresolution,direction=0) 
    
    direction=0
    if fov_zigzag==True:
        fov_set=15 # sets the number of fov after the motor direction will be flipped
        defaultFOVresolution_gain=3 # coarse angular resolution=defaultFOVresolution_gain*defaultFOVresolution
        coarse_angular_resolution=defaultFOVresolution_gain*defaultFOVresolution
        effective_fov=fov%fov_set

        if ((fov)//fov_set)%2==0:
            fov_dir=1
            direction=0
        else:
            fov_dir=-1
            direction=1
            
        if fov_dir==1:
            home_deg=home_deg+coarse_angular_resolution*fov_dir # rotate in positive direction

            logging.info('Rotating in positive direction')
        else:
            home_deg=home_deg+coarse_angular_resolution*fov_dir # rotate in the negative direction
            logging.info('Rotating in negative direction')
    else:
        home_deg=home_deg+defaultFOVresolution # calculate new home_deg based on a previously written file

    if disable_motor is False:
        asyncio.run(go(home_deg,direction=direction))# move the motor
    
    if fov_zigzag==True and effective_fov==fov_set-1 :      
        home_deg=home_deg+defaultFOVresolution # adding a phase shift so that the zigzag action donot make fovs overlap
    
    
    
    with open(home_deg_filename,'wb') as fp:
            logging.warning(f'Writing home_deg value : {home_deg} to the filename')
            pickle.dump(home_deg,fp)    
    

    return temp   
      

def run(): # running the entire subsystem    

    startTime = time.time()
    initialize()
    logging.info("Stepping through FOVs")
    list_fov=[]
    
    #FOR EACH FOV IN RANGE 0 to # FOVS
    for fov in tqdm(range (0, defaultFOVrange, 1),disable=disable_progress_bar):
#     fov=0
        logging.info("Investigating FOV " + str(fov))

        temp=run_oneFOV(fov)
        list_fov.append(temp)
#         logging.info("Analysis images captured! - advancing FOV")
#         #step forward to new FOV position
#         logging.debug("Finding new FOV")
#         move_motor(defaultFOVresolution=defaultFOVresolution,direction=0)


        logging.info("FOV " + str(fov) + "completed!")        
    
    #After all FOVs are checked, the VCM is in the last position of greatest focus, only the BF image of sharpest image remains, and the analysis images are captured
    logging.info("All FOVs have been analysed - beginning the shutdown routine")
    df_fov=pd.DataFrame(list_fov,columns=['FOV#','DF','FL','AF','VCM'])
    logging.info(df_fov)
    csv_filename=os.path.join(defaultFolderName,'stats_time_vcm.csv')
    csv_filename_=os.path.join(defaultFolderName,'exif_data.csv')
    logging.debug('saving CSVs')
    f=open(csv_filename,'w')
    df_fov.to_csv(f,index=False)
    f.close()
    f=open(csv_filename_,'w')
    exif_df.to_csv(f,index=True,index_label='Function')
    f.close()
    ##### SHUTDOWN SEQUENCE BEGINS HERE #####
    #finally drive the VCM to 0
    logging.info("driving VCM to MIN")
    for i in range ((defaultZstep - defaultZrange), 0, -1):
        driveVCM(i)
    logging.debug("VCM Driven to MIN")

    
    endTime = time.time()
    
    #drive the disk motor to the default fov if possible
#     gpio.output(DIRPin, 1) #reverse the direction of the motor
#     logging.info("Returning to origin FOV")
#     
#     if init_accuation:
#         
#         logging.debug("Init:Accuation ACTIVE")
#         move_motor(direction=1,initialization=True,init_range=init_range)
# 
#     
#     for i in range (0, defaultFOVrange):
#         sleep(0.1) 
#         move_motor(defaultFOVresolution=defaultFOVresolution,direction=1)
# 
#         logging.debug("FOV returning" + str(i))
#     logging.info("FOV Returned")

#         print(defaultZstep)
    time_elapsed=endTime - startTime
    logging.info(f'Total Script RunTime : {time_elapsed} seconds')
    disable_trigger()
def help():
    dict_help={'init':'Initialize Cyclops','full': 'Run all FOV together','one': 'Run one FOV at a time [After Initialization]','off' :'Disable Trigger', 'help': 'Show Help menu', 'fov#':'Order of FOV'}
    print('The following Modes are available with passing arguments')
    print('python3 <script_name> arg1 arg2')
    print('arg1: [full, init, one, off, help]')
    print('arg2: [fov#]')
    print(dict_help)   
    
    print('################Notes############')
    print('arg2 is only needed when arg1=one')
    print('For running one FOV at a time, Initialization must be done at the very beggining by passing arg1= init')

    
            
if __name__ == "__main__":
    #start by setting the motor to middle position and request user to align
    logging.info('####################### CYCLOPS LOADING ###########################')
    try:
        if disable_motor is False:
            asyncio.run(setHome())
#        asyncio.run(go(1800))
        logging.info('GG')
        if sys.argv[1]=='full':
            run()
        elif sys.argv[1]=='init':
            try:
                initialize()
                disable_trigger()
            except:
                logging.info('Initialization failed, Trying again')
                initialize()
                disable_trigger()
        elif sys.argv[1]=='one':
            try:
                run_oneFOV(int(sys.argv[2]))
            except:
                print('Please pass a arg2 as fov number')
                run_oneFOV(0)
            disable_trigger()
        elif sys.argv[1]=='off':
            disable_trigger()
        elif sys.argv[1]=='help':
            help()

#         run()
#         initialize()
#         run_oneFOV(0)
#         disable_trigger()

    except:
        print(sys.exc_info())
        print('#################### SCRIPT FAILED ####################')
        disable_trigger()
        help()

    logging.info('############################# THE END ####################################')


