# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 12:17:19 2021

@author: imrul
"""
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
class CyclopsTest:
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
    
    defaultFileName = ' -o testimage_000' #filepath and name
    defaultAWB = ' -awb off -awbg 1.0,1.0' #AWB settings
    defaultWidth = ' -w 4056'      #sensor width in pixels
    defaultHeight = ' -h 3040'     #sensor height in pixels
    defaultShutterSpeed = ' -ss 20000' #shutter speed in us
    defaultISO = ' -ag 1.0 -dg 1.0'            #ISO setting
    defaultExposure = ' -ex auto'   #exposure mode
    defaultDelay = ' -t 1'
    defaultPreview = ' -n'
    defaultFileExtention='.png'

    
    def __init__(self):
        print('INIT')  
        
               
    def driveVCM(self,i=500):
        # drive the VCM to adjust lens position
        print("setting VCM position to " + str(i))
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
        self.i2cbus.write_byte_data(0x0c, byte1, byte2)
    
    def testVCM(self,maxIter=20):
        # toggles back and forth between min and max VCM position until maxIter is reached
        for i in range(maxIter):
            if i%2==0:
                print('Driving VCM to Max')
                self.driveVCM(1000)
            else:
                print('Driving VCM to Min')
                self.driveVCM(0)
            sleep(0.5)
            
    def testLEDs(self,maxIter=200):
        for i in range(maxIter):
            if i%4==0:
                gp.output(self.BFPin, 1)
                gp.output(self.DFPin, 0)
                gp.output(self.SSPin, 0)
                gp.output(self.FLPin, 0)
                print("BF Source ON")
            elif i%4==1:
                gp.output(self.BFPin, 0)
                gp.output(self.DFPin, 1)
                gp.output(self.SSPin, 0)
                gp.output(self.FLPin, 0)
                print("DF Source ON")
            elif i%4==2:
                gp.output(self.BFPin, 0)
                gp.output(self.DFPin, 0)
                gp.output(self.SSPin, 1)
                gp.output(self.FLPin, 0)
                print("SS Source ON")
            elif i%4==3:
                gp.output(self.BFPin, 0)
                gp.output(self.DFPin, 0)
                gp.output(self.SSPin, 0)
                gp.output(self.FLPin, 1)
                print("FL Source ON")
            gp.output(self.BFPin, 0)
            gp.output(self.DFPin, 0)
            gp.output(self.SSPin, 0)
            gp.output(self.FLPin, 0)
            print('All source OFF')
                    
    
        
    def selectCamera(self,channel='A'):
        # Select the Camera Multiplexer A,B or C
        if channel=='A':
            print("Selecting camera A")
            i2c = "i2cset -y 1 0x70 0x00 0x04"
            os.system(i2c)
            gp.output(7, False)
            gp.output(11, False)
            gp.output(12, True)
            
        elif channel=='B':
            print("Selecting camera B")
            i2c = "i2cset -y 1 0x70 0x00 0x05"
            os.system(i2c)
            gp.output(7, True)
            gp.output(11, False)
            gp.output(12, True)   
            
    def capture(self,color_channel='R'):
        #starts preview by calling capture without defining the delay time - END WITH CTRL + C
        filename=self.defaultFileName+color_channel+self.defaultFileExtention
        cmd = "raspistill -t 0 -o capture_" + filename + ' -set'
        os.system(cmd)
        #captures an image once the preview has been cancelled

        #capture the image by calling the raspistill function from python
        start_time=time()
        cmd = "raspistill" +self.defaultDelay+ self.defaultWidth + self.defaultHeight+ self.defaultAWB + self.defaultShutterSpeed + self.defaultISO + filename  +' -set'  
        os.system(cmd)
        end_time=time()
        elapsed_time=end_time-start_time
        print(f'{filename}: {elapsed_time}')
        
    def capture_BF(self):
        # capturing RED camera
        self.selectCamera('A')
        gp.output(self.BFPin,True)
        self.capture('BF_R')
        gp.output(self.BFPin,False)
        # capturing Green camera
        self.selectCamera('B')
        gp.output(self.BFPin,True)
        self.capture('BF_G')
        gp.output(self.BFPin,False)
    
    async def go(self,deg): 
    # moves the BLDC motor through network
        reads = []
        temps = []
        # print("@@@ before move")
        # on_alert = eng_on_alert_handler(reads, temps)
#         on_alert = lambda c: ""
        on_alert=eng_on_alert_handler(reads, temps)
        async with PhoenixClient("testbench-demo-cyclops.vital.company", on_alert=on_alert) as client:
            programA = [
                #home_spindle(duration=0),
                position_spindle(duration=0.1, position=deg, velocity_limit=400, acceleration=150, deceleration=150),
    #             set_spindle_velocity(duration=10, velocity=50, acceleration=25, deceleration=25),
                power_off_spindle(duration=0.1)
            ]
            await client.run_custom_program(program=programA, title="motor_control_test") 
    
    def testMotor(self):
        for deg in range(0,300,10):
            print(deg)
            t0=time()
            asyncio.run(self.go(deg))
            t1=time()
            dt=t1-t0
            print(f'Elapsed time for sequence: {dt} sec')
            sleep(0.1)
        
        
    def run(self):
        self.testLEDs()
        self.testVCM() 
        self.driveVCM(500)
        self.capture_BF()
        self.testMotor()

if __name__ == "__main__":
    testCy=CyclopsTest()
    testCy.run()


        