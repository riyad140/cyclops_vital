# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:40:12 2021

@author: imrul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import os


#%%
def flatFieldCorrection(binPath,key='SS',channel='R',blurSize=(301,301)):
    imList=[]
    imStack=[]
    fName=[]
    blurSize=(301,301)
    for file in os.listdir(binPath):
        if file.endswith('png')==True and file.find(key)!=-1:
            print(file)
            fName.append(file)
            im=imread(os.path.join(binPath,file))
            
            if channel=='R':
                imR=im[:,:,0]
            else:
                imR=im[:,:,1]
                    
            imList.append(imR)
            imR=cv2.blur(imR,blurSize)
            imStack.append(imR)
    
    imStack_arr=np.array(imStack)
    imFlatField=np.mean(imStack,axis=0)
    print(imFlatField.shape)
    
    imFlatfield=np.copy(imFlatField).astype(np.uint8)+1 # adding one to avoid divide by zero error
    # rad=(201,201)
    # imFlatField_b=cv2.blur(imFlatField,rad)
    
    # plt.figure()
    # plt.imshow(imFlatfield)
    
    dstPath=os.path.join(binPath,'ffc')
    if os.path.isdir(dstPath)==False:
        os.mkdir(dstPath)
    print('Making Directory and writing files')
    imFinal=np.zeros(im.shape,dtype=np.uint8)
    for count,im in enumerate(imList):
        print(fName[count])
        imCorrected=(imList[count]/imFlatfield*np.mean(imFlatfield)).clip(0,255)
        imCorrected=imCorrected.astype(np.uint8)
        if channel=='R':
           imFinal[:,:,2]=np.copy(imCorrected) # need to make them RGB
        else:
           imFinal[:,:,1]=np.copy(imCorrected)
        filename=os.path.join(dstPath,fName[count])
        cv2.imwrite(filename, imFinal)
#%%
if __name__ == "__main__":
    superbinPath=r'Z:\controller\cyclops\2021-09-22'
    for folder in os.listdir(superbinPath):
        print(folder)
        binPath=os.path.join(superbinPath,folder)
        flatFieldCorrection(binPath,key='SS')
        flatFieldCorrection(binPath,key='R_FL')
        flatFieldCorrection(binPath,key='G_FL',channel='G')
        
    # binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\Spencer - CC\16-23-56_sample_476_wbc_disc'
    # flatFieldCorrection(binPath,key='BF')
    # flatFieldCorrection(binPath,key='DF')
    # flatFieldCorrection(binPath,key='SS')
    # flatFieldCorrection(binPath,key='FL')