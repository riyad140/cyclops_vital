# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:25:32 2021

@author: imrul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import os


#%%
def flatFieldCorrection(binPath,key='SS',blurSize=(301,301)):
    imList=[]
    imStack=[]
    fName=[]
    blurSize=(301,301)
    for file in os.listdir(binPath):
        if file.endswith('png')==True and file.find(key)!=-1:
            print(file)
            fName.append(file)
            im=imread(os.path.join(binPath,file))
            imR=im[:,:,0]
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
    for count,im in enumerate(imList):
        print(fName[count])
        imCorrected=(imList[count]/imFlatfield*np.mean(imFlatfield)).clip(0,255)
        imCorrected=imCorrected.astype(np.uint8)
        filename=os.path.join(dstPath,fName[count])
        cv2.imwrite(filename, imCorrected)
    



#%%
ind=0
binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\Spencer - CC\16-23-56_sample_476_wbc_disc'
key='FL'
flatFieldCorrection(binPath,key)

#%%
imList=[]
imStack=[]
fName=[]
blurSize=(301,301)
for file in os.listdir(binPath):
    if file.endswith('png')==True and file.find(key)!=-1:
        print(file)
        fName.append(file)
        im=imread(os.path.join(binPath,file))
        imR=im[:,:,0]
        imList.append(imR)
        imR=cv2.blur(imR,blurSize)
        imStack.append(imR)

imStack_arr=np.array(imStack)
imFlatField=np.mean(imStack,axis=0)
print(imFlatField.shape)

imFlatfield=np.copy(imFlatField).astype(np.uint8)+1 # adding one to avoid divide by zero error
# rad=(201,201)
# imFlatField_b=cv2.blur(imFlatField,rad)

plt.figure()
plt.imshow(imFlatfield)

dstPath=os.path.join(binPath,'ffc')
if os.path.isdir(dstPath)==False:
    os.mkdir(dstPath)
for count,im in enumerate(imList):
    print(fName[count])
    imCorrected=(imList[ind]/imFlatfield*np.mean(imFlatfield)).clip(0,255)
    imCorrected=imCorrected.astype(np.uint8)
    filename=os.path.join(dstPath,fName[count])
    cv2.imwrite(filename, imCorrected)
    
    

    
    
    
#%%
# plt.figure()
# plt.imshow(imList[ind])

#%%


#%%

# plt.figure()
# plt.imshow(imFlatfield)

#%%
imCorrected=(imList[ind]/imFlatfield*np.mean(imFlatfield)).clip(0,255)
imCorrected=imCorrected.astype(np.uint8)

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(imList[ind])
ax[1].imshow(imCorrected)


#%%
fudge=1.99

imCorrected_=(imStack[ind]-imFlatField/fudge).clip(0,255)

plt.figure()
plt.imshow(imCorrected_)


#%%
imOriginal=np.zeros(im.shape,dtype=np.uint8)
imFinal=np.zeros(im.shape,dtype=np.uint8)
imFinal_=np.copy(imFinal)

imOriginal[:,:,0]=imStack[ind]
imFinal[:,:,0]=imCorrected
imFinal_[:,:,0]=imCorrected_

fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
ax[0].imshow(imOriginal)
ax[0].set_title('Orignal')
ax[1].imshow(imFinal)
ax[1].set_title('Division with synthetic flatfield')
ax[2].imshow(imFinal_)
ax[2].set_title('Substraction with a synthetic flatfield')
#%%

# 
    
    
    
