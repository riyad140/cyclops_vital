# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:35:29 2021

@author: imrul
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

#%% affine transform

im1=cv2.imread(r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\affine transform\cyclops_G_BF-FOV_9.png",1)
im2=cv2.imread(r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\affine transform\cyclops_R_BF-FOV_9.png",1)

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(im1)
ax[0].set_title('G')
ax[1].imshow(im2)
ax[1].set_title('R')


imRed=np.zeros(im1.shape,dtype=np.uint16)
imGreen=np.zeros(im2.shape,dtype=np.uint16)
imRed[:,:,0]=im2[:,:,2]
imGreen[:,:,1]=im1[:,:,1]

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(imGreen)
ax[0].set_title('G')
ax[1].imshow(imRed)
ax[1].set_title('R')


imG=imGreen[:,:,1]
imR=imRed[:,:,0]
#%%



imOverlay=cv2.addWeighted(imGreen,0.5,imRed,0.75,0)
plt.figure()
plt.imshow(imOverlay)

#%%
# imG=im1[:,:,1]
# imR=im2[:,:,0]
# fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
# ax[0].imshow(imG)
# ax[1].imshow(imR)


#%%

pts1 = np.float32([[1064,667],
                    [2565,2574], 
                    [ 3795,1385]])
  
pts2 = np.float32([[ 733,586],
                    [2188,2528], 
                    [3445,1371]])

# pt1=np.array([pts1[:,1],pts1[:,0]]).T
# pt2=np.array([pts2[:,1],pts1[:,0]]).T

# pts1 = np.float32([[1996.0, 1443.6],
#                    [1244.3, 441.1], 
#                    [3242.8, 2533.9]])
  
# pts2 = np.float32([[2346.3, 1490.0],
#                    [1569.2, 509.7], 
#                    [3617.6, 2556.1]])

# pt1=np.array([pts1[:,1],pts1[:,0]]).T
# pt2=np.array([pts2[:,1],pts1[:,0]]).T

rows, cols = imG.shape  
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(imG, M, (cols, rows))

#%%



fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(dst)
ax[0].set_title('G')
ax[1].imshow(imR)
ax[1].set_title('R')

#%%
imM = cv2.addWeighted(imR,0.5,dst,0.7,0)
plt.figure()
plt.imshow(imM)