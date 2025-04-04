# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 11:36:04 2021

@author: imrul.kayes
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle
from ROI_manager_user import userROI
import cv2



def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def image_rescaling(img,scale_percent=100):
    # scale_percent = 99.51 # percent of original size
    width = int(np.round((img.shape[1] * scale_percent / 100),0))
    height = int(np.round((img.shape[0] * scale_percent / 100),0))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def get_lateral_CA(im_raw,boxSize=[25,25]):

    imRed=im_raw[:,:,0]
    imGreen=im_raw[:,:,1]
    imBlue=im_raw[:,:,2]

    classROI=userROI(imGreen, no_of_corners=1, sort=False, rectangle=False)   
    
    roi={'x0':classROI.coords[0][0]-boxSize[0],'x1':classROI.coords[0][0]+boxSize[0],'y0':classROI.coords[0][1]-boxSize[1],'y1':classROI.coords[0][1]+boxSize[1]}
           
    r=np.mean([roi['y0'],roi['y1']]).astype(int)
    sG=imGreen[r,roi['x0']:roi['x1']]
    sR=imRed[r,roi['x0']:roi['x1']]
    sB=imBlue[r,roi['x0']:roi['x1']]
    plt.figure()
    plt.plot(sG,'g')
    plt.plot(sR,'r')
    plt.plot(sB,'b')
    
    level=np.mean([np.percentile(sG,90),np.percentile(sG,10)])   

    #%
    plt.figure()
    plt.plot(np.abs(sG-level),'g')
    plt.plot(np.abs(sB-level),'b')
    plt.plot(np.abs(sR-level),'r')
    
    #%
    edge_location={'red':np.argmin(np.abs(sR-level)),'green':np.argmin(np.abs(sG-level)),'blue':np.argmin(np.abs(sB-level))}
    print(f'edge_location: {edge_location}')
    
    lca_count=[edge_location['red']-edge_location['green'],edge_location['blue']-edge_location['green']]
    
    imHeight,imWidth=imGreen.shape
    cx,cy=[imWidth/2,imHeight/2]
    imRadius_max=np.abs(cx+1j*cy)
    
    rx,ry=classROI.coords[0][0]-cx,classROI.coords[0][1]-cy
    
    if rx>=0:
        sgn=+1
    else:
        sgn=-1
    
    imRadius=np.abs(rx+1j*ry).astype(int)
    imRadius_pt=(imRadius/imRadius_max*100).astype(int)
    lca_pixel={'red':lca_count[0],'blue':lca_count[1]}
    
    return lca_pixel,sgn*imRadius_pt

def compensate_LCA(im_raw,scale_percent=100,channel='blue'):
    # this function compensate for LCA for one color channel. Scale percent must be less than 100 for now.
    # imRed=im_raw[:,:,0]
    # imGreen=im_raw[:,:,1]
    # imBlue=im_raw[:,:,2]
    
    channels=['red','green','blue']
    ch_idx=channels.index(channel) 
    
    img=np.copy(imBlue)
    resized=image_rescaling(img,scale_percent)
    print('Original Dimensions : ',img.shape)
    print('Resized Dimensions : ',resized.shape)
    dim0=np.array(img.shape)
    dim1=np.array(resized.shape)
    ctr_shift=dim0-dim1   # shift in the image center
    crop=[ctr_shift[0]//2,ctr_shift[0]-ctr_shift[0]//2,ctr_shift[1]//2,ctr_shift[1]-ctr_shift[1]//2]  # crop location to realign the two image center
    # M=np.float32([[1,0,ctr_shift[1]],[0,1,ctr_shift[0]]])
    # dst=cv2.warpAffine(resized,M,(dim1[1],dim1[0]))
    dst0=np.zeros(img.shape)  # place holder for compensated channel
    dst0[crop[0]:dim0[0]-crop[1],crop[2]:dim0[1]-crop[3]]=np.copy(resized)
    # im_compensated=np.zeros(im_raw.shape)
    im_compensated=np.copy(im_raw)
    # im_compensated[:,:,0]=np.copy(imRed)
    # im_compensated[:,:,1]=np.copy(imGreen)
    im_compensated[:,:,ch_idx]=np.copy(dst0)
    im_compensated=im_compensated.astype(np.uint8)
    fig, ax = plt.subplots(1,2, figsize=(10,6),sharex=True,sharey=True)
    ax[0].imshow(im_raw)
    ax[0].set_title('Original Image')
    ax[1].imshow(im_compensated)
    ax[1].set_title('compensated Image')
    
    return im_compensated
#%%
filename=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\20210719_FlatFieldCorrection_YellowandWhiteTest1\20210719_FFImage_White_30mm_USAF_10msISO100_.bmp"
im_raw = imread(filename)
# plt.figure()
# plt.imshow(im_raw)

flatfield=r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\20210719_FlatFieldCorrection_YellowandWhiteTest1\20210719_FFImage_White_30mm_FlatField_10msISO100_.bmp"

imUfield=imread(flatfield)


im_final=((im_raw/imUfield).clip(0,1)*255).astype(np.uint8)

# plt.figure()
# plt.imshow(im_final)

fig, ax = plt.subplots(1,2, figsize=(10,6),sharex=True,sharey=True)
ax[0].imshow(im_raw)
ax[0].set_title('Original Image')
# dinner_max = (dinner*1.0 / dinner.max(axis=(0,1)))
ax[1].imshow(im_final);
ax[1].set_title('Processed Image')


#%% dividing by flatfield
im_raw=np.copy(im_final)
# im_raw=im_raw.clip(0,255)

#%% quantifying chormatic aberration with image height.





#%%  To characterize LCA by selecting certain number of edges across the FOV
lca_blue=[]
lca_red=[]
imRadius_list=[]

no_of_edge=0

for i in range(no_of_edge):
    print(f'select the edge number # {i+1}')
    lca,imR=get_lateral_CA(im_raw)
    lca_blue.append(lca['blue'])
    lca_red.append(lca['red'])
    imRadius_list.append(imR)
    
    

#%%
   
plt.figure()
plt.plot(imRadius_list,lca_blue,'o-',color='b')
plt.plot(imRadius_list,lca_red,'*-',color='r')
plt.xlabel('Image Radius (%)')
plt.ylabel('Chromatic Aberration [pixel]')
plt.title('Lateral CA w.r.t Green channel')
plt.grid(True)

#%% LCA compensation
imRed=im_raw[:,:,0]
imGreen=im_raw[:,:,1]
imBlue=im_raw[:,:,2]

#%%

#%%

scale_pt_arr=[99.51]#np.linspace(99.4,100,10)

for scale_pt in scale_pt_arr:
    im_compensated=compensate_LCA(im_raw,scale_percent=scale_pt)
    
    roi_eval={'x0':860,'x1':2020,'y0':710,'y1':2200}
    
    im_eval=np.copy(im_compensated[roi_eval['y0']:roi_eval['y1'],roi_eval['x0']:roi_eval['x1'],:])
    
    err=mse(im_eval[:,:,1],im_eval[:,:,2])
    
    print(f'MSE: {err} for scale_percentage of {scale_pt}')    
    
    
    
    
    
    
    
    



#%%


#%%
