# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:49:05 2022

@author: imrul
"""
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
def square_detector(img,range_area=[50000,90000]):
    # It detects the squares present in an image which has an area within range [minArea,MaxArea]
    
    minArea=range_area[0]
    maxArea=range_area[1]    
   
    
    blurred=cv2.GaussianBlur(img,(1,1),0)
    edged=cv2.Canny(blurred,30,150,apertureSize=3)
    
    nRow,nCol=img.shape
    
    canvas=np.zeros(edged.shape)
    
    # lines=cv2.HoughLinesP(edged,1,np.pi/180,1,minLineLength=200,maxLineGap=90)
    rho, theta, thresh = 2, np.pi/180, 200
    lines = cv2.HoughLines(edged, rho, theta, thresh)  # drawing lines based on Hough transofrmation
    
      
    if lines is not None:
        for i in range(0, len(lines)):
            
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            
            if True is True: #np.degrees(theta)<70 and np.degrees(theta)>30:  # filtering only horizontal and vertical Hough lines
            
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + nCol*(-b)), int(y0 + nRow*(a)))
                pt2 = (int(x0 - nCol*(-b)), int(y0 - nRow*(a)))
                canvas=cv2.line(canvas, pt1, pt2, (255,255,255), 10)
                # print(pt1)
    

    canvas_=np.copy(canvas).astype(np.uint8)  # saving the line marked image as 8 bit
    thresh=cv2.adaptiveThreshold(canvas_,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                cv2.THRESH_BINARY_INV,11,2)     # Thresholding the line marked image

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  #finding contours of line-image
    
    contour_plot=cv2.drawContours(np.zeros(np.shape(img)),contours,-1,(255,255,255),10) # drawing all the contours

    
    canvas=np.copy(img)  #saving a copy of the input image so that square could be drawn on them
    valid_contours=[]
    count=0
    centroid=[]
    vertex=[]
    
    for contour in contours:
        area=cv2.contourArea(contour)
        
        if area>minArea and area<maxArea:
            x, y, w, h = cv2.boundingRect(contour)
            canvas=cv2.rectangle(canvas, (x, y), (x+w, y+h), (120, 0, 0), 10) # drawing the valid rectangle on to the image
            # print([x,y,w,h])
            
            cx,cy=x+w//2,y+h//2
            
            canvas=cv2.circle(canvas,(cx,cy),10,(120,0,0),5)
            
            # canvas=cv2.drawContours(canvas,[contour],0,(255,255,255),10)
            # print(area)
            if w-h<=(w+h)/2*5/100:  # Condition for being a square.
                valid_contours.append(contour)
                count=count+1
                centroid.append([cx,cy])  # only detected square's centroid is saved
                vertex.append([[x,y],[x+w,y+h]])
                
                
                
            
            # rect = cv2.minAreaRect(contour)
            # box = cv2.boxPoints(rect)
            # # convert all coordinates floating point values to int
            # box = np.int0(box)
            # # draw a red 'nghien' rectangle
            # cv2.drawContours(canvas, [box], 0, (255,255,255),10)
            
            
    # print(count)        
                    
    
    plt.figure()
    plt.imshow(canvas,cmap='gray')
    plt.title('After Square Detection')
    
    centroid=[i for i in centroid[::2]] # because for each square shape, two concentrating squares are detected [one is the inner square, another is the outer square]
    vertex=[i for i in vertex[::2]]
    return vertex,centroid,canvas



#%% Reading the input image

file=r"Z:\raspberrypi\photos\FAT_Captures\cyc5Apollo\2022-08-05\run003_motor_on_sample_grid_Ronchi_FATv4_Cyc5Apollo\subset\cyc5-DMSP-G_BF-FOV_0.bmp"
im=cv2.imread(file,0)
img=im[400:600,400:600]


# v,c,cv=square_detector(im_crop,range_area=[100,90000])

#%


blurred=cv2.GaussianBlur(img,(1,1),0)
edged=cv2.Canny(blurred,10,60,apertureSize=3)
# plt.figure()
# plt.imshow(edged)

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(img)
ax[1].imshow(edged)
#%

nRow,nCol=img.shape

canvas=np.zeros(edged.shape)
#%
# lines=cv2.HoughLinesP(edged,1,np.pi/180,1,minLineLength=200,maxLineGap=90)
rho, theta, thresh = 2, np.pi/180/4, 100
lines = cv2.HoughLines(edged, rho, theta, thresh) 

if lines is not None:
    for i in range(0, len(lines)):
        
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        theta_deg=theta/np.pi*180
        
        
        if theta_deg<57 and theta_deg>53: # is True: #np.degrees(theta)<70 and np.degrees(theta)>30:  # filtering only horizontal and vertical Hough lines
            print(f'Theta in degrees {theta_deg}')
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + nCol*(-b)), int(y0 + nRow*(a)))
            pt2 = (int(x0 - nCol*(-b)), int(y0 - nRow*(a)))
            canvas=cv2.line(canvas, pt1, pt2, (255,255,255), 1)
            # print(pt1)


canvas_=np.copy(canvas).astype(np.uint8) 

plt.figure()
plt.imshow(canvas_)