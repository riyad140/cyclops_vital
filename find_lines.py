# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:32:03 2021

@author: imrul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
#%%
binPath=r"Z:\raspberrypi\photos\2021-10-15\demo_run02_from_pi_sample_s522__Cyc4Demo_Stardisk_NoSheet_drc_off\AF\subset\cyclops-_AF_0_475.bmp"
img=cv2.imread(binPath,0)
plt.figure()
plt.imshow(img)

start=time.time()
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/2,200,min_theta=np.pi/2)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    
end=time.time()-start
print(end)
plt.figure()
plt.imshow(img)
#%%
import os
def find_lines(img):
    if img.shape[-1]==3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi/2,200,min_theta=np.pi/2)
    # print(lines)
    if lines is None:
        lines=[]
    if len(lines)>=1:
        return True
    else:
        return False
    

start_time=time.time()    
binPath=r'Z:\raspberrypi\photos\2021-10-15\demo_run02_from_pi_sample_s522__Cyc4Demo_Stardisk_NoSheet_drc_off\AF'
for file in os.listdir(binPath):
    if file.endswith('bmp'):
        im=cv2.imread(os.path.join(binPath,file),0)
        img=im[im.shape[0]//3:im.shape[0]*2//3,im.shape[1]//3:im.shape[1]*2//3]
        lineCheck=find_lines(img)
        print(f'Filename: {file}  line: {lineCheck}')
end_time=time.time()
ETA=end_time-start_time
print(f'ETA: {ETA}')