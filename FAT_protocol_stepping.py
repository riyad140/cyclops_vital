# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:06:22 2022

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

from ROI_manager_user import userROI
import os
import sys
import cv2
import math
#%%

# binPath=r'Z:\raspberrypi\photos\FAT_Captures\run03_sample_FAT_v3_usaf_Cyc4Juno\stepping'
binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\FAT_Captures\run03_sample_FAT_v3_usaf_Cyc4Juno\stepping'
key='G_BF'
channel=1 # BGR

ims=[]
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith('bmp'):
        print(file)
        im=cv2.imread(os.path.join(binPath,file),1)
        ims.append(im[:,:,channel])
        



#%%


def onKey(event):
        count = 0
        if event.key == " ":
            print('click to select an edge')
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            

        elif event.key == "enter":
            # showROI()
            fig.canvas.stop_event_loop()
            fig.canvas.mpl_disconnect(cidk)
            showROI()

        else:
            print("Press Space and then select the same edge across all the frames" )
            
def onclick(event):
        ix, iy = event.xdata, event.ydata
        coords.append([int(ix), int(iy)])

def printInstructions():
    print("INSTRUCTIONS:")
    print("Press spacebar, then click the same elements in all the frames")
    print("Press enter if you are satisfied with the selection")
            
    

#self.patches.append(self.axes.add_patch(Circle((self.coords[0][0], self.coords[0][1]), 2, edgeColor="r", fill=True)))


def showROI():
    
        # for patch in self.patches:
        #     patch.remove()
        #     self.patches = []

    patches=[]
    patches.append(ax[0,0].add_patch(Circle((coords[0][0], coords[0][1]), 20, edgeColor="r", fill=False)))
    patches.append(ax[0,1].add_patch(Circle((coords[1][0], coords[1][1]), 20, edgeColor="r", fill=False)))
    patches.append(ax[1,0].add_patch(Circle((coords[2][0], coords[2][1]), 20, edgeColor="r", fill=False)))
    patches.append(ax[1,1].add_patch(Circle((coords[3][0], coords[3][1]), 20, edgeColor="r", fill=False)))
    
    patches.append(ax[0,0].add_patch(Circle((coords[1][0], coords[1][1]), 20, edgeColor="k", fill=False)))
    patches.append(ax[0,0].add_patch(Circle((coords[2][0], coords[2][1]), 20, edgeColor="k", fill=False)))
    patches.append(ax[0,0].add_patch(Circle((coords[3][0], coords[3][1]), 20, edgeColor="k", fill=False)))
        # else:
        #     self.patches.append(self.axes.add_patch(Polygon(self.coords, edgeColor="r", fill=False)))
    fig.canvas.draw()
    
def get_angle(coords0,coords1):
    dy=coords0[1]-coords1[1]
    dx=coords0[0]-coords1[0]
    
    return np.degrees(math.atan(dy/dx))
#%%    
coords=[]
nIm=4
r,c=2,nIm//2
printInstructions() 
        
fig, ax = plt.subplots(2,2,sharex=True,sharey=True)
for i in range(4):
    b=bin(i)[2:]
    if len(b)==1:
        r=0
        c=int(b)
    else:
        r=int(b[0])
        c=int(b[1])
    ax[r,c].imshow(ims[i+1])
    ax[r,c].set_title(f'Frame Count {i}')
      



           
cidk=fig.canvas.mpl_connect('key_press_event', onKey)
fig.canvas.start_event_loop()

#%
step_size=[]
for i in range(len(coords)-1):
    step_size.append(math.dist(coords[i],coords[i+1]))

print('Motor Stepping size in pixels')    
print(step_size)  

m,s=np.median(step_size),np.std(step_size)
print(f'Median: {m}\nStd: {s}')

angles=[]
for i in range(len(coords)-1):
    angles.append(abs(get_angle(coords[i],coords[i+1])))
print(f'Angles: {angles}')
#%%
# u=coords[0]
# v=coords[3]

# c=np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)
# angle=np.degrees(np.arccos(c))

  

# #%%


# ang=get_angle(coords[2],coords[3])