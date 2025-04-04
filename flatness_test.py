# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 12:17:20 2022

@author: imrul
"""

import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt

#%%
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_average2(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def horizontal_xSection(imR):
    roi={'x0':0,
     'x1':-1,
     'y0':1500,
     'y1':2000
     }
    
    imCrop=imR[roi['y0']:roi['y1'],roi['x0']:roi['x1']]
    xSection=np.nanmean(imCrop,axis=0)
    xSection_f=moving_average2(xSection,100)
    return xSection_f

#%%


binPath=r'Z:\raspberrypi\photos\temp\2022-04-06_flatness_Test\cyc-Metal\subset_pp'    


key='R_BF'
plt.figure()
for file in os.listdir(binPath):
    if file.endswith('png') and file.find(key)>0:
        print(file)
        im=cv2.imread(os.path.join(binPath,file),1)
        imR=im[:,:,2]#red channel
        xSection_f=horizontal_xSection(imR)
        plt.plot(xSection_f,label=file[:-3])
plt.legend()
        

# binPath=r"Z:\raspberrypi\photos\temp\2022-04-06_flatness_Test\cyc-Metal\ppp_054321_sample_Beads_Cyc4Metal\cyclops-R_BF-FOV_5.png"
# #r"Z:\raspberrypi\photos\temp\2022-04-06_flatness_Test\run00_sample_beads_p3_vcm440_Cyc4Luna_FlatnessTest\cyclops-R_BF-FOV_0.png"

# im=cv2.imread(binPath,1)
# imR=im[:,:,2]# getting red channel image

# roi={'x0':0,
#      'x1':-1,
#      'y0':1500,
#      'y1':2000
#      }

# imCrop=imR[roi['y0']:roi['y1'],roi['x0']:roi['x1']]

# xSection=np.nanmean(imCrop,axis=0)
# #%%
# plt.figure()
# plt.plot(xSection)

# xSection_f=moving_average2(xSection,500)
# plt.plot(xSection_f,'k')
# plt.title(binPath[-14:])