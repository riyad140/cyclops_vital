# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:11:49 2024

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.patches import Rectangle, Polygon, Circle

# from ROI_manager_user import userROI
import os
#import sys
import cv2
import math
from scipy import ndimage, signal
#from ROI_manager_user import userROI
import pandas as pd
import time
from tqdm import tqdm




pathLog=r"C:\Users\imrul\Downloads\External_encoder_data_20240905_165714.csv"
df = pd.read_csv(pathLog, names = ['Time','Encoder Count','Position Measure','Velocity Measure'],skiprows=(1))

#%%
plt.figure()
plt.plot(df['Time'],df['Position Measure'])

#%%


signs = np.sign(df['Position Measure']-359.95)  # to un wrap values after 360 degrees
zero_crossings = np.where(np.diff(signs) != 0)[0]
unique_zero_crossings = zero_crossings[1::2]
#%%
degrees = np.copy(df['Position Measure'].values)
for n,index in enumerate(unique_zero_crossings[:-1]):
    print (n)
    degrees[unique_zero_crossings[n]:unique_zero_crossings[n+1]] = degrees[unique_zero_crossings[n]:unique_zero_crossings[n+1]]-360*(n+1)
degrees[unique_zero_crossings[n+1]:] = degrees[unique_zero_crossings[n+1]:]-360*(n+2)    

plt.figure()
plt.plot(df['Time'],degrees)
