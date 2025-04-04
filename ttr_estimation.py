# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:10:29 2023

@author: imrul
"""





import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime



def time_difference(wbcTimes):
    t1 = datetime.strptime(wbcTimes[0], "%H:%M:%S")
    t2 = datetime.strptime(wbcTimes[-1], "%H:%M:%S")
    
    delta = t2 - t1
    
    # time difference in seconds
    print(f"Time difference is {delta.total_seconds()} seconds")
    return delta.total_seconds()
    
#%%    
    
txtFilePath = r"C:\Users\imrul\Downloads\log_A10.rtf"

with open(txtFilePath) as f:
    lines = f.readlines()
    
hgbKey = "HGB Image gathering"   
rbcKey = "RBC Image gathering"    
wbcKey = "WBC Image gathering"
pltKey = "PLT Image gathering"

keys = [hgbKey,rbcKey,wbcKey,pltKey]

hgbTimes = []
rbcTimes = []
wbcTimes = []
pltTimes = []

for line in lines:
    if line.find(hgbKey) >=0:
        print(line)
        hgbTimes.append(line[7:15])
    if line.find(rbcKey) >=0:
        rbcTimes.append(line[7:15])
    if line.find(wbcKey) >=0:
        wbcTimes.append(line[7:15])
    if line.find(pltKey) >=0:
        pltTimes.append(line[7:15])
        
#%%        
a=time_difference(hgbTimes)
b=time_difference(rbcTimes)
c=time_difference(wbcTimes)
d=time_difference(pltTimes)

plt.figure(20)
plt.plot([a,b,c,d],'o')