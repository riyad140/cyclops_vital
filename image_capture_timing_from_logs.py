# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:21:09 2023

@author: imrul
"""
#%%

txtFilePath = r"C:\Users\imrul\Downloads\captured_logs_imaging (1).txt"


import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd



with open(txtFilePath) as f:
    lines = f.readlines()
word1 = "Configure(Config"    
word2 = "capture_ok"   
# word2 = "double chosen_focus_sharpness /* 2 */ ="

# dacVals = [10] + list(np.arange(100,1100,100))

pdVals = []
shapness_values = []

initTimes = []
captureTimes = []
for line in lines:
    
    if line.find(word1) >=0:
        print(line)
        initTimes.append(line[:26])
    
    
    if line.find(word2) >=0:
        print(line)
        captureTimes.append(line[:26])
        # pdVals.append(float(line[line.find('=')+2:line.find(';')]))
        # try:
        #     pdVals.append(np.int32(line[-5:-2]))
        # except:
        #     pdVals.append(np.int32(line[-3:-2]))
                
    # if line.find(word2) >=0:
    #     shapness_values.append(np.double(line[line.find(word2)+len(word2):-2]))
        
# plt.figure()
# plt.plot(dacVals, pdVals, 'o-')  
# plt.xlabel('LED DAC VALUE')
# plt.ylabel('PD VLAUE')     
# plt.title(os.path.split(txtFilePath)[-1])
# pngPath = txtFilePath[:-3]+'png'
# plt.savefig(pngPath)

    
#%%