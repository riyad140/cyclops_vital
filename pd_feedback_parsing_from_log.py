# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:21:09 2023

@author: imrul
"""
#%%

txtFilePath = r"W:\raspberrypi\photos\FAT_Captures\Beta\PD_response\green_PD_response_b003_2-240819.txt"


import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd



with open(txtFilePath) as f:
    lines = f.readlines()
word = "response.illuminator_resp.feedback_resp"   
# word2 = "double chosen_focus_sharpness /* 2 */ ="

dacVals = [10] + list(np.arange(100,1100,100))

pdVals = []
shapness_values = []
for line in lines:
    if line.find(word) >=0:
        print(line)
        pdVals.append(float(line[line.find('=')+2:line.find(';')]))
        # try:
        #     pdVals.append(np.int32(line[-5:-2]))
        # except:
        #     pdVals.append(np.int32(line[-3:-2]))
                
    # if line.find(word2) >=0:
    #     shapness_values.append(np.double(line[line.find(word2)+len(word2):-2]))
        
plt.figure()
plt.plot(dacVals, pdVals, 'o-')  
plt.xlabel('LED DAC VALUE')
plt.ylabel('PD VLAUE')     
plt.title(os.path.split(txtFilePath)[-1])
pngPath = txtFilePath[:-3]+'png'
plt.savefig(pngPath)

    
#%%