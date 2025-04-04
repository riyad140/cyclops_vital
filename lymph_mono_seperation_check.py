# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:50:49 2023

@author: imrul
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
def get_lymph_mono_seperation(csvPath, figNum = 100, legend = 'sXXX', color = 'r'):
    df = pd.read_csv(csvPath)
    lymph = df['mean_intensity_red_1.0']
    mono = df['mean_intensity_red_2.0']
    
    meanIntRed = df['mean_intensity_red']
    
    windowSize = 3
    beginInt = np.mean(meanIntRed[:windowSize])
    endInt = np.mean(meanIntRed[-windowSize:])
    
    intDrop = (beginInt-endInt)/beginInt*100
    print(f'% intensity drop {intDrop}')
    
    
    plt.figure(figNum)
    plt.plot(lymph,'o-', color = color)
    plt.plot(mono, color = color, label = legend)
    plt.xlabel('FOV#')
    plt.ylabel('Intensity [a.u.]')
    plt.legend()
    plt.ylim([400,900])
    plt.grid(True,axis = 'y')
    
    plt.figure(figNum+1)
    plt.plot(meanIntRed,color = color, label = legend)
    plt.xlabel('FOV#')
    plt.ylabel('Intensity [a.u.]')
    plt.legend()
    plt.ylim([400,900])
    plt.grid(True,axis = 'y')

#%%
# binPath = r'Z:\raspberrypi\photos\Alpha_plus\CYC7_A10\20230920\WBC_Cyc7-A10_s594_run03_LRV2.nonPC\results-20230920_202338-WBC_Cyc7-A10_s594_run03_LRV2nonPC'
# filename = 'results-20230920_202338-WBC_Cyc7-A10_s594_run03_LRV2nonPC-qc_checks.csv'

# df = pd.read_csv(os.path.join(binPath,filename))

# lymph = df['mean_intensity_red_1.0']
# mono = df['mean_intensity_red_2.0']

# plt.figure()
# plt.plot(lymph,'o-',color = 'r')
# plt.plot(mono, color = 'r')
# plt.legend()


#%% 

csvPath = r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A10\20230920\WBC_Cyc7-A10_s594_run03_LRV2.nonPC\results-20230920_202338-WBC_Cyc7-A10_s594_run03_LRV2nonPC\results-20230920_202338-WBC_Cyc7-A10_s594_run03_LRV2nonPC-qc_checks.csv"
get_lymph_mono_seperation(csvPath, figNum = 200, legend = 'LRV2.nonPC', color = 'r')

csvPath = r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A10\20230920\WBC_Cyc7-A10_s594_run02_LRV1_\results-20230920_191629-WBC_Cyc7-A10_s594_run02_LRV1_\results-20230920_191629-WBC_Cyc7-A10_s594_run02_LRV1_-qc_checks.csv"
get_lymph_mono_seperation(csvPath, figNum = 200, legend = 'LRV1', color = 'b')    

csvPath = r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A10\20230920\WBC_Cyc7-A10_s594_run01_LRV2.6_\results-20230920_173339-WBC_Cyc7-A10_s594_run01_LRV26_\results-20230920_173339-WBC_Cyc7-A10_s594_run01_LRV26_-qc_checks.csv"
get_lymph_mono_seperation(csvPath, figNum = 200, legend = 'LRV2.6', color = 'g')     

csvPath = r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A10\20230920\WBC_Cyc7-A10_s594_run00_LRV2_RB01_\results-20230920_153747-WBC_Cyc7-A10_s594_run00_LRV2_RB01_\results-20230920_153747-WBC_Cyc7-A10_s594_run00_LRV2_RB01_-qc_checks.csv"
get_lymph_mono_seperation(csvPath, figNum = 200, legend = 'LRV2.RB01', color = 'k')     

    