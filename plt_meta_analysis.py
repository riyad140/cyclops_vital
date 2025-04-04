# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 07:41:12 2024

@author: imrul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#%%






def plt_qc_analysis (csvPath, label = 'gg', figNum = 99):
    df = pd.read_csv(csvPath)
    df_filtered = df[df['omit_fov']==False]
    
    
    fig,ax = plt.subplots(2,1,sharex=False)
    ax[0].plot(df_filtered['fov_num'],df_filtered['cell_count'],label = label)
    ax[0].legend()
    ax[0].set_title('Cell Count')
    ax[1].plot(df_filtered['fov_num'],df_filtered['red_background'],'o-')
    ax[1].set_title('FL mean Background')
    
    return df_filtered['fov_num'],df_filtered['cell_count'],df_filtered['red_background']



csvPath = r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A2\2024-02-29\S824-PLT3\plt-results-20240229_164345\S824-PLT3-perfov_qc_checks.csv"




csvPaths = [ r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A2\2024-02-29\S823-PLT-CoarseAFLap\plt-results-20240229_163407\S823-PLT-CoarseAFLap-perfov_qc_checks.csv",
            r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A2\2024-02-29\S823-PLT-Re-CoarseAFLap\plt-results-20240229_163512\S823-PLT-Re-CoarseAFLap-perfov_qc_checks.csv",
            r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A2\2024-02-29\S823-PLT-ReRe-CoarseAFLap\plt-results-20240229_163617\S823-PLT-ReRe-CoarseAFLap-perfov_qc_checks.csv"
    ]

stats=[]

for csvPath in csvPaths:
    print(csvPath)
    stats.append(plt_qc_analysis (csvPath, label = os.path.split(csvPath)[-1]))
    
plt.figure()
plt.plot(stats[0][0],stats[0][1],'o--')
plt.plot(stats[1][0],stats[1][1],'o--')
plt.plot(stats[2][0],stats[2][1],'o--')
plt.xlabel('FOV#')
plt.ylabel('Cell Count')
plt.title('Cell count vs fov count')


plt.figure()
plt.plot(stats[0][0],stats[0][2],'o--')
plt.plot(stats[1][0],stats[1][2],'o--')
plt.plot(stats[2][0],stats[2][2],'o--')
plt.xlabel('FOV#')
plt.ylabel('FL Background')
plt.title('Background vs fov count')


#%%

def plt_qc_analysis2 (csvPaths, label = 'gg', figNum = 99):
    fig,ax = plt.subplots(2,1,sharex=False,num = figNum)
    for csvPath in csvPaths:   
        df = pd.read_csv(csvPath)
        df_filtered = df[df['omit_fov']==False] 
    
    
        ax[0].plot(df_filtered['fov_num'],df_filtered['cell_count'],label = os.path.split(csvPath)[-1])
        
        ax[0].set_title('Cell Count')
        ax[1].plot(df_filtered['fov_num'],df_filtered['red_background'], 'o--',label = os.path.split(csvPath)[-1])
        ax[1].set_title('FL mean Background')
        ax[1].legend()
    
    return 0

plt_qc_analysis2(csvPaths)