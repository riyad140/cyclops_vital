# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:14:40 2024

@author: imrul
"""

import numpy as np
import pandas as pd
import os

def plt_count(cellCount):
    (0.006*(3120*4096*(0.0001375)**2))*6/10
    channelDepth = 0.006
    dillRatio = 6
    mFactor = 10
    pxToDisRatio = 0.0001375
    
    roiArea = 2400*3400
    
    return cellCount/(channelDepth*roiArea*(pxToDisRatio)**2)*dillRatio/mFactor


#%%
runFolders = [
              r'Z:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PV\2024-06-28\MC_S1074W-S1075R_AS1',
             #  r'Z:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PV\2024-06-10\Prec_S1030_Run2_AS1',
             #  r'Z:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PV\2024-06-10\Prec_S1030_Run3_AS1',
             #   r'Z:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PV\2024-06-10\Prec_S1030_Run4_AS1',
             # r'Z:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PV\2024-06-10\Prec_S1030_Run5_AS1',
             #  r'Z:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\PV\2024-06-10\Prec_S1030_Run6_AS1',
              
              ]

key= 'pl_bf'# pl_bf' 
sample='1030'
bcCount = 244.9

fovToRemove = [3,5,6,8]

indexToRemove = list(np.array(fovToRemove)-1)

df_list = []

for n,runFolder in enumerate(runFolders):
    analysisFolder = ""
    for file in os.listdir(runFolder):
        if file.endswith('tiff') is False:
            if file.find(key) >= 0:
                analysisFolder = os.path.join(runFolder,file)
                
    print(analysisFolder)
    
    for file in os.listdir(analysisFolder):    
        if file.find('perfov_qc_checks')>=0:
            csvFile = os.path.join(analysisFolder,file)
    
    #%%
    
    # csvFile = r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A8\PV\2024-06-05\PREC_S1024_RUN2_PM\pl_bf-results-20240606_174254\PREC_S1024_RUN2_PM-plt_perfov_qc_checks.csv"
    
    df = pd.read_csv(csvFile)
    
    df_ = df.drop(indexToRemove)                    #[df['omit_fov']==False]
    
    
    meanCount = np.mean(df_['cell_count'])
    maxCount = np.max(df_['cell_count'])
    
    
    pltMeanCount = plt_count(meanCount)/1000
    pltMaxCount = plt_count(maxCount)/1000
    
    pltMeanCountPt = pltMeanCount / bcCount * 100
    pltMaxCountPt = pltMaxCount / bcCount * 100
    
    
    print(f'mean count {pltMeanCount}\nMax count {pltMaxCount}')
    
    
    resultDict = {'path': csvFile,
                  'meanCount': pltMeanCount,
                  'maxCount': pltMaxCount,
                  'BC': bcCount,
                  'meanCountRatio': pltMeanCountPt,
                  'maxCountRatio': pltMaxCountPt
        }

    df_list.append(pd.DataFrame(resultDict,index=[n]))

df_results = pd.concat(df_list)
print(df_results)

saveFolder = os.path.join(os.path.split(runFolder)[0],key)+'_'+sample+'.csv'

# df_results.to_csv(saveFolder)