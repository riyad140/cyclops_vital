# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:09:28 2025

@author: imrul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = r"G:\My Drive\Working\TTR\timings.csv"

df = pd.read_csv(csv_path)
#%%
nRow = len(df)


index_FLR = np.arange(1,nRow,5)

capture_readout_time = []
set_filter_time = []
readout_time = []
storage_time = []
configuration_time = []

for i in index_FLR:
    print(df.loc[i]['path'])
    capture_readout_time.append(df.loc[i]['capture & readout time']/1e6)
    set_filter_time.append(df.loc[i]['set filter time']/1e6)
    
    
#%%
plt.figure()
plt.hist(set_filter_time, bins = 5)