# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:41:58 2025

@author: imrul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

files = [r"W:\raspberrypi\photos\Vibration_Study\2025-03-04\run19\sharpness_analysis\BF_sharpness.csv",
        r"W:\raspberrypi\photos\Vibration_Study\2025-03-04\run20\sharpness_analysis\BF_sharpness.csv",
        r"W:\raspberrypi\photos\Vibration_Study\2025-03-06\run21\sharpness_analysis\BF_sharpness.csv",
        r"W:\raspberrypi\photos\Vibration_Study\2025-03-06\run22\sharpness_analysis\BF_sharpness.csv"]

plt.figure()
for file in files:
    df = pd.read_csv(file)
    plt.plot(df['FOV'],df['Sharpness']-np.max(df['Sharpness']),'.-')

plt.xlabel('FOV count')
plt.ylabel('Normalized Sharpness')
plt.ylim([-0.03,0])
# plt.title(os.path.join(tiffPath,keyTiff))
# plt.savefig(os.path.join(analysis_folder_name,keyTiff +'_plot.png'))