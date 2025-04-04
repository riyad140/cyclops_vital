# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:00:22 2024

@author: imrul
"""

import pandas as pd
import matplotlib.pyplot as plt

csvFile = r"W:\raspberrypi\photos\Alpha_plus\Libra_all_motors_20240724_mod.csv"
df = pd.read_csv(csvFile)





fig,ax = plt.subplots(3,1,sharex=True)
ax[0].plot(df['TS_HT'],df['Position Measure HT'])
ax[0].set_title('HT Position')
ax[1].plot(df['TS_CC'],df['Velocity Measure CC'])
ax[1].set_title('CC Velocity')
ax[2].plot(df['TS_IA'],df['Velocity Measure IA'])
ax[2].set_title('IA velocity')
