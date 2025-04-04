# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:04:51 2024

@author: imrul
"""

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

unpickled_df = pd.read_pickle(r"Z:\raspberrypi\photos\Alpha_plus\CYC7_A8\PV\2024-06-05\PREC_S1024_RUN2\wbc-results-20240605_150354\PREC_S1024_RUN2-df_features.pickle")

plt.figure()
plt.plot(unpickled_df['fov'],unpickled_df['mean_intensity_red'],'.')
plt.xlabel('FOVs')
plt.ylabel('Cell Intensity')