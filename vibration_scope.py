# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:47:21 2024

@author: imrul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import os
# dt_string = "2024-4-5 10:31:46:426"

# dt_obj = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S:%f")
# millisec = dt_obj.timestamp() * 1000

# print(millisec)



def ts_to_ms(dt_strings):
    millisecs=[]
    for dt_string in dt_strings:
        dt_obj = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S:%f")
        millisecs.append( dt_obj.timestamp() * 1000)
    return np.array(millisecs)

import pandas as pd
pathLog=r"C:\Users\imrul\Downloads\20250122152040.txt"
df = pd.read_csv(pathLog, delimiter = "\t")

# plt.figure()
# plt.plot(df['AccX(g)'])
# plt.plot(df['AccY(g)'])



time = ts_to_ms(df['time'].values)

dTime = np.median(np.diff(time))  # ms


accX=df['AccX(g)'].values
accY= df['AccY(g)'].values
accZ = 1-df['AccZ(g)'].values



accX_=accX-np.median(accX)
accY_=accY-np.median(accY)
accZ_=accZ-np.median(accZ)



acc = np.sqrt(accX**2+1*accY**2+accZ**2)

acc_ = np.sqrt(accX_**2+1*accY_**2+accZ_**2)

offsetY = 0.05

plt.figure()
plt.plot(accX_,label='X')
plt.plot(accY_-offsetY,label='Y-offset')
plt.plot(accZ_-2*offsetY,label='Z-offset*2')
# plt.plot(acc,label='R')
plt.plot(acc_+2*offsetY,label='R+offset*2')
plt.ylim([-0.2, 0.2])
plt.grid(True)
plt.ylabel('Acc. [g]')
plt.xlabel('Data points')
plt.title(os.path.split(pathLog)[-1])
plt.legend()

# plt.figure()
# plt.plot(time,acc)
# plt.ylim([-0.07, 0.07])
# plt.grid(True)
# plt.ylabel('Acc. [g]')
# plt.xlabel('Time[ms]')
# plt.title(os.path.split(pathLog)[-1])
# plt.legend()
#%%
# angleX = df['AngleX(°)']
# angleY = df['AngleY(°)']
# angleZ = df['AngleZ(°)']

# plt.figure()
# plt.plot(angleX)
# plt.plot(angleY)
# plt.plot(angleZ)


#%%
plt.figure()
plt.plot(accY)