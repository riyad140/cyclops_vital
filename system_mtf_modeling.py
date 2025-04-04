# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:10:38 2022

@author: imrul
"""

#MTF graphs
import numpy as np
import math
import matplotlib.pyplot as plt
#%%


px_size=1.15e-3 #mm


def get_sensorMTF(px_size):
    f_lpmm=np.linspace(1,500,400)
    fc= 1/2/px_size
    phi=[]
    for f in f_lpmm:
        try:
            phi.append(math.acos(f/fc))
        except:
            phi.append(0)
                
    phi=np.array(phi)
    
    mtf=2/np.pi*(phi-np.cos(phi)*np.sin(phi))
    return f_lpmm,mtf


f,sensorMTF=get_sensorMTF(px_size) 

plt.figure(33)
plt.plot(f,sensorMTF,label='sensor')

# mtf_tubelenstxt

lensMTF=np.interp(f,mtf_tubelenstxt[:,0],mtf_tubelenstxt[:,1])

plt.figure(33)
plt.plot(f,lensMTF,label='lens')



system_mtf=lensMTF*sensorMTF

plt.figure(33)
plt.plot(f,system_mtf,label='system')
plt.legend()