# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 15:30:38 2023

@author: imrul
"""
import os
import sys

binPath=r'C:\Users\imrul\Downloads\cyclops7_vdo\subset\gif'
#r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2023-03-27_Volpi_Visit\Cyc7_A2\BC_Ctrl_normal_run04'
#r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2023-03-27_Volpi_Visit\Cyc7_A2\PS_beads_run04_df_allblue'
#binPath=r'Z:\raspberrypi\photos\FAT_Captures\cyc6Dionysus\2023-02-09\run100_FAT5b_sample_Ronchi-1_Red_puck_dionysus'
# binPath=r'\\files.vital.company\cyclops\raspberrypi\photos\Erics Sandbox\2022-10-04_RONCHI_CollimatedvsDiffused\run05_FOV2_Diffused_exp10000_012A_sample_SNA_RONCHI__Cyc5Artemis'

key='img'
# zoom_window=250 # sets the window of the zoom for the ROI selectoin

# resultPath=os.path.join(binPath,f'analysis_final_{key}')

# ts=str(int(np.round(time.time(),0)))
# resultPath=resultPath+'_'+ts
# try:
#     os.mkdir(resultPath)
# except:
#     pass


files=[]
for file in os.listdir(binPath):
    if file.find(key)>-1 and file.endswith('tiff'):
        print(file)
        files.append(os.path.join(binPath,file))
#         im=plt.imread(os.path.join(binPath,file))
#         ims.append(im)
#%      reading the images after fov sorting
files.sort(key=os.path.getctime) 
files.reverse()

for n,file in enumerate(files):
    # print(n)
    path,f=os.path.split(file)
    new_f=str(n)+"_"+f
    print(new_f)
    os.rename(file,os.path.join(path,new_f))