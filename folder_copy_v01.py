# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:34:47 2021

@author: imrul
"""

import os
import sys
import shutil
from datetime import datetime
#%%%

# now=datetime.now()
# current_date=now.strftime('%Y-%m-%d')

# srcPath0=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\a'
# dstPath0=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\b'

# srcPath=os.path.join(srcPath0,current_date)
# dstPath=os.path.join(dstPath0,current_date)

# if os.isdir(dstPath) is False:
#     os.mkdir(dstPath)


# for folder in os.listdir(srcPath):
#     if os.path.isdir(os.path.join(srcPath,folder))==True:
#         # print(folder)
#         if folder not in os.listdir(dstPath):
#             print(folder)
#             shutil.move(os.path.join(srcPath,folder),os.path.join(dstPath,folder))
            
#%% file copy for cyc 7
src_path=r'Z:\raspberrypi\photos\imrul_sandbox\cyc7captures\2023-03-07\second_run'
dst_path=r'Z:\raspberrypi\photos\imrul_sandbox\cyc7captures\combined_folder\combined'

folder_key='run'

if os.path.exists(dst_path) is False:
    os.mkdir(dst_path)

for count,folder in enumerate(os.listdir(src_path)):
    fov=count
    
    if folder[:3]==folder_key:
        print(folder)
        for file in os.listdir(os.path.join(src_path,folder)):
            if file.endswith('raw'):
                new_file_name=file[:-4]+'-FOV_'+str(count)+'.raw'
                shutil.copy(os.path.join(src_path,folder,file),os.path.join(dst_path,new_file_name))
            
            


            