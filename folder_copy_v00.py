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

now=datetime.now()
current_date=now.strftime('%Y-%m-%d')

srcPath0=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\a'
dstPath0=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\b'

srcPath=os.path.join(srcPath0,current_date)
dstPath=os.path.join(dstPath0,current_date)

if os.isdir(dstPath) is False:
    os.mkdir(dstPath)


for folder in os.listdir(srcPath):
    if os.path.isdir(os.path.join(srcPath,folder))==True:
        # print(folder)
        if folder not in os.listdir(dstPath):
            print(folder)
            shutil.move(os.path.join(srcPath,folder),os.path.join(dstPath,folder))
            