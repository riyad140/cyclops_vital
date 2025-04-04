# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:34:47 2021

@author: imrul
"""

import os
import sys
import shutil
#%%%

srcPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\a'
dstPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\b'


for folder in os.listdir(srcPath):
    if os.path.isdir(os.path.join(srcPath,folder))==True:
        # print(folder)
        if folder not in os.listdir(dstPath):
            print(folder)
            shutil.move(os.path.join(srcPath,folder),os.path.join(dstPath,folder))
            