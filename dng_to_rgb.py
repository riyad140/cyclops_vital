# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:02:00 2021

@author: imrul
"""

import rawpy
import imageio
import matplotlib.pyplot as plt
import os

path = r"Z:\raspberrypi\photos\trash\20211207_dng_raw\14-33-44_run00_sample_High_25mM_1-20_0.1Bead_100um_Hb\cyclops-G_exp_0.0_BF-FOV_0.dng"
with rawpy.imread(path) as raw:
    rgb = raw.postprocess(no_auto_bright=True,no_auto_scale=True, output_bps=16,use_auto_wb=False,use_camera_wb=False)


dst_folder=r'Z:\raspberrypi\photos\trash\20211207_dng_raw\dng2raw_repo'    
dst_filename='linear16bit.tiff'
imageio.imsave(os.path.join(dst_folder,dst_filename),rgb)

