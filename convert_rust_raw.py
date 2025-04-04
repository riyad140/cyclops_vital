# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:02:12 2022

@author: Jeff
"""

import rawpy
import imageio
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
from glob import glob


folder = r'Z:\raspberrypi\photos\Misc\2022-06-17_light_ring_characterization\2022-06-17\run000_LRIM_sample_s868_Cyc4Minerva\DF'
filenames = glob(os.path.join(folder, '*.raw'))

save_as_png_flag = True
#R camera RGGB
#G camera GBRG

for filename in filenames:
    if 'cyclops-G' in os.path.split(filename)[1]:
        bayer = 'GBRG'
    else:
        bayer = 'RGGB'
        
    im_flat = np.fromfile(filename, dtype='int16', sep="")
    im = im_flat.reshape((3040,4048))


    if bayer=='BGGR':
        B=im[0::2,0::2]
        G1=im[0::2,1::2]
        G2=im[1::2,0::2]
        R=im[1::2,1::2]
        G=(G1+G2)//2
    elif bayer=='GRBG':
        G1=im[0::2,0::2]
        R=im[0::2,1::2]
        B=im[1::2,0::2]
        G2=im[1::2,1::2]
        G=(G1+G2)//2
    elif bayer=='RGGB':
        R=im[0::2,0::2]
        G1=im[0::2,1::2]
        G2=im[1::2,0::2]
        B=im[1::2,1::2]
        G=(G1+G2)//2
    elif bayer=='GBRG':
        G1=im[0::2,0::2]
        B=im[0::2,1::2]
        R=im[1::2,0::2]
        G2=im[1::2,1::2]
        G=(G1+G2)//2
    else:
        print("No Bayer pattern not found.")

    rgb=np.dstack((R,G,B))
    rgb_r8 = np.uint8((rgb+1)/2**12*255)

    if save_as_png_flag:
        out_file =  filename[:-4] + '.bmp'
        io.imsave(out_file,rgb_r8)