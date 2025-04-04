from pidng.core import RPICAM2DNG
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters
import os
import tqdm

import rawpy
import imageio
import matplotlib.pyplot as plt
import os

 
#%%
# raw_path=r'/home/pi/Desktop/Final_Scripts/semi_final/custom_scripts/2021-12-07/14-33-44_run00_sample_High_25mM_1-20_0.1Bead_100um_Hb/cyclops-G_exp_0.0_BF-FOV_0.jpg'

d = RPICAM2DNG()
# d.convert(raw_path)

binPath=r'G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\20220201_IMX219_BinnigTest_RED'
for file in os.listdir(binPath):
    if file.endswith('bmp'):
        raw_path=os.path.join(binPath,file)
        d.convert(raw_path)        
        

src_path = binPath#r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2021-01-26_SensorCharacterization\imx477"
dst_path= os.path.join(src_path,'TIFFs_8bit_no_auto_brightness')
if os.path.isdir(dst_path) is False:
    os.mkdir(dst_path)

for file in os.listdir(src_path):
    if file.endswith('dng'):
        # print(file)
        print(f'filename: {file}')
        with rawpy.imread(os.path.join(src_path,file)) as raw:
            rgb = raw.postprocess(no_auto_bright=False,no_auto_scale=False, output_bps=8,use_auto_wb=False,use_camera_wb=False)
        dst_filename=file[:-3]+'tiff'    
        imageio.imsave(os.path.join(dst_path,dst_filename),rgb)   
