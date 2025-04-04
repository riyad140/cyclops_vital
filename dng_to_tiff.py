# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:49:41 2021

@author: imrul
"""

import rawpy
import imageio
import matplotlib.pyplot as plt
import os

src_path = r"G:\Shared drives\Engineering\Optics_Sensing_Imaging\CC-Cyclops\Imrul_python_script_ISP\image_repo\2021-01-26_SensorCharacterization\imx477"
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

#%%            
# path=r'Z:\raspberrypi\photos\Misc\2021-12-09\run00-80percent_sample_s601-1to40diluted_cyc4Metal\cyclops-G_0_exp_10.0_ms__BF-FOV_0.dng'
# with rawpy.imread(path) as raw:
#     rgb = raw.postprocess(no_auto_bright=True,no_auto_scale=True, output_bps=8,use_auto_wb=False,use_camera_wb=False)
# plt.figure()
# plt.imshow(rgb)

# dst_folder=r'Z:\raspberrypi\photos\trash\20211207_dng_raw\dng2raw_repo'    
# dst_filename='linear16bit.tiff'
# imageio.imsave(os.path.join(dst_folder,dst_filename),rgb)