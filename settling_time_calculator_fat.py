# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:49:21 2024

@author: imrul
"""

from PIL import Image
import tifffile
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def read_tiff_metadata(image_path):
    # Open the image using PIL
    dictTiff = {}
    # with Image.open(image_path) as img:
    #     # Print basic info using PIL
    #     print("PIL Info:")
    #     print(f"Format: {img.format}")
    #     print(f"Size: {img.size}")
    #     print(f"Mode: {img.mode}")
    #     print(f"Info: {img.info}")
        # print(f"Time: {img.time}")

    # Open the image using tifffile
    with tifffile.TiffFile(image_path) as tif:
        # print("\nTiffFile Info:")
        # Print TIFF file information
        for page in tif.pages:
            for tag in page.tags.values():
                tag_name, tag_value = tag.name, tag.value
                # print(f"{tag_name}: {tag_value}")
                dictTiff[tag_name] = tag_value
    return dictTiff


def read_time_stamp(image_path):
    dd = read_tiff_metadata(image_path)
    a=dd['ImageDescription']
    key = "time"    
    time = int(a[a.find(key)+len(key)+2:a.find('width')-2])
    
    return time
    
def find_time_difference(fileName0, fileName1):
    t0 = read_time_stamp(os.path.join(tiffPath,fileName0))
    t1 = read_time_stamp(os.path.join(tiffPath,fileName1))


    dTime = (t1-t0)/1000 # in seconds

    print(f'Time Difference: {dTime} seconds')
    
    return dTime
    
# Replace 'your_image.tiff' with the path to your TIFF image

tiffPath = r"W:\raspberrypi\photos\FAT_Captures\Beta\Unit-8\2024-12-18\ps_beads_benchmark_3m_tp_speaker_1\subset"


# fileName0 = "img_RBC_blank_BF_fov5_offset_3.tiff"
# fileName1 = "img_RBC_blank_BF_fov4_offset_3.tiff"

# find_time_difference(fileName0, fileName1) 
#img_WBC_blank_BF_fov1
#img_WBC_red_FLR_fov1
#img_RBC_blank_BF_fov{i}_offset_3
#%%
fovCount = 11
dTime = []

for i in range(1,fovCount-1):
    j=i+1
    fileName0 = f"img_FAT_beads_blank_BF_ledPWR_100_fov{i}.tiff"                     #f"img_OD_A_greenNB_BF_fov1_{i}.tiff"         #f"img_FAT_beads_blank_BF_ledPWR_100_fov{i}.tiff"               # f"img_RBC_blank_BF_fov{i}_offset_3.tiff"                   # f"img_HGB_green_BF_fov{i}_1.tiff"           #f"img_PLT_blank_BF_fov{i}_offset_0.tiff"
    fileName1 = f"img_FAT_beads_blank_BF_ledPWR_100_fov{j}.tiff"                      #f"img_OD_A_greenNB_BF_fov1_{j}.tiff"           #f"img_FAT_beads_blank_BF_ledPWR_100_fov{j}.tiff"                # f"img_RBC_blank_BF_fov{j}_offset_3.tiff"                   # f"img_HGB_green_BF_fov{j}_1.tiff"           #f"img_PLT_blank_BF_fov{j}_offset_0.tiff"
    dTime.append(find_time_difference(fileName0, fileName1))


plt.figure()
plt.plot(dTime, 'o')
plt.xlabel('FOV Count')
plt.ylabel('Time [s]')
plt.title(tiffPath)


print(np.mean(dTime))
print(np.std(dTime))

print('Total Time')
print(np.mean(dTime)*fovCount)

#%%


