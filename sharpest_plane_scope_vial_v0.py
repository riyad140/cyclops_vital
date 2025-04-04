# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:10:27 2024

@author: imrul
"""

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

def read_sharp_plane(image_path, key = 'sharp_plane'):
    dd = read_tiff_metadata(image_path)
    a=dd['ImageDescription']
    key = "sharp_plane"    
    key2 = "gain_cal"
    time = float(a[a.find(key)+len(key)+2:a.find(key2)-2])
    
    return time

    
def find_time_difference(image_path, fileName0, fileName1):
    t0 = read_time_stamp(os.path.join(image_path,fileName0))
    t1 = read_time_stamp(os.path.join(image_path,fileName1))


    dTime = (t1-t0)/1000 # in seconds

    print(f'Time Difference: {dTime} seconds')
    
    return dTime
    
# Replace 'your_image.tiff' with the path to your TIFF image

# image_path = r"Z:\raspberrypi\photos\Alpha_plus\Leo\2024-06-27\1071R2\img_RBC_blank_BF_fov18_offset_3.tiff"

# metaData = read_sharp_plane(image_path)


# fileName0 = "img_RBC_blank_BF_fov5_offset_3.tiff"
# fileName1 = "img_RBC_blank_BF_fov4_offset_3.tiff"

# find_time_difference(fileName0, fileName1)
#%%
# fovCount = 8
# dTime = []

# for i in range(1,fovCount-1):
#     j=i+1
#     fileName0 =f"img_PLT_blank_BF_fov{i}_offset_0.tiff"
#     fileName1 =f"img_PLT_blank_BF_fov{j}_offset_0.tiff"
#     dTime.append(find_time_difference(fileName0, fileName1))


# plt.figure()
# plt.plot(dTime, 'o')
# plt.xlabel('FOV Count')
# plt.ylabel('Time [s]')
#%%

binPath = r'Z:\raspberrypi\photos\Alpha_plus\Libra\2024-07-11\2024-07-11T14-46-23_rbc'
keyTiff = 'img_RBC_blank_BF'

fileKey1 = 'fov'
fileKey2 = '_offset'

focusPlanes = []
fovs = []

for file in os.listdir(binPath):
    if file.find(keyTiff)>=0:
        print(file)
        fov = int(file[file.find(fileKey1)+len(fileKey1):file.find(fileKey2)])
        fovs.append(fov)
        tiffPath = os.path.join(binPath,file)
        focusPlanes.append(read_sharp_plane(tiffPath))
        
plt.figure()
plt.plot(fovs,focusPlanes,'o')
plt.ylabel('focus planes')
plt.xlabel('fov count')
        

plt.title(binPath)
