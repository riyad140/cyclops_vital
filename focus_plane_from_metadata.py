# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:19:42 2024

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
import json
import numpy as np

def read_tiff_metadata(imagePath):
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
    with tifffile.TiffFile(imagePath) as tif:
        # print("\nTiffFile Info:")
        # Print TIFF file information
        for page in tif.pages:
            for tag in page.tags.values():
                tag_name, tag_value = tag.name, tag.value
                # print(f"{tag_name}: {tag_value}")
                dictTiff[tag_name] = tag_value
    return dictTiff



def read_sharp_plane(imagePath):
    metadata = read_tiff_metadata(imagePath)
    imageDescription = metadata['ImageDescription']
    json_object = json.loads(imageDescription)
    
    focusPlane = json_object['sharp_plane']
    
    return focusPlane
    
    
    



# def read_time_stamp(image_path):
#     dd = read_tiff_metadata(image_path)
#     a=dd['ImageDescription']
#     key = "time"    
#     time = int(a[a.find(key)+len(key)+2:a.find('width')-2])
    
#     return time
    
# def find_time_difference(fileName0, fileName1):
#     t0 = read_time_stamp(os.path.join(tiffPath,fileName0))
#     t1 = read_time_stamp(os.path.join(tiffPath,fileName1))


#     dTime = (t1-t0)/1000 # in seconds

#     print(f'Time Difference: {dTime} seconds')
    
#     return dTime
    
# # Replace 'your_image.tiff' with the path to your TIFF image
#%%


directory = r'W:\raspberrypi\photos\Alpha_sharp\CYC7_AS3\2025-01-08'
keyword = 'S'


assayName = 'wbc' # 'rbc'

if assayName == 'wbc':
    keyTiff = 'img_WBC_blank_BF'
    keyAfterFov = '.tiff'
elif assayName == 'rbc':
    keyTiff = 'img_RBC_blank_BF'    
    keyAfterFov = '_offset'  # '_offset'  ,  .tiff  (for WBC)





subfolders = [f.path for f in os.scandir(directory) if f.is_dir()and keyword in f.name]

for tiffPath in subfolders:

    # tiffPath = r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-09-24\S1203_PREC_RUN3_IM5.1_PBS_AS1"
    print(tiffPath)
    
    
    
    fovs = []
    focusPlanes = []
    
    for file in os.listdir(tiffPath):
        if file.find(keyTiff)>=0 and file.endswith('tiff'):
            print(file)
            fovCount = int(file[file.find('fov')+3:file.find(keyAfterFov)])
            fovs.append(fovCount)
            focusPlanes.append(read_sharp_plane(os.path.join(tiffPath,file)))
            
    
    
    sorted_indices = sorted(range(len(fovs)), key=lambda i: fovs[i])
    
    sorted_fovs = [fovs[i] for i in sorted_indices]
    sorted_focusPlanes = [focusPlanes[i] for i in sorted_indices]
    
    
    
    
    
    minVariation = np.min(np.diff(sorted_focusPlanes))
    maxVariation = np.max(np.diff(sorted_focusPlanes))
            
    plt.figure()
    plt.plot(sorted_fovs, sorted_focusPlanes, 'o')
    plt.xlabel('Fov Count')
    plt.ylabel('Sharpest Plane')
    plt.title(tiffPath)
    pngName = 'focus_variation_of_'+keyTiff+'.png'
    plt.savefig(os.path.join(tiffPath,pngName))
            
    
    dictFocus = {'path': tiffPath,
                 'key': keyTiff,
                 
                 'meanFocusPlane': np.mean(sorted_focusPlanes),
                 'stdFocusPlane': np.std(sorted_focusPlanes),
                 'maxVariation': maxVariation,
                 'minVariation': minVariation ,
                 'focusPlanes': sorted_focusPlanes,             
                 
                 }
    
    
    jsonFileName = keyTiff +'_focus_info.json'
    
    with open(os.path.join(tiffPath,jsonFileName), 'w') as json_file:
        json.dump(dictFocus, json_file, indent=4) 
    
    print(dictFocus)    
# focus = read_sharp_plane(os.path.join(tiffPath,fileName))