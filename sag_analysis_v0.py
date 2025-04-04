# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:59:21 2025

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


def get_sag_params(image_path):
    info = read_tiff_metadata(image_path)['ImageDescription']
    targetKey = 'gain_cal'
    
    sagString = info[info.find(targetKey)+len(targetKey)+3:-2]
    
    # print(sagString)
    
    slope,intercept = float(sagString[sagString.find('m')+3:sagString.find(',')]),float(sagString[sagString.find('b')+3:-1])
    
    
    targetKey0 = 'gain'
    targetKey1 = 'ss'
    
    gain = float(info[info.find(targetKey0)+len(targetKey0)+2:info.find(targetKey1)-2])
    
    print(f' slope {slope}\n intercept {intercept}\n Gain {gain}')
    
    return slope, intercept , gain
    
    


tiffPath = r"W:\raspberrypi\photos\Juravinski\2025-01-22\01-22-S35"


imageTypes = {
    
    'BF': 'img_WBC_blank_BF_fov1.tiff',
    'FLR': 'img_WBC_red_FLR_fov1.tiff',
    'FLG': 'img_WBC_green_FLG_fov1.tiff',
    'DF': 'img_WBC_blue_DF_fov1.tiff'
    
    }

data = []
for key in imageTypes.keys():
    print(key)
    slope, intercept, gain = get_sag_params(os.path.join(tiffPath,imageTypes[key]))
    data.append([slope, intercept, gain])
    
    
df = pd.DataFrame(data, index = imageTypes.keys(),columns = ['slope','intercept','gain'])


folderName = os.path.join(tiffPath,'SAG_Analysis')

try:
    os.mkdir(folderName)
except:
    print('Folder Already Exist?')

csvName = 'sag_statistics.csv'
pngName = 'sag_analysis.png'

df.to_csv(os.path.join(folderName,csvName))
#%%


imageModality = ['BF','FLR','FLG','DF']
colors = ['black','red','green','blue']

x = np.arange(1,8,0.25)

plt.figure()
for n,modality in enumerate(imageModality):
    

# modality = 'BF'

    y = df['slope'][modality]*x  + df['intercept'][modality]

    g = df['gain'][modality]

# plt.figure(100)
    plt.plot(x,y,label = modality, color = colors[n])
# ax = plt.gca()
# last_line = ax.get_lines()[-1]
# last_color = last_line.get_color()
    plt.plot(g,y[np.argwhere(x==g)[0][0]],marker = 'o', color = colors[n])

    plt.legend()
plt.xlabel('Gain')
plt.ylabel('SAG intensity')
plt.grid(axis='y', linestyle='--', color='gray', alpha=0.7)
plt.ylim([0,1500])
plt.title(os.path.split(tiffPath)[-1])
plt.savefig(os.path.join(folderName,pngName))