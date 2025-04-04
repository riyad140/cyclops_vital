# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:19:17 2025

@author: imrul
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import os
import pandas as pd



run_path = r"W:\raspberrypi\photos\Beta\B011\2025-02-25\S048_1"

assayName = 'WBC'


fovs = 30
fovs_to_skip = []

analysis_folder = os.path.join(run_path,f'{assayName}_Area_Analysis')

df= pd.DataFrame(columns = ['green_intensities', 'red_intensities', 'nucleus_areas', 'cytoplasm_areas', 'fov' ])

data=[]


from datetime import datetime

# Get the current timestamp with high precision
unique_timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")


try:
    os.mkdir(os.path.join(run_path,analysis_folder))
    
except:
    print('Analysis Folder Already Exists. Overwritting Files')    

for fov in range(1,fovs+1):
    
    if fov in fovs_to_skip:
        continue  # Skip the current FOV
    print(f"Processing FOV {fov}")
    
    green_channel_path = os.path.join(run_path ,f"img_WBC_green_FLG_fov{fov}.tiff" ) # Path to green channel image
    red_channel_path =os.path.join(run_path ,f"img_WBC_red_FLR_fov{fov}.tiff")

    # Step 1: Load the green and red channel images (10-bit TIFF)
    # green_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S15\img_WBC_green_FLG_fov9.tiff"  # Path to green channel image
    # red_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S15\img_WBC_red_FLR_fov9.tiff" 
    
    
    green_image = cv2.imread(os.path.join(run_path,green_channel_path), cv2.IMREAD_UNCHANGED)
    red_image = cv2.imread(os.path.join(run_path,red_channel_path), cv2.IMREAD_UNCHANGED)
    
    # blue_image = np.zeros_like(red_image, dtype=red_image.dtype)

    # # Combine the channels into an RGB image
    # rgb_image = np.stack((red_image, green_image, blue_image), axis=-1)
    
    # # Save the resulting RGB image
    # io.imsave(os.path.join(analysis_folder,f"img_WBC_rgb_fov{fov}.tiff"), rgb_image)
    
        
    _, green_mask = cv2.threshold(green_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, red_mask = cv2.threshold(red_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine the binary masks to create a combined mask
    combined_mask = (green_mask | red_mask).astype(np.uint8)
    
    
    
    
    # Step 1: Load the images
    green_channel = io.imread(green_channel_path).astype(np.float32)
    red_channel = io.imread(red_channel_path).astype(np.float32)
    
    # Step 2: Normalize each channel to [0, 1]
    green_normalized = rescale_intensity(green_channel, in_range='image', out_range=(0, 1))
    red_normalized = rescale_intensity(red_channel, in_range='image', out_range=(0, 1))
    
    # Step 3: Subtract red from green
    difference_image = green_normalized - red_normalized
    
    # Step 4: Rescale the difference image for visualization
    difference_rescaled = rescale_intensity(difference_image, in_range='image', out_range=(0, 1))
    
    # Visualize the results
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5),sharex=True,sharey=True)
    # ax[0].imshow(green_normalized, cmap='gray')
    # ax[0].set_title("Green Channel (Normalized)")
    # ax[1].imshow(red_normalized, cmap='gray')
    # ax[1].set_title("Red Channel (Normalized)")
    # ax[2].imshow(difference_rescaled, cmap='gray')
    # ax[2].set_title("Difference Image (Rescaled)")
    # plt.tight_layout()
    # plt.show()
    
    # Optionally save the result
    # io.imsave("difference_image.tiff", (difference_rescaled * 255).astype(np.uint8))  # Save as 8-bit TIFF
    
    # plt.show()
    
    
    #%%
    
    
    
    
    
    #%%
    
    threshold_value = threshold_otsu(difference_rescaled)
    
    # Create a binary mask for dark spots
    mask = difference_rescaled <= threshold_value  # Dark spots are below the threshold
    
    # Replace dark spots with 0
    difference_with_zero = np.where(mask, 0, difference_rescaled)
    
    nucleus_image = difference_with_zero*green_mask
    
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5),sharex=True,sharey=True)
    # ax[0].imshow(difference_rescaled, cmap='gray')
    # ax[0].set_title("Original Difference Image")
    # ax[1].imshow(mask, cmap='gray')
    # ax[1].set_title("Dark Spot Mask")
    # ax[2].imshow(difference_with_zero*green_mask, cmap='gray')
    # ax[2].set_title("Difference Image with Dark Spots Set to 0")
    # plt.tight_layout()
    
    
    
    
    
    nonzero_mask = nucleus_image > 0
    
    nucleus_mask = np.where(nonzero_mask,1,nucleus_image).astype(np.uint8)
    
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5),sharex=True,sharey=True)
    # ax[0].imshow(green_channel, cmap='gray')
    # ax[0].set_title("Original Green Image")
    # ax[1].imshow(combined_mask, cmap='gray')
    # ax[1].set_title("Combined Mask")
    # ax[2].imshow(nucleus_mask, cmap='gray')
    # ax[2].set_title("Nucleus Mask")
    # plt.tight_layout()
    
    
    #%% Cytoplasm Mask
    
    nand_mask = 1 - (combined_mask & nucleus_mask).astype(np.uint8)
    
    
    cytoplasm_mask = nand_mask*red_mask
    
    
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5),sharex=True,sharey=True)
    # ax[0].imshow(red_channel, cmap='gray')
    # ax[0].set_title("Original red Image")
    # ax[1].imshow(cytoplasm_mask, cmap='gray')
    # ax[1].set_title("Cytoplasm Mask")
    # ax[2].imshow(nucleus_mask, cmap='gray')
    # ax[2].set_title("Nucleus Mask")
    # plt.tight_layout()
    
    
    #%%
    
    color_image = np.zeros((nucleus_mask.shape[0], nucleus_mask.shape[1], 3), dtype=np.float)
    color_image[..., 0] = cytoplasm_mask  # Red channel
    color_image[..., 1] = nucleus_mask    # Green channel
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5),sharex=True,sharey=True)
    ax[0].imshow(red_channel, cmap='gray')
    ax[0].set_title("Original red Image")
    ax[1].imshow(green_channel, cmap='gray')
    ax[1].set_title("Orignal green Image")
    ax[2].imshow(color_image, cmap='gray')
    ax[2].set_title("Color Mask Image")
    plt.tight_layout()
    
    
    io.imsave(os.path.join(analysis_folder,f"img_WBC_mask_rg_fov{fov}.png"), color_image)
    plt.close()
    
    #%%
    from skimage.measure import label, regionprops
    
    labeled_combined = label(combined_mask)
    regions = regionprops(labeled_combined)
    
    min_area_threshold = 50
    max_area_threshold = 5000
    
    cn_ratios = []
    
    cytoplasm_areas = []
    nucleus_areas = []
    
    red_intensities = []
    green_intensities = []
    
    # Step 3: Create a mask for large regions
    # filtered_mask = np.zeros_like(labeled_combined, dtype=bool)
    
    # Iterate over the regions and keep only the ones with area above the threshold
    cell_count = 0
    for region in regions:
        if region.area > min_area_threshold and region.area < max_area_threshold and region.label > 0:
            cell_count = cell_count + 1
            
            min_row, min_col, max_row, max_col = region.bbox
        
            # Extract the nucleus and combined regions for this cell
            # nucleus_region = green_mask[min_row:max_row, min_col:max_col]
            
            cytoplasm_region = cytoplasm_mask[min_row:max_row, min_col:max_col]
            
            nucleus_region = nucleus_mask[min_row:max_row, min_col:max_col]
            
            
            red_region = red_channel[min_row:max_row, min_col:max_col]
            green_region = green_channel[min_row:max_row, min_col:max_col]
            
            red_intensities.append(np.mean(red_region))
            green_intensities.append(np.mean(green_region))
        
            # Calculate the area for this nucleus
            # cell_nucleus_area = np.sum(nucleus_region)
            
            cell_cytoplasm_area = np.sum(cytoplasm_region)
            
            cell_nucleus_area = np.sum(nucleus_region)
            
            
            
            cytoplasm_areas.append(cell_cytoplasm_area)
            nucleus_areas.append(cell_nucleus_area)
            
            if cell_nucleus_area > 0:
            
                cn_ratio = cell_cytoplasm_area / cell_nucleus_area
                cn_ratios.append(cn_ratio)
    
    fov_array = np.array([fov]*cell_count)    
    array = np.array([green_intensities, red_intensities, nucleus_areas, cytoplasm_areas,fov_array])
    array_df = pd.DataFrame(array.T, columns=df.columns)
    df = pd.concat([df, array_df], ignore_index=True)
    

print(df)

df.to_csv(os.path.join(analysis_folder,'master_stats.csv'))


#%% Plot From Master Stats
    
    
    
plt.figure()
plt.plot(df['nucleus_areas'],df['cytoplasm_areas'],'.')
plt.xlabel('Nucleus Area')
plt.ylabel('Cytoplasm Area')

plt.xlim([-10,4000])
plt.ylim([-10,4000])

plt.title('Cytoplasm vs Nucleus Area')
plt.savefig(os.path.join(analysis_folder,'Cytoplasm vs Nucleus Area.png'))






fig,ax=plt.subplots(1,3, sharey = True,figsize=(19.2, 10.8))

ax[0].plot(df['green_intensities'],df['red_intensities'],'*')
ax[0].set_title('RFL vs GFL')
ax[0].set_xlabel('Green FL')
ax[0].set_ylabel('Red FL')

ax[1].plot(df['nucleus_areas'],df['red_intensities'],'*')
ax[1].set_title('RFL vs Nucleus Area')
ax[1].set_xlabel('Nucleus Area')
ax[1].set_ylabel('Red FL')

ax[2].plot(df['cytoplasm_areas'],df['red_intensities'],'*')
ax[2].set_title('RFL vs Cytoplasm Area')
ax[2].set_xlabel('Cytoplasm Area')
ax[2].set_ylabel('Red FL')

fig.suptitle(os.path.split(run_path)[-1])

fig = plt.gca()

plt.savefig(os.path.join(analysis_folder,'RFL vs Areas.png'))




# plt.figure()
# plt.plot(green_intensities,red_intensities,'+')
# plt.xlabel('GFL')
# plt.ylabel('RFL')


# #%%

# plt.figure()
# plt.plot(nucleus_areas,green_intensities,'+')
# plt.xlabel('area')
# plt.ylabel('FL')