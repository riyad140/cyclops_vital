# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:59:16 2024

@author: imrul
"""

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

def combine_tiff_to_color(red_channel_path, green_channel_path, blue_channel_path, output_path):
    # Open the red, green, and blue channel TIFF images
    red_channel = Image.open(red_channel_path).convert('L')  # Convert to grayscale
    green_channel = Image.open(green_channel_path).convert('L')  # Convert to grayscale
    blue_channel = Image.open(blue_channel_path).convert('L')  # Convert to grayscale

    # Ensure that all images have the same size
    if red_channel.size != green_channel.size or red_channel.size != blue_channel.size:
        raise ValueError("Input images must have the same dimensions")

    # Merge the three channels into an RGB image
    rgb_image = Image.merge("RGB", (red_channel, green_channel, blue_channel))

    # Save the combined RGB image
    rgb_image.save(output_path)
    print(f"Color image saved at {output_path}")
    
    
def combine_tiff_to_color_with_blue_adjustment(red_channel_path, green_channel_path, blue_channel_path, output_path, blue_factor=0.5):
    # Open the red, green, and blue channel TIFF images
    red_channel = Image.open(red_channel_path).convert('L')  # Convert to grayscale
    green_channel = Image.open(green_channel_path).convert('L')  # Convert to grayscale
    blue_channel = Image.open(blue_channel_path).convert('L')  # Convert to grayscale

    # Ensure that all images have the same size
    if red_channel.size != green_channel.size or red_channel.size != blue_channel.size:
        raise ValueError("Input images must have the same dimensions")

    # Convert the images to NumPy arrays to manipulate pixel values
    red_array = np.array(red_channel)
    green_array = np.array(green_channel)
    blue_array = np.array(blue_channel)

    # Apply the blue factor to reduce the intensity of the blue channel
    blue_array = (blue_array * blue_factor).astype(np.uint8)  # Ensure values stay in the 0-255 range

    # Convert arrays back to PIL images
    red_channel = Image.fromarray(red_array)
    green_channel = Image.fromarray(green_array)
    blue_channel = Image.fromarray(blue_array)

    # Merge the three channels into an RGB image
    rgb_image = Image.merge("RGB", (red_channel, green_channel, blue_channel))

    # Save the combined RGB image
    rgb_image.save(output_path)
    print(f"Color image saved at {output_path}")
    
    

from PIL import Image
import numpy as np

def combine_tiff_to_color_with_blue_adjustment_16bit(red_channel_path, green_channel_path, blue_channel_path, output_path, blue_factor=0):
    # Open the red, green, and blue channel TIFF images as 16-bit (mode 'I;16')
    red_channel = Image.open(red_channel_path).convert('I;16')  # Convert to 16-bit grayscale
    green_channel = Image.open(green_channel_path).convert('I;16')  # Convert to 16-bit grayscale
    blue_channel = Image.open(blue_channel_path).convert('I;16')  # Convert to 16-bit grayscale

    # Ensure that all images have the same size
    if red_channel.size != green_channel.size or red_channel.size != blue_channel.size:
        raise ValueError("Input images must have the same dimensions")

    # Convert the images to NumPy arrays to manipulate 16-bit pixel values
    red_array = np.array(red_channel, dtype=np.uint16)
    green_array = np.array(green_channel, dtype=np.uint16)
    blue_array = np.array(blue_channel, dtype=np.uint16)

    # Apply the blue factor to reduce the intensity of the blue channel
    blue_array = (blue_array * blue_factor).astype(np.uint16)

    # Convert the 16-bit arrays to 8-bit by downscaling (divide by 256)
    red_array_8bit = (red_array // 4).astype(np.uint8)
    green_array_8bit = (green_array // 4).astype(np.uint8)
    blue_array_8bit = (blue_array // 4).astype(np.uint8)

    # Convert arrays back to PIL images in 8-bit mode ('L')
    red_channel_8bit = Image.fromarray(red_array_8bit, mode='L')
    green_channel_8bit = Image.fromarray(green_array_8bit, mode='L')
    blue_channel_8bit = Image.fromarray(blue_array_8bit, mode='L')

    # Merge the three channels into an 8-bit RGB image
    rgb_image = Image.merge("RGB", (red_channel_8bit, green_channel_8bit, blue_channel_8bit))

    # Save the combined 8-bit RGB image
    rgb_image.save(output_path)
    print(f"8-bit color image saved at {output_path}")

    return rgb_image

# Example usage
# combine_tiff_to_color_with_blue_adjustment("red_channel.tiff", "green_channel.tiff", "blue_channel.tiff", "output_color_image.tiff", blue_factor=0.5)


# Example usage
# Example usage


binPath = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R"

# fovCount = 2

for fovCount in range(1,31):
    print(fovCount)

    tiffPaths = [f"img_WBC_red_FLR_fov{fovCount}.tiff",f"img_WBC_green_FLG_fov{fovCount}.tiff",f"img_WBC_blue_DF_fov{fovCount}.tiff"]
    
    outputPath = os.path.join(binPath,'compositeColorImages')
    try:
        os.mkdir(outputPath)
    except:
        print('Output Folder Exists')
            
    
    outputFileName = 'color_rg'+ tiffPaths[0][-10:-4] + '.tiff'
    
    rgb_image= combine_tiff_to_color_with_blue_adjustment_16bit(os.path.join(binPath,tiffPaths[0]), os.path.join(binPath,tiffPaths[1]), os.path.join(binPath,tiffPaths[2]),os.path.join(outputPath,outputFileName))
    
    plt.figure()
    plt.imshow(rgb_image)
    plt.title(tiffPaths[0])