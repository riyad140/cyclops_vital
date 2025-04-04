import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def calculate_cn_ratios_and_areas(green_channel_path, red_channel_path):
    # Load the green and red channel images
    green_image = cv2.imread(green_channel_path, cv2.IMREAD_UNCHANGED)
    red_image = cv2.imread(red_channel_path, cv2.IMREAD_UNCHANGED)

    if green_image is None or red_image is None:
        raise ValueError("Error loading images. Check file paths.")

    # Apply thresholding to create binary masks for green and red channels
    _, green_mask = cv2.threshold(green_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, red_mask = cv2.threshold(red_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine the binary masks to create a combined mask
    combined_mask = (green_mask | red_mask).astype(np.uint8)
    
    fig,ax = plt.subplots(3,1,sharex=True,sharey=True)
    ax[0].imshow(combined_mask)
    ax[1].imshow(green_mask)
    ax[2].imshow(red_mask)

    # Label connected components in the combined mask
    labeled_combined = label(combined_mask)
    regions = regionprops(labeled_combined)

    cn_ratios = []

    for region in regions:
        # Extract the bounding box for each region
        min_row, min_col, max_row, max_col = region.bbox

        # Extract the nucleus and combined regions for this cell
        nucleus_region = green_mask[min_row:max_row, min_col:max_col]
        combined_region = combined_mask[min_row:max_row, min_col:max_col]

        # Calculate the area for this nucleus
        cell_nucleus_area = np.sum(nucleus_region)

        # Calculate the combined area for this cell
        cell_combined_area = np.sum(combined_region)

        # Calculate the cytoplasm area for this cell by subtracting nucleus area from combined area
        cell_cytoplasm_area = cell_combined_area - cell_nucleus_area

        # Skip if the cytoplasm area is zero or negative
        if cell_nucleus_area == 0:
            continue

        # Calculate the Cytoplasm-to-Nucleus ratio for this cell
        cn_ratio = cell_cytoplasm_area / cell_nucleus_area
        cn_ratios.append(cn_ratio)

    return cn_ratios

# Example usage
green_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R\img_WBC_green_FLG_fov1.tiff"  # Path to green channel image
red_channel_path = r"W:\raspberrypi\photos\Juravinski\2025-01-16\01-16-S13-2_S16R\img_WBC_red_FLR_fov1.tiff"     # Path to red channel image

try:
    cn_ratios = calculate_cn_ratios_and_areas(green_channel_path, red_channel_path)
    
    plt.figure()
    plt.plot(cn_ratios,'o')

    # Specify bin edges explicitly
    
    plt.figure()
    bin_edges = np.arange(0, max(cn_ratios) + 0.1, 0.1)  # Bins of width 0.1

    # Plot histogram of C:N ratios
    plt.hist(cn_ratios, bins=bin_edges, color='green', edgecolor='black')
    plt.title("Cytoplasm-to-Nucleus Ratio Histogram")
    plt.xlabel("C:N Ratio")
    plt.ylabel("Frequency")
    plt.show()

except ValueError as e:
    print(e)
