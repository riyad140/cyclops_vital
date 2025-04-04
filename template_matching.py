import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_image(tiffPath,keyTiff='tiff',extension = 'tiff'): # to read an image and return a numpy array

    
    ims=[]
    # files=[]
    # for file in os.listdir(binPath):
    if tiffPath.find(keyTiff)>-1 and tiffPath.endswith(extension):
        print(tiffPath)
        im=plt.imread(tiffPath)
        image_8bit = (im / 1023.0 * 255).astype(np.uint8)
        ims.append(image_8bit) 
            # files.append(file)
    
    return ims[0]

def template_matching(image_array, template_array):
    """
    Perform template matching using OpenCV's matchTemplate function with 2D arrays.
    Returns the matching result and the location of the best match.
    """
    # Perform template matching
    result = cv2.matchTemplate(image_array, template_array, cv2.TM_CCOEFF_NORMED)

    # Get the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Draw a rectangle around the best match
    top_left = max_loc
    h, w = template_array.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Draw rectangle on image array (you can use any visualization of your choice)
    image_with_match = image_array.copy()
    image_with_match[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255  # Highlight match

    # Return the result and the image with the match marked
    return result, image_with_match, max_loc, max_val

# Example usage with 2D arrays (e.g., numpy arrays)
# Define two 2D arrays (example)
# image_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Example 100x100 array
# template_array = np.random.randint(0, 256, (20, 20), dtype=np.uint8)  # Example 20x20 template


tiffPaths = [r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-12-02\S009-16min-A\img_PLT_blank_BF_fov11_offset_0.tiff",
             r"W:\raspberrypi\photos\Alpha_sharp\CYC7_AS1\2024-12-02\S009-16min-A\img_PLT_blank_BF_fov10_offset_0.tiff"]


# image1 = np.random.random((200, 300))  # Placeholder: Replace with imread('image1_path')
# image2 = np.random.random((200, 300))  # Placeholder: Replace with imread('image2_path')

image1 = read_image(tiffPaths[0])
image2 = read_image(tiffPaths[1])

image_array = image1
template_array = image2[-200:,500:1000]

# Perform template matching
result, image_with_match, best_match_loc, match_value = template_matching(image_array, template_array)

# Display results
print(f"Best match location: {best_match_loc}")
print(f"Matching score: {match_value:.2f}")

# Plot the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(result, cmap='hot')
plt.title("Template Matching Result")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(image_with_match, cmap='gray')
plt.title("Matched Image with Rectangle")
plt.axis('off')

plt.tight_layout()
plt.show()
