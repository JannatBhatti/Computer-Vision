import cv2
from PIL import Image, ImageFilter
import numpy as np
import os

# Ensure output directory exists
if not os.path.exists("output"):
    os.makedirs("output")

# Read the original image
image_path = "images/tiger.jpg"  # Change this if using another image
img = cv2.imread(image_path, flags=0)  # Load image in grayscale

if img is None:
    print("Error: Image not found. Make sure the path is correct!")
    exit()

# Convert image to uint8 type
img_u8 = img.astype(np.uint8)

# Apply Gaussian Blur to reduce noise
img_blur = cv2.GaussianBlur(img_u8, (3, 3), 0)

# Apply Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

# Convert OpenCV image format to PIL format
color_converted = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
pil_image = Image.fromarray(color_converted)

# Save the processed image
output_path = "output/Edge_Sample_Canny.png"
pil_image.save(output_path)

print(f"Edge-detected image saved at: {output_path}")
