import os
import numpy as np
from PIL import Image

def generate_gray_images(output_dir, width, height, num_images):
    """
    Generate a set of single gray contrast images with uniform pixel values.
    
    Parameters:
        output_dir (str): Directory to save the images.
        width (int): Width of each image.
        height (int): Height of each image.
        num_images (int): Number of images to generate.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        # Generate a random gray value (0 to 255)
        gray_value = np.random.randint(0, 256)
        
        # Create an image with all pixels set to the gray value
        image_array = np.full((height, width), gray_value, dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Save the image as PNG
        image_path = os.path.join(output_dir, f'gray_image_{i+1:03d}.png')
        image.save(image_path)
        print(f"Saved image: {image_path}")

# Parameters
output_directory =  '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/single-contrast/'
image_width = 640
image_height = 480
number_of_images = 32

# Generate images
generate_gray_images(output_directory, image_width, image_height, number_of_images)
