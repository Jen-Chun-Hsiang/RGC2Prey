import os
from PIL import Image

def convert_images_to_png(input_folder, output_folder):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Supported input image extensions
    valid_extensions = ('.jpeg', '.jpg', '.webp', '.bmp', '.jfif', '.avif')
    
    # Check the highest existing number in the output folder to continue naming from there
    existing_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    existing_numbers = [int(os.path.splitext(f)[0]) for f in existing_files if f.split('.')[0].isdigit()]
    start_index = max(existing_numbers) + 1 if existing_numbers else 0
    
    # Iterate through the files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(valid_extensions):
            # Open the image file
            input_path = os.path.join(input_folder, file_name)
            try:
                with Image.open(input_path) as img:
                    # Create the output file path with iteration order
                    output_file_name = f"{start_index}.png"
                    output_path = os.path.join(output_folder, output_file_name)
                    
                    # Convert and save the image as PNG
                    img.save(output_path, 'PNG')
                    print(f"Converted: {file_name} -> {output_file_name}")
                    
                    start_index += 1
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")


def process_and_crop_images(input_folder, output_folder, crop_size=(480, 640), scale_factor=0.5):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Supported input image extensions
    valid_extensions = ('.jpeg', '.jpg', '.webp', '.bmp', '.jfif', '.avif', '.png')
    
    # Check the highest existing number in the output folder to continue naming from there
    existing_files = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    existing_numbers = [int(f.split('_')[0]) for f in existing_files if f.split('_')[0].isdigit()]
    start_index = max(existing_numbers) + 1 if existing_numbers else 0
    
    # Iterate through the files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(valid_extensions):
            input_path = os.path.join(input_folder, file_name)
            try:
                with Image.open(input_path) as img:
                    # Step 1: Rotate the image to ensure the shortest side is the height
                    if img.width < img.height:
                        img = img.rotate(90, expand=True)
                    
                    # Step 2: Calculate scaling factors
                    scale_factor_height = crop_size[0] / img.height
                    scale_factor_width = crop_size[1] / img.width
                    overall_scale_factor = max(scale_factor_height, scale_factor_width)
                    
                    # Step 3: Scale the image using the larger scaling factor
                    new_size = (int(img.width * overall_scale_factor), int(img.height * overall_scale_factor))
                    img = img.resize(new_size, Image.ANTIALIAS)
                    
                    # Step 4: Crop the image if it is larger than the fixed crop size
                    num_crops_x = max(1, int((img.width-crop_size[1]) / (crop_size[1] * scale_factor)))
                    num_crops_y = max(1, int((img.height-crop_size[0]) / (crop_size[0] * scale_factor)))
                    
                    crop_index = 0
                    for i in range(num_crops_y):
                        for j in range(num_crops_x):
                            left = int(j * crop_size[1] * scale_factor)
                            upper = int(i * crop_size[0] * scale_factor)
                            right = left + crop_size[1]
                            lower = upper + crop_size[0]

                            # Ensure we don't go out of bounds
                            if right > img.width:
                                right = img.width
                                left = right - crop_size[1]
                            if lower > img.height:
                                lower = img.height
                                upper = lower - crop_size[0]
                            
                            cropped_img = img.crop((left, upper, right, lower))
                            
                            # Save the cropped image with the new naming convention
                            output_file_name = f"{start_index}_{crop_index}.png"
                            output_path = os.path.join(output_folder, output_file_name)
                            cropped_img.save(output_path, 'PNG')
                            print(f"Saved cropped image: {output_file_name}")
                            
                            crop_index += 1

                    start_index += 1
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")



