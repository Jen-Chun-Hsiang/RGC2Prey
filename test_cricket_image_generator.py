import os
import random 
import numpy as np
import matplotlib.pyplot as plt
from datasets.sim_cricket import overlay_images_with_jitter_and_scaling, synthesize_image_with_params


def get_random_file_path(folder_path):
    """
    Returns the path of a randomly sampled file from the specified folder.

    Parameters:
    - folder_path: str, the path to the folder to sample from.

    Returns:
    - file_path: str, the full path of the randomly sampled file, or None if the folder is empty.
    """
    if not os.path.isdir(folder_path):
        raise ValueError("The provided path is not a valid directory.")

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        return None  # Return None if the folder is empty

    random_file = random.choice(files)
    return os.path.join(folder_path, random_file)


def plot_tensor_and_save(tensor, output_folder, file_name='tensor_plot.png'):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert the tensor to a NumPy array for plotting
    tensor_np = tensor.numpy()

    # Plot the tensor
    plt.figure(figsize=(8, 6))
    plt.imshow(tensor_np, cmap='gray')  # You can choose a different colormap if needed
    # plt.colorbar()  # Optional: adds a color bar to the plot
    plt.title('Synthesize Image Visualization')
    
    # Save the plot to the assigned folder
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid displaying it if running in an interactive environment

    print(f"Plot saved to {output_path}")

# Below unit in pixel
num_syn_img = 20
image_size = np.array([640, 480])
crop_size = np.array([320, 240])
rgc_canvas_size = np.array([240, 180])  
cricket_size_range = np.array([40, 100])  # visual angle (~ 20 cm for 1.5~2 cm cricket, 5.56 to 13.89 degree)
bottom_img_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/grass/'
top_img_folder    = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/cricket/'
temp_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/syn_img/'

pixel_in_um = 4.375  # [task] make sure all recordings have the same value; if not, normalization is required

top_img_pos = np.array([0, 0])
bottom_img_pos = np.array([0, 0])
bottom_img_jitter_range = np.array([0, 0])
top_img_jitter_range = np.array([0, 0])
top_img_scale_range = np.array([cricket_size_range[0]/cricket_size_range[1], 1])

# Generate and save n synthesized images
for i in range(num_syn_img):
    bottom_img_path = get_random_file_path(bottom_img_folder)
    top_img_path = get_random_file_path(top_img_folder)
    scale_factor = random.uniform(*top_img_scale_range)

    if bottom_img_path and top_img_path:
        # syn_image = overlay_images_with_jitter_and_scaling(
        #    bottom_img_path, top_img_path, top_img_pos, bottom_img_pos,
        #    bottom_img_jitter_range, top_img_jitter_range,
        #    top_img_scale_range, crop_size, alpha=1.0
        #)
        syn_image = synthesize_image_with_params(bottom_img_path, top_img_path, top_img_pos, bottom_img_pos,
                                 scale_factor, crop_size, alpha=1.0)
        Timg = syn_image[1, :, :]
        plot_tensor_and_save(Timg, temp_save_folder, f'synthesized_image_{i + 1}.png')
    else:
        print(f"Skipping iteration {i + 1} due to missing image file(s).")