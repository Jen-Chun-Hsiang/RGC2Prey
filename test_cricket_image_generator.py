import os
import random 
import numpy as np
import matplotlib.pyplot as plt
from datasets.sim_cricket import overlay_images_with_jitter_and_scaling, synthesize_image_with_params, random_movement
from utils.utils import get_random_file_path, plot_tensor_and_save, plot_movement_and_velocity, create_video_from_specific_files


if __name__ == "__main__":
    run_task_id = 3

    # Below unit in pixel
    num_syn_img = 20
    image_size = np.array([640, 480])
    crop_size = np.array([320, 240])
    rgc_canvas_size = np.array([240, 180])  
    cricket_size_range = np.array([40, 100])  # visual angle (~ 20 cm for 1.5~2 cm cricket, 5.56 to 13.89 degree)
    bottom_img_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/grass/'
    top_img_folder    = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/cricket/'
    syn_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/syn_img/'
    plot_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Figures/'
    video_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Videos/'

    pixel_in_um = 4.375  # [task] make sure all recordings have the same value; if not, normalization is required

    top_img_pos = np.array([0, 0])
    bottom_img_pos = np.array([0, 0])
    bottom_img_jitter_range = np.array([0, 0])
    top_img_jitter_range = np.array([0, 0])
    top_img_scale_range = np.array([cricket_size_range[0]/cricket_size_range[1], 1])

    if run_task_id == 1:
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
                plot_tensor_and_save(Timg, syn_save_folder, f'synthesized_image_{i + 1}.png')
            else:
                print(f"Skipping iteration {i + 1} due to missing image file(s).")

    elif run_task_id == 2:
        boundary_size = np.array([220, 140]) 
        center_ratio = np.array([0.2, 0.2])
        max_steps = 200
        prob_stay = 0.95
        prob_mov = 0.975
        initial_velocity = 6
        mov_id = 110602
        path, velocity = random_movement(boundary_size, center_ratio, max_steps, prob_stay, prob_mov, initial_velocity=initial_velocity,
                                          momentum_decay=0.95, velocity_randomness=0.02, angle_range=0.5)
        path_bg, velocity_bg = random_movement(boundary_size, center_ratio, max_steps, prob_stay=0.98, prob_mov=0.98,
                                                initial_velocity=initial_velocity, momentum_decay=0.9, velocity_randomness=0.01,
                                                  angle_range=0.25)
        
        # Determine the minimum length
        min_length = min(len(path), len(path_bg))

        # Trim both paths and velocities to the minimum length
        path = path[:min_length]
        velocity = velocity[:min_length]
        path_bg = path_bg[:min_length]
        velocity_bg = velocity_bg[:min_length]
        
        plot_movement_and_velocity(path, velocity, boundary_size, f'{mov_id}_obj', output_folder=plot_save_folder)
        plot_movement_and_velocity(path_bg, velocity_bg, boundary_size=np.array([320, 240]), name=f'{mov_id}_bg',
                                    output_folder=plot_save_folder)

        bottom_img_path = get_random_file_path(bottom_img_folder)
        top_img_path = get_random_file_path(top_img_folder)
        scale_factor = 1  #random.uniform(*top_img_scale_range)
        num_syn_img = len(path)
        for i in range(num_syn_img):
            top_img_pos = path[i,:].round().astype(int)
            bottom_img_pos = path_bg[i,:].round().astype(int)
            syn_image = synthesize_image_with_params(bottom_img_path, top_img_path, top_img_pos, bottom_img_pos,
                                        scale_factor, crop_size, alpha=1.0)
            Timg = syn_image[1, :, :]
            plot_tensor_and_save(Timg, syn_save_folder, f'synthesized_movement_{mov_id}_{i + 1}.png')
            
    elif run_task_id == 3:
        video_file_name = "synthesized_movement_110602.mp4"
        create_video_from_specific_files(syn_save_folder, video_save_folder, video_file_name, 
                                         filename_template="synthesized_movement_110602_{}.png", fps=20)



