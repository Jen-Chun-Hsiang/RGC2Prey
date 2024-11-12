import os
import random 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datasets.sim_cricket import overlay_images_with_jitter_and_scaling, synthesize_image_with_params, random_movement
from utils.utils import get_random_file_path, plot_tensor_and_save, plot_movement_and_velocity, create_video_from_specific_files


if __name__ == "__main__":
    run_task_id = 3
    mov_id = 111201
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
    syn_data_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/SynData/'
    # video params
    boundary_size = np.array([220, 140]) 
    center_ratio = np.array([0.2, 0.2])
    scale_factor = 1  #random.uniform(*top_img_scale_range)
    max_steps = 200
    fps = 20  # Frames per second
    prob_stay = 0.95
    prob_mov = 0.975
    initial_velocity = 6
    min_video_value, max_video_value = None, None  # Value range for the color map
    #
    pixel_in_um = 4.375  # [task] make sure all recordings have the same value; if not, normalization is required
    top_img_pos = np.array([0, 0])
    bottom_img_pos = np.array([0, 0])
    bottom_img_jitter_range = np.array([0, 0])
    top_img_jitter_range = np.array([0, 0])
    top_img_scale_range = np.array([cricket_size_range[0]/cricket_size_range[1], 1])

    if run_task_id == 1:   # randomly synthesize images given number of images
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

    elif run_task_id == 2:   # synthesize series of image based on the position of object and background
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
        num_syn_img = len(path)
        syn_movie = np.zeros((crop_size[1], crop_size[0], num_syn_img)) 
        for i in range(num_syn_img):
            top_img_pos = path[i,:].round().astype(int)
            bottom_img_pos = path_bg[i,:].round().astype(int)
            syn_image = synthesize_image_with_params(bottom_img_path, top_img_path, top_img_pos, bottom_img_pos,
                                        scale_factor, crop_size, alpha=1.0)
            Timg = syn_image[1, :, :]
            syn_movie[:, :, i] = Timg
            plot_tensor_and_save(Timg, syn_save_folder, f'synthesized_movement_{mov_id}_{i + 1}.png')

        syn_file = os.path.join(syn_data_save_folder, 'syn_movie.npz')
        np.savez_compressed(syn_file, syn_movie=syn_movie)
        print('npz data is saved!')

    elif run_task_id == 3:
        frame_width, frame_height = crop_size  # Set frame dimensions based on crop_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Ensure output folder exists
        os.makedirs(video_save_folder, exist_ok=True)
        output_filename = os.path.join(video_save_folder, f'synthesized_movement_{mov_id}.mp4')
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

        path, velocity = random_movement(boundary_size, center_ratio, max_steps, prob_stay, prob_mov, 
                                        initial_velocity=initial_velocity, momentum_decay=0.95, velocity_randomness=0.02, angle_range=0.5)
        path_bg, velocity_bg = random_movement(boundary_size, center_ratio, max_steps, prob_stay=0.98, prob_mov=0.98, 
                                            initial_velocity=initial_velocity, momentum_decay=0.9, velocity_randomness=0.01, angle_range=0.25)

        # Determine the minimum length for consistency
        min_length = min(len(path), len(path_bg))
        path = path[:min_length]
        velocity = velocity[:min_length]
        path_bg = path_bg[:min_length]
        velocity_bg = velocity_bg[:min_length]

        bottom_img_path = get_random_file_path(bottom_img_folder)
        top_img_path = get_random_file_path(top_img_folder)
        num_syn_img = len(path)
        syn_movie = np.zeros((crop_size[1], crop_size[0], num_syn_img)) 
        # Loop over each synthesized image to generate frames for the video
        for i in range(num_syn_img):
            top_img_pos = path[i, :].round().astype(int)
            bottom_img_pos = path_bg[i, :].round().astype(int)
            syn_image = synthesize_image_with_params(bottom_img_path, top_img_path, top_img_pos, bottom_img_pos,
                                                    scale_factor, crop_size, alpha=1.0)
            Timg = syn_image[1, :, :]  # Extract relevant image layer for video frame
            syn_movie[:, :, i] = Timg
            # Generate a plot with Matplotlib for better visual results
            fig, ax = plt.subplots(figsize=(8, 8))
            canvas = FigureCanvas(fig)
            
            # Plot data using imshow and add colorbar
            if min_video_value is not None:
                cax = ax.imshow(np.rot90(Timg, k=1), cmap='viridis', vmin=min_video_value, vmax=max_video_value)
            else:
                cax = ax.imshow(np.rot90(Timg, k=1), cmap='viridis')
            fig.colorbar(cax, ax=ax, label="Value")
            ax.set_title(f"Frame {i}")
            
            # Render the plot to an image array
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(canvas.get_width_height()[::-1] + (3,))

            # Resize the image to fit video dimensions
            img = cv2.resize(img, (frame_width, frame_height))

            # Write the frame to the video
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Close the figure to free memory
            plt.close(fig)

        # Release the video writer
        video_writer.release()
        print("Video generation complete!")

        syn_file = os.path.join(syn_data_save_folder, 'syn_movie_{mov_id}.npz')
        np.savez_compressed(syn_file, syn_movie=syn_movie)
            
    elif run_task_id == 4:   # create the movie based on the task 2
        video_file_name = "synthesized_movement_111002.mp4"
        create_video_from_specific_files(syn_save_folder, video_save_folder, video_file_name, 
                                         filename_template="synthesized_movement_110602_{}.png", fps=20)



