
import torch
import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from datasets.rgc_rf import create_hexagonal_centers, precompute_grid_centers, get_closest_indices, map_to_fixed_grid_closest
from datasets.rgc_rf import compute_distance_decay_matrix, map_to_fixed_grid_decay, gaussian_multi
from utils.utils import plot_position_and_save, plot_map_and_save, plot_gaussian_model



if __name__ == "__main__":
    task_id = 1
    plot_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Figures/'
    video_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Videos/'
    rf_params_file = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/SimulationParams.xlsx'
    file_name = 'rgc_rf_position_plot.png'
    xlim = (-120, 120)
    ylim = (-90, 90)
    rgc_array_rf_size = (320, 240)
    target_num_centers = 140
    num_step = 20
    num_gauss_example = 1
    tau = 30
    grid_generate_method = 'decay'  #'closest', 'decay'
    points = create_hexagonal_centers(xlim, ylim, target_num_centers=50, rand_seed=42)
    
    number_samples = len(points)
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]
    grid_centers = precompute_grid_centers(target_height, target_width, x_min=xlim[0], x_max=xlim[1],
                                            y_min=ylim[0], y_max=ylim[1])

    if grid_generate_method == 'closest':
        grid2value_mapping = get_closest_indices(grid_centers, points)
        map_func = map_to_fixed_grid_closest
    elif grid_generate_method == 'decay':
        grid2value_mapping = compute_distance_decay_matrix(grid_centers, points, tau)
        map_func = map_to_fixed_grid_decay
    else:
        raise ValueError("Invalid grid_generate_method. Use 'closest' or 'decay'.")


    if task_id == 0:
        sf_param_table = pd.read_excel(rf_params_file, sheet_name='SF_params', usecols='A:L')
        num_sim_data = len(sf_param_table)
        pid = random.randint(0, num_sim_data - 1)
        row = sf_param_table.iloc[pid]
        sf_params = np.array([0, 0, row['sigma_x'], row['sigma_y'],
                        row['theta'], row['bias'], row['c_scale'], row['s_sigma_x'], row['s_sigma_y'], row['s_scale']])
        opt_sf = gaussian_multi(sf_params, rgc_array_rf_size, num_gauss_example)
        opt_sf -= np.median(opt_sf)
        plot_gaussian_model(opt_sf, rgc_array_rf_size, plot_save_folder, file_name='gaussian_model_plot.png')


    elif task_id == 1:
        sf_param_table = pd.read_excel(rf_params_file, sheet_name='SF_params', usecols='A:L')
        num_sim_data = len(sf_param_table)
        pid = random.randint(0, num_sim_data - 1)
        row = sf_param_table.iloc[pid]

        # Initialize a list to store each opt_sf for concatenation
        multi_opt_sf = []

        # Loop over each row in grid_centers to generate multiple opt_sf
        for i in range(grid_centers.shape[0]):
            # Set up sf_params, using the current grid center for the first two entries
            sf_params = np.array([grid_centers[i, 0], grid_centers[i, 1], row['sigma_x'], row['sigma_y'],
                                row['theta'], row['bias'], row['c_scale'], row['s_sigma_x'], row['s_sigma_y'], row['s_scale']])
            
            # Generate opt_sf using gaussian_multi function
            opt_sf = gaussian_multi(sf_params, rgc_array_rf_size, num_gauss_example)
            opt_sf -= np.median(opt_sf)  # Center opt_sf around zero
            
            # Append to multi_opt_sf list
            multi_opt_sf.append(opt_sf)

        # Stack opt_sf along the third dimension to form multi_opt_sf
        multi_opt_sf = np.stack(multi_opt_sf, axis=-1)

        # Sum along the last dimension to create assemble_opt_sf
        assemble_opt_sf = np.sum(multi_opt_sf, axis=-1)

        # Plot the assembled result
        plot_gaussian_model(assemble_opt_sf, rgc_array_rf_size, plot_save_folder, file_name='gaussian_model_assemble_plot.png')


    elif task_id == 2:
        values = np.random.uniform(0, 1, size=(number_samples, 1))
        plot_position_and_save(points, values=values, output_folder=plot_save_folder, file_name=file_name)
        # Call the selected mapping function
        grid_values = map_func(values, grid2value_mapping, target_width, target_height)

        file_name = f'rgc_rf_gridmap_{grid_generate_method}_plot.png'
        plot_map_and_save(grid_values, plot_save_folder, file_name=file_name)

    elif task_id == 3:
        # Parameters for video
        video_id = 110701
        frame_width, frame_height = 640, 480  # Example resolution
        fps = 5  # Frames per second
        min_video_value, max_video_value = 0, 4  # Value range for the color map
        os.makedirs(video_save_folder, exist_ok=True)
        output_filename = os.path.join(video_save_folder, f'RGC_proj_map_{grid_generate_method}_{video_id}.mp4')
        # Initialize OpenCV video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

        for i in range(num_step):
            values = np.random.uniform(0, 1, size=(number_samples, 1))
            grid_values = map_func(values, grid2value_mapping, target_width, target_height)

            # Create the figure and render the plot in memory
            fig, ax = plt.subplots(figsize=(8, 8))
            canvas = FigureCanvas(fig)  # Use canvas to render the plot to an image

            # Plot the data
            cax = ax.imshow(np.rot90(grid_values, k=1), cmap='viridis', vmin=min_video_value, vmax=max_video_value)
            fig.colorbar(cax, ax=ax, label="Value")
            ax.set_title(f"Frame {i}")

            # Draw the canvas and convert to an image
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(canvas.get_width_height()[::-1] + (3,))

            # Resize the image to fit video dimensions
            img = cv2.resize(img, (frame_width, frame_height))

            # Write the frame to the video
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Close the figure to free memory
            plt.close(fig)

        # Release video writer
        video_writer.release()
            