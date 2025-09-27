# utils.py

import time
from contextlib import contextmanager
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch.nn.functional as F
import logging

@contextmanager
def timer(log_values, tau=0.99, n=100):
    """Context manager to time a code block and update log values with min, max, and moving average."""

    # Only update every nth iteration
    log_values['counter'] += 1
    if log_values['counter'] < n:
        yield  # No timing needed for this iteration
        return

    # Start timing
    start_time = time.perf_counter()
    yield  # Run the block of code inside the `with` statement
    end_time = time.perf_counter()

    duration = end_time - start_time

    # Handle the first-time setting of values
    if log_values['min'] is None:
        log_values['min'] = duration
        log_values['max'] = duration
        log_values['moving_avg'] = duration
    else:
        # Update minimum and maximum
        log_values['min'] = min(log_values['min'], duration)
        log_values['max'] = max(log_values['max'], duration)

        # Update moving average with exponential weighting
        log_values['moving_avg'] = tau * log_values['moving_avg'] + (1 - tau) * duration

    log_values['counter'] = 0  # Reset the counter after the update


class MovieGenerator:
    def __init__(self, frame_width, frame_height, fps, video_save_folder, bls_tag, grid_generate_method):
        """
        Initialize the MovieGenerator.

        Args:
            frame_width (int): Width of the video frames.
            frame_height (int): Height of the video frames.
            fps (int): Frames per second for the output video.
            video_save_folder (str): Folder to save the generated video.
            bls_tag (str): Identifier for the video.
            grid_generate_method (str): Method used for grid generation.
            video_id (str): Identifier for the video.
            grid_id (str): Identifier for the grid.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.video_save_folder = video_save_folder
        self.bls_tag = bls_tag
        self.grid_generate_method = grid_generate_method

    def create_figure(self, syn_movie_frame, rgc_output, path_history, path_predict_history, path_bg_history, coord_x_history, 
                      coord_y_history, scaling_history, y_min, y_max, centerRF_history, rgcout_min, rgcout_max, movie_min, movie_max, 
                      is_path_predict=True, is_centerRF=False,
                      ):
        """
        Create the figure layout with subplots for the movie frame.

        Args:
            syn_movie_frame (np.array): Frame for the synthesized movie.
            rgc_output (np.array): Data for the RGC output subplot.
            path_history (np.array): History of path coordinates.
            path_predict_history (np.array): History of predicted path coordinates.
            path_bg_history (np.array): History of background path coordinates.
            coord_x_history (list): History of X-coordinates for the path.
            coord_y_history (list): History of Y-coordinates for the path.
            scaling_history (list): History of scaling factor values.

        Returns:
            np.array: RGB image of the figure.
        """
        scalar_width = 120
        scalar_height = 90
        desired_width, desired_height = 160, 120
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(4, 6, height_ratios=[1, 1, 1, 1])

        # Synthesized Movie Subplot
        ax1 = fig.add_subplot(gs[:2, :3])
        ax1.set_title("Synthesized Movie")
        ax1.imshow(syn_movie_frame, cmap='gray', vmin=movie_min, vmax=movie_max)
        path_coord= path_history[-1] * np.array([scalar_width, -scalar_height]) 
        path_coord += np.array([desired_width, desired_height])
        ax1.scatter(path_coord[0], path_coord[1], color='blue', marker='x', s=50, label="target")
        if is_path_predict:
            path_pred_coord= path_predict_history[-1] * np.array([scalar_width, -scalar_height]) 
            path_pred_coord += np.array([desired_width, desired_height])
            ax1.scatter(path_pred_coord[0], path_pred_coord[1], color='orange', marker='x', s=50, label="Pred")

        if is_centerRF:
            centerRF_coord = centerRF_history[-1] * np.array([1, -1]) 
            centerRF_coord += np.array([desired_width, desired_height])
            ax1.scatter(centerRF_coord[0], centerRF_coord[1], color='red', marker='x', s=50, label="centerRF")
        ax1.legend()
        ax1.set_xticks([])
        ax1.set_yticks([])


        # RGC Outputs Subplot
        ax2 = fig.add_subplot(gs[:2, 3:6])
        ax2.set_title("RGC Outputs")

        if rgc_output.ndim == 3 and rgc_output.shape[0] > 1:
            rgc_output = rgc_output[0, :, :]  # Use only the first channel

        image_width, image_height = rgc_output.shape[1], rgc_output.shape[0]
        x_min = (desired_width - image_width) / 2
        x_max = x_min + image_width
        y_min = (desired_height - image_height) / 2
        y_max = y_min + image_height

        ax2.imshow(rgc_output, cmap='gray', extent=[x_min, x_max, y_min, y_max], vmin=rgcout_min, vmax=rgcout_max)
        path_coord= path_history[-1] * np.array([scalar_width / 2, scalar_height / 2])
        path_coord += np.array([desired_width / 2, desired_height /2 ])
        ax2.scatter(path_coord[0], path_coord[1], color='blue', marker='x', s=50, label="target")
        if is_path_predict:
            path_pred_coord= path_predict_history[-1] * np.array([scalar_width / 2, scalar_height / 2])
            path_pred_coord += np.array([desired_width / 2, desired_height / 2])
            ax2.scatter(path_pred_coord[0], path_pred_coord[1], color='orange', marker='x', s=50, label="Pred")
        if is_centerRF:
            centerRF_coord = centerRF_history[-1] * np.array([1 / 2, 1 / 2])
            centerRF_coord += np.array([desired_width / 2, desired_height / 2])
            ax2.scatter(centerRF_coord[0], centerRF_coord[1], color='red', marker='x', s=50, label="centerRF")
        ax2.set_xlim(0, desired_width)
        ax2.set_ylim(0, desired_height)
        ax2.legend()
        ax2.set_xticks([])
        ax2.set_yticks([])

        # ax2.imshow(rgc_output, cmap='gray')

        # Path Subplot
        ax3 = fig.add_subplot(gs[2:4, :3])
        ax3.set_title("Path")
        scaled_path_history = path_history * np.array([scalar_width, scalar_height])
        ax3.plot(scaled_path_history[:, 0], scaled_path_history[:, 1], label='Ground Truth Path', color='blue')
        if is_path_predict:
            scaled_path_predict_history = path_predict_history * np.array([scalar_width, scalar_height])
            ax3.plot(scaled_path_predict_history[:, 0], scaled_path_predict_history[:, 1], label='Predicted path', color='orange')
        else:
            scaled_path_bg_history = path_bg_history * np.array([scalar_width, scalar_height])
            ax3.plot(scaled_path_bg_history[:, 0], scaled_path_bg_history[:, 1], label='Background path', color='green')

        if is_centerRF:
            ax3.plot(centerRF_history[:, 0], centerRF_history[:, 1], label='Center RF path', color='red')

        ax3.legend()
        ax3.set_xlim(-desired_width,  desired_width)  # X-axis range
        ax3.set_ylim(-desired_height, desired_height)    # Y-axis range
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Coordinate X Subplot
        ax4 = fig.add_subplot(gs[2:3, 3:5])
        ax4.set_title("Coord X")
        ax4.plot(coord_x_history * np.array([scalar_width]), label='Ground Truth', color='blue')
        ax4.plot(path_bg_history[:, 0] * np.array([scalar_width]), label='Background Path', color='green')
        if is_path_predict:
            ax4.plot(path_predict_history[:, 0] * np.array([scalar_width]), label='Predicted Path', color='orange')
        if is_centerRF:
            ax4.plot(centerRF_history[:, 0], label='Center RF path', color='red')
        # ax4.legend()
        ax4.set_ylim(-scalar_width, scalar_width)
        ax4.set_ylabel("Location (pixels)")
        ax4.set_xlabel("Time (frames)")

        # Coordinate Y Subplot
        ax5 = fig.add_subplot(gs[3:4, 3:5])
        ax5.set_title("Coord Y")
        ax5.plot(coord_y_history * np.array([scalar_height]), label='Ground Truth', color='blue')
        ax5.plot(path_bg_history[:, 1] * np.array([scalar_height]), label='Background Path', color='green')
        if is_path_predict:
            ax5.plot(path_predict_history[:, 1] * np.array([scalar_height]), label='Predicted Path', color='orange')
        if is_centerRF:
            ax5.plot(centerRF_history[:, 1], label='Center RF path', color='red')
        # ax5.legend()
        ax5.set_ylim(-scalar_height, scalar_height)
        ax4.set_ylabel("Location (pixels)")
        ax4.set_xlabel("Time (frames)")

        # Scaling Factor Subplot
        ax6 = fig.add_subplot(gs[2:4, 5:6])
        ax6.set_title("Scaling")
        ax6.plot(scaling_history, color='purple')
        ax6.set_ylim(0, 2.1)

        plt.tight_layout()

        # Render the figure to an image
        canvas = FigureCanvas(fig)
        canvas.draw()
        if hasattr(canvas, "tostring_rgb"):
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        else:
            # For newer matplotlib versions
            img = np.asarray(canvas.buffer_rgba())
            img = img[..., :3]  # Drop alpha channel if present
        
        plt.close(fig)  # Close the figure to free memory
        return img

    def generate_movie(self, image_sequence, syn_movie, path, path_bg, path_predict, scaling_factors, video_id, weighted_coords):
        """
        Generate the video based on the provided input data.

        Args:
            image_sequence (list): List of image sequences.
            syn_movie (list): List of synthesized movie frames.
            path (list): List of (x, y) coordinates for the path.
            path_bg (list): Background path data.
            path_predict (list): Predicted path data.
            scaling_factors (list): List of scaling factors.
        """
        # flip y for visualization
        path[:, 1] *= -1
        path_bg[:, 1] *= -1

        # Ensure all inputs have the same length based on the length of `inputs`
        num_steps = image_sequence.shape[0]
        syn_movie = syn_movie[-num_steps:]
        scaling_factors = scaling_factors[-num_steps:]

        os.makedirs(self.video_save_folder, exist_ok=True)
        output_filename = os.path.join(
            self.video_save_folder,
            f'RGC_proj_map_{self.bls_tag}_{self.grid_generate_method}_{video_id}.mp4'
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.frame_width, self.frame_height))

        if path_predict is not None:
            is_path_predict = True
            path_predict[:, 1] *= -1
            all_y_values = np.concatenate((path, path_bg, path_predict), axis=0)
        else:
            is_path_predict = False
            path_predict = [None] * len(image_sequence)
            all_y_values = np.concatenate((path, path_bg), axis=0)

        if weighted_coords is not None:
            is_centerRF = True
            weighted_coords[:, 1] *= -1
        else:
            is_centerRF = False
            weighted_coords = [None] * len(image_sequence)
        y_min, y_max = np.min(all_y_values), np.max(all_y_values)

        rgcout_min = np.min(image_sequence)
        rgcout_max = np.max(image_sequence)
        movie_min = np.min(syn_movie)
        movie_max = np.max(syn_movie)

        path_history = []
        path_predict_history = []
        path_bg_history = []
        coord_x_history = []
        coord_y_history = []
        scaling_history = []
        centerRF_history = []

        if syn_movie.shape[1] == 2:
            syn_movie = syn_movie[:, 0, :, :]

        for i, (frame, syn_frame, coord, bg_coord, pred_coord, scaling, centerRF) in enumerate(zip(image_sequence, syn_movie, path, path_bg, 
                                                                                         path_predict, scaling_factors, weighted_coords)):
            path_history.append(coord)
            path_bg_history.append(bg_coord)

            if is_path_predict:  #
                path_predict_history.append(pred_coord)

            if is_centerRF:
                centerRF_history.append(centerRF)

            coord_x_history.append(coord[0])
            coord_y_history.append(coord[1])
            scaling_history.append(scaling)

            img = self.create_figure(
                syn_movie_frame=syn_frame,
                rgc_output=frame,
                path_history=np.array(path_history),
                path_predict_history=np.array(path_predict_history),
                path_bg_history=np.array(path_bg_history),
                coord_x_history=coord_x_history,
                coord_y_history=coord_y_history,
                scaling_history=scaling_history, 
                y_min = y_min,
                y_max = y_max,
                is_path_predict = is_path_predict,
                is_centerRF = is_centerRF,
                centerRF_history = np.array(centerRF_history), 
                rgcout_min = rgcout_min, 
                rgcout_max = rgcout_max,
                movie_min = movie_min,
                movie_max = movie_max
            )

            # Resize the image to fit video dimensions
            img = cv2.resize(img, (self.frame_width, self.frame_height))
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        video_writer.release()
        logging.info(f"Video saved at {output_filename}")


def save_distributions(train_loader, n, folder_name, file_name, logging=None):
    """
    Accumulates values from batches, plots 1D distributions, and saves the plots.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        n (int): Number of batches to process.
        folder_name (str): Directory where the plots will be saved.
        file_name (str): Base name for the saved plot file.
    """
    # Create folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Accumulators for each variable
    all_random_matrix = []
    all_output_value = []

    random_id = np.random.randint(0, 10001, size=1)
    file_name = f'{random_id[0]}_{file_name}'

    # Collect data over n batches
    with torch.no_grad():
        for batch_idx, (inputs, true_path, _) in enumerate(train_loader):
            if batch_idx >= n:
                break  # Stop after processing n batches

            # Flatten tensors and accumulate
            all_random_matrix.append(inputs.view(-1).cpu().numpy())
            all_output_value.append(true_path.view(-1).cpu().numpy())
            if logging is not None:
                logging.debug(f'batch_idx: {batch_idx} \n')

    # Concatenate accumulated values
    all_random_matrix = np.concatenate(all_random_matrix)
    all_output_value = np.concatenate(all_output_value)

    # Plot separate distributions in subplots
    plt.figure(figsize=(12, 6))

    # Subplot 1: Random Matrix
    plt.subplot(1, 2, 1)
    plt.hist(all_random_matrix, bins=50, alpha=0.7, color='blue')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Random Matrix")

    # Subplot 2: Output Value
    plt.subplot(1, 2, 2)
    plt.hist(all_output_value, bins=50, alpha=0.7, color='green')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Output Value")

    # Adjust layout and save the plot
    plt.tight_layout()
    save_path = os.path.join(folder_name, file_name)
    plt.savefig(save_path)
    plt.close()

    logging.info(f"Plot saved to {save_path}")


def generate_causal_gaussian_kernel(kernel_length, sampling_rate, gaussian_std):
    """
    Generate a 1D causal Gaussian kernel for convolution.
    Ensures no influence is added to future points (causal).
    The kernel will always have the specified size `kernel_length`.

    Args:
        kernel_length (int): The length of the kernel in time points.
        sampling_rate (float): The sampling rate in Hz (e.g., samples per second).
        gaussian_std (float): The standard deviation of the Gaussian in seconds.

    Returns:
        torch.Tensor: A 1D causal Gaussian kernel of shape (kernel_length,).
    """
    # Convert standard deviation from seconds to time points
    std_in_points = gaussian_std * sampling_rate
    
    # Generate a range of values: [0, 1, ..., kernel_length - 1]
    x = torch.arange(kernel_length)
    center = kernel_length - 1  # Causality: Center shifted to the last index
    
    # Compute the Gaussian function (unnormalized, causal)
    gaussian_kernel = torch.exp(-0.5 * ((x - center) / std_in_points) ** 2)
    
    # Normalize the kernel to ensure it sums to 1
    gaussian_kernel /= gaussian_kernel.sum()
    
    return x, gaussian_kernel


def gaussian_smooth_1d(input_tensor, gaussian_kernel=None, kernel_size=50, 
                       sampleing_rate=100, sigma=0.05):
    """
    Apply Gaussian smoothing along a specified dimension for a 2D tensor.
    
    Args:
        input_tensor (torch.Tensor): The 2D input tensor to smooth.
        dim (int): The dimension along which to apply smoothing (0 or 1).
        kernel_size (int): Size of the Gaussian kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian kernel.
        
    Returns:
        torch.Tensor: Smoothed tensor.
    """
    
    if gaussian_kernel is None:
        _, gaussian_kernel = generate_causal_gaussian_kernel(kernel_size, sampleing_rate, sigma)
    
    # Ensure kernel is a tensor
    gaussian_kernel = gaussian_kernel.unsqueeze(0)

    kernel = gaussian_kernel.view(1, 1, -1)  # Shape: (1, 1, kernel_size)
    input_tensor = input_tensor.unsqueeze(0)  # Add channel dimension for conv1d
    kernel = np.repeat(kernel, input_tensor.shape[1], axis=0)
    input_tensor = F.pad(input_tensor, (kernel_size-1, 0))
    smoothed = F.conv1d(input_tensor, kernel, padding=0, groups=input_tensor.shape[1]).squeeze()
    
    return smoothed

