# utils.py

import time
from contextlib import contextmanager
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
                      coord_y_history, scaling_history, y_min, y_max, path_predict=None):
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
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(4, 6, height_ratios=[1, 1, 1, 1])

        # Synthesized Movie Subplot
        ax1 = fig.add_subplot(gs[:2, :3])
        ax1.set_title("Synthesized Movie")
        ax1.imshow(syn_movie_frame, cmap='gray')

        # RGC Outputs Subplot
        ax2 = fig.add_subplot(gs[:2, 3:6])
        ax2.set_title("RGC Outputs")
        ax2.imshow(rgc_output, cmap='gray')

        # Path Subplot
        ax3 = fig.add_subplot(gs[2:4, :3])
        ax3.set_title("Path")
        ax3.plot(path_history[:, 0], path_history[:, 1], label='Ground Truth Path', color='blue')
        if path_predict is not None:
            ax3.plot(path_predict_history[:, 0], path_predict_history[:, 1], label='Predicted Path', color='orange')
        ax3.legend()
        ax3.set_ylim(y_min, y_max)
        ax3.set_xlim(y_min, y_max)

        # Coordinate X Subplot
        ax4 = fig.add_subplot(gs[2:3, 3:5])
        ax4.set_title("Coord X")
        ax4.plot(coord_x_history, label='Ground Truth', color='blue')
        ax4.plot(path_bg_history[:, 0], label='Background Path', color='green')
        if path_predict is not None:
            ax4.plot(path_predict_history[:, 0], label='Predicted Path', color='orange')
        # ax4.legend()
        ax4.set_ylim(y_min, y_max)

        # Coordinate Y Subplot
        ax5 = fig.add_subplot(gs[3:4, 3:5])
        ax5.set_title("Coord Y")
        ax5.plot(coord_y_history, label='Ground Truth', color='blue')
        ax5.plot(path_bg_history[:, 1], label='Background Path', color='green')
        if path_predict is not None:
            ax5.plot(path_predict_history[:, 1], label='Predicted Path', color='orange')
        # ax5.legend()
        ax5.set_ylim(y_min, y_max)

        # Scaling Factor Subplot
        ax6 = fig.add_subplot(gs[2:4, 5:6])
        ax6.set_title("Scaling")
        ax6.plot(scaling_history, color='purple')
        ax6.set_ylim(0, 2.1)

        plt.tight_layout()

        # Render the figure to an image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)  # Close the figure to free memory
        return img

    def generate_movie(self, image_sequence, syn_movie, path, path_bg, path_predict, scaling_factors, video_id):
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
        # Ensure all inputs have the same length based on the length of `inputs`
        num_steps = image_sequence.shape[0]
        syn_movie = syn_movie[-num_steps:]
        scaling_factors = scaling_factors[-num_steps:]

        print(f'image_sequence type: {type(image_sequence)}')
        print(f'syn_movie type: {type(syn_movie)}')
        print(f'path type: {type(path)}')
        print(f'path_bg type: {type(path_bg)}')
        print(f'scaling_factors type: {type(scaling_factors)}')

        print(f'image_sequence shape: {image_sequence.shape}')
        print(f'syn_movie shape: {syn_movie.shape}')
        print(f'path shape: {path.shape}')
        print(f'path_bg shape: {path_bg.shape}')
        print(f'scaling_factors shape: {scaling_factors.shape}')


        os.makedirs(self.video_save_folder, exist_ok=True)
        output_filename = os.path.join(
            self.video_save_folder,
            f'RGC_proj_map_{self.bls_tag}_{self.grid_generate_method}_{video_id}.mp4'
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_filename, fourcc, self.fps, (self.frame_width, self.frame_height))

        if path_predict is not None:
            all_y_values = np.concatenate((path, path_bg, path_predict), axis=0)
        else:
            all_y_values = np.concatenate((path, path_bg), axis=0)
        y_min, y_max = np.min(all_y_values), np.max(all_y_values)

        path_history = []
        path_predict_history = []
        path_bg_history = []
        coord_x_history = []
        coord_y_history = []
        scaling_history = []

        for i, (frame, syn_frame, coord, bg_coord, pred_coord, scaling) in enumerate(zip(image_sequence, syn_movie, path, path_bg, path_predict or [], scaling_factors)):
            print(f'i: {i}')
            path_history.append(coord)
            path_bg_history.append(bg_coord)

            if path_predict is not None:  #
                path_predict_history.append(pred_coord)

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
                path_predict = path_predict
            )

            # Resize the image to fit video dimensions
            img = cv2.resize(img, (self.frame_width, self.frame_height))
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        video_writer.release()
        print(f"Video saved at {output_filename}")

