import os
import re
import cv2
import torch
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt


def get_filename_without_extension(image_path: str) -> str:
    """
    Extracts the file name without the extension from an image path.

    Args:
        image_path (str): The full path of the image.

    Returns:
        str: The file name without extension.
    """
    return os.path.splitext(os.path.basename(image_path))[0]


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

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith('.png')]
    
    if not files:
        return None  # Return None if the folder is empty

    random_file = random.choice(files)
    return os.path.join(folder_path, random_file)


def get_image_number(image_path: str) -> int:
    """
    Extracts and returns the number from the image filename.

    Args:
        image_path (str): The full path to the image file.

    Returns:
        int: The number extracted from the filename.
    """
    # Get the filename from the path
    filename = os.path.basename(image_path)
    # Use regex to find the first sequence of digits in the filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("No number found in the filename")


def load_mat_to_dataframe(mat_file_path):
    # Load the .mat file
    mat_data = loadmat(mat_file_path)

    # Extract the 2D array and column names
    summary_array = mat_data.get('summary_array')
    summary_array_name = mat_data.get('summary_array_name')

    # Ensure both variables are present
    if summary_array is None or summary_array_name is None:
        raise ValueError("The .mat file does not contain the required variables 'summary_array' and 'summary_array_name'.")

    # Convert summary_array_name to a list of strings
    if isinstance(summary_array_name, list):
        column_names = [str(name[0]) for name in summary_array_name]
    else:
        column_names = [str(name[0]) for name in summary_array_name.squeeze()]

    # Create a pandas DataFrame
    df = pd.DataFrame(summary_array, columns=column_names)

    # Convert the 'image_id' column to int if it exists
    if 'image_id' in df.columns:
        df['image_id'] = df['image_id'].astype(int)

    return df




def plot_tensor_and_save(tensor, output_folder, file_name='tensor_plot.png'):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert the tensor to a NumPy array for plotting
    tensor_np = tensor.numpy()

    # Plot the tensor
    plt.figure(figsize=(8, 6))
    plt.imshow(tensor_np, cmap='gray')  # You can choose a different colormap if needed
    plt.colorbar()  # Optional: adds a color bar to the plot
    plt.title('Synthesize Image Visualization')
    
    # Save the plot to the assigned folder
    output_path = os.path.join(output_folder, file_name)
    import logging
    logging.info(f'{file_name}')
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid displaying it if running in an interactive environment

    # print(f"Plot saved to {output_path}")


def plot_vector_and_save(vector, output_folder, file_name='vector_plot.png'):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Convert the input to a NumPy array if it is a PyTorch tensor
    if isinstance(vector, torch.Tensor):
        vector_np = vector.numpy()
    elif isinstance(vector, np.ndarray):
        vector_np = vector
    else:
        raise TypeError("Input should be a PyTorch tensor or a NumPy array.")
    
    # If 1D, behave as before. If 2D (T x N), overlay all and save individual plots (limit to 50)
    if vector_np.ndim == 1:
        plt.figure(figsize=(8, 6))
        plt.plot(vector_np, color='b', linestyle='-', marker='o')  # Customize the line style as desired
        plt.title('1D Vector Plot')
        plt.xlabel('Index')
        plt.ylabel('Value')
        # Save the plot to the assigned folder
        output_path = os.path.join(output_folder, file_name)
        plt.savefig(output_path)
        plt.close()
    elif vector_np.ndim == 2:
        T, N = vector_np.shape
        # Overlay plot of all vectors
        plt.figure(figsize=(10, 6))
        for i in range(N):
            plt.plot(range(T), vector_np[:, i], alpha=0.6, linewidth=1)
        plt.title(f'{file_name} - All vectors (N={N})')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.tight_layout()
        base_name, ext = os.path.splitext(file_name)
        overlay_path = os.path.join(output_folder, f"{base_name}_all{ext}")
        plt.savefig(overlay_path)
        plt.close()

        # Individual plots (limit to first 50)
        max_individual = min(N, 50)
        for i in range(max_individual):
            plt.figure(figsize=(8, 6))
            plt.plot(range(T), vector_np[:, i], color='b', linestyle='-', marker='o')
            plt.title(f'Vector {i+1} of {N}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            indiv_path = os.path.join(output_folder, f"{base_name}_{i+1}{ext}")
            plt.savefig(indiv_path)
            plt.close()
    else:
        raise ValueError('Input array must be 1D or 2D for plotting.')


def plot_movement_and_velocity(path, velocity_history, boundary_size, name, output_folder="plots"):
    """
    Plots the random movement path and velocity history, saving the plots as images.

    Parameters:
    - path: numpy array of shape (n_steps, 2), with each row representing an (x, y) position.
    - velocity_history: numpy array of velocities for each step.
    - boundary_size: numpy array of shape (2,), defining the total size of the boundary.
    - output_folder: str, the directory to save the plot images. Default is "plots".
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract x and y coordinates from the path
    x_coords, y_coords = path[:, 0], path[:, 1]
    steps = len(path)

    # Plot the path with a color gradient for time progression
    plt.figure(1, figsize=(8, 8))
    plt.plot([-boundary_size[0] / 2, boundary_size[0] / 2, boundary_size[0] / 2, -boundary_size[0] / 2, -boundary_size[0] / 2],
             [-boundary_size[1] / 2, -boundary_size[1] / 2, boundary_size[1] / 2, boundary_size[1] / 2, -boundary_size[1] / 2],
             'k--', label="Boundary")
    plt.scatter(x_coords, y_coords, c=range(steps), cmap='viridis', edgecolor='k', s=20)
    plt.xlim(-boundary_size[0] / 2 - 20, boundary_size[0] / 2 + 20)
    plt.ylim(-boundary_size[1] / 2 - 20, boundary_size[1] / 2 + 20)
    plt.colorbar(label='Time Progression')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Random Movement Path within Boundary")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"movement_path_{name}.png"))
    plt.close()

    # Plot of velocity over iterations
    plt.figure(2, figsize=(8, 6))
    plt.plot(velocity_history)
    plt.xlabel('Iteration')
    plt.ylabel('Speed')
    plt.title('Plot of Velocity at Each Iteration')
    plt.savefig(os.path.join(output_folder, f"velocity_history_{name}.png"))
    plt.close()


def create_video_from_specific_files(folder_path, output_path, video_file_name, filename_template="synthesized_movement_{}.png", fps=10):
    """
    Combines specific images in a folder into a video, using an incremental naming pattern.

    Parameters:
    - folder_path: str, path to the folder containing the images.
    - output_path: str, path where the video should be saved.
    - filename_template: str, template for the filenames with `{}` as a placeholder for incrementing numbers.
    - fps: int, frames per second for the video.
    """
    video_file = os.path.join(output_path, video_file_name)
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Collect images based on the template
    images = []
    i = 1
    while True:
        file_path = os.path.join(folder_path, filename_template.format(i))
        if os.path.isfile(file_path):
            images.append(file_path)
            i += 1
        else:
            break

    # Ensure we have images to create the video
    if not images:
        raise ValueError("No images found with the specified template.")

    # Read the first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

    # Write each image to the video
    for image_path in images:
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()


def plot_position_and_save(positions, values=None, output_folder='', file_name='rgc_rf_position_plot.png'):
    """
    Plots the positions with color-coded values (if provided) and saves the plot.

    Parameters:
    - positions (np.ndarray): Array of shape (N, 2) containing x and y coordinates.
    - values (np.ndarray, optional): Array of shape (N,) containing values for each position to color code.
                                     If None, all points are the same color.
    - output_folder (str): Folder to save the plot.
    - file_name (str): Name of the output file (default is 'rgc_rf_position_plot.png').
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate the min and max for each axis
    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])

    # Calculate the ranges and expand limits by 10%
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_lim = (x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    y_lim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # Plotting
    plt.figure(figsize=(8, 6))

    if values is not None:
        # If values are provided, use them for coloring
        scatter = plt.scatter(positions[:, 0], positions[:, 1], c=values, s=100, cmap='viridis', edgecolor='k', alpha=0.6)
        plt.colorbar(scatter, label="Value")  # Add a color bar
    else:
        # If values are None, use a default color for all points
        plt.scatter(positions[:, 0], positions[:, 1], color='blue', s=100, edgecolor='k', alpha=0.6)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.title("Hexagonal Center Points in 2D Space")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the plot to the assigned folder
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path)
    plt.close() 


def plot_map_and_save(grid_values, output_folder, file_name='rgc_rf_gridmap_plot.png'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.rot90(grid_values, k=1), cmap='viridis')
    # plt.imshow(grid_values, cmap='viridis')
    plt.colorbar()
    plt.title("Mapped Values on Fixed-Size Grid by Closest Coordinate")
     
     # Save the plot to the assigned folder
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path)
    plt.close() 


def plot_gaussian_model(grid_values, rgc_array_rf_area, output_folder, file_name='gaussian_model_plot.png'):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract height and width from rgc_array_rf_area
    height, width = rgc_array_rf_area

    # Plot the Gaussian model
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_values, cmap='viridis', origin='lower')
    plt.colorbar(label="Gaussian Intensity (Median Adjusted)")

    # Add dashed lines at specified positions using height and width from rgc_array_rf_area
    plt.plot([width // 2, width // 2], [0, grid_values.shape[0]], '--k', label=f"Vertical Line at x={width // 2}")
    plt.plot([0, grid_values.shape[1]], [height // 2, height // 2], '--k', label=f"Horizontal Line at y={height // 2}")

    # Label and display the plot
    plt.title("Gaussian Model with Median Adjustment")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()

    # Save the plot to the assigned folder
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path)
    plt.close()
    import logging
    logging.info(f"Plot saved to {output_path}")


def plot_two_path_comparison(array1, array2, save_folder, file_name):
    """
    Plot traces from two 2D numpy arrays and save the figure.
    
    Parameters:
    - array1: numpy.ndarray, shape (n, 2), first array with x and y coordinates
    - array2: numpy.ndarray, shape (n, 2), second array with x and y coordinates
    - save_folder: str, folder path to save the plot
    - file_name: str, name of the file to save the plot (e.g., "plot.png")
    """
    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    # Extract x and y coordinates
    x1, y1 = array1[:, 0], array1[:, 1]
    x2, y2 = array2[:, 0], array2[:, 1]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(x1, y1, label="Trace 1", color="blue", linestyle="-", linewidth=2)
    plt.plot(x2, y2, label="Trace 2", color="red", linestyle="--", linewidth=2)
    
    # Add labels and legend
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Traces Plot")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    save_path = os.path.join(save_folder, file_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_temporal_filters(tf, output_folder, base_file_name='temporal_filter', max_individual=50):
    """
    Plot temporal filter(s) which can be 1D (T,) or 2D (T, N).

    - If 1D: saves single plot named '<base_file_name>.png'
    - If 2D: saves overlay '<base_file_name>_all.png' and individual files
      '<base_file_name>_1.png', '<base_file_name>_2.png', ... up to max_individual.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Convert to numpy array if torch tensor
    try:
        if isinstance(tf, torch.Tensor):
            tf_np = tf.cpu().numpy()
        else:
            tf_np = np.array(tf)
    except Exception:
        # Fallback: try to coerce via list
        tf_np = np.array(tf)

    # Delegate to plot_vector_and_save for 1D or individual plots
    if tf_np.ndim == 1:
        plot_vector_and_save(tf_np, output_folder, file_name=f'{base_file_name}.png')
        return

    if tf_np.ndim == 2:
        T, N = tf_np.shape
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            for i in range(N):
                plt.plot(range(T), tf_np[:, i], alpha=0.6, linewidth=1)
            plt.title(f'{base_file_name} - All temporal filters (N={N})')
            plt.xlabel('Time')
            plt.ylabel('Filter value')
            plt.tight_layout()
            overlay_path = os.path.join(output_folder, f'{base_file_name}_all.png')
            plt.savefig(overlay_path)
            plt.close()

            # Individual plots (limit)
            max_ind = min(N, max_individual)
            for i in range(max_ind):
                plot_vector_and_save(tf_np[:, i], output_folder, file_name=f'{base_file_name}_{i+1}.png')

            # Also create a tiled image sampling up to rows x cols (default 3x4)
            try:
                rows, cols = 3, 4
                num_tiles = rows * cols
                sample_N = min(N, num_tiles)
                # sample indices evenly if more filters than tiles
                if N <= num_tiles:
                    sample_idx = list(range(N))
                else:
                    # pick evenly spaced indices across N
                    sample_idx = [int(round(i * (N - 1) / (num_tiles - 1))) for i in range(num_tiles)]

                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5), squeeze=False)
                for k in range(num_tiles):
                    r = k // cols
                    c = k % cols
                    ax = axes[r][c]
                    if k < len(sample_idx):
                        idx = sample_idx[k]
                        ax.plot(range(T), tf_np[:, idx], color='b', linewidth=1)
                        ax.set_title(f'#{idx+1}')
                    else:
                        ax.axis('off')
                    ax.set_xlabel('t')
                    ax.set_ylabel('val')
                plt.tight_layout()
                tiled_path = os.path.join(output_folder, f'{base_file_name}_tiles_{rows}x{cols}.png')
                fig.savefig(tiled_path)
                plt.close(fig)
            except Exception:
                pass
        except Exception:
            # If plotting fails, fallback to saving first column as a single plot
            plot_vector_and_save(tf_np[:, 0], output_folder, file_name=f'{base_file_name}.png')
        return

    raise ValueError('tf must be 1D or 2D array-like')


def causal_moving_average(seq, window_size):
    """
    Computes a causal moving average for 2D coordinates using PyTorch with optimized computation.
    Uses cumulative sum for O(N) complexity instead of O(N * W).
    """
    assert seq.ndim == 3
    seq = seq.to(dtype=torch.float32)  # Ensure input is a tensor
    cumsum_seq = torch.cumsum(seq, dim=1)  # Compute cumulative sum

    # Compute the moving average using cumulative sum differences
    ma = cumsum_seq.clone()
    ma[:, window_size:] -= cumsum_seq[:, :-window_size]  # Efficient subtraction for moving window
    time_indices = torch.arange(1, seq.shape[1] + 1, dtype=torch.float32, device=seq.device)
    divisor = time_indices.clamp(max=window_size).view(1, -1, 1)  # Shape: [1, T, 1] for broadcasting

    # Normalize moving average
    ma /= divisor

    return ma


def causal_moving_average_numpy(seq, window_size):
    """
    Computes a causal moving average for 2D coordinates using NumPy.
    The input array shape is expected to be [time, coordinates] → (T, C).
    
    Uses cumulative sum for O(N) complexity instead of O(N * W).

    Parameters:
    - seq: NumPy array of shape (T, C), where T = time steps, C = coordinate dimensions
    - window_size: Size of the moving average window

    Returns:
    - ma: NumPy array of shape (T, C) with causal moving averages applied
    """
    assert seq.ndim == 2, "Input sequence must have shape (time, coordinates)"

    # Compute cumulative sum along the time axis (axis=0)
    cumsum_seq = np.cumsum(seq, axis=0)

    # Compute moving average using cumulative sum differences
    ma = cumsum_seq.copy()
    ma[window_size:] -= cumsum_seq[:-window_size]  # Efficient subtraction for moving window

    # Generate divisors for correct normalization
    time_indices = np.arange(1, seq.shape[0] + 1)  # Shape: (T,)
    divisor = np.minimum(time_indices, window_size).reshape(-1, 1)  # Shape: (T, 1) for broadcasting

    # Normalize moving average
    ma /= divisor

    return ma


def plot_coordinate_and_save(rgc_locs, rgc_locs_off=None, plot_save_folder=None, file_name=None):
    """
    Plots one or two sets of coordinates in separate panels and optionally saves the plot.
    
    Parameters:
    rgc_locs (numpy.ndarray): A 2D array of shape (N, 2) containing x, y coordinates.
    rgc_locs_off (numpy.ndarray, optional): A 2D array of shape (M, 2) containing x, y coordinates. If None, only rgc_locs is plotted.
    plot_save_folder (str, optional): Folder where the plot will be saved. If None, the plot is only displayed.
    file_name (str, optional): Name of the file to save the plot as, if saving is enabled.
    """
    if rgc_locs_off is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot ON locations
        axes[0].scatter(rgc_locs[:, 0], rgc_locs[:, 1], color='blue', alpha=0.7)
        axes[0].set_xlabel("X Coordinate")
        axes[0].set_ylabel("Y Coordinate")
        axes[0].set_title("RGC Locations (ON)")
        
        # Plot OFF locations
        axes[1].scatter(rgc_locs_off[:, 0], rgc_locs_off[:, 1], color='red', alpha=0.7)
        axes[1].set_xlabel("X Coordinate")
        axes[1].set_ylabel("Y Coordinate")
        axes[1].set_title("RGC Locations (OFF)")
        
        plt.tight_layout()
    else:
        # Create the plot for single set
        plt.figure(figsize=(8, 6))
        plt.scatter(rgc_locs[:, 0], rgc_locs[:, 1], color='blue', label='ON', alpha=0.7)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("RGC Locations")
        plt.legend()
    
    # Save or show the figure
    if plot_save_folder is not None and file_name is not None:
        os.makedirs(plot_save_folder, exist_ok=True)
        # Save rasterized PNG (useful for quick previews) and an EPS vector file for Illustrator
        save_path_png = os.path.join(plot_save_folder, file_name)
        try:
            plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        except Exception:
            # Fallback: try without bbox adjustment
            plt.savefig(save_path_png)

        # Also save a vectorized EPS version using the same base name but .eps extension
        base, _ext = os.path.splitext(file_name)
        save_path_eps = os.path.join(plot_save_folder, f"{base}.eps")
        try:
            plt.savefig(save_path_eps, format='eps', bbox_inches='tight')
        except Exception:
            # If EPS save fails, try saving without bbox and with default format inference
            try:
                plt.savefig(save_path_eps)
            except Exception:
                pass

        plt.close()
        import logging
        logging.info(f"Plot saved at: {save_path_png} (raster) and {save_path_eps} (vector eps)")
    else:
        plt.show()


