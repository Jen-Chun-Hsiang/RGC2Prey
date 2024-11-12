import os
import random
import matplotlib.pyplot as plt
import cv2
import os
import torch
import numpy as np


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
    
    # Plot the vector
    plt.figure(figsize=(8, 6))
    plt.plot(vector_np, color='b', linestyle='-', marker='o')  # Customize the line style as desired
    plt.title('1D Vector Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    # Save the plot to the assigned folder
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path)
    plt.close() 


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
    print(f"Plot saved to {output_path}")
