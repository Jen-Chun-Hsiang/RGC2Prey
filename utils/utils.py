import os
import random
import matplotlib.pyplot as plt
import cv2
import os
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

    print(f"Plot saved to {output_path}")


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


def plot_position_and_save(positions, output_folder, file_name='rgc_rf_position_plot.png'):
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

    plt.figure(figsize=(8, 6))
    plt.scatter(positions[:, 0], positions[:, 1], c='blue', alpha=0.6)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.title("Hexagonal Center Points in 2D Space")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the plot to the assigned folder
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid displaying it if running in an interactive environment


def plot_map_and_save(grid_values, output_folder, file_name='rgc_rf_gridmap_plot.png'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.figure(figsize=(6, 6))
    # plt.imshow(grid_values.rot90(k=1).numpy(), cmap='viridis')
    plt.imshow(grid_values, cmap='viridis')
    plt.colorbar()
    plt.title("Mapped Values on Fixed-Size Grid by Closest Coordinate")
     
     # Save the plot to the assigned folder
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path)
    plt.close() 
