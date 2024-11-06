import os
import random
import matplotlib.pyplot as plt


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


def plot_movement_and_velocity(path, velocity_history, boundary_size, output_folder="plots"):
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
    plt.savefig(os.path.join(output_folder, "movement_path.png"))
    plt.close()

    # Plot of velocity over iterations
    plt.figure(2, figsize=(8, 6))
    plt.plot(velocity_history)
    plt.xlabel('Iteration')
    plt.ylabel('Speed')
    plt.title('Plot of Velocity at Each Iteration')
    plt.savefig(os.path.join(output_folder, "velocity_history.png"))
    plt.close()