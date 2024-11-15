import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datasets.movingdot import RandomMovingSpotDataset

# Set paths and filenames
video_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Videos/'
file_name = 'demo_moving_spot'
os.makedirs(video_save_folder, exist_ok=True)
save_path = os.path.join(video_save_folder, file_name + ".mp4")

# Initialize dataset
dataset = RandomMovingSpotDataset(sequence_length=20, grid_height=24, grid_width=32, prob_vis=0.5, num_samples=1)
sequence, coords, visibility = dataset[0]  # Retrieve a sample sequence

# Convert tensors to numpy arrays
sequence = (sequence.squeeze(1).numpy() * 255).astype(np.uint8)
coords = coords.numpy()
visibility = visibility.numpy()

# Video settings
frame_height, frame_width = sequence.shape[1:3]
fps = 5
video_scalar = 10  # Scaling factor for video resolution

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width * video_scalar, frame_height * video_scalar))

# Generate and write frames to video
for frame in range(len(sequence)):
    # Set up matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    canvas = FigureCanvas(fig)
    
    # Plot grayscale image for the current frame in the first subplot
    ax1.imshow(sequence[frame], cmap='gray', vmin=0, vmax=255)
    ax1.set_title("Current Frame")
    ax1.axis('off')  # Hide axes for a cleaner look

    # Plot x, y trajectory in the second subplot
    for i in range(frame + 1):
        color = 'blue' if visibility[i] == 1 else 'lightblue'
        ax2.plot(coords[i, 1], coords[i, 0], 'o', color=color, markersize=3)
    
    ax2.set_title("Trajectory Path")
    ax2.set_xlabel("X Coordinate")
    ax2.set_ylabel("Y Coordinate")
    ax2.invert_yaxis()  # Invert Y axis to match image coordinate system
    ax2.set_aspect('equal', adjustable='box')  # Keep aspect ratio square for clarity
    
    # Render the plot to an image array
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))

    # Resize the image for the video dimensions
    img = cv2.resize(img, (frame_width * video_scalar, frame_height * video_scalar))

    # Write the frame to the video
    video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Close the plot to free memory
    plt.close(fig)

# Release the video writer
video_writer.release()
print(f"Video saved to {save_path}")
