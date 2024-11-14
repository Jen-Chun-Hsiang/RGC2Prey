import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets.movingdot import RandomMovingSpotDataset

# Set paths and filenames
video_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Videos/'
file_name = 'demo_moving_spot'
os.makedirs(video_save_folder, exist_ok=True)
save_path = os.path.join(video_save_folder, file_name + ".mp4")

# Initialize dataset
dataset = RandomMovingSpotDataset(sequence_length=20, grid_height=24, grid_width=32, prob_vis=0.8, num_samples=1)
sequence, coords, visibility = dataset[0]  # Retrieve a sample sequence

# Convert tensors to numpy arrays
sequence = (sequence.squeeze(1).numpy() * 255).astype(np.uint8)
coords = coords.numpy()
visibility = visibility.numpy()

# Video settings
frame_height, frame_width = sequence.shape[1:3]
fps = 5

# Colors for visible and invisible trajectory
color_visible = 'blue'       # Blue for visible
color_invisible = 'lightblue' # Light blue for invisible

# Prepare a list to store frames for video creation
frames_for_video = []

# Generate frames
for frame in range(len(sequence)):
    plt.figure(figsize=(5, 5))
    plt.imshow(sequence[frame], cmap='gray', vmin=0, vmax=255)
    plt.title("Moving Spot Trajectory")
    
    # Plot the trajectory points
    for i in range(frame + 1):
        if visibility[i] == 1:
            plt.plot(coords[i, 1], coords[i, 0], 'o', color=color_visible, markersize=3)
        else:
            plt.plot(coords[i, 1], coords[i, 0], 'o', color=color_invisible, markersize=3)
    
    # Add text to indicate the current frame
    plt.text(1, 2, f"Frame {frame+1}", color="white", fontsize=10, backgroundcolor="black")

    # Remove axis for cleaner visualization
    plt.axis('off')
    
    # Save the frame to a temporary location as an image
    temp_path = os.path.join(video_save_folder, f"frame_{frame}.png")
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Read the image back as a frame for the video
    img = cv2.imread(temp_path)
    frames_for_video.append(img)

# Initialize video writer
video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Write frames to video
for frame in frames_for_video:
    video_writer.write(frame)

# Release the video writer
video_writer.release()

# Clean up temporary images
for frame_path in frames_for_video:
    os.remove(frame_path)

print(f"Video saved to {save_path}")
