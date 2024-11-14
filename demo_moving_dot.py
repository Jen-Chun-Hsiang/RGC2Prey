import os
import cv2
import numpy as np
import torch
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
video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Colors for visible and invisible trajectory
color_visible = (255, 0, 0)      # Blue for visible
color_invisible = (173, 216, 230) # Light blue for invisible

# Generate video frames
for frame in range(len(sequence)):
    # Get the current frame
    img = cv2.cvtColor(sequence[frame], cv2.COLOR_GRAY2BGR)

    # Plot the trajectory points
    for i in range(frame + 1):
        if visibility[i] == 1:
            cv2.circle(img, (int(coords[i, 1]), int(coords[i, 0])), 2, color_visible, -1)
        else:
            cv2.circle(img, (int(coords[i, 1]), int(coords[i, 0])), 2, color_invisible, -1)

    # Add text to indicate the current frame
    cv2.putText(img, f"Frame {frame+1}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Write frame to video
    video_writer.write(img)

# Release the video writer
video_writer.release()
print(f"Video saved to {save_path}")
