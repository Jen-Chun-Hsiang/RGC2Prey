import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datasets.movingdot import RandomMovingSpotDataset

# Assuming you have the RandomMovingSpotDataset class defined above
# Create an instance of the dataset
video_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Videos/'
file_name = 'demo_moving_spot'
os.makedirs(video_save_folder, exist_ok=True)
save_path = os.path.join(video_save_folder, file_name)

dataset = RandomMovingSpotDataset(sequence_length=20, grid_height=24, grid_width=32, prob_vis=0.8, num_samples=1)

# Retrieve a sample sequence
sequence, coords, visibility = dataset[0]  # Get the first item from the dataset

# Convert tensors to numpy arrays for plotting
sequence = sequence.squeeze(1).numpy()
coords = coords.numpy()
visibility = visibility.numpy()

# Initialize the figure and axes for the animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the trajectory in ax2
ax2.set_xlim(0, dataset.grid_width)
ax2.set_ylim(0, dataset.grid_height)
ax2.invert_yaxis()
ax2.set_title("Trajectory (Blue = Visible, Light Blue = Invisible)")

# Define two lines: one for the visible trajectory and one for invisible
visible_line, = ax2.plot([], [], 'o-', color='darkblue', lw=2)
invisible_line, = ax2.plot([], [], 'o-', color='lightblue', lw=1)

# Initialize the frame in ax1
im = ax1.imshow(sequence[0], cmap='gray', vmin=0, vmax=1)
ax1.set_title("Current Frame")

# Animation update function
def update(frame):
    # Update the frame display in ax1
    im.set_data(sequence[frame])
    ax1.set_title(f"Current Frame (Step {frame})")

    # Separate visible and invisible points in the trajectory
    visible_points = coords[:frame+1][visibility[:frame+1] == 1]
    invisible_points = coords[:frame+1][visibility[:frame+1] == 0]

    # Update the trajectory plot
    if len(visible_points) > 0:
        visible_line.set_data(visible_points[:, 1], visible_points[:, 0])
    if len(invisible_points) > 0:
        invisible_line.set_data(invisible_points[:, 1], invisible_points[:, 0])

    return im, visible_line, invisible_line

# Set up the animation writer to save as MP4
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

# Create and save the animation
ani = animation.FuncAnimation(fig, update, frames=len(sequence), interval=200, blit=True)
ani.save(f"{save_path}.mp4", writer=writer)


plt.show()
