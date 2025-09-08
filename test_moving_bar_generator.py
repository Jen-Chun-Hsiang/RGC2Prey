#!/usr/bin/env python3
"""
Test script for MovingBarMovieGenerator class.
This script demonstrates how to use the MovingBarMovieGenerator to create
moving bar stimuli and visualize the results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import cv2
from datasets.movie_generator import MovingBarMovieGenerator

def test_moving_bar_generator():
    """
    Test the MovingBarMovieGenerator with various configurations
    and create visualizations to inspect the functionality.
    """
    
    # Configuration parameters (similar to those used in train_preycapture.py)
    crop_size = (320, 240)  # (width, height)
    boundary_size = (220, 140)  # boundary for movement
    center_ratio = (0.2, 0.2)  # center region ratio
    max_steps = 100
    num_ext = 10  # number of extended static frames
    
    # Moving bar specific parameters
    bar_width_range = (10, 40)      # width range in pixels
    bar_height_range = (60, 200)    # height range in pixels  
    speed_range = (2.0, 8.0)        # speed range in pixels/frame
    direction_range = (0.0, 360.0)  # direction range in degrees
    num_episodes = 3                # number of episodes to generate
    
    print("Initializing MovingBarMovieGenerator...")
    
    # Initialize the generator
    movie_generator = MovingBarMovieGenerator(
        crop_size=crop_size,
        boundary_size=boundary_size,
        center_ratio=center_ratio,
        max_steps=max_steps,
        num_ext=num_ext,
        bar_width_range=bar_width_range,
        bar_height_range=bar_height_range,
        speed_range=speed_range,
        direction_range=direction_range,
        num_episodes=num_episodes,
        margin=2.0,
        is_binocular=False,  # Start with monocular for simplicity
        bottom_contrast=1.0,
        top_contrast=1.0
    )
    
    print("Generating moving bar episodes...")
    
    # Generate episodes
    result = movie_generator.generate()
    episodes = result["episodes"]
    
    print(f"Generated {len(episodes)} episodes")
    
    # Visualize each episode
    for ep_idx, episode in enumerate(episodes):
        frames = episode["frames"]  # Shape: (H, W, T)
        path = episode["path"]      # Shape: (T, 2) - (x, y) positions
        meta = episode["meta"]
        
        H, W, T = frames.shape
        print(f"\nEpisode {ep_idx + 1}:")
        print(f"  Frames shape: {frames.shape}")
        print(f"  Path shape: {path.shape}")
        print(f"  Metadata: {meta}")
        
        # Create visualization
        visualize_episode(frames, path, meta, ep_idx)
        
        # Save frames as images for inspection
        save_episode_frames(frames, path, meta, ep_idx)
        
        # Create animated GIF
        create_episode_animation(frames, path, meta, ep_idx)
    
    print("\nMoving bar generator test completed!")

def visualize_episode(frames, path, meta, episode_idx):
    """
    Create a static visualization showing key frames and trajectory.
    """
    H, W, T = frames.shape
    
    # Select a few key frames to display
    key_frame_indices = [0, T//4, T//2, 3*T//4, T-1]
    
    fig, axes = plt.subplots(2, len(key_frame_indices), figsize=(15, 8))
    fig.suptitle(f'Episode {episode_idx + 1}: Moving Bar Visualization\n'
                f'Width: {meta["bar_width"]:.1f}, Height: {meta["bar_height"]:.1f}, '
                f'Speed: {meta["speed"]:.1f}, Direction: {meta["move_dir"]:.1f}°', 
                fontsize=14)
    
    # Plot key frames
    for i, frame_idx in enumerate(key_frame_indices):
        ax = axes[0, i]
        ax.imshow(frames[:, :, frame_idx], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Frame {frame_idx}')
        ax.axis('off')
        
        # Mark bar center position
        if frame_idx < len(path):
            x, y = path[frame_idx]
            ax.plot(x, y, 'r+', markersize=10, markeredgewidth=2)
    
    # Plot trajectory
    ax_traj = axes[1, :]
    fig.delaxes(axes[1, 1])
    fig.delaxes(axes[1, 2])
    fig.delaxes(axes[1, 3])
    fig.delaxes(axes[1, 4])
    
    ax_traj = plt.subplot(2, 1, 2)
    ax_traj.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Bar trajectory')
    ax_traj.plot(path[0, 0], path[0, 1], 'go', markersize=8, label='Start')
    ax_traj.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, label='End')
    ax_traj.set_xlim(0, W)
    ax_traj.set_ylim(0, H)
    ax_traj.set_xlabel('X (pixels)')
    ax_traj.set_ylabel('Y (pixels)')
    ax_traj.set_title('Bar Center Trajectory')
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)
    ax_traj.invert_yaxis()  # Invert y-axis to match image coordinates
    
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs('test_results', exist_ok=True)
    plt.savefig(f'test_results/moving_bar_episode_{episode_idx + 1}_summary.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

def save_episode_frames(frames, path, meta, episode_idx, save_every=5):
    """
    Save individual frames as PNG images for detailed inspection.
    """
    H, W, T = frames.shape
    os.makedirs(f'test_results/episode_{episode_idx + 1}_frames', exist_ok=True)
    
    for t in range(0, T, save_every):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Display frame
        im = ax.imshow(frames[:, :, t], cmap='gray', vmin=0, vmax=1)
        
        # Mark bar center
        if t < len(path):
            x, y = path[t]
            ax.plot(x, y, 'r+', markersize=12, markeredgewidth=3)
            
            # Add bar outline for visualization
            bar_w, bar_h = meta["bar_width"], meta["bar_height"]
            angle = meta["bar_angle"]
            
            # Create rotated rectangle (approximate visualization)
            rect = Rectangle((x - bar_w/2, y - bar_h/2), bar_w, bar_h, 
                           angle=angle, fill=False, edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(rect)
        
        ax.set_title(f'Episode {episode_idx + 1}, Frame {t}\n'
                    f'Bar center: ({path[t, 0]:.1f}, {path[t, 1]:.1f})')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(f'test_results/episode_{episode_idx + 1}_frames/frame_{t:03d}.png', 
                   dpi=100, bbox_inches='tight')
        plt.close()

def create_episode_animation(frames, path, meta, episode_idx):
    """
    Create an animated GIF showing the moving bar.
    """
    H, W, T = frames.shape
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Initial frame
    im = ax.imshow(frames[:, :, 0], cmap='gray', vmin=0, vmax=1, animated=True)
    point, = ax.plot([], [], 'r+', markersize=12, markeredgewidth=3, animated=True)
    
    ax.set_title(f'Episode {episode_idx + 1}: Moving Bar Animation')
    ax.axis('off')
    
    def animate(frame_idx):
        # Update frame
        im.set_array(frames[:, :, frame_idx])
        
        # Update bar center position
        if frame_idx < len(path):
            x, y = path[frame_idx]
            point.set_data([x], [y])
        
        return [im, point]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=T, interval=100, 
                                 blit=True, repeat=True)
    
    # Save as GIF
    os.makedirs('test_results', exist_ok=True)
    anim.save(f'test_results/moving_bar_episode_{episode_idx + 1}_animation.gif', 
              writer='pillow', fps=10)
    
    plt.close()
    print(f"Animation saved: test_results/moving_bar_episode_{episode_idx + 1}_animation.gif")

def test_binocular_mode():
    """
    Test the generator in binocular mode.
    """
    print("\nTesting binocular mode...")
    
    movie_generator = MovingBarMovieGenerator(
        crop_size=(320, 240),
        boundary_size=(220, 140),
        bar_width_range=(20, 30),
        bar_height_range=(80, 120),
        speed_range=(4.0, 6.0),
        direction_range=(0.0, 180.0),
        num_episodes=1,
        is_binocular=True,
        interocular_dist=1.0  # cm
    )
    
    result = movie_generator.generate()
    episode = result["episodes"][0]
    
    print(f"Binocular frames shape: {episode['frames'].shape}")
    print(f"Binocular path shape: {episode['path'].shape}")
    print(f"Binocular metadata: {episode['meta']}")

def test_parameter_ranges():
    """
    Test different parameter ranges to ensure robustness.
    """
    print("\nTesting different parameter ranges...")
    
    test_configs = [
        {
            "name": "Small fast bars",
            "bar_width_range": (5, 15),
            "bar_height_range": (20, 60),
            "speed_range": (8.0, 15.0)
        },
        {
            "name": "Large slow bars", 
            "bar_width_range": (40, 80),
            "bar_height_range": (100, 200),
            "speed_range": (1.0, 3.0)
        },
        {
            "name": "Horizontal movement only",
            "bar_width_range": (20, 40),
            "bar_height_range": (60, 120),
            "speed_range": (4.0, 8.0),
            "direction_range": (-10.0, 10.0)  # Nearly horizontal
        }
    ]
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        
        movie_generator = MovingBarMovieGenerator(
            crop_size=(240, 180),
            boundary_size=(180, 120),
            **{k: v for k, v in config.items() if k != "name"},
            num_episodes=1
        )
        
        result = movie_generator.generate()
        episode = result["episodes"][0]
        
        print(f"  Generated {episode['frames'].shape[2]} frames")
        print(f"  Metadata: {episode['meta']}")

if __name__ == "__main__":
    print("=== MovingBarMovieGenerator Test ===")
    
    # Main test
    test_moving_bar_generator()
    
    # Additional tests
    test_binocular_mode()
    test_parameter_ranges()
    
    print("\n=== All tests completed! ===")
    print("Check the 'test_results' directory for output files:")
    print("  - Static summary plots")
    print("  - Individual frame images") 
    print("  - Animated GIF files")
