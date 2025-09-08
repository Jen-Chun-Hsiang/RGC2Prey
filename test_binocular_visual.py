#!/usr/bin/env python3
"""
Visual test for MovingBarMovieGenerator binocular functionality.
This script creates simple visualizations to verify the binocular disparity implementation.
"""

import os
import sys
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, '/Users/emilyhsiang/Desktop/Documents/RGC2Prey')

from datasets.movie_generator import MovingBarMovieGenerator

def visual_binocular_test():
    """
    Create a visual test of binocular moving bar generation.
    Saves frames as text representations for inspection.
    """
    print("=== Visual Binocular Test ===")
    
    # Create a small, simple test case
    movie_generator = MovingBarMovieGenerator(
        crop_size=(40, 30),  # Small for easy inspection
        boundary_size=(30, 20),
        bar_width_range=(8, 8),    # Fixed width
        bar_height_range=(20, 20), # Fixed height  
        speed_range=(2.0, 2.0),    # Fixed speed
        direction_range=(0.0, 0.0), # Horizontal movement only
        num_episodes=1,
        is_binocular=True,
        fix_disparity=4.0  # Fixed disparity for predictable results
    )
    
    result = movie_generator.generate()
    episode = result["episodes"][0]
    
    frames = episode["frames"]  # Shape: (H, W, 2, T)
    path = episode["path"]
    meta = episode["meta"]
    
    H, W, n_eyes, T = frames.shape
    
    print(f"Generated frames: {frames.shape}")
    print(f"Disparity: {meta['disparity'][0]:.1f} pixels")
    print(f"Bar size: {meta['bar_width']:.0f} x {meta['bar_height']:.0f}")
    print(f"Movement direction: {meta['move_dir']:.1f} degrees")
    
    # Create output directory
    os.makedirs('visual_test_output', exist_ok=True)
    
    # Save a few frames as text for inspection
    frames_to_show = [0, T//4, T//2, 3*T//4, T-1]
    
    with open('visual_test_output/binocular_frames.txt', 'w') as f:
        f.write("BINOCULAR MOVING BAR TEST\n")
        f.write("=" * 50 + "\n")
        f.write(f"Disparity: {meta['disparity'][0]:.1f} pixels\n")
        f.write(f"Bar: {meta['bar_width']:.0f}x{meta['bar_height']:.0f}, Speed: {meta['speed']:.1f}\n")
        f.write("=" * 50 + "\n\n")
        
        for frame_idx in frames_to_show:
            f.write(f"FRAME {frame_idx}\n")
            f.write("-" * 40 + "\n")
            
            left_frame = frames[:, :, 0, frame_idx]
            right_frame = frames[:, :, 1, frame_idx]
            
            left_pos = meta['left_positions'][frame_idx]
            right_pos = meta['right_positions'][frame_idx]
            center_pos = path[frame_idx]
            
            f.write(f"Center position: ({center_pos[0]:.1f}, {center_pos[1]:.1f})\n")
            f.write(f"Left position:   ({left_pos[0]:.1f}, {left_pos[1]:.1f})\n") 
            f.write(f"Right position:  ({right_pos[0]:.1f}, {right_pos[1]:.1f})\n")
            f.write(f"Horizontal diff: {right_pos[0] - left_pos[0]:.1f}\n\n")
            
            # Show left eye
            f.write("LEFT EYE:\n")
            for y in range(H):
                for x in range(W):
                    if left_frame[y, x] > 0.6:  # Bar pixels
                        f.write("██")
                    elif left_frame[y, x] > 0.4:  # Background 
                        f.write("░░")
                    else:  # Dark background
                        f.write("  ")
                f.write("\n")
            
            f.write("\nRIGHT EYE:\n")
            for y in range(H):
                for x in range(W):
                    if right_frame[y, x] > 0.6:  # Bar pixels
                        f.write("██")
                    elif right_frame[y, x] > 0.4:  # Background
                        f.write("░░") 
                    else:  # Dark background
                        f.write("  ")
                f.write("\n")
            
            f.write("\n" + "=" * 40 + "\n\n")
    
    print("✓ Visual test frames saved to: visual_test_output/binocular_frames.txt")
    
    # Also save numerical data for analysis
    np.save('visual_test_output/left_frames.npy', frames[:, :, 0, :])
    np.save('visual_test_output/right_frames.npy', frames[:, :, 1, :])
    np.save('visual_test_output/positions.npy', path)
    np.save('visual_test_output/left_positions.npy', meta['left_positions'])
    np.save('visual_test_output/right_positions.npy', meta['right_positions'])
    
    print("✓ Numerical data saved to visual_test_output/")
    
    # Verify disparity implementation
    left_pos = meta['left_positions']
    right_pos = meta['right_positions']
    disparity = meta['disparity']
    
    # Check horizontal shifts
    h_diff = right_pos[:, 0] - left_pos[:, 0]
    expected_diff = disparity
    
    print(f"\nDisparity verification:")
    print(f"Expected horizontal difference: {expected_diff[0]:.2f}")
    print(f"Actual horizontal difference: {h_diff[0]:.2f}")
    print(f"Max deviation: {np.abs(h_diff - expected_diff).max():.6f}")
    
    # Check vertical alignment
    v_diff = np.abs(right_pos[:, 1] - left_pos[:, 1])
    print(f"Max vertical difference: {v_diff.max():.6f} (should be ~0)")
    
    # Check center position
    center_calc = (left_pos + right_pos) / 2.0
    center_diff = np.abs(center_calc - path)
    print(f"Center position error: {center_diff.max():.6f} (should be ~0)")
    
    return True

def compare_monocular_binocular():
    """
    Compare monocular and binocular versions side by side.
    """
    print("\n=== Monocular vs Binocular Comparison ===")
    
    # Same parameters for both
    params = {
        'crop_size': (40, 30),
        'boundary_size': (30, 20),
        'bar_width_range': (6, 6),
        'bar_height_range': (15, 15),
        'speed_range': (1.5, 1.5),
        'direction_range': (45.0, 45.0),  # Diagonal movement
        'num_episodes': 1
    }
    
    # Monocular version
    mono_gen = MovingBarMovieGenerator(**params, is_binocular=False)
    mono_result = mono_gen.generate()
    mono_episode = mono_result["episodes"][0]
    
    # Binocular version  
    bino_gen = MovingBarMovieGenerator(**params, is_binocular=True, fix_disparity=3.0)
    bino_result = bino_gen.generate()
    bino_episode = bino_result["episodes"][0]
    
    print(f"Monocular frames shape: {mono_episode['frames'].shape}")
    print(f"Binocular frames shape: {bino_episode['frames'].shape}")
    
    # Save comparison
    with open('visual_test_output/mono_vs_bino_comparison.txt', 'w') as f:
        f.write("MONOCULAR vs BINOCULAR COMPARISON\n")
        f.write("=" * 50 + "\n")
        f.write(f"Movement direction: {mono_episode['meta']['move_dir']:.1f} degrees\n")
        f.write(f"Bar size: {mono_episode['meta']['bar_width']:.0f}x{mono_episode['meta']['bar_height']:.0f}\n")
        if bino_episode['meta']['is_binocular']:
            f.write(f"Binocular disparity: {bino_episode['meta']['disparity'][0]:.1f} pixels\n")
        f.write("\n")
        
        # Show middle frame
        mono_frames = mono_episode['frames']
        bino_frames = bino_episode['frames']
        
        mid_frame = mono_frames.shape[-1] // 2
        
        f.write(f"FRAME {mid_frame} COMPARISON\n")
        f.write("-" * 30 + "\n")
        
        f.write("MONOCULAR:\n")
        mono_frame = mono_frames[:, :, mid_frame]
        for y in range(mono_frame.shape[0]):
            for x in range(mono_frame.shape[1]):
                if mono_frame[y, x] > 0.6:
                    f.write("██")
                else:
                    f.write("░░")
            f.write("\n")
        
        f.write("\nBINOCULAR LEFT:\n")
        left_frame = bino_frames[:, :, 0, mid_frame]
        for y in range(left_frame.shape[0]):
            for x in range(left_frame.shape[1]):
                if left_frame[y, x] > 0.6:
                    f.write("██")
                else:
                    f.write("░░")
            f.write("\n")
        
        f.write("\nBINOCULAR RIGHT:\n")
        right_frame = bino_frames[:, :, 1, mid_frame]
        for y in range(right_frame.shape[0]):
            for x in range(right_frame.shape[1]):
                if right_frame[y, x] > 0.6:
                    f.write("██")
                else:
                    f.write("░░")
            f.write("\n")
    
    print("✓ Comparison saved to: visual_test_output/mono_vs_bino_comparison.txt")
    return True

if __name__ == "__main__":
    print("MovingBarMovieGenerator Binocular Visual Test")
    print("=" * 50)
    
    success = True
    success &= visual_binocular_test()
    success &= compare_monocular_binocular()
    
    if success:
        print("\n🎉 Visual tests completed successfully!")
        print("Check the 'visual_test_output' directory for detailed visualizations.")
    else:
        print("\n❌ Some visual tests failed.")
