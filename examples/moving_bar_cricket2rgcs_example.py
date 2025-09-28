#!/usr/bin/env python3
"""
Example showing how to use MovingBarAdapter with Cricket2RGCs.
This replaces the original cricket movie generator with moving bar stimuli.
"""

import sys
import os
import numpy as np
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets.movie_generator import MovingBarMovieGenerator
from datasets.moving_bar_adapter import MovingBarAdapter
from datasets.sim_cricket import Cricket2RGCs

def create_moving_bar_dataset():
    """Example of creating a Cricket2RGCs dataset with moving bar stimuli."""
    
    print("=== Creating Cricket2RGCs with Moving Bar Stimuli ===")
    
    # Step 1: Create MovingBarMovieGenerator
    moving_bar_gen = MovingBarMovieGenerator(
        crop_size=(320, 240),  # (width, height)
        boundary_size=(220, 140),
        bar_width_range=(15, 35),
        bar_height_range=(60, 120),
        speed_range=(3.0, 7.0),  # pixels/frame
        direction_range=(0.0, 360.0),  # degrees
        num_episodes=1,  # Cricket2RGCs calls generate() once per sample
        is_binocular=False,  # Can be True for binocular stimuli
        margin=2.0
    )
    
    # Step 2: Wrap with adapter
    movie_generator = MovingBarAdapter(moving_bar_gen)
    
    # Step 3: Create dummy RGC parameters (you would load these from your actual data)
    # For this example, we'll create minimal placeholder arrays
    
    # Spatial filters: shape (W, H, N_rgcs)
    target_width, target_height = 64, 48  # Grid dimensions
    n_rgcs = 100  # Number of RGC cells
    multi_opt_sf = np.random.randn(320, 240, n_rgcs) * 0.1  # Placeholder spatial filters
    surround_sf = np.random.randn(320, 240, n_rgcs) * 0.05  # Placeholder surround filters
    
    # Temporal filter: shape (n_rgcs, temporal_length) or (temporal_length,)
    temporal_length = 30
    tf = np.random.randn(temporal_length) * 0.1  # Single temporal filter for all RGCs
    
    # Grid mapping function and parameters (using closest mapping for simplicity)
    from datasets.rgc_rf import map_to_fixed_grid_closest_batch, get_closest_indices
    
    # Create grid centers (placeholder)
    grid_centers = np.random.rand(target_height * target_width, 2) * [320, 240]
    rgc_centers = np.random.rand(n_rgcs, 2) * [320, 240]
    
    closest_indices = get_closest_indices(grid_centers, rgc_centers)
    grid2value_mapping = torch.tensor(closest_indices, dtype=torch.long)
    map_func = map_to_fixed_grid_closest_batch
    
    # Step 4: Create Cricket2RGCs dataset
    dataset = Cricket2RGCs(
        num_samples=10,  # Number of samples in dataset
        multi_opt_sf=multi_opt_sf,
        tf=tf,
        map_func=map_func,
        grid2value_mapping=grid2value_mapping,
        target_width=target_width,
        target_height=target_height,
        movie_generator=movie_generator,  # Our adapted moving bar generator
        grid_size_fac=1,
        is_norm_coords=False,
        is_syn_mov_shown=True,  # Return synthetic movie for visualization
        add_noise=False,
        rgc_noise_std=0.0,
        smooth_data=False,
        is_rectified=True,
        is_direct_image=False,
        # LNK parameters (optional)
        use_lnk=False,
        surround_sigma_ratio=4.0,
        surround_sf=surround_sf,
        lnk_params=None,
        # Random seed for consistent data generation
        rnd_seed=None
    )
    
    print(f"✓ Created Cricket2RGCs dataset with {len(dataset)} samples")
    
    # Step 5: Test the dataset
    print("\n--- Testing Dataset ---")
    sample = dataset[0]  # Get first sample
    
    if dataset.is_syn_mov_shown:
        grid_seq, path, path_bg, syn_movie, meta, weighted_coords = sample
        print(f"grid_seq shape: {grid_seq.shape}")
        print(f"path shape: {path.shape}")
        print(f"path_bg shape: {path_bg.shape}")
        print(f"syn_movie shape: {syn_movie.shape}")
        print(f"meta type: {type(meta)}")
        print(f"Moving bar metadata: {meta}")
    else:
        grid_seq, path, path_bg = sample
        print(f"grid_seq shape: {grid_seq.shape}")
        print(f"path shape: {path.shape}")
        print(f"path_bg shape: {path_bg.shape}")
    
    print("✓ Dataset test completed successfully!")
    
    return dataset

def create_binocular_moving_bar_dataset():
    """Example with binocular moving bar stimuli."""
    
    print("\n=== Creating Binocular Moving Bar Dataset ===")
    
    # Create binocular moving bar generator
    moving_bar_gen = MovingBarMovieGenerator(
        crop_size=(320, 240),
        boundary_size=(220, 140),
        bar_width_range=(20, 30),
        bar_height_range=(80, 120),
        speed_range=(4.0, 6.0),
        direction_range=(0.0, 360.0),
        num_episodes=1,
        is_binocular=True,  # Enable binocular mode
        fix_disparity=2.0,  # Fixed disparity in pixels
        interocular_dist=1.0,  # cm
        margin=2.0
    )
    
    movie_generator = MovingBarAdapter(moving_bar_gen)
    
    # Dummy parameters for binocular case
    target_width, target_height = 64, 48
    n_rgcs = 50
    multi_opt_sf = np.random.randn(320, 240, n_rgcs) * 0.1
    surround_sf = np.random.randn(320, 240, n_rgcs) * 0.05
    tf = np.random.randn(30) * 0.1
    
    from datasets.rgc_rf import map_to_fixed_grid_closest_batch, get_closest_indices
    grid_centers = np.random.rand(target_height * target_width, 2) * [320, 240]
    rgc_centers = np.random.rand(n_rgcs, 2) * [320, 240]
    closest_indices = get_closest_indices(grid_centers, rgc_centers)
    grid2value_mapping = torch.tensor(closest_indices, dtype=torch.long)
    map_func = map_to_fixed_grid_closest_batch
    
    # Create dataset with binocular support
    dataset = Cricket2RGCs(
        num_samples=5,
        multi_opt_sf=multi_opt_sf,
        tf=tf,
        map_func=map_func,
        grid2value_mapping=grid2value_mapping,
        target_width=target_width,
        target_height=target_height,
        movie_generator=movie_generator,
        is_syn_mov_shown=True,
        use_lnk=False,
        surround_sf=surround_sf,
    )
    
    # Test binocular sample
    sample = dataset[0]
    grid_seq, path, path_bg, syn_movie, meta, weighted_coords = sample
    
    print(f"Binocular syn_movie shape: {syn_movie.shape}")
    print(f"Number of channels: {syn_movie.shape[1]} (should be 2 for binocular)")
    print(f"Binocular grid_seq shape: {grid_seq.shape}")
    
    if syn_movie.shape[1] == 2:
        print("✓ Binocular processing detected correctly")
    else:
        print("⚠ Expected binocular (2 channels), got different number")
    
    return dataset

def main():
    """Run moving bar adapter examples."""
    try:
        # Test monocular moving bar dataset
        mono_dataset = create_moving_bar_dataset()
        
        # Test binocular moving bar dataset  
        bino_dataset = create_binocular_moving_bar_dataset()
        
        print("\n🎉 All examples completed successfully!")
        print("\nUsage Summary:")
        print("1. Create MovingBarMovieGenerator with desired parameters")
        print("2. Wrap it with MovingBarAdapter")
        print("3. Pass the adapter as movie_generator to Cricket2RGCs")
        print("4. Use the dataset normally - it will generate moving bar stimuli instead of cricket images")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
