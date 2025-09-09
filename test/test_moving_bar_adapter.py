#!/usr/bin/env python3
"""
Test script to verify MovingBarAdapter compatibility with Cricket2RGCs.
"""

import os
import sys
import numpy as np
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets.movie_generator import MovingBarMovieGenerator
from datasets.moving_bar_adapter import MovingBarAdapter

def test_adapter_basic():
    """Test basic adapter functionality."""
    print("=== Testing MovingBarAdapter ===")
    
    # Create a MovingBarMovieGenerator instance
    mb_gen = MovingBarMovieGenerator(
        crop_size=(160, 120),  # (width, height)
        boundary_size=(120, 80),
        bar_width_range=(10, 20),
        bar_height_range=(40, 80),
        speed_range=(2.0, 4.0),
        num_episodes=1,
        is_binocular=False
    )
    
    # Wrap it with the adapter
    adapter = MovingBarAdapter(mb_gen)
    
    # Test monocular output
    print("\n--- Testing Monocular Output ---")
    syn_movie, path, path_bg, meta = adapter.generate()
    
    print(f"syn_movie type: {type(syn_movie)}")
    print(f"syn_movie shape: {syn_movie.shape}")
    print(f"syn_movie dtype: {syn_movie.dtype}")
    print(f"syn_movie value range: [{syn_movie.min():.3f}, {syn_movie.max():.3f}]")
    print(f"path shape: {path.shape}, dtype: {path.dtype}")
    print(f"path_bg shape: {path_bg.shape}, dtype: {path_bg.dtype}")
    print(f"meta type: {type(meta)}")
    
    # Verify expected format for Cricket2RGCs
    assert isinstance(syn_movie, torch.Tensor), "syn_movie should be torch.Tensor"
    assert syn_movie.ndim == 4, f"syn_movie should be 4D, got {syn_movie.ndim}D"
    assert syn_movie.shape[1] == 1, f"Expected 1 channel for monocular, got {syn_movie.shape[1]}"
    assert syn_movie.dtype == torch.float32, f"Expected float32, got {syn_movie.dtype}"
    assert isinstance(path, np.ndarray), "path should be numpy array"
    assert isinstance(path_bg, np.ndarray), "path_bg should be numpy array"
    assert path.shape[1] == 2, f"path should have 2 columns (x,y), got {path.shape[1]}"
    assert path_bg.shape[1] == 2, f"path_bg should have 2 columns (x,y), got {path_bg.shape[1]}"
    assert path.shape[0] == syn_movie.shape[0], f"Time dimension mismatch: path {path.shape[0]} vs frames {syn_movie.shape[0]}"
    
    print("✓ Monocular test passed!")
    
    # Test binocular output
    print("\n--- Testing Binocular Output ---")
    mb_gen_bino = MovingBarMovieGenerator(
        crop_size=(160, 120),
        boundary_size=(120, 80),
        bar_width_range=(10, 20),
        bar_height_range=(40, 80),
        speed_range=(2.0, 4.0),
        num_episodes=1,
        is_binocular=True,
        fix_disparity=2.0
    )
    
    adapter_bino = MovingBarAdapter(mb_gen_bino)
    syn_movie_bino, path_bino, path_bg_bino, meta_bino = adapter_bino.generate()
    
    print(f"syn_movie_bino shape: {syn_movie_bino.shape}")
    print(f"syn_movie_bino dtype: {syn_movie_bino.dtype}")
    print(f"path_bino shape: {path_bino.shape}, dtype: {path_bino.dtype}")
    
    # Verify binocular format
    assert syn_movie_bino.shape[1] == 2, f"Expected 2 channels for binocular, got {syn_movie_bino.shape[1]}"
    assert path_bino.shape[0] == syn_movie_bino.shape[0], f"Time dimension mismatch: path {path_bino.shape[0]} vs frames {syn_movie_bino.shape[0]}"
    
    # Check that left and right eye frames are different
    left_frames = syn_movie_bino[:, 0, :, :]
    right_frames = syn_movie_bino[:, 1, :, :]
    are_different = not torch.equal(left_frames, right_frames)
    print(f"Left and right eye frames are different: {are_different}")
    assert are_different, "Left and right eye frames should be different due to disparity"
    
    print("✓ Binocular test passed!")
    
    return True

def test_adapter_with_cricket2rgcs_interface():
    """Test that the adapter works with Cricket2RGCs interface expectations."""
    print("\n--- Testing Cricket2RGCs Interface Compatibility ---")
    
    # Create adapter
    mb_gen = MovingBarMovieGenerator(
        crop_size=(200, 150),
        boundary_size=(150, 100),
        bar_width_range=(15, 25),
        bar_height_range=(60, 100),
        speed_range=(3.0, 5.0),
        num_episodes=1,
        is_binocular=False
    )
    
    adapter = MovingBarAdapter(mb_gen)
    
    # Simulate what Cricket2RGCs.__getitem__ does
    syn_movie, path, path_bg, *rest = adapter.generate()
    
    print(f"Unpacked successfully: syn_movie {syn_movie.shape}, path {path.shape}, path_bg {path_bg.shape}")
    print(f"Rest contains: {len(rest)} additional items")
    
    # Test the binocular detection logic from Cricket2RGCs
    is_binocular = (syn_movie.ndim == 4 and syn_movie.shape[1] == 2)
    print(f"Detected as binocular: {is_binocular}")
    
    if is_binocular:
        movies = [syn_movie[:, i] for i in range(2)]
    else:
        mono = syn_movie[:, 0] if syn_movie.ndim == 4 else syn_movie
        movies = [mono]
        syn_movie = syn_movie[:, 0, :, :] 
    
    print(f"Number of movies extracted: {len(movies)}")
    print(f"Movie shape: {movies[0].shape}")
    
    # Verify the movie shape is what Cricket2RGCs expects: (T, H, W)
    assert movies[0].ndim == 3, f"Expected 3D movie (T,H,W), got {movies[0].ndim}D"
    T, H, W = movies[0].shape
    print(f"Movie dimensions: T={T}, H={H}, W={W}")
    
    print("✓ Cricket2RGCs interface compatibility test passed!")
    
    return True

def main():
    """Run all adapter tests."""
    print("Testing MovingBarAdapter compatibility with Cricket2RGCs...")
    
    try:
        success = True
        success &= test_adapter_basic()
        success &= test_adapter_with_cricket2rgcs_interface()
        
        if success:
            print("\n🎉 All adapter tests passed! MovingBarAdapter is compatible with Cricket2RGCs.")
        else:
            print("\n❌ Some adapter tests failed.")
            
        return success
        
    except Exception as e:
        print(f"\n❌ Adapter test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
