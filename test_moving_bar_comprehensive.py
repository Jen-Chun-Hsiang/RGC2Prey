#!/usr/bin/env python3
"""
Comprehensive test script for MovingBarMovieGenerator class.
This script tests basic functionality and optionally creates visualizations.
Can run in basic mode (no visualization dependencies) or full mode (with plots/animations).
"""

import os
import sys
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, '/Users/emilyhsiang/Desktop/Documents/RGC2Prey')

from datasets.movie_generator import MovingBarMovieGenerator

# Check for optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available - visualization features disabled")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Note: cv2 not available - some video features disabled")

VISUALIZATION_AVAILABLE = HAS_MATPLOTLIB


class MovingBarTester:
    """Comprehensive tester for MovingBarMovieGenerator with optional visualization."""
    
    def __init__(self, enable_visualization=True):
        self.enable_viz = enable_visualization and VISUALIZATION_AVAILABLE
        if enable_visualization and not VISUALIZATION_AVAILABLE:
            print("Warning: Visualization requested but dependencies not available")
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("=== MovingBarMovieGenerator Comprehensive Test ===")
        
        success = True
        success &= self.basic_functionality_test()
        success &= self.parameter_variation_test()
        success &= self.binocular_test()
        
        if self.enable_viz:
            success &= self.visualization_test()
        
        return success
    
    def basic_functionality_test(self):
        """Test basic MovingBarMovieGenerator functionality."""
        print("\n=== Basic Functionality Test ===")
        
        # Configuration parameters based on train_preycapture.py usage
        crop_size = (320, 240)  # (width, height)
        boundary_size = (220, 140)  # boundary for movement
        center_ratio = (0.2, 0.2)  # center region ratio
        max_steps = 50  # shorter for quick testing
        num_ext = 5     # fewer extended frames
        
        # Moving bar specific parameters
        bar_width_range = (15, 35)      # width range in pixels
        bar_height_range = (60, 120)    # height range in pixels  
        speed_range = (3.0, 7.0)        # speed range in pixels/frame
        direction_range = (0.0, 360.0)  # direction range in degrees
        num_episodes = 2                # number of episodes to generate
        
        print("Initializing MovingBarMovieGenerator...")
        
        try:
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
                is_binocular=False,
                bottom_contrast=1.0,
                top_contrast=1.0
            )
            
            print("✓ Generator initialized successfully")
            
        except Exception as e:
            print(f"✗ Error initializing generator: {e}")
            return False
        
        print("Generating moving bar episodes...")
        
        try:
            # Generate episodes
            result = movie_generator.generate()
            episodes = result["episodes"]
            
            print(f"✓ Generated {len(episodes)} episodes successfully")
            
        except Exception as e:
            print(f"✗ Error generating episodes: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Inspect the generated data
        for ep_idx, episode in enumerate(episodes):
            frames = episode["frames"]  # Shape: (H, W, T)
            path = episode["path"]      # Shape: (T, 2) - (x, y) positions
            path_bg = episode["path_bg"]  # Should be None for moving bar
            meta = episode["meta"]
            
            print(f"\n--- Episode {ep_idx + 1} ---")
            print(f"Frames shape: {frames.shape}")
            print(f"Path shape: {path.shape}")
            print(f"Path_bg: {path_bg}")
            print(f"Frame data type: {frames.dtype}")
            print(f"Frame value range: [{frames.min():.3f}, {frames.max():.3f}]")
            
            print("Metadata:")
            for key, value in meta.items():
                if isinstance(value, (np.ndarray, list)) and len(str(value)) > 50:
                    print(f"  {key}: {type(value).__name__} shape {getattr(value, 'shape', len(value))}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            
            # Validate basic properties
            H, W, T = frames.shape
            assert H == crop_size[1], f"Height mismatch: {H} != {crop_size[1]}"
            assert W == crop_size[0], f"Width mismatch: {W} != {crop_size[0]}"
            assert len(path) == T, f"Path length mismatch: {len(path)} != {T}"
            
            # Check that path coordinates are reasonable
            assert np.all(path[:, 0] >= -W), "X coordinates too negative"
            assert np.all(path[:, 0] <= 2*W), "X coordinates too positive"
            assert np.all(path[:, 1] >= -H), "Y coordinates too negative"
            assert np.all(path[:, 1] <= 2*H), "Y coordinates too positive"
            
            # Check frame values are in reasonable range
            assert np.all(frames >= 0), "Frame values below 0"
            assert np.all(frames <= 1), "Frame values above 1"
            
            print("✓ Episode validation passed")
        
        print("✓ Basic functionality test completed successfully!")
        return True
    
    def parameter_variation_test(self):
        """Test different parameter configurations to ensure robustness."""
        print("\n=== Parameter Variation Tests ===")
        
        test_configs = [
            {
                "name": "Minimal configuration",
                "crop_size": (160, 120),
                "boundary_size": (100, 80),
                "bar_width_range": (10, 20),
                "bar_height_range": (40, 80),
                "speed_range": (2.0, 4.0),
                "num_episodes": 1
            },
            {
                "name": "Large configuration",
                "crop_size": (640, 480),
                "boundary_size": (500, 400),
                "bar_width_range": (30, 60),
                "bar_height_range": (100, 200),
                "speed_range": (5.0, 10.0),
                "num_episodes": 1
            },
            {
                "name": "Constrained direction",
                "crop_size": (320, 240),
                "boundary_size": (220, 140),
                "bar_width_range": (20, 40),
                "bar_height_range": (80, 120),
                "speed_range": (4.0, 6.0),
                "direction_range": (80.0, 100.0),  # Nearly vertical
                "num_episodes": 1
            },
            {
                "name": "Small fast bars",
                "crop_size": (240, 180),
                "boundary_size": (180, 120),
                "bar_width_range": (5, 15),
                "bar_height_range": (20, 60),
                "speed_range": (8.0, 15.0),
                "num_episodes": 1
            },
            {
                "name": "Horizontal movement only",
                "crop_size": (300, 200),
                "boundary_size": (200, 150),
                "bar_width_range": (20, 40),
                "bar_height_range": (60, 120),
                "speed_range": (4.0, 8.0),
                "direction_range": (-10.0, 10.0),  # Nearly horizontal
                "num_episodes": 1
            }
        ]
        
        for config in test_configs:
            print(f"\nTesting: {config['name']}")
            
            try:
                movie_generator = MovingBarMovieGenerator(**config)
                result = movie_generator.generate()
                episode = result["episodes"][0]
                
                frames = episode["frames"]
                meta = episode["meta"]
                
                print(f"  ✓ Generated {frames.shape[2]} frames")
                print(f"  ✓ Bar dimensions: {meta['bar_width']:.1f} x {meta['bar_height']:.1f}")
                print(f"  ✓ Speed: {meta['speed']:.1f}, Direction: {meta['move_dir']:.1f}°")
                
            except Exception as e:
                print(f"  ✗ Error with {config['name']}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("✓ All parameter variation tests passed")
        return True
    
    def binocular_test(self):
        """Comprehensive test of binocular mode functionality."""
        print("\n=== Binocular Mode Test ===")
        
        try:
            movie_generator = MovingBarMovieGenerator(
                crop_size=(320, 240),
                boundary_size=(220, 140),
                bar_width_range=(20, 30),
                bar_height_range=(80, 120),
                speed_range=(4.0, 6.0),
                num_episodes=1,
                is_binocular=True,
                interocular_dist=1.0,  # cm
                disparity_range=(1.0, 3.0),  # pixels
                fix_disparity=2.0  # fixed disparity for testing
            )
            
            result = movie_generator.generate()
            episode = result["episodes"][0]
            
            frames = episode["frames"]
            path = episode["path"]
            meta = episode["meta"]
            
            print(f"✓ Binocular frames shape: {frames.shape}")
            print(f"✓ Binocular path shape: {path.shape}")
            
            # Check binocular-specific properties
            if frames.ndim == 4:  # (H, W, 2_eyes, T)
                H, W, n_eyes, T = frames.shape
                assert n_eyes == 2, f"Expected 2 eyes, got {n_eyes}"
                print(f"✓ Correct binocular format: (H={H}, W={W}, eyes={n_eyes}, T={T})")
                
                # Check that left and right eye frames are different (due to disparity)
                left_frames = frames[:, :, 0, :]
                right_frames = frames[:, :, 1, :]
                
                # They should be different due to horizontal shift
                are_different = not np.array_equal(left_frames, right_frames)
                print(f"✓ Left and right eye frames are different: {are_different}")
                
                # Check metadata contains binocular info
                assert 'is_binocular' in meta and meta['is_binocular'], "Missing binocular flag in metadata"
                assert 'disparity' in meta, "Missing disparity in metadata"
                assert 'left_positions' in meta, "Missing left_positions in metadata"
                assert 'right_positions' in meta, "Missing right_positions in metadata"
                
                left_pos = meta['left_positions']
                right_pos = meta['right_positions']
                disparity = meta['disparity']
                
                print(f"✓ Disparity values: min={disparity.min():.2f}, max={disparity.max():.2f}")
                
                # Verify horizontal shift pattern
                # Left should be center - disparity/2, right should be center + disparity/2
                expected_diff = disparity[0]  # use first frame disparity
                actual_diff = right_pos[0, 0] - left_pos[0, 0]  # horizontal difference
                
                print(f"✓ Expected horizontal difference: {expected_diff:.2f}")
                print(f"✓ Actual horizontal difference: {actual_diff:.2f}")
                
                diff_error = abs(actual_diff - expected_diff)
                assert diff_error < 0.01, f"Horizontal difference error too large: {diff_error}"
                
                # Check that vertical positions are the same
                vertical_diff = np.abs(right_pos[:, 1] - left_pos[:, 1]).max()
                print(f"✓ Max vertical difference (should be ~0): {vertical_diff:.6f}")
                assert vertical_diff < 1e-6, f"Vertical positions should be identical"
                
                # Check that average position equals the reported path
                avg_pos = (left_pos + right_pos) / 2.0
                path_diff = np.abs(avg_pos - path).max()
                print(f"✓ Max difference between average and path (should be ~0): {path_diff:.6f}")
                assert path_diff < 1e-6, f"Average position should equal reported path"
                
                # Test with random disparity as well
                print("Testing with random disparity...")
                movie_generator_random = MovingBarMovieGenerator(
                    crop_size=(200, 150),
                    boundary_size=(150, 100),
                    bar_width_range=(15, 15),
                    bar_height_range=(50, 50),
                    speed_range=(3.0, 3.0),
                    num_episodes=1,
                    is_binocular=True,
                    disparity_range=(0.5, 4.0),  # Random disparity
                    fix_disparity=None  # Let it be random
                )
                
                result_random = movie_generator_random.generate()
                episode_random = result_random["episodes"][0]
                meta_random = episode_random["meta"]
                
                print(f"✓ Random disparity: {meta_random['disparity'][0]:.2f}")
                
            else:
                print(f"✗ Unexpected frame shape for binocular: {frames.shape}")
                return False
                
        except Exception as e:
            print(f"✗ Binocular mode test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("✓ Binocular mode test passed!")
        return True
    
    def visualization_test(self):
        """Create visualizations to inspect the MovingBarMovieGenerator functionality."""
        if not self.enable_viz:
            print("\n=== Visualization Test Skipped (dependencies not available) ===")
            return True
            
        print("\n=== Visualization Test ===")
        
        try:
            # Generate episodes for visualization
            movie_generator = MovingBarMovieGenerator(
                crop_size=(320, 240),
                boundary_size=(220, 140),
                bar_width_range=(20, 40),
                bar_height_range=(80, 160),
                speed_range=(3.0, 7.0),
                direction_range=(0.0, 360.0),
                num_episodes=2,
                is_binocular=False
            )
            
            result = movie_generator.generate()
            episodes = result["episodes"]
            
            print(f"Creating visualizations for {len(episodes)} episodes...")
            
            # Create output directory
            os.makedirs('test_results', exist_ok=True)
            
            for ep_idx, episode in enumerate(episodes):
                frames = episode["frames"]
                path = episode["path"]
                meta = episode["meta"]
                
                print(f"Processing episode {ep_idx + 1}...")
                
                # Create summary visualization
                self._create_episode_summary(frames, path, meta, ep_idx)
                
                # Save some frames
                self._save_sample_frames(frames, path, meta, ep_idx)
                
                # Create animation if requested
                if ep_idx == 0:  # Only create animation for first episode
                    self._create_animation(frames, path, meta, ep_idx)
            
            # Test binocular visualization
            self._test_binocular_visualization()
            
            print("✓ Visualization test completed!")
            print("Check the 'test_results' directory for output files")
            
        except Exception as e:
            print(f"✗ Visualization test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def _create_episode_summary(self, frames, path, meta, episode_idx):
        """Create a static visualization showing key frames and trajectory."""
        H, W, T = frames.shape
        
        # Select key frames to display
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
        # Remove individual subplot axes
        for i in range(1, len(key_frame_indices)):
            fig.delaxes(axes[1, i])
        
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
        plt.savefig(f'test_results/moving_bar_episode_{episode_idx + 1}_summary.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_sample_frames(self, frames, path, meta, episode_idx, save_every=10):
        """Save sample frames as PNG images."""
        H, W, T = frames.shape
        os.makedirs(f'test_results/episode_{episode_idx + 1}_frames', exist_ok=True)
        
        # Save only a few frames to avoid too many files
        frame_indices = range(0, T, save_every)
        
        for t in frame_indices:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Display frame
            im = ax.imshow(frames[:, :, t], cmap='gray', vmin=0, vmax=1)
            
            # Mark bar center
            if t < len(path):
                x, y = path[t]
                ax.plot(x, y, 'r+', markersize=12, markeredgewidth=3)
            
            ax.set_title(f'Episode {episode_idx + 1}, Frame {t}\n'
                        f'Bar center: ({path[t, 0]:.1f}, {path[t, 1]:.1f})')
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            plt.tight_layout()
            plt.savefig(f'test_results/episode_{episode_idx + 1}_frames/frame_{t:03d}.png', 
                       dpi=100, bbox_inches='tight')
            plt.close()
    
    def _create_animation(self, frames, path, meta, episode_idx):
        """Create an animated GIF showing the moving bar."""
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
        
        # Create animation (subsample for smaller file)
        subsample = max(1, T // 30)  # Limit to ~30 frames
        frame_indices = list(range(0, T, subsample))
        
        anim = animation.FuncAnimation(fig, animate, frames=frame_indices, interval=100, 
                                     blit=True, repeat=True)
        
        # Save as GIF
        anim.save(f'test_results/moving_bar_episode_{episode_idx + 1}_animation.gif', 
                  writer='pillow', fps=5)
        
        plt.close()
        print(f"Animation saved: test_results/moving_bar_episode_{episode_idx + 1}_animation.gif")
    
    def _test_binocular_visualization(self):
        """Create a visualization comparing monocular and binocular modes."""
        print("Creating binocular comparison visualization...")
        
        # Generate binocular episode
        bino_gen = MovingBarMovieGenerator(
            crop_size=(200, 150),
            boundary_size=(150, 100),
            bar_width_range=(15, 15),
            bar_height_range=(60, 60),
            speed_range=(2.0, 2.0),
            direction_range=(30.0, 30.0),
            num_episodes=1,
            is_binocular=True,
            fix_disparity=3.0
        )
        
        result = bino_gen.generate()
        episode = result["episodes"][0]
        
        frames = episode["frames"]  # (H, W, 2, T)
        meta = episode["meta"]
        
        # Create comparison plot
        H, W, n_eyes, T = frames.shape
        mid_frame = T // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Binocular Moving Bar Comparison\n'
                    f'Disparity: {meta["disparity"][0]:.1f} pixels', fontsize=14)
        
        # Left eye
        axes[0].imshow(frames[:, :, 0, mid_frame], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Left Eye')
        axes[0].axis('off')
        left_pos = meta['left_positions'][mid_frame]
        axes[0].plot(left_pos[0], left_pos[1], 'r+', markersize=12, markeredgewidth=3)
        
        # Right eye
        axes[1].imshow(frames[:, :, 1, mid_frame], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Right Eye')
        axes[1].axis('off')
        right_pos = meta['right_positions'][mid_frame]
        axes[1].plot(right_pos[0], right_pos[1], 'r+', markersize=12, markeredgewidth=3)
        
        # Difference (highlight disparity)
        diff = np.abs(frames[:, :, 1, mid_frame] - frames[:, :, 0, mid_frame])
        im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title('Difference (Right - Left)')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('test_results/binocular_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run tests with optional visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MovingBarMovieGenerator')
    parser.add_argument('--no-viz', action='store_true', 
                       help='Disable visualization features (useful when matplotlib unavailable)')
    parser.add_argument('--basic-only', action='store_true',
                       help='Run only basic functionality tests')
    
    args = parser.parse_args()
    
    enable_viz = not args.no_viz
    
    # Create tester
    tester = MovingBarTester(enable_visualization=enable_viz)
    
    if args.basic_only:
        success = tester.basic_functionality_test()
    else:
        success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! MovingBarMovieGenerator is working correctly.")
        if enable_viz and VISUALIZATION_AVAILABLE:
            print("Check the 'test_results' directory for visualization outputs.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    return success


if __name__ == "__main__":
    main()
