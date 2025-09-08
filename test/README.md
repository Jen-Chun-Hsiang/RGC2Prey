# MovingBarMovieGenerator Tests

This directory contains comprehensive tests for the `MovingBarMovieGenerator` class.

## Main Test Script

### `test_moving_bar_comprehensive.py`

A comprehensive test script that includes:

- **Basic functionality tests**: Core MovingBarMovieGenerator functionality
- **Parameter variation tests**: Different configurations and edge cases  
- **Binocular mode tests**: Detailed testing of binocular disparity implementation
- **Text-based visual tests**: Text visualizations that work without matplotlib
- **Full visualizations**: Plots, animations, and detailed visual output (requires matplotlib)

## Usage

### Basic Testing (No Visualization Dependencies)
```bash
cd test
python test_moving_bar_comprehensive.py --no-viz
```

### Basic Tests Only
```bash
cd test  
python test_moving_bar_comprehensive.py --basic-only
```

### Full Testing with Visualizations (Requires matplotlib, cv2)
```bash
cd test
python test_moving_bar_comprehensive.py
```

## Features

### Always Available (No Dependencies)
- Basic functionality validation
- Parameter variation testing
- Comprehensive binocular mode testing
- Text-based frame visualizations
- Numerical data validation

### With Visualization Dependencies
- Static summary plots showing key frames and trajectories
- Individual frame images saved as PNG files
- Animated GIF files
- Binocular comparison visualizations
- Matplotlib-based plots and analysis

## Output

The script creates a `test_results` directory containing:

- `binocular_frames_text.txt`: Text-based visualization of binocular frames
- `mono_vs_bino_comparison.txt`: Text comparison between monocular and binocular modes
- `*.npy` files: Numerical data for further analysis
- `moving_bar_episode_*_summary.png`: Summary visualizations (if matplotlib available)
- `episode_*_frames/`: Individual frame images (if matplotlib available)
- `moving_bar_episode_*_animation.gif`: Animated sequences (if matplotlib available)
- `binocular_comparison.png`: Side-by-side binocular comparison (if matplotlib available)

## Binocular Disparity Testing

The comprehensive test includes detailed validation of the binocular implementation:

1. **Horizontal shift verification**: Ensures left/right eye positions differ by exactly the disparity amount
2. **Vertical alignment**: Confirms vertical positions are identical between eyes
3. **Center position accuracy**: Validates that the average of left/right positions equals the reported center
4. **Frame difference verification**: Confirms left and right eye frames are actually different
5. **Metadata completeness**: Checks all binocular-specific metadata is present

## Requirements

### Minimal (Always Required)
- numpy
- MovingBarMovieGenerator class

### Optional (For Full Visualization)
- matplotlib
- cv2 (opencv-python)
- pillow (for GIF creation)

The script automatically detects available dependencies and adjusts functionality accordingly.
