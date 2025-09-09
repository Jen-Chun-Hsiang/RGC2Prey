# MovingBarAdapter for Cricket2RGCs

The `MovingBarAdapter` allows you to use `MovingBarMovieGenerator` as a drop-in replacement for cricket movie generators in the `Cricket2RGCs` dataset class.

## Problem Solved

`MovingBarMovieGenerator` and `Cricket2RGCs` have incompatible interfaces:

- **MovingBarMovieGenerator**: Returns `dict` with `"episodes"` containing frames shaped `(H, W, T)` or `(H, W, 2, T)` as numpy arrays
- **Cricket2RGCs**: Expects `generate()` to return tuple `(syn_movie, path, path_bg, ...)` where `syn_movie` is torch tensor shaped `(T, C, H, W)`

## Usage

### Basic Example (Monocular)

```python
from datasets.movie_generator import MovingBarMovieGenerator
from datasets.moving_bar_adapter import MovingBarAdapter
from datasets.sim_cricket import Cricket2RGCs

# Create moving bar generator
mb_gen = MovingBarMovieGenerator(
    crop_size=(320, 240),
    boundary_size=(220, 140),
    bar_width_range=(15, 35),
    bar_height_range=(60, 120),
    speed_range=(3.0, 7.0),  # pixels/frame
    direction_range=(0.0, 360.0),
    num_episodes=1,
    is_binocular=False
)

# Wrap with adapter
movie_generator = MovingBarAdapter(mb_gen)

# Use with Cricket2RGCs (replace cricket movie generator)
dataset = Cricket2RGCs(
    num_samples=100,
    multi_opt_sf=multi_opt_sf,
    tf=tf,
    map_func=map_func,
    grid2value_mapping=grid2value_mapping,
    target_width=target_width,
    target_height=target_height,
    movie_generator=movie_generator,  # <-- Adapted moving bar generator
    # ... other parameters
)
```

### Binocular Example

```python
# Enable binocular moving bars
mb_gen = MovingBarMovieGenerator(
    crop_size=(320, 240),
    boundary_size=(220, 140),
    bar_width_range=(20, 30),
    bar_height_range=(80, 120),
    speed_range=(4.0, 6.0),
    is_binocular=True,           # <-- Enable binocular
    fix_disparity=2.0,           # Fixed disparity in pixels
    interocular_dist=1.0,        # Interocular distance in cm
    num_episodes=1
)

movie_generator = MovingBarAdapter(mb_gen)

# Cricket2RGCs automatically handles binocular input
dataset = Cricket2RGCs(
    # ... parameters same as above
    movie_generator=movie_generator
)
```

## What the Adapter Does

1. **Format Conversion**: 
   - Extracts frames from MovingBar episodes dict
   - Converts numpy arrays to torch tensors
   - Reshapes from `(H, W, T)` → `(T, 1, H, W)` or `(H, W, 2, T)` → `(T, 2, H, W)`

2. **Path Alignment**:
   - Ensures path and path_bg have same time length as frames
   - Converts to numpy arrays with shape `(T, 2)`
   - Pads or trims as needed

3. **Value Normalization**:
   - Ensures values are in `[0, 1]` range
   - Converts to `float32` dtype

4. **Interface Compatibility**:
   - Returns tuple `(syn_movie, path, path_bg, meta)` as expected by Cricket2RGCs

## Frame Rate Considerations

`MovingBarMovieGenerator` treats speed as pixels/frame with no explicit frame rate. If you need real-time semantics:

1. **Document units**: Speed is in pixels/frame
2. **Or add fps conversion**: Modify MovingBarMovieGenerator to accept fps parameter and convert speeds from pixels/second to pixels/frame

## Testing

Run the test suite to verify compatibility:

```bash
python test/test_moving_bar_adapter.py
```

Run the example to see it in action:

```bash
python examples/moving_bar_cricket2rgcs_example.py
```

## Files Created

- `datasets/moving_bar_adapter.py` - The adapter class
- `test/test_moving_bar_adapter.py` - Test suite
- `examples/moving_bar_cricket2rgcs_example.py` - Usage examples
- `docs/moving_bar_adapter.md` - This documentation

## Benefits

- **Non-invasive**: No changes needed to existing MovingBarMovieGenerator or Cricket2RGCs classes
- **Drop-in replacement**: Simply replace `movie_generator=cricket_gen` with `movie_generator=MovingBarAdapter(mb_gen)`
- **Full compatibility**: Supports both monocular and binocular modes
- **Maintains metadata**: Preserves moving bar parameters in the output
