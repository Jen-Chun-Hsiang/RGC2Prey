# LNK vs LN Model Comparison Example

This example demonstrates the differences between LN and LNK models in the RGC2Prey framework.

## Quick Start Examples

### Example 1: Basic LN Model (Default)
```bash
# Run with standard LN model
python train_preycapture.py \
  --experiment_name "ln_baseline" \
  --num_samples 100 \
  --batch_size 4 \
  --num_epochs 10
```

### Example 2: Basic LNK Model 
```bash
# Run with LNK model using default parameters
python train_preycapture.py \
  --experiment_name "lnk_basic" \
  --use_lnk_model \
  --lnk_sheet_name "LNK_params" \
  --num_samples 100 \
  --batch_size 4 \
  --num_epochs 10
```

### Example 3: Advanced LNK with Separate Surrounds
```bash
# LNK with automatically generated surround filters
python train_preycapture.py \
  --experiment_name "lnk_advanced" \
  --use_lnk_model \
  --lnk_sheet_name "LNK_params" \
  --use_separate_surround \
  --surround_sigma_ratio 4.0 \
  --lnk_adapt_mode "divisive" \
  --num_samples 100
```

## Parameter File Setup

### Create LNK Parameters Excel Sheet

Add a new sheet called "LNK_params" to your `SimulationParams.xlsx` file:

| tau | alpha_d | theta | sigma0 | alpha | beta | b_out | g_out | w_xs | dt   |
|-----|---------|-------|--------|-------|------|-------|-------|------|------|
| 0.1 | 1.0     | 0.0   | 1.0    | 0.1   | 0.0  | 0.0   | 1.0   | -0.1 | 0.01 |

**Parameter Meanings:**
- `tau`: Adaptation time constant (seconds)
- `alpha_d`: Drive strength for adaptation  
- `theta`: Threshold for adaptation drive
- `sigma0`: Baseline normalization factor
- `alpha`: Coupling between adaptation and normalization
- `beta`: Additive adaptation term
- `b_out`: Output bias
- `g_out`: Output gain
- `w_xs`: Center-surround interaction weight (negative = suppressive)
- `dt`: Time step (seconds)

## Comparison Script

Here's a Python script to compare LN vs LNK model responses:

```python
#!/usr/bin/env python3
"""
Compare LN vs LNK model responses for the same stimulus.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.sim_cricket import Cricket2RGCs, RGCrfArray, SynMovieGenerator
from datasets.lnk_utils import create_cricket2rgcs_config, load_unified_parameters
import pandas as pd

def create_test_parameters():
    """Create test LNK parameters."""
    lnk_params = {
        'tau': [0.1],        # Fast adaptation
        'alpha_d': [1.0],    # Standard drive
        'theta': [0.0],      # No threshold
        'sigma0': [1.0],     # Baseline normalization
        'alpha': [0.1],      # Weak adaptation coupling
        'beta': [0.0],       # No additive adaptation
        'b_out': [0.0],      # No output bias
        'g_out': [1.0],      # Unity gain
        'w_xs': [-0.1],      # Weak suppressive surround
        'dt': [0.01]         # 10ms time step
    }
    return pd.DataFrame(lnk_params)

def compare_models(num_samples=5):
    """Compare LN vs LNK model responses."""
    
    # Setup common parameters
    rf_params_file = 'SimulationParams.xlsx'  # Update path as needed
    
    # Load spatial and temporal filter parameters
    param_tables = load_unified_parameters(
        rf_params_file=rf_params_file,
        sf_sheet_name='SF_params_modified',
        tf_sheet_name='TF_params',
        num_rgcs=500,
        optional_sheets={'lnk': 'LNK_params'}
    )
    
    # Create RGC array
    rgc_array = RGCrfArray(
        param_tables['sf'], param_tables['tf'],
        rgc_array_rf_size=(320, 240),
        xlim=(-120, 120), ylim=(-90, 90),
        target_num_centers=500,
        sf_scalar=0.2,
        grid_generate_method='circle',
        tau=3, mask_radius=30,
        rgc_rand_seed=42
    )
    
    multi_opt_sf, tf, grid2value_mapping, map_func, rgc_locs = rgc_array.get_results()
    
    # Create movie generator
    movie_generator = SynMovieGenerator(
        top_img_folder='datasets/cricketimages/',
        bottom_img_folder='single-contrast/',  # Update paths as needed
        crop_size=(320, 240),
        boundary_size=(220, 140),
        center_ratio=(0.2, 0.2),
        max_steps=200
    )
    
    # Common configuration
    base_config = {
        'num_samples': num_samples,
        'multi_opt_sf': multi_opt_sf,
        'tf': tf,
        'map_func': map_func,
        'grid2value_mapping': grid2value_mapping,
        'target_width': 240,
        'target_height': 180,
        'movie_generator': movie_generator,
        'grid_size_fac': 1.0,
        'is_norm_coords': False,
        'is_syn_mov_shown': True,
        'fr2spikes': False,
        'add_noise': False,
        'smooth_data': False,
        'is_rectified': True,
        'is_direct_image': False,
    }
    
    # Create LN dataset (no LNK config)
    ln_config = create_cricket2rgcs_config(**base_config)
    ln_dataset = Cricket2RGCs(**ln_config)
    
    # Create LNK dataset
    if param_tables.get('lnk') is not None:
        from datasets.lnk_utils import process_parameter_table
        lnk_params = process_parameter_table(param_tables['lnk'], 500, 'lnk')
        lnk_config_dict = {
            'lnk_params': lnk_params,
            'adapt_mode': 'divisive'
        }
    else:
        # Use test parameters if no Excel sheet
        lnk_params = {
            'tau': 0.1, 'alpha_d': 1.0, 'theta': 0.0,
            'sigma0': 1.0, 'alpha': 0.1, 'beta': 0.0,
            'b_out': 0.0, 'g_out': 1.0, 'w_xs': -0.1, 'dt': 0.01
        }
        lnk_config_dict = {
            'lnk_params': lnk_params,
            'adapt_mode': 'divisive'
        }
    
    lnk_config = create_cricket2rgcs_config(**base_config, lnk_config=lnk_config_dict)
    lnk_dataset = Cricket2RGCs(**lnk_config)
    
    # Generate and compare responses
    results = []
    
    for i in range(min(num_samples, 3)):  # Limit for visualization
        print(f"Processing sample {i+1}/{num_samples}")
        
        # Get responses from both models
        ln_sequence, ln_path, _, ln_movie, *_ = ln_dataset[i]
        lnk_sequence, lnk_path, _, lnk_movie, *_ = lnk_dataset[i]
        
        results.append({
            'sample': i,
            'ln_sequence': ln_sequence,
            'lnk_sequence': lnk_sequence,
            'path': ln_path,  # Same path for both
            'movie': ln_movie  # Same movie for both
        })
    
    return results

def plot_comparison(results, save_folder='./'):
    """Plot comparison between LN and LNK responses."""
    
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 4*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Plot RGC responses
        axes[i, 0].plot(result['ln_sequence'][0, :50, 0].numpy(), 'b-', label='LN Model', alpha=0.7)
        axes[i, 0].plot(result['lnk_sequence'][0, :50, 0].numpy(), 'r-', label='LNK Model', alpha=0.7)
        axes[i, 0].set_title(f'Sample {result["sample"]+1}: First RGC Response')
        axes[i, 0].set_xlabel('Time')
        axes[i, 0].set_ylabel('Response')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot trajectory
        path = result['path']
        axes[i, 1].plot(path[:, 0], path[:, 1], 'k-', alpha=0.7, linewidth=2)
        axes[i, 1].scatter(path[0, 0], path[0, 1], c='green', s=100, marker='o', label='Start')
        axes[i, 1].scatter(path[-1, 0], path[-1, 1], c='red', s=100, marker='x', label='End')
        axes[i, 1].set_title(f'Cricket Trajectory')
        axes[i, 1].set_xlabel('X Position')
        axes[i, 1].set_ylabel('Y Position')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # Plot response statistics
        ln_mean = result['ln_sequence'].mean(dim=1).numpy()
        lnk_mean = result['lnk_sequence'].mean(dim=1).numpy()
        
        axes[i, 2].scatter(ln_mean, lnk_mean, alpha=0.6, s=20)
        axes[i, 2].plot([0, max(ln_mean.max(), lnk_mean.max())], 
                       [0, max(ln_mean.max(), lnk_mean.max())], 'k--', alpha=0.5)
        axes[i, 2].set_title('LN vs LNK Mean Response')
        axes[i, 2].set_xlabel('LN Model Response')
        axes[i, 2].set_ylabel('LNK Model Response')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_folder}/ln_vs_lnk_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_differences(results):
    """Analyze key differences between LN and LNK responses."""
    
    print("\n" + "="*50)
    print("LN vs LNK MODEL ANALYSIS")
    print("="*50)
    
    for i, result in enumerate(results):
        ln_seq = result['ln_sequence']
        lnk_seq = result['lnk_sequence']
        
        print(f"\nSample {i+1}:")
        print(f"  LN  - Mean: {ln_seq.mean():.4f}, Std: {ln_seq.std():.4f}, Max: {ln_seq.max():.4f}")
        print(f"  LNK - Mean: {lnk_seq.mean():.4f}, Std: {lnk_seq.std():.4f}, Max: {lnk_seq.max():.4f}")
        
        # Calculate correlation
        correlation = torch.corrcoef(torch.stack([
            ln_seq.flatten(), 
            lnk_seq.flatten()
        ]))[0, 1]
        print(f"  Correlation: {correlation:.4f}")
        
        # Calculate temporal dynamics
        ln_temporal_var = ln_seq.var(dim=1).mean()
        lnk_temporal_var = lnk_seq.var(dim=1).mean()
        print(f"  Temporal variance - LN: {ln_temporal_var:.4f}, LNK: {lnk_temporal_var:.4f}")

if __name__ == "__main__":
    print("Comparing LN vs LNK models...")
    
    # Run comparison
    results = compare_models(num_samples=3)
    
    # Plot results
    plot_comparison(results)
    
    # Analyze differences
    analyze_differences(results)
    
    print("\nComparison complete! Check 'ln_vs_lnk_comparison.png' for visualization.")
```

## Expected Differences

### 1. **Temporal Dynamics**
- **LN Model**: Static responses, no adaptation
- **LNK Model**: Dynamic adaptation, responses decrease for sustained stimuli

### 2. **Response Magnitude**
- **LN Model**: Can maintain high firing rates indefinitely
- **LNK Model**: Adapts down from high initial responses

### 3. **Center-Surround Interaction**
- **LN Model**: Static surround suppression via `s_scale`
- **LNK Model**: Dynamic surround interaction via `w_xs`

### 4. **Computational Cost**
- **LN Model**: ~1x baseline (fastest)
- **LNK Model**: ~2-3x baseline (iterative adaptation computation)

## Key Insights for Your Research

### When LNK Shows Clear Advantages:
1. **Moving stimuli** (prey capture scenarios)
2. **Sustained stimuli** (stationary prey)
3. **Contrast changes** (lighting variations)
4. **Natural image statistics** (complex visual scenes)

### When LN Is Sufficient:
1. **Simple stimuli** (flashes, gratings)
2. **Quick prototyping** 
3. **Computational constraints**
4. **Baseline comparisons**

## Troubleshooting Common Issues

### 1. LNK Parameters Not Found
```bash
# Make sure Excel sheet exists and has correct name
--lnk_sheet_name "LNK_params"  # Must match Excel sheet name exactly
```

### 2. Parameter Length Mismatch
```python
# Parameters must match number of RGC cells or be scalars
tau = [0.1] * 500  # For 500 RGC cells
# OR
tau = 0.1  # Single value for all cells
```

### 3. Slow Performance
```bash
# Reduce batch size or sequence length
--batch_size 2  # Instead of 4
--max_steps 100  # Instead of 200
```

This example provides a comprehensive comparison framework to help you understand the practical differences between LN and LNK models in your specific research context.
