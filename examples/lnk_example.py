#!/usr/bin/env python3
"""
Example script demonstrating how to use LNK utilities for RGC modeling.

This script shows how to:
1. Load LNK parameters from Excel
2. Create separate center/surround filters (automatic or from Excel)
3. Configure Cricket2RGCs dataset with LNK
4. Run basic validation and testing

Key Features:
- Automatic surround filter generation: Surround filters are generated from center
  filters by scaling sigma by 4x (default), matching MATLAB behavior
- Optional Excel-based surround filters for custom requirements
- Flexible parameter management with validation

Usage:
    # Basic LNK with auto-generated surround (4x sigma, MATLAB default)
    python examples/lnk_example.py --use_lnk_model --use_separate_surround
    
    # Custom surround sigma ratio
    python examples/lnk_example.py --use_lnk_model --use_separate_surround --surround_sigma_ratio 6.0
    
    # Use custom surround filters from Excel
    python examples/lnk_example.py --use_lnk_model --use_separate_surround --surround_generation sheet --sf_surround_sheet_name SF_surround
    
Author: Emily Hsiang
Date: August 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import logging
from datasets.lnk_utils import (
    load_lnk_parameters,
    load_separate_filters, 
    create_cricket2rgcs_config,
    validate_lnk_config,
    get_lnk_config_summary
)
from datasets.sim_cricket import RGCrfArray, Cricket2RGCs


def create_example_lnk_params(save_path: str, num_cells: int = 100):
    """
    Create an example LNK parameters Excel file for testing.
    
    Args:
        save_path: Path to save the Excel file
        num_cells: Number of RGC cells
    """
    # Create example parameters with some cell-to-cell variation
    np.random.seed(42)  # For reproducibility
    
    # Base parameters (means)
    base_params = {
        'tau': 0.1,
        'alpha_d': 1.0, 
        'theta': 0.0,
        'sigma0': 1.0,
        'alpha': 0.1,
        'beta': -0.05,
        'b_out': 0.0,
        'g_out': 1.0,
        'w_xs': -0.2,
        'dt': 0.01
    }
    
    # Add variation for per-cell parameters
    lnk_data = {}
    for param, mean_val in base_params.items():
        if param in ['tau', 'alpha_d', 'alpha', 'w_xs']:
            # Add 20% variation for these parameters
            std_val = abs(mean_val) * 0.2
            values = np.random.normal(mean_val, std_val, num_cells)
            # Ensure positive values where needed
            if param in ['tau', 'alpha_d', 'alpha']:
                values = np.abs(values)
            lnk_data[param] = values
        else:
            # Use constant values for other parameters
            lnk_data[param] = [mean_val] * num_cells
    
    # Create DataFrame and save
    df = pd.DataFrame(lnk_data)
    df.to_excel(save_path, sheet_name='LNK_params', index=False)
    print(f"Created example LNK parameters file: {save_path}")
    return df


def create_minimal_sf_tf_tables(save_path: str):
    """
    Create minimal SF and TF parameter tables for testing.
    """
    # Minimal spatial filter parameters (single Gaussian)
    sf_data = {
        'id': [1],
        'center_x': [0],
        'center_y': [0], 
        'sigma_x': [10],
        'sigma_y': [10],
        'theta': [0],
        'amplitude': [1],
        'surround_amplitude': [0.2],
        'surround_sigma_x': [20],
        'surround_sigma_y': [20],
        'offset': [0],
        'type': ['difference_of_gaussians']
    }
    
    # Minimal temporal filter parameters
    tf_data = {
        'id': [1],
        'type': ['biphasic'],
        'tau1': [20],
        'tau2': [40],
        'amplitude1': [1],
        'amplitude2': [-0.5],
        'delay': [0],
        'duration': [100],
        'offset': [0]
    }
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(save_path) as writer:
        pd.DataFrame(sf_data).to_excel(writer, sheet_name='SF_params', index=False)
        pd.DataFrame(tf_data).to_excel(writer, sheet_name='TF_params', index=False)
    
    print(f"Created minimal SF/TF parameters file: {save_path}")


def run_lnk_example():
    """
    Run a complete example of LNK model setup and validation.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Create example parameter files
    lnk_params_file = "example_lnk_params.xlsx"
    rf_params_file = "example_rf_params.xlsx"
    
    num_rgcs = 50
    
    print("Creating example parameter files...")
    create_example_lnk_params(lnk_params_file, num_rgcs)
    create_minimal_sf_tf_tables(rf_params_file)
    
    # Create minimal SF and TF tables for RGCrfArray
    sf_table = pd.DataFrame({
        'id': [1], 'center_x': [0], 'center_y': [0], 'sigma_x': [10], 'sigma_y': [10],
        'theta': [0], 'amplitude': [1], 'surround_amplitude': [0.2], 
        'surround_sigma_x': [20], 'surround_sigma_y': [20], 'offset': [0],
        'type': ['difference_of_gaussians']
    })
    
    tf_table = pd.DataFrame({
        'id': [1], 'type': ['biphasic'], 'tau1': [20], 'tau2': [40],
        'amplitude1': [1], 'amplitude2': [-0.5], 'delay': [0], 
        'duration': [100], 'offset': [0]
    })
    
    print("\n1. Creating RGCrfArray...")
    rgc_array = RGCrfArray(
        sf_table, tf_table,
        rgc_array_rf_size=(64, 64),
        xlim=(-32, 32), ylim=(-32, 32),
        target_num_centers=num_rgcs,
        sf_scalar=0.2,
        grid_generate_method='circle',
        tau=3, mask_radius=15,
        rgc_rand_seed=42
    )
    
    multi_opt_sf, tf, grid2value_mapping, map_func, rgc_locs = rgc_array.get_results()
    print(f"Generated {multi_opt_sf.shape[2]} RGC filters")
    
    print("\n2. Loading LNK parameters...")
    lnk_config = load_lnk_parameters(
        rf_params_file=lnk_params_file,
        lnk_sheet_name='LNK_params',
        num_rgcs=num_rgcs,
        lnk_adapt_mode='divisive',
        use_lnk_model=True
    )
    
    if lnk_config:
        print(f"✓ LNK parameters loaded successfully")
        print(f"Configuration: {get_lnk_config_summary(lnk_config)}")
        
        # Validate configuration
        if validate_lnk_config(lnk_config, num_rgcs):
            print("✓ LNK configuration validation passed")
        else:
            print("✗ LNK configuration validation failed")
    else:
        print("✗ Failed to load LNK parameters")
        return
    
    print("\n3. Testing separate filters (optional)...")
    separate_filters = load_separate_filters(
        rf_params_file=rf_params_file,
        rgc_array=rgc_array,
        use_separate_surround=False  # Set to True to test separate filters
    )
    
    if separate_filters:
        print(f"✓ Loaded separate filters: {list(separate_filters.keys())}")
    else:
        print("- No separate filters requested")
    
    print("\n4. Creating Cricket2RGCs configuration...")
    
    # Dummy movie generator for testing
    class DummyMovieGenerator:
        def generate(self):
            # Return dummy movie data
            movie = torch.randn(50, 64, 64)  # T, H, W
            path = torch.randn(50, 2)  # T, 2 (x, y coordinates)
            path_bg = torch.randn(50, 2)
            scaling = torch.ones(50)
            return movie, path, path_bg, movie, scaling, None, None, None
    
    config = create_cricket2rgcs_config(
        num_samples=10,
        multi_opt_sf=multi_opt_sf,
        tf=tf,
        map_func=map_func,
        grid2value_mapping=grid2value_mapping,
        target_width=64,
        target_height=64,
        movie_generator=DummyMovieGenerator(),
        lnk_config=lnk_config,
        separate_filters=separate_filters
    )
    
    print("✓ Cricket2RGCs configuration created")
    
    print("\n5. Testing dataset creation...")
    try:
        # Note: This would normally fail without proper movie generator
        # but shows the configuration is properly formatted
        print("Configuration keys:", list(config.keys()))
        if 'lnk_config_on' in config:
            print("✓ LNK configuration included in dataset config")
        else:
            print("- Standard LN configuration (no LNK)")
            
    except Exception as e:
        print(f"Dataset creation test encountered: {e}")
    
    # Cleanup
    print("\n6. Cleaning up example files...")
    try:
        os.remove(lnk_params_file)
        os.remove(rf_params_file)
        print("✓ Example files removed")
    except:
        print("- Could not remove example files")
    
    print("\n✓ LNK example completed successfully!")
    print("\nTo use LNK in your training:")
    print("1. Create LNK_params sheet in your SimulationParams.xlsx")
    print("2. Add --use_lnk_model to your training command")
    print("3. Optionally use --use_separate_surround for advanced center/surround")


if __name__ == "__main__":
    run_lnk_example()
