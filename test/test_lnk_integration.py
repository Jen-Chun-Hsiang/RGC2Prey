#!/usr/bin/env python3
"""
Test script to verify the new LNK model integration.
This script tests that compute_lnk_response_from_convolved works correctly.
"""

import numpy as np
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Conditional import for torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. New LNK model will be disabled.")

from utils.dynamic_fitting import LNKParams

if TORCH_AVAILABLE:
    from datasets.simple_lnk import compute_lnk_response_from_convolved

def lnk_params_to_dict(params: LNKParams, dt: float) -> dict:
    """Convert LNKParams dataclass to dictionary format."""
    return {
        'tau': np.array([params.tau]),
        'alpha_d': np.array([params.alpha_d]),
        'theta': np.array([params.theta]),
        'sigma0': np.array([params.sigma0]),
        'alpha': np.array([params.alpha]),
        'beta': np.array([params.beta]),
        'b_out': np.array([params.b_out]),
        'g_out': np.array([params.g_out]),
        'w_xs': np.array([params.w_xs]),
        'dt': np.array([dt])
    }

if TORCH_AVAILABLE:
    def compute_lnk_new(x_center: np.ndarray, x_surround: np.ndarray, params: LNKParams, dt: float) -> np.ndarray:
        """Wrapper function for compute_lnk_response_from_convolved."""
        device = torch.device('cpu')
        dtype = torch.float32
        
        # Convert to torch tensors [N=1, T]
        x_center_torch = torch.from_numpy(x_center).unsqueeze(0).to(device=device, dtype=dtype)
        x_surround_torch = torch.from_numpy(x_surround).unsqueeze(0).to(device=device, dtype=dtype)
        
        # Convert parameters
        lnk_params_dict = lnk_params_to_dict(params, dt)
        
        # Compute response
        rgc_response = compute_lnk_response_from_convolved(
            x_center_torch, x_surround_torch, lnk_params_dict, device, dtype
        )
        
        return rgc_response.squeeze(0).detach().cpu().numpy()

def test_new_lnk_model():
    """Test the new LNK model implementation."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot test new LNK model.")
        return
    
    print("Testing New LNK Model Integration")
    print("=" * 40)
    
    # Create synthetic data
    np.random.seed(42)
    T = 1000
    dt = 0.001
    
    t = np.arange(T) * dt
    x_center = 2 * np.sin(2 * np.pi * t) + np.random.randn(T) * 0.1
    x_surround = 1.5 * np.sin(2 * np.pi * t + np.pi/4) + np.random.randn(T) * 0.05
    
    # Scale inputs (as done in lnk_verify.py)
    x_center_scaled = x_center * 1e6
    x_surround_scaled = x_surround * 1e6
    
    # Create parameters
    params = LNKParams(
        tau=0.1, alpha_d=1.0, theta=0.0, sigma0=1.0,
        alpha=0.1, beta=0.0, b_out=0.0, g_out=1.0, w_xs=-0.1
    )
    
    print(f"Input: T={T}, dt={dt}")
    print(f"Parameters: tau={params.tau}, alpha_d={params.alpha_d}, w_xs={params.w_xs}")
    
    # Test the function
    try:
        start_time = time.time()
        response = compute_lnk_new(x_center_scaled, x_surround_scaled, params, dt)
        execution_time = time.time() - start_time
        
        print(f"✓ Function executed successfully in {execution_time:.4f}s")
        print(f"  Output shape: {response.shape}")
        print(f"  Output range: [{response.min():.3f}, {response.max():.3f}]")
        print(f"  Output mean: {response.mean():.3f}, std: {response.std():.3f}")
        
        # Basic sanity checks
        if np.any(np.isnan(response)):
            print("✗ Output contains NaN values")
        elif np.any(np.isinf(response)):
            print("✗ Output contains infinite values")
        elif response.mean() < 0:
            print("⚠ Negative mean response (unexpected for firing rates)")
        else:
            print("✓ Output appears reasonable")
            
    except Exception as e:
        print(f"✗ Function failed with error: {e}")
    
    print("=" * 40)

if __name__ == "__main__":
    test_new_lnk_model()
