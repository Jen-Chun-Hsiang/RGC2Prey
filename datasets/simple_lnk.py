"""
Simple LNK Implementation for RGC2Prey

This module provides a simplified LNK (Linear-Nonlinear-Kinetics) model implementation
that replaces the overcomplicated lnk_utils.py. The goal is to make LNK just a simple
extension of the LN model, not a complete restructuring.

Key principles:
1. LNK = LN + simple adaptation dynamics  
2. Surround filters = center filters with 4x larger sigma
3. Minimal configuration overhead
4. Easy debugging and understanding
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import logging
from typing import Optional, Dict, Any
from scipy.stats import skew
import sys


def load_lnk_parameters(rf_params_file: str, lnk_sheet_name: str, num_rgcs: int) -> Optional[Dict[str, np.ndarray]]:
    """
    Simple LNK parameter loader - just loads the essentials.
    
    Args:
        rf_params_file: Path to Excel file
        lnk_sheet_name: Sheet name with LNK parameters  
        num_rgcs: Number of RGC cells
        
    Returns:
        Dict with LNK parameters as numpy arrays, or None if failed
    """
    try:
        df = pd.read_excel(rf_params_file, sheet_name=lnk_sheet_name).dropna()
        if len(df) == 0:
            return None
    except Exception as e:
        logging.warning(f"Could not load LNK parameters: {e}")
        return None
    
    # Sample parameters for each RGC
    idx = np.random.choice(len(df), num_rgcs, replace=True)
    
    params = {}
    # Core LNK parameters with defaults
    param_defaults = {
        'tau': 0.1, 'alpha_d': 1.0, 'theta': 0.0, 'sigma0': 1.0, 
        'alpha': 0.1, 'beta': 0.0, 'b_out': 0.0, 'g_out': 1.0, 
        'w_xs': -0.1, 'dt': 0.01
    }
    
    for param, default in param_defaults.items():
        if param in df.columns:
            params[param] = df.iloc[idx][param].values.astype(float)
        else:
            params[param] = np.full(num_rgcs, default)
    
    logging.info(f"Loaded LNK parameters for {num_rgcs} cells")
    return params


def generate_surround_filters(center_sf: np.ndarray, center_tf: np.ndarray, sigma_ratio: float = 4.0) -> tuple:
    """
    Generate surround filters by scaling center filters with larger sigma.
    Simple approach using Gaussian blur approximation.
    
    Args:
        center_sf: Center spatial filters [W, H, N]
        center_tf: Center temporal filters [N, 1, T] or [N, T]
        sigma_ratio: Factor to scale sigma for surround (default 4.0)
        
    Returns:
        (surround_sf, surround_tf): Surround filters with same shapes as center
    """
    W, H, N = center_sf.shape
    surround_sf = np.zeros_like(center_sf)
    
    # Generate surround spatial filters by blurring center
    for i in range(N):
        # Apply Gaussian blur to approximate larger sigma
        surround_sf[:, :, i] = gaussian_filter(center_sf[:, :, i], sigma=sigma_ratio)
        
        # Normalize to similar magnitude as center
        center_norm = np.abs(center_sf[:, :, i]).max()
        surround_norm = np.abs(surround_sf[:, :, i]).max()
        if surround_norm > 0:
            surround_sf[:, :, i] *= center_norm / surround_norm
    
    # Use same temporal filters for surround (typical assumption)
    surround_tf = center_tf.copy()
    
    return surround_sf, surround_tf


def compute_lnk_response(movie: torch.Tensor, 
                        center_sf: torch.Tensor, 
                        center_tf: torch.Tensor,
                        surround_sf: torch.Tensor, 
                        surround_tf: torch.Tensor,
                        lnk_params: Dict[str, np.ndarray],
                        device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
    """
    Compute LNK response with simplified implementation.
    
    LNK Model:
        1. Center: x_c = movie ⊗ center_sf ⊗ center_tf
        2. Surround: x_s = movie ⊗ surround_sf ⊗ surround_tf  
        3. Adaptation: a_{t+1} = a_t + dt * (alpha_d * max(0, x_c - theta) - a_t) / tau
        4. Output: r_t = softplus(g_out * (x_c + w_xs * x_s) / (sigma0 + alpha * a_t) + beta * a_t + b_out)
    
    Args:
        movie: Input movie [T, H, W]
        center_sf: Center spatial filters [W, H, N] 
        center_tf: Center temporal filters [N, 1, T_filter]
        surround_sf: Surround spatial filters [W, H, N]
        surround_tf: Surround temporal filters [N, 1, T_filter]
        lnk_params: Dict with LNK parameters
        device: Device for computation
        dtype: Data type for computation
        
    Returns:
        RGC responses [N, T_out]
    """
    # Step 1: Center spatial-temporal convolution
    x_c = torch.einsum('whn,thw->nt', center_sf, movie)  # [N, T]
    x_c = x_c.unsqueeze(0)  # [1, N, T]
    x_center = F.conv1d(x_c, center_tf, stride=1, padding=0, groups=x_c.shape[1]).squeeze(0)  # [N, T_out]
    
    # Step 2: Surround spatial-temporal convolution
    x_s = torch.einsum('whn,thw->nt', surround_sf, movie)  # [N, T]
    x_s = x_s.unsqueeze(0)  # [1, N, T] 
    x_surround = F.conv1d(x_s, surround_tf, stride=1, padding=0, groups=x_s.shape[1]).squeeze(0)  # [N, T_out]
    
    # Align time lengths
    T_out = min(x_center.shape[1], x_surround.shape[1])
    x_center = x_center[:, :T_out]
    x_surround = x_surround[:, :T_out]
    
    N = x_center.shape[0]
    
    # Step 3: Convert parameters to tensors
    def to_tensor(param_name: str) -> torch.Tensor:
        return torch.from_numpy(lnk_params[param_name]).to(device=device, dtype=dtype)
    
    tau = to_tensor('tau')
    alpha_d = to_tensor('alpha_d') 
    theta = to_tensor('theta')
    sigma0 = to_tensor('sigma0')
    alpha = to_tensor('alpha')
    beta = to_tensor('beta')
    b_out = to_tensor('b_out')
    g_out = to_tensor('g_out')
    w_xs = to_tensor('w_xs')
    dt = float(lnk_params['dt'][0] if isinstance(lnk_params['dt'], np.ndarray) else lnk_params['dt'])
    
    # Step 4: Adaptation dynamics
    a = torch.zeros((N, T_out), device=device, dtype=dtype)
    
    for t in range(T_out):
        if t == 0:
            a_prev = torch.zeros(N, device=device, dtype=dtype)
        else:
            a_prev = a[:, t-1]
        
        # Drive signal: F(x_c) = max(0, x_c - theta)
        drive = torch.relu(x_center[:, t] - theta)
        
        # Adaptation update: da/dt = (alpha_d * drive - a) / tau
        da_dt = (alpha_d * drive - a_prev) / tau
        a_t = a_prev + dt * da_dt
        a[:, t] = torch.clamp_min(a_t, 0.0)
    
    # Step 5: Divisive normalization
    den = sigma0[:, None] + alpha[:, None] * a  # [N, T_out]
    den = torch.clamp_min(den, 1e-9)  # Avoid division by zero
    
    # Step 6: Combined response
    combined_input = x_center + w_xs[:, None] * x_surround  # Center-surround interaction
    y = combined_input / den + beta[:, None] * a + b_out[:, None]
    
    # Step 7: Output nonlinearity
    rgc_response = F.softplus(g_out[:, None] * y, beta=1.0, threshold=20.0)
    
    # Log distribution statistics
    def log_distribution_stats(tensor, name):
        """Log mean, std, and skewness of a tensor"""
        tensor_np = tensor.detach().cpu().numpy()
        mean_val = np.mean(tensor_np)
        std_val = np.std(tensor_np)
        try:
            from scipy.stats import skew
            skew_val = skew(tensor_np.flatten())
        except ImportError:
            # Fallback skewness calculation
            tensor_flat = tensor_np.flatten()
            mean_flat = np.mean(tensor_flat)
            std_flat = np.std(tensor_flat)
            if std_flat > 0:
                skew_val = np.mean(((tensor_flat - mean_flat) / std_flat) ** 3)
            else:
                skew_val = 0.0
        
        logging.info(f"{name} distribution - Mean: {mean_val:.6f}, Std: {std_val:.6f}, Skew: {skew_val:.6f}, Shape: {tensor.shape}")
    
    logging.info("=" * 60)
    logging.info("LNK Response Distribution Analysis")
    logging.info("=" * 60)
    log_distribution_stats(x_center, "x_center")
    log_distribution_stats(x_surround, "x_surround") 
    log_distribution_stats(y, "y (pre-nonlinearity)")
    log_distribution_stats(rgc_response, "rgc_response (final)")
    logging.info("=" * 60)
    
    # Stop the process
    logging.info("Stopping process for distribution analysis...")
    sys.exit(0)
    
    return rgc_response


def add_lnk_to_cricket2rgcs(cricket_dataset, 
                           rf_params_file: str, 
                           lnk_sheet_name: str = 'LNK_params',
                           surround_sigma_ratio: float = 4.0) -> bool:
    """
    Add LNK functionality to existing Cricket2RGCs dataset.
    Simple modification that doesn't require restructuring.
    
    Args:
        cricket_dataset: Existing Cricket2RGCs instance
        rf_params_file: Path to Excel file with LNK parameters
        lnk_sheet_name: Sheet name for LNK parameters
        surround_sigma_ratio: Ratio for surround filter generation
        
    Returns:
        True if LNK was successfully added, False otherwise
    """
    # Load LNK parameters
    num_rgcs = cricket_dataset.multi_opt_sf.shape[2]  # [W, H, N]
    lnk_params = load_lnk_parameters(rf_params_file, lnk_sheet_name, num_rgcs)
    
    if lnk_params is None:
        logging.warning("Failed to load LNK parameters, keeping LN model")
        return False
    
    # Generate surround filters
    center_sf_np = cricket_dataset.multi_opt_sf.cpu().numpy()
    center_tf_np = cricket_dataset.tf.cpu().numpy()
    surround_sf_np, surround_tf_np = generate_surround_filters(
        center_sf_np, center_tf_np, surround_sigma_ratio
    )
    
    # Add to dataset
    cricket_dataset.lnk_params = lnk_params
    cricket_dataset.surround_sf = torch.from_numpy(surround_sf_np).to(cricket_dataset.multi_opt_sf.device)
    cricket_dataset.surround_tf = torch.from_numpy(surround_tf_np).to(cricket_dataset.tf.device)
    cricket_dataset.use_lnk = True
    
    logging.info(f"Added LNK functionality with surround sigma ratio {surround_sigma_ratio}")
    return True
