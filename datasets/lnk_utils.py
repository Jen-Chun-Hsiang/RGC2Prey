"""
LNK (Linear-Nonlinear-Kinetic) Model Utilities for RGC Response Generation

This module provides utilities for loading and configuring LNK model parameters
and filters for use with the Cricket2RGCs dataset. The LNK model extends the
standard LN model with kinetic adaptation states.

Author: Emily Hsiang
Date: August 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, Any, Tuple
from datasets.sim_cricket import RGCrfArray


def generate_surround_from_center(sf_center: np.ndarray, 
                                 surround_sigma_ratio: float = 4.0,
                                 rgc_array: Optional[RGCrfArray] = None) -> np.ndarray:
    """
    Generate surround spatial filters from center filters by scaling sigma.
    
    This implements the default MATLAB behavior where surround filters are
    created by multiplying the center filter's sigma by a factor (default 4).
    
    Args:
        sf_center: Center spatial filters [W, H, N]
        surround_sigma_ratio: Factor to multiply center sigma (default 4.0)
        rgc_array: Optional RGCrfArray for regenerating filters with new params
        
    Returns:
        Surround spatial filters [W, H, N]
    """
    if rgc_array is None:
        # Simple scaling approach - approximate by spatial interpolation
        # This is less accurate but doesn't require regenerating Gaussians
        try:
            import scipy.ndimage as ndimage
        except ImportError:
            logging.warning("scipy not available, using simple surround approximation")
            # Fallback: just return center filters (better than failing)
            return sf_center.copy()
            
        W, H, N = sf_center.shape
        sf_surround = np.zeros_like(sf_center)
        
        for i in range(N):
            # Scale the spatial extent of each filter
            # This approximates increasing sigma by interpolating
            zoom_factor = 1.0 / surround_sigma_ratio
            center_filter = sf_center[:, :, i]
            
            # Zoom out (makes features larger)
            zoomed = ndimage.zoom(center_filter, zoom_factor, order=3)
            
            # Pad or crop to original size
            if zoomed.shape[0] < W or zoomed.shape[1] < H:
                # Pad if smaller
                pad_w = (W - zoomed.shape[0]) // 2
                pad_h = (H - zoomed.shape[1]) // 2
                sf_surround[:, :, i] = np.pad(zoomed, 
                                             ((pad_w, W - zoomed.shape[0] - pad_w),
                                              (pad_h, H - zoomed.shape[1] - pad_h)),
                                             mode='constant')
            else:
                # Crop if larger
                crop_w = (zoomed.shape[0] - W) // 2
                crop_h = (zoomed.shape[1] - H) // 2
                sf_surround[:, :, i] = zoomed[crop_w:crop_w+W, crop_h:crop_h+H]
            
            # Normalize like in MATLAB
            sf_surround[:, :, i] -= np.median(sf_surround[:, :, i])
            max_val = np.max(np.abs(sf_surround[:, :, i]))
            if max_val > 0:
                sf_surround[:, :, i] /= max_val
                
        logging.info(f"Generated surround filters with sigma ratio {surround_sigma_ratio} using interpolation")
    else:
        # More accurate: regenerate Gaussians with scaled sigma
        sf_param_table = rgc_array.sf_param_table.copy()
        
        # Scale sigma_x and sigma_y by the ratio
        if 'sigma_x' in sf_param_table.columns:
            sf_param_table['sigma_x'] = sf_param_table['sigma_x'] * surround_sigma_ratio
        if 'sigma_y' in sf_param_table.columns:
            sf_param_table['sigma_y'] = sf_param_table['sigma_y'] * surround_sigma_ratio
        if 'surround_sigma_x' in sf_param_table.columns:
            sf_param_table['surround_sigma_x'] = sf_param_table['surround_sigma_x'] * surround_sigma_ratio
        if 'surround_sigma_y' in sf_param_table.columns:
            sf_param_table['surround_sigma_y'] = sf_param_table['surround_sigma_y'] * surround_sigma_ratio
        
        # Create new RGCrfArray with modified parameters for surround
        try:
            surround_array = _create_rgc_array_copy(rgc_array, sf_param_table, rgc_array.tf_param_table)
            sf_surround, _, _, _, _ = surround_array.get_results()
            logging.info(f"Generated surround filters with sigma ratio {surround_sigma_ratio} using Gaussian regeneration")
        except Exception as e:
            logging.warning(f"Could not regenerate Gaussians: {e}, using interpolation fallback")
            # Fallback to interpolation method
            return generate_surround_from_center(sf_center, surround_sigma_ratio, None)
    
    return sf_surround


def load_lnk_parameters(rf_params_file: str, 
                       lnk_sheet_name: str,
                       num_rgcs: int,
                       lnk_adapt_mode: str = 'divisive',
                       use_lnk_model: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load LNK parameters from Excel sheet and process them for each RGC cell.
    
    The LNK model implements the following dynamics:
        a_{t+1} = a_t + dt * (alpha_d * F(x_t) - a_t) / tau
        den_t   = sigma0 + alpha * a_t  
        ỹ_t     = x_t / den_t + w_xs * x_s_t / den_t + beta * a_t + b_out
        r_t     = softplus(g_out * ỹ_t)
    
    Where F(x) = max(0, x - theta) and all parameters can be cell-specific.
    
    Args:
        rf_params_file: Path to Excel file containing parameters
        lnk_sheet_name: Name of Excel sheet with LNK parameters
        num_rgcs: Number of RGC cells
        lnk_adapt_mode: Adaptation mode ('divisive' or 'subtractive')
        use_lnk_model: Whether to use LNK model (if False, returns None)
        
    Returns:
        Dictionary with LNK parameters and adaptation mode, or None if LNK disabled
        
    Raises:
        FileNotFoundError: If Excel file or sheet not found
        ValueError: If parameter dimensions don't match num_rgcs
    """
    if not use_lnk_model:
        return None
        
    # Load LNK parameter table
    try:
        lnk_param_table = pd.read_excel(rf_params_file, sheet_name=lnk_sheet_name)
        logging.info(f"Loaded LNK parameters from sheet: {lnk_sheet_name}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file not found: {rf_params_file}")
    except ValueError as e:
        logging.warning(f"Could not load LNK parameters: {e}. Using default LN model.")
        return None
    
    # Extract parameters - convert to per-cell arrays or scalars
    def get_param(param_name: str, 
                  default_value: Union[float, np.ndarray], 
                  num_cells: int = num_rgcs) -> Union[float, np.ndarray]:
        """Extract parameter with proper broadcasting and validation."""
        if param_name in lnk_param_table.columns:
            param_values = lnk_param_table[param_name].values
            
            # Handle NaN values
            if np.any(pd.isna(param_values)):
                logging.warning(f"NaN values found in {param_name}, using default")
                return default_value
                
            if len(param_values) == 1:
                # Single value - broadcast to all cells
                return float(param_values[0])
            elif len(param_values) == num_cells:
                # Per-cell values
                return param_values.astype(float)
            else:
                logging.warning(f"LNK parameter {param_name} length mismatch "
                               f"(got {len(param_values)}, expected {num_cells}). Using default.")
                return default_value
        else:
            logging.warning(f"LNK parameter {param_name} not found. Using default: {default_value}")
            return default_value
    
    # Default LNK parameters (matching MATLAB fitLNK_rate_scw)
    lnk_params = {
        'tau': get_param('tau', 0.1),           # Time constant for adaptation
        'alpha_d': get_param('alpha_d', 1.0),   # Drive strength for adaptation
        'theta': get_param('theta', 0.0),       # Threshold for drive function
        'sigma0': get_param('sigma0', 1.0),     # Baseline divisive denominator
        'alpha': get_param('alpha', 0.1),       # Adaptation coupling to denominator
        'beta': get_param('beta', 0.0),         # Additive adaptation term
        'b_out': get_param('b_out', 0.0),       # Output bias
        'g_out': get_param('g_out', 1.0),       # Output gain
        'w_xs': get_param('w_xs', -0.1),        # Center-surround interaction weight
        'dt': get_param('dt', 0.01)             # Time step (sampling period)
    }
    
    # Validate adaptation mode
    if lnk_adapt_mode not in ['divisive', 'subtractive']:
        logging.warning(f"Invalid adaptation mode {lnk_adapt_mode}, using 'divisive'")
        lnk_adapt_mode = 'divisive'
    
    logging.info(f"LNK parameters loaded for {num_rgcs} cells with adaptation mode: {lnk_adapt_mode}")
    
    return {
        'lnk_params': lnk_params,
        'adapt_mode': lnk_adapt_mode
    }


def load_separate_filters(rf_params_file: str,
                         rgc_array: RGCrfArray,
                         use_separate_surround: bool = False,
                         sf_center_sheet_name: Optional[str] = None,
                         sf_surround_sheet_name: Optional[str] = None,
                         tf_center_sheet_name: Optional[str] = None,
                         tf_surround_sheet_name: Optional[str] = None,
                         surround_sigma_ratio: float = 4.0,
                         surround_generation: str = 'auto') -> Optional[Dict[str, np.ndarray]]:
    """
    Load or generate separate center and surround filters for LNK model.
    
    By default, surround filters are generated from center filters by scaling
    sigma values by surround_sigma_ratio (default 4.0), matching MATLAB behavior.
    
    Args:
        rf_params_file: Path to Excel file with filter parameters
        rgc_array: RGCrfArray instance for generating filters
        use_separate_surround: Whether to use separate center/surround filters
        sf_center_sheet_name: Sheet name for center spatial filters (optional)
        sf_surround_sheet_name: Sheet name for surround spatial filters (optional)
        tf_center_sheet_name: Sheet name for center temporal filters (optional)
        tf_surround_sheet_name: Sheet name for surround temporal filters (optional)
        surround_sigma_ratio: Ratio to scale center sigma for surround (default 4.0)
        surround_generation: 'auto' (from center), 'sheet' (from Excel), or 'both'
        
    Returns:
        Dictionary with separate filter arrays or None if not requested
    """
    if not use_separate_surround:
        return None
        
    separate_filters = {}
    
    # Get center filters - use existing filters from rgc_array by default
    sf_center, tf_center, _, _, _ = rgc_array.get_results()
    separate_filters['sf_center'] = sf_center
    separate_filters['tf_center'] = tf_center
    
    # Override with custom center filters if specified
    if sf_center_sheet_name or tf_center_sheet_name:
        try:
            if sf_center_sheet_name:
                sf_center_table = pd.read_excel(rf_params_file, 
                                               sheet_name=sf_center_sheet_name, 
                                               usecols='A:L')
                center_array = _create_rgc_array_copy(rgc_array, sf_center_table, rgc_array.tf_param_table)
                sf_center, _, _, _, _ = center_array.get_results()
                separate_filters['sf_center'] = sf_center
                logging.info(f"Loaded center spatial filters from sheet: {sf_center_sheet_name}")
                
            if tf_center_sheet_name:
                tf_center_table = pd.read_excel(rf_params_file, 
                                               sheet_name=tf_center_sheet_name, 
                                               usecols='A:I')
                center_array = _create_rgc_array_copy(rgc_array, rgc_array.sf_param_table, tf_center_table)
                _, tf_center, _, _, _ = center_array.get_results()
                separate_filters['tf_center'] = tf_center
                logging.info(f"Loaded center temporal filters from sheet: {tf_center_sheet_name}")
        except Exception as e:
            logging.warning(f"Could not load custom center filters: {e}, using defaults")
    
    # Generate or load surround filters
    if surround_generation == 'auto' or (surround_generation == 'both' and not sf_surround_sheet_name):
        # Generate surround from center by scaling sigma (MATLAB default behavior)
        sf_surround = generate_surround_from_center(
            separate_filters['sf_center'], 
            surround_sigma_ratio=surround_sigma_ratio,
            rgc_array=rgc_array
        )
        separate_filters['sf_surround'] = sf_surround
        
        # Use same temporal filter for surround as center (typical assumption)
        separate_filters['tf_surround'] = separate_filters['tf_center']
        
        logging.info(f"Generated surround filters from center with sigma ratio {surround_sigma_ratio}")
        
    elif surround_generation == 'sheet' or surround_generation == 'both':
        # Load surround filters from Excel sheets if provided
        if sf_surround_sheet_name:
            try:
                sf_surround_table = pd.read_excel(rf_params_file, 
                                                 sheet_name=sf_surround_sheet_name, 
                                                 usecols='A:L')
                surround_array = _create_rgc_array_copy(rgc_array, sf_surround_table, rgc_array.tf_param_table)
                sf_surround, _, _, _, _ = surround_array.get_results()
                separate_filters['sf_surround'] = sf_surround
                logging.info(f"Loaded surround spatial filters from sheet: {sf_surround_sheet_name}")
            except Exception as e:
                logging.warning(f"Could not load surround spatial filters: {e}, generating from center")
                sf_surround = generate_surround_from_center(
                    separate_filters['sf_center'], 
                    surround_sigma_ratio=surround_sigma_ratio,
                    rgc_array=rgc_array
                )
                separate_filters['sf_surround'] = sf_surround
        else:
            # No sheet specified, generate from center
            sf_surround = generate_surround_from_center(
                separate_filters['sf_center'], 
                surround_sigma_ratio=surround_sigma_ratio,
                rgc_array=rgc_array
            )
            separate_filters['sf_surround'] = sf_surround
        
        if tf_surround_sheet_name:
            try:
                tf_surround_table = pd.read_excel(rf_params_file, 
                                                 sheet_name=tf_surround_sheet_name, 
                                                 usecols='A:I')
                surround_array = _create_rgc_array_copy(rgc_array, rgc_array.sf_param_table, tf_surround_table)
                _, tf_surround, _, _, _ = surround_array.get_results()
                separate_filters['tf_surround'] = tf_surround
                logging.info(f"Loaded surround temporal filters from sheet: {tf_surround_sheet_name}")
            except Exception as e:
                logging.warning(f"Could not load surround temporal filters: {e}, using center")
                separate_filters['tf_surround'] = separate_filters['tf_center']
        else:
            # Default: use same temporal filter for surround
            separate_filters['tf_surround'] = separate_filters['tf_center']
    
    return separate_filters


def _create_rgc_array_copy(original_array: RGCrfArray, 
                          sf_table: pd.DataFrame, 
                          tf_table: pd.DataFrame) -> RGCrfArray:
    """
    Create a copy of RGCrfArray with different SF/TF tables.
    
    Args:
        original_array: Original RGCrfArray instance
        sf_table: New spatial filter parameter table
        tf_table: New temporal filter parameter table
        
    Returns:
        New RGCrfArray instance with updated parameters
    """
    return RGCrfArray(
        sf_table, tf_table,
        rgc_array_rf_size=original_array.rgc_array_rf_size,
        xlim=original_array.xlim, 
        ylim=original_array.ylim,
        target_num_centers=original_array.target_num_centers,
        sf_scalar=original_array.sf_scalar,
        grid_generate_method=original_array.grid_generate_method,
        tau=original_array.tau,
        mask_radius=original_array.mask_radius,
        rgc_rand_seed=original_array.rgc_rand_seed,
        num_gauss_example=original_array.num_gauss_example,
        sf_constraint_method=original_array.sf_constraint_method,
        temporal_filter_len=original_array.temporal_filter_len,
        grid_size_fac=original_array.grid_size_fac,
        sf_mask_radius=original_array.sf_mask_radius,
        is_pixelized_tf=original_array.is_pixelized_tf,
        set_s_scale=original_array.set_s_scale,
        is_rf_median_subtract=original_array.is_rf_median_subtract,
        is_rescale_diffgaussian=original_array.is_rescale_diffgaussian,
        grid_noise_level=original_array.grid_noise_level,
        is_reversed_tf=original_array.is_reversed_tf,
        sf_id_list=original_array.sf_id_list,
        syn_tf_sf=original_array.syn_tf_sf
    )


def build_channel_config(multi_opt_sf: np.ndarray,
                        tf: np.ndarray, 
                        grid2value_mapping: Any,
                        map_func: Any,
                        rect_thr: float,
                        lnk_config: Optional[Dict[str, Any]] = None,
                        separate_filters: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """
    Build channel configuration dictionary for Cricket2RGCs dataset.
    
    This function creates a standardized configuration dictionary that can be
    passed to Cricket2RGCs to set up either LN or LNK model processing.
    
    Args:
        multi_opt_sf: Spatial filters array [W, H, N]
        tf: Temporal filters array [N, T] or [T, N]
        grid2value_mapping: Grid to value mapping function/data
        map_func: Mapping function for RGC responses to grid
        rect_thr: Rectification threshold
        lnk_config: LNK configuration dict with 'lnk_params' and 'adapt_mode'
        separate_filters: Dict with separate center/surround filters
    
    Returns:
        Dictionary with complete channel configuration
    """
    # Base channel configuration
    channel_config = {
        'sf': multi_opt_sf,
        'tf': tf,
        'map_func': map_func,
        'grid2value': grid2value_mapping,
        'rect_thr': rect_thr
    }
    
    # Add LNK configuration if provided
    if lnk_config is not None:
        channel_config.update(lnk_config)
        num_params = len(lnk_config['lnk_params'])
        logging.info(f"Added LNK configuration with {num_params} parameters")
    
    # Add separate filters if provided
    if separate_filters is not None:
        channel_config.update(separate_filters)
        filter_types = list(separate_filters.keys())
        logging.info(f"Added separate filters: {', '.join(filter_types)}")
    
    return channel_config


def create_cricket2rgcs_config(num_samples: int,
                              multi_opt_sf: np.ndarray,
                              tf: np.ndarray,
                              map_func: Any,
                              grid2value_mapping: Any,
                              target_width: int,
                              target_height: int,
                              movie_generator: Any,
                              grid_size_fac: float = 1.0,
                              is_norm_coords: bool = False,
                              is_syn_mov_shown: bool = False,
                              fr2spikes: bool = False,
                              quantize_scale: float = 1.0,
                              add_noise: bool = False,
                              rgc_noise_std: float = 0.0,
                              smooth_data: bool = False,
                              is_rectified: bool = True,
                              is_direct_image: bool = False,
                              is_reversed_OFF_sign: bool = False,
                              is_both_ON_OFF: bool = False,
                              is_two_grids: bool = False,
                              rectified_thr_ON: float = 0.0,
                              rectified_thr_OFF: float = 0.0,
                              # OFF channel parameters
                              multi_opt_sf_off: Optional[np.ndarray] = None,
                              tf_off: Optional[np.ndarray] = None,
                              map_func_off: Optional[Any] = None,
                              grid2value_mapping_off: Optional[Any] = None,
                              # LNK parameters
                              lnk_config: Optional[Dict[str, Any]] = None,
                              separate_filters: Optional[Dict[str, np.ndarray]] = None,
                              lnk_config_off: Optional[Dict[str, Any]] = None,
                              separate_filters_off: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """
    Create comprehensive configuration dictionary for Cricket2RGCs dataset with LNK support.
    
    This function provides a clean interface for creating Cricket2RGCs configurations
    with support for both standard LN and advanced LNK models, ON/OFF channels,
    and separate center/surround processing.
    
    Args:
        num_samples: Number of samples in dataset
        multi_opt_sf: ON channel spatial filters [W, H, N]
        tf: ON channel temporal filters
        map_func: ON channel mapping function
        grid2value_mapping: ON channel grid-to-value mapping
        target_width: Target grid width
        target_height: Target grid height  
        movie_generator: Movie generator instance
        grid_size_fac: Grid size scaling factor
        is_norm_coords: Whether to normalize coordinates
        is_syn_mov_shown: Whether to return synthetic movies
        fr2spikes: Whether to convert firing rates to spikes
        quantize_scale: Quantization scaling for spikes
        add_noise: Whether to add noise to RGC outputs
        rgc_noise_std: Standard deviation of added noise
        smooth_data: Whether to smooth RGC outputs
        is_rectified: Whether to rectify RGC outputs
        is_direct_image: Whether to bypass RGC convolution
        is_reversed_OFF_sign: Whether to reverse OFF channel sign
        is_both_ON_OFF: Whether to include both ON and OFF channels
        is_two_grids: Whether to create two separate grids
        rectified_thr_ON: Rectification threshold for ON channel
        rectified_thr_OFF: Rectification threshold for OFF channel
        multi_opt_sf_off: OFF channel spatial filters
        tf_off: OFF channel temporal filters
        map_func_off: OFF channel mapping function
        grid2value_mapping_off: OFF channel grid-to-value mapping
        lnk_config: LNK configuration for ON channel
        separate_filters: Separate center/surround filters for ON channel
        lnk_config_off: LNK configuration for OFF channel
        separate_filters_off: Separate center/surround filters for OFF channel
        
    Returns:
        Complete configuration dictionary for Cricket2RGCs initialization
    """
    # Build channel configurations
    channel_on_config = build_channel_config(
        multi_opt_sf, tf, grid2value_mapping, map_func, rectified_thr_ON,
        lnk_config=lnk_config, separate_filters=separate_filters
    )
    
    channel_off_config = None
    if multi_opt_sf_off is not None:
        channel_off_config = build_channel_config(
            multi_opt_sf_off, tf_off, grid2value_mapping_off, map_func_off, rectified_thr_OFF,
            lnk_config=lnk_config_off, separate_filters=separate_filters_off
        )
    
    # Base configuration
    config = {
        'num_samples': num_samples,
        'multi_opt_sf': multi_opt_sf,
        'tf': tf,
        'map_func': map_func,
        'grid2value_mapping': grid2value_mapping,
        'multi_opt_sf_off': multi_opt_sf_off,
        'tf_off': tf_off,
        'map_func_off': map_func_off,
        'grid2value_mapping_off': grid2value_mapping_off,
        'target_width': target_width,
        'target_height': target_height,
        'movie_generator': movie_generator,
        'grid_size_fac': grid_size_fac,
        'is_norm_coords': is_norm_coords,
        'is_syn_mov_shown': is_syn_mov_shown,
        'fr2spikes': fr2spikes,
        'is_both_ON_OFF': is_both_ON_OFF,
        'quantize_scale': quantize_scale,
        'add_noise': add_noise,
        'rgc_noise_std': rgc_noise_std,
        'smooth_data': smooth_data,
        'is_rectified': is_rectified,
        'is_direct_image': is_direct_image,
        'is_reversed_OFF_sign': is_reversed_OFF_sign,
        'is_two_grids': is_two_grids,
        'rectified_thr_ON': rectified_thr_ON,
        'rectified_thr_OFF': rectified_thr_OFF
    }
    
    # Add LNK configurations to dataset arguments
    if lnk_config is not None:
        config['lnk_config_on'] = channel_on_config
        if channel_off_config is not None:
            config['lnk_config_off'] = channel_off_config
            
        logging.info("Created Cricket2RGCs config with LNK model enabled")
    else:
        logging.info("Created Cricket2RGCs config with standard LN model")
    
    return config


def validate_lnk_config(lnk_config: Dict[str, Any], num_rgcs: int) -> bool:
    """
    Validate LNK configuration parameters.
    
    Args:
        lnk_config: LNK configuration dictionary
        num_rgcs: Expected number of RGC cells
        
    Returns:
        True if configuration is valid, False otherwise
    """
    if 'lnk_params' not in lnk_config:
        logging.error("LNK config missing 'lnk_params' key")
        return False
        
    if 'adapt_mode' not in lnk_config:
        logging.error("LNK config missing 'adapt_mode' key")
        return False
        
    # Check parameter dimensions
    lnk_params = lnk_config['lnk_params']
    required_params = ['tau', 'alpha_d', 'theta', 'sigma0', 'alpha', 'beta', 'b_out', 'g_out', 'w_xs', 'dt']
    
    for param_name in required_params:
        if param_name not in lnk_params:
            logging.error(f"Missing required LNK parameter: {param_name}")
            return False
            
        param_val = lnk_params[param_name]
        if isinstance(param_val, np.ndarray):
            if len(param_val) != num_rgcs:
                logging.error(f"Parameter {param_name} length {len(param_val)} != {num_rgcs}")
                return False
                
    return True


def get_lnk_config_summary(lnk_config: Dict[str, Any]) -> str:
    """
    Generate a summary string of LNK configuration.
    
    Args:
        lnk_config: LNK configuration dictionary
        
    Returns:
        Human-readable summary string
    """
    if lnk_config is None:
        return "LN model (no adaptation)"
        
    lnk_params = lnk_config['lnk_params']
    adapt_mode = lnk_config['adapt_mode']
    
    # Check if parameters are scalar or per-cell
    param_types = []
    for key, val in lnk_params.items():
        if isinstance(val, np.ndarray):
            param_types.append(f"{key}[{len(val)}]")
        else:
            param_types.append(f"{key}={val:.3f}")
    
    return f"LNK model ({adapt_mode} adaptation): {', '.join(param_types[:3])}..."
