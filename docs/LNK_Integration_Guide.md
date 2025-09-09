# LNK Model Integration Documentation

This document describes the modular LNK (Linear-Nonlinear-Kinetic) model implementation for RGC response generation in the RGC2Prey project.

## Overview

The LNK model extends the standard Linear-Nonlinear (LN) model by adding kinetic adaptation states, implementing the following dynamics:

```
a_{t+1} = a_t + dt * (alpha_d * F(x_t) - a_t) / tau
den_t   = sigma0 + alpha * a_t  
ỹ_t     = x_t / den_t + w_xs * x_s_t / den_t + beta * a_t + b_out
r_t     = softplus(g_out * ỹ_t)
```

Where `F(x) = max(0, x - theta)` and all parameters can be cell-specific.

## Modular Structure

### 1. Core LNK Utilities (`datasets/lnk_utils.py`)

This module provides the main LNK functionality:

#### Functions:

- **`load_lnk_parameters()`**: Load LNK parameters from Excel sheet
- **`load_separate_filters()`**: Load separate center/surround filters  
- **`build_channel_config()`**: Build channel configuration for Cricket2RGCs
- **`create_cricket2rgcs_config()`**: Create complete dataset configuration
- **`validate_lnk_config()`**: Validate LNK parameter configuration
- **`get_lnk_config_summary()`**: Generate human-readable configuration summary

#### Key Features:
- Type hints for better code documentation
- Comprehensive error handling and validation
- Support for both scalar and per-cell parameters
- Modular design for easy testing and reuse

### 2. Updated Training Script (`train_preycapture.py`)

The main training script now imports and uses the modular LNK functions:

```python
from datasets.lnk_utils import (
    load_lnk_parameters, 
    load_separate_filters, 
    create_cricket2rgcs_config,
    validate_lnk_config,
    get_lnk_config_summary
)
```

### 3. Example Usage (`examples/lnk_example.py`)

Demonstrates how to use the LNK utilities with complete examples and testing.

## Usage Guide

### Basic LNK Model

1. **Create LNK parameters Excel sheet** with columns:
   ```
   tau | alpha_d | theta | sigma0 | alpha | beta | b_out | g_out | w_xs | dt
   ```

2. **Run training with LNK**:
   ```bash
   python train_preycapture.py \
     --experiment_name "lnk_test" \
     --use_lnk_model \
     --lnk_sheet_name "LNK_params" \
     --lnk_adapt_mode "divisive"
   ```

### Advanced: Separate Center/Surround

#### Automatic Surround Generation (Default)
By default, surround filters are automatically generated from center filters by scaling the sigma parameter by 4x, matching MATLAB behavior:

```bash
python train_preycapture.py \
  --experiment_name "lnk_auto_surround" \
  --use_lnk_model \
  --use_separate_surround
```

**Custom surround sigma ratio:**
```bash
python train_preycapture.py \
  --experiment_name "lnk_custom_surround" \
  --use_lnk_model \
  --use_separate_surround \
  --surround_sigma_ratio 6.0
```

#### Manual Surround from Excel Sheets
For custom surround filters, create separate filter sheets in Excel:
- `SF_center`: Center spatial filter parameters
- `SF_surround`: Surround spatial filter parameters  
- `TF_center`: Center temporal filter parameters
- `TF_surround`: Surround temporal filter parameters

```bash
python train_preycapture.py \
  --experiment_name "lnk_separate" \
  --use_lnk_model \
  --use_separate_surround \
  --surround_generation "sheet" \
  --sf_center_sheet_name "SF_center" \
  --sf_surround_sheet_name "SF_surround" \
  --lnk_sheet_name "LNK_params"
```

#### Hybrid Approach
Use automatic generation with Excel fallback:
```bash
python train_preycapture.py \
  --experiment_name "lnk_hybrid" \
  --use_lnk_model \
  --use_separate_surround \
  --surround_generation "both" \
  --sf_surround_sheet_name "SF_surround"
```

## Parameter Specifications

### LNK Parameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `tau` | Time constant for adaptation | 0.01 - 1.0 | 0.1 |
| `alpha_d` | Drive strength | 0.1 - 10.0 | 1.0 |
| `theta` | Threshold for drive function | -1.0 - 1.0 | 0.0 |
| `sigma0` | Baseline divisive denominator | 0.1 - 10.0 | 1.0 |
| `alpha` | Adaptation coupling to denominator | 0.01 - 1.0 | 0.1 |
| `beta` | Additive adaptation | -1.0 - 1.0 | 0.0 |
| `b_out` | Output bias | -5.0 - 5.0 | 0.0 |
| `g_out` | Output gain | 0.1 - 10.0 | 1.0 |
| `w_xs` | Center-surround weight | -2.0 - 2.0 | -0.1 |
| `dt` | Time step (seconds) | 0.001 - 0.1 | 0.01 |

### Surround Generation Parameters

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `--surround_sigma_ratio` | Ratio to scale center sigma for surround | Float > 0 | 4.0 |
| `--surround_generation` | How to generate surround filters | auto, sheet, both | auto |

**Surround Generation Modes:**
- `auto`: Generate surround from center filters using sigma scaling (default)
- `sheet`: Load surround filters from Excel sheets only
- `both`: Use auto-generation if no Excel sheet specified, otherwise use sheet

### Parameter Formats

- **Scalar values**: Same parameter for all RGC cells
- **Array values**: Different parameter per cell (length must match number of RGCs)

## Testing and Validation

### Run Example Script
```bash
cd examples
python lnk_example.py
```

This will:
1. Create example parameter files
2. Load and validate LNK configuration
3. Test dataset configuration creation
4. Clean up temporary files

### Validation Features
- Parameter dimension checking
- Configuration validation
- Error handling with fallbacks
- Comprehensive logging

## Integration with Existing Code

### Backward Compatibility
- When `--use_lnk_model` is not specified, uses original LN model
- All existing functionality preserved
- No changes required to existing scripts

### Performance Considerations
- Vectorized operations across all RGC cells
- Efficient parameter broadcasting
- Minimal memory overhead

## Troubleshooting

### Common Issues

1. **"LNK parameters not found"**
   - Check Excel sheet name matches `--lnk_sheet_name`
   - Verify all required columns are present

2. **"Parameter length mismatch"**
   - Ensure parameter arrays match number of RGC cells
   - Use single values for scalar parameters

3. **"Configuration validation failed"**
   - Check parameter ranges and types
   - Verify adaptation mode is 'divisive' or 'subtractive'

### Debug Tips
- Enable detailed logging with `logging.basicConfig(level=logging.DEBUG)`
- Use `validate_lnk_config()` to check configuration
- Check `get_lnk_config_summary()` output for parameter overview

## Future Extensions

The modular design allows for easy extensions:

- **Multiple adaptation states**: Extend kinetic state vector
- **Nonlinear adaptation**: Custom adaptation functions
- **Spatial adaptation**: Position-dependent parameters
- **Learning**: Adaptive parameter optimization

## API Reference

See docstrings in `datasets/lnk_utils.py` for detailed function documentation with type hints and examples.

---

# New LNK Model Integration (compute_lnk_response_from_convolved)

## Overview

The `lnk_verify.py` script has been enhanced to support both the original LNK implementation and the new PyTorch-based implementation from `simple_lnk.py`. **Note: These represent different modeling approaches and are not expected to produce identical results.**

This integration provides:

1. **Dual Model Support**: Switch between implementations via a simple flag
2. **Automatic Detection**: PyTorch availability check with graceful fallback
3. **Parameter Conversion**: Seamless mapping between parameter formats
4. **Independent Verification**: Each model can be validated against experimental data

## Key Integration Features

### Model Selection Flag
```python
USE_NEW_LNK_MODEL = True  # Set to False to use original model
```

### Parameter Mapping
- Automatic conversion from `LNKParams` dataclass to dictionary format
- Preserves all parameter values and meanings
- Handles single vs. dual input cases appropriately

### Signal Correspondence
- `x_center` ← `sim` (center/primary signal)
- `x_surround` ← `sim_s` (surround signal)
- Both signals scaled by 1e6 for compatibility

### Wrapper Functions
- `lnk_params_to_dict()`: Convert parameter formats
- `compute_lnk_new()`: Unified interface for new implementation
- Error handling and device management included

## Usage Example

```python
# In lnk_verify.py
USE_NEW_LNK_MODEL = True  # Enable new implementation

# The script automatically:
# 1. Detects PyTorch availability
# 2. Converts parameters to appropriate format
# 3. Runs timing comparisons
# 4. Generates correlation analysis
# 5. Creates visualization plots
```

## Testing Integration

Run the integration test to verify the new implementation works correctly:
```bash
python test/test_lnk_integration.py
```

This test creates synthetic data and validates that the new implementation:
- Executes without errors
- Produces reasonable output values
- Handles parameter conversion correctly
- Manages tensor operations properly

## Benefits

1. **Backward Compatibility**: Existing code continues to work unchanged
2. **Model Choice**: Select implementation based on specific needs
3. **Easy Integration**: Simple flag-based switching between models
4. **Independent Validation**: Each model can be tested against experimental data
5. **Future Ready**: GPU acceleration available with new implementation
