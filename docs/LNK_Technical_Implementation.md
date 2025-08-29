# Technical Implementation Guide: LNK Model in RGC2Prey

## Overview

This document provides technical details for implementing and using the LNK (Linear-Nonlinear-Kinetic) model in the RGC2Prey codebase for realistic retinal ganglion cell response modeling.

## Architecture

### Code Organization
```
RGC2Prey/
├── datasets/
│   ├── lnk_utils.py          # LNK parameter loading and configuration
│   └── sim_cricket.py        # Core LNK implementation in Cricket2RGCs
├── train_preycapture.py      # Main training script with LNK integration
├── docs/
│   ├── LNK_Integration_Guide.md    # General LNK usage guide
│   └── LNK_vs_LN_Model_Guide.md    # Model selection guide
└── examples/
    └── lnk_example.py        # LNK usage examples
```

### LNK Model Implementation

The LNK model is implemented in `Cricket2RGCs._rgc_activation()` method:

```python
def _rgc_activation(self, movie, sf, tf, rect_thr,
                   sf_surround=None, tf_surround=None,
                   lnk_params=None):
    """
    LNK Model Dynamics (vectorized across N cells):
    
    1. Linear filtering: x_t = conv(movie, sf, tf)
    2. Surround processing: x_s_t = conv(movie, sf_surround, tf_surround)  
    3. Adaptation state: a_{t+1} = a_t + dt * (α_d * F(x_t) - a_t) / τ
    4. Normalization: den_t = σ_0 + α * a_t
    5. Pre-output: ỹ_t = x_t/den_t + w_xs*x_s_t/den_t + β*a_t + b_out
    6. Output: r_t = softplus(g_out * ỹ_t)
    """
```

## Parameter Configuration

### Excel Parameter Sheet Format

Create an Excel sheet (e.g., "LNK_params") with these columns:

```
| tau | alpha_d | theta | sigma0 | alpha | beta | b_out | g_out | w_xs | dt |
|-----|---------|-------|--------|-------|------|-------|-------|------|----| 
| 0.1 |   1.0   |  0.0  |  1.0   |  0.1  | 0.0  |  0.0  |  1.0  |-0.1  |0.01|
```

### Parameter Broadcasting
- **Scalar values**: Same parameter for all RGC cells
- **Vector values**: Different parameter per cell (length must match number of RGCs)

Example for heterogeneous population:
```python
# 500 RGC cells with different adaptation time constants
tau_values = np.random.uniform(0.05, 0.2, 500)  # Different tau for each cell
```

### Parameter Validation
The system validates parameters automatically:
```python
from datasets.lnk_utils import validate_lnk_config

if not validate_lnk_config(lnk_config, num_rgcs):
    logging.warning("LNK validation failed, falling back to LN model")
```

## Advanced Features

### 1. Separate Center/Surround Filters

#### Automatic Surround Generation (Default)
```bash
python train_preycapture.py \
  --use_lnk_model \
  --use_separate_surround \
  --surround_sigma_ratio 4.0  # Surround sigma = 4x center sigma
```

#### Manual Surround from Excel
```bash
python train_preycapture.py \
  --use_lnk_model \
  --use_separate_surround \
  --surround_generation "sheet" \
  --sf_center_sheet_name "SF_center" \
  --sf_surround_sheet_name "SF_surround"
```

### 2. Adaptation Modes
```python
--lnk_adapt_mode "divisive"     # den_t = σ_0 + α * a_t (default)
--lnk_adapt_mode "subtractive"  # y_t = x_t - α * a_t
```

### 3. Binocular Processing
```bash
python train_preycapture.py \
  --use_lnk_model \
  --is_binocular \
  --interocular_dist 1.0  # cm
```

## Performance Optimization

### 1. Computational Efficiency
```python
# Vectorized computation across all RGC cells
a = torch.zeros((N, T_out), device=device, dtype=dtype)
for t in range(T_out):
    drive = torch.relu(x[:, t] - theta)  # F(x) for all cells
    da_dt = (alpha_d * drive - a_prev) / tau
    a[:, t] = torch.clamp_min(a_prev + dt * da_dt, 0.0)
```

### 2. Memory Management
- Uses in-place operations where possible
- Clamps adaptation states to prevent unbounded growth
- Efficient parameter broadcasting

### 3. GPU Acceleration
```python
# Automatically uses GPU if available and --is_GPU flag is set
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Integration with Existing Pipeline

### 1. Backward Compatibility
```python
# LN model (original)
if lnk_params is None:
    rgc_time = self._original_ln_computation(movie, sf, tf)
    
# LNK model (new)  
else:
    rgc_time = self._lnk_computation(movie, sf, tf, lnk_params)
```

### 2. Configuration Pipeline
```python
# 1. Load parameters
param_tables = load_unified_parameters(rf_params_file, ...)

# 2. Process LNK config
if args.use_lnk_model:
    lnk_params = process_parameter_table(param_tables['lnk'], num_rgcs, 'lnk')
    lnk_config = {'lnk_params': lnk_params, 'adapt_mode': args.lnk_adapt_mode}

# 3. Create dataset
train_config = create_cricket2rgcs_config(..., lnk_config=lnk_config)
train_dataset = Cricket2RGCs(**train_config)
```

## Debugging and Validation

### 1. Parameter Inspection
```python
from datasets.lnk_utils import get_lnk_config_summary
logging.info(f"LNK config: {get_lnk_config_summary(lnk_config)}")
```

### 2. Response Visualization
```python
# Compare LN vs LNK responses
sequence_ln, _, _, _ = dataset_ln[0]  # LN model
sequence_lnk, _, _, _ = dataset_lnk[0]  # LNK model

# Plot comparison
plot_two_path_comparison(sequence_ln, sequence_lnk, save_folder, 
                        "ln_vs_lnk_comparison.png")
```

### 3. Adaptation State Monitoring
```python
# Monitor adaptation states during computation
print(f"Adaptation range: [{a.min():.3f}, {a.max():.3f}]")
print(f"Normalization range: [{den.min():.3f}, {den.max():.3f}]")
```

## Common Implementation Patterns

### 1. Cell-Type Specific Parameters
```python
# Different parameters for ON vs OFF cells
if cell_type == 'ON':
    tau_default = 0.1
    w_xs_default = -0.1  # Suppressive surround
elif cell_type == 'OFF':
    tau_default = 0.15   # Slower adaptation
    w_xs_default = -0.2  # Stronger surround
```

### 2. Stimulus-Dependent Parameters
```python
# Adjust parameters based on stimulus statistics
if stimulus_contrast > 0.5:
    alpha_d *= 1.5  # Stronger adaptation for high contrast
```

### 3. Population Heterogeneity
```python
# Create diverse RGC population
tau_population = np.random.lognormal(np.log(0.1), 0.3, num_rgcs)
alpha_population = np.random.gamma(2.0, 0.05, num_rgcs)
```

## Testing and Validation

### 1. Unit Tests
```python
# Test parameter loading
def test_lnk_parameter_loading():
    params = load_lnk_parameters(test_file, "LNK_params", 100)
    assert params is not None
    assert 'lnk_params' in params
    
# Test response computation  
def test_lnk_response_shape():
    rgc_time = dataset._rgc_activation(movie, sf, tf, 0.0, lnk_params=lnk_params)
    assert rgc_time.shape == (num_rgcs, time_points)
```

### 2. Biological Validation
```python
# Check adaptation time constants
def validate_adaptation_timescale(rgc_responses, expected_tau):
    fitted_tau = fit_exponential_decay(rgc_responses)
    assert abs(fitted_tau - expected_tau) < 0.05
```

### 3. Performance Benchmarks
```python
# Compare computation time
import time
start = time.time()
_ = dataset[0]  # Generate one sample
elapsed = time.time() - start
print(f"LNK computation time: {elapsed:.3f}s")
```

## Error Handling

### 1. Parameter Validation
```python
def validate_lnk_params(lnk_params, num_rgcs):
    for key, value in lnk_params.items():
        if isinstance(value, np.ndarray):
            if len(value) != num_rgcs:
                raise ValueError(f"Parameter {key} length mismatch")
        elif not np.isfinite(value):
            raise ValueError(f"Invalid parameter {key}: {value}")
```

### 2. Numerical Stability
```python
# Prevent division by zero
den = torch.clamp_min(sigma0 + alpha * a, 1e-9)

# Prevent overflow in softplus
y_tilde = torch.clamp(g_out * y_tilde, -20, 20)
```

### 3. Graceful Fallback
```python
try:
    lnk_config = load_lnk_parameters(...)
except Exception as e:
    logging.warning(f"LNK loading failed: {e}. Using LN model.")
    lnk_config = None
```

## Best Practices

### 1. Parameter Selection
- Start with biological defaults
- Validate against experimental data
- Use population heterogeneity for realism
- Document parameter choices

### 2. Code Organization
- Keep LNK logic in `datasets/lnk_utils.py`
- Use type hints for clarity
- Add comprehensive docstrings
- Maintain backward compatibility

### 3. Performance
- Profile computational bottlenecks
- Use vectorized operations
- Consider memory vs speed tradeoffs
- Cache expensive computations

### 4. Documentation
- Document parameter biological meaning
- Provide usage examples
- Explain adaptation dynamics
- Reference literature sources

## Future Extensions

### 1. Multiple Adaptation States
```python
# Extend to multiple timescales
a_fast = torch.zeros((N, T_out))  # Fast adaptation
a_slow = torch.zeros((N, T_out))  # Slow adaptation
```

### 2. Spatial Adaptation
```python
# Position-dependent parameters
tau_map = create_spatial_parameter_map(tau_center, tau_periphery, positions)
```

### 3. Learning
```python
# Adaptive parameter optimization
tau_learned = optimize_parameters(responses_target, responses_model)
```

This technical guide provides the implementation details needed to effectively use and extend the LNK model in your RGC2Prey research.
