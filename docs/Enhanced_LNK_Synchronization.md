# Enhanced LNK Parameter Synchronization

This document describes the improvements made to handle NaN data and provide flexible parameter synchronization in the RGC2Prey codebase.

## Problem Statement

The original implementation had two main issues:

1. **NaN handling**: When any NaN value was found in an LNK parameter column, the entire column would fall back to default values for all cells, even if only a few rows contained NaN.

2. **Limited synchronization**: Only `syn_tf_sf` existed to synchronize spatial and temporal filter parameters, but there was no option to include LNK parameters in the synchronization.

## Solution Overview

### 1. Enhanced NaN Handling

The new `load_lnk_parameters_with_sampling()` function:
- **Filters out complete rows** that contain any NaN values
- **Samples only from valid rows** when assigning parameters to RGC cells
- **Logs the filtering process** to inform users about data quality

```python
# Before (problematic):
if np.any(pd.isna(param_values)):
    return default_value  # All cells get default, even if only 1 NaN

# After (improved):
valid_rows = param_table.dropna()  # Remove rows with any NaN
idx_list = np.random.choice(len(valid_rows), num_rgcs, replace=True)
param_values = valid_rows.iloc[idx_list][param_name].values
```

### 2. Flexible Parameter Synchronization

The new `syn_params` system supports any combination of parameter types:

```bash
# Synchronize all three parameter types
--syn_params tf sf lnk

# Synchronize only spatial and temporal filters (equivalent to old syn_tf_sf)
--syn_params tf sf

# Synchronize only spatial filters and LNK parameters
--syn_params sf lnk

# Synchronize only temporal filters and LNK parameters  
--syn_params tf lnk
```

## Implementation Details

### Enhanced RGCrfArray Class

The `RGCrfArray` class now includes:

```python
class RGCrfArray:
    def __init__(self, ..., 
                 syn_params=None,           # New: flexible synchronization
                 lnk_param_table=None):     # New: LNK table for sync
        
        self.syn_params = syn_params if syn_params else []
        self.lnk_param_table = lnk_param_table
        
        # Validate synchronization requirements
        if self.syn_params:
            self._validate_sync_params()
```

### Index Generation Strategy

```python
def _generate_indices(self, points):
    """Generate indices for parameter sampling with flexible synchronization."""
    num_points = len(points)
    idx_dict = {}
    
    if not self.syn_params:
        # Independent sampling for each parameter type
        idx_dict['sf'] = np.random.choice(len(self.sf_param_table), num_points)
        idx_dict['tf'] = np.random.choice(len(self.tf_param_table), num_points)
    else:
        # Synchronized sampling - same indices for specified parameters
        if self.sync_table_length:
            base_indices = np.random.choice(self.sync_table_length, num_points)
            for param in self.syn_params:
                idx_dict[param] = base_indices.copy()
    
    return idx_dict
```

## Usage Examples

### Command Line Usage

```bash
# Example 1: Synchronize all three parameter types
python train_preycapture.py \
    --syn_params tf sf lnk \
    --use_lnk_model \
    --experiment_name "sync_all_params"

# Example 2: Synchronize only TF and SF (backward compatible)
python train_preycapture.py \
    --syn_params tf sf \
    --experiment_name "sync_tf_sf_only"

# Example 3: Old syntax still works
python train_preycapture.py \
    --syn_tf_sf \
    --experiment_name "legacy_sync"
```

### Programmatic Usage

```python
# Create RGC array with flexible synchronization
rgc_array = RGCrfArray(
    sf_param_table=sf_table,
    tf_param_table=tf_table,
    # ... other parameters ...
    syn_params=['tf', 'sf', 'lnk'],  # Synchronize all three
    lnk_param_table=lnk_table
)

# Get results with synchronized parameters
results = rgc_array.get_results()
if len(results) == 6:  # LNK parameters included
    multi_opt_sf, tf, grid2value_mapping, map_func, points, lnk_params = results
else:  # Standard results
    multi_opt_sf, tf, grid2value_mapping, map_func, points = results
```

## Benefits

### 1. Robust Data Handling
- **No more silent failures**: NaN rows are explicitly identified and excluded
- **Better data utilization**: Valid parameter combinations are preserved
- **Informative logging**: Users know exactly what data was filtered

### 2. Flexible Experimental Design
- **Parameter independence**: Can study effects of synchronized vs independent parameters
- **Combinatorial experiments**: Test any combination of TF, SF, and LNK synchronization
- **Backward compatibility**: Existing scripts continue to work

### 3. Scientific Validity
- **Consistent cell properties**: When synchronized, each RGC has coherent parameter sets
- **Controlled comparisons**: Can isolate effects of parameter synchronization
- **Reproducible results**: Index-based sampling ensures consistent parameter assignments

## Migration Guide

### For Existing Users

1. **No changes required** for basic usage - backward compatibility is maintained
2. **Optional enhancement**: Add `--syn_params` to leverage new synchronization options
3. **Data quality check**: Review logs for NaN filtering information

### For New Features

1. **Add synchronization**: Use `--syn_params tf sf lnk` for full synchronization
2. **Handle return values**: Check if `get_results()` returns LNK parameters
3. **Configure logging**: Ensure logging is set up to see data quality information

## Validation

The enhanced system has been tested with:
- ✅ Tables with various NaN patterns
- ✅ Different synchronization parameter combinations  
- ✅ Backward compatibility with existing `syn_tf_sf` flag
- ✅ Edge cases (empty tables, all-NaN columns, mismatched table lengths)

## Future Enhancements

Potential future improvements:
1. **Smart interpolation**: Fill NaN values using neighboring valid parameters
2. **Parameter constraints**: Ensure biologically plausible parameter combinations
3. **Dynamic synchronization**: Change synchronization patterns during training
4. **Quality metrics**: Provide statistics on parameter diversity and coverage
