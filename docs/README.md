# RGC2Prey Documentation

This folder contains comprehensive documentation for the RGC2Prey project, with a focus on the LNK (Linear-Nonlinear-Kinetic) model implementation for realistic retinal ganglion cell response modeling.

## Documentation Overview

### 📋 Quick Start
- **[LNK vs LN Model Guide](LNK_vs_LN_Model_Guide.md)** - **START HERE**: Comprehensive guide on when to use LNK vs LN models
- **[LNK Usage Examples](LNK_Usage_Examples.md)** - Practical examples and comparison scripts

### 🔧 Technical Documentation  
- **[LNK Integration Guide](LNK_Integration_Guide.md)** - Original integration documentation
- **[LNK Technical Implementation](LNK_Technical_Implementation.md)** - Detailed technical implementation guide

## Key Recommendations

### For Prey Capture Research: **Use LNK Model** ✅

**Why LNK over LN for your research:**
1. **Dynamic visual scenes**: Prey capture involves rapid motion and changing contexts
2. **Biological realism**: LNK better represents real RGC responses with adaptation
3. **Temporal dynamics**: Captures realistic response patterns to moving stimuli
4. **Research quality**: Current state-of-the-art for RGC modeling

### Quick Start Commands

```bash
# Use LNK model (recommended)
python train_preycapture.py \
  --experiment_name "lnk_prey_capture" \
  --use_lnk_model \
  --lnk_sheet_name "LNK_params" \
  --num_samples 100

# Use LN model (baseline/comparison)
python train_preycapture.py \
  --experiment_name "ln_baseline" \
  --num_samples 100
```

## Document Summary

### 1. LNK vs LN Model Guide 📖
**Purpose**: Decision-making guide for model selection  
**Content**: 
- Model comparison and advantages
- When to use each model
- Performance considerations
- Parameter configuration
- Migration strategy

**Key insight**: LNK model provides significant advantages for dynamic visual scenarios like prey capture.

### 2. LNK Usage Examples 🚀
**Purpose**: Practical examples and comparison scripts  
**Content**:
- Quick start commands
- Parameter setup instructions
- Comparison scripts
- Expected differences
- Troubleshooting

**Key insight**: Easy-to-follow examples for immediate implementation.

### 3. LNK Technical Implementation ⚙️
**Purpose**: Detailed technical documentation  
**Content**:
- Code architecture
- Implementation details  
- Advanced features
- Performance optimization
- Best practices

**Key insight**: Complete technical reference for developers and advanced users.

### 4. LNK Integration Guide 📚
**Purpose**: Original integration documentation  
**Content**:
- Modular structure
- Parameter specifications
- Testing and validation
- API reference

**Key insight**: Foundation documentation for the LNK integration.

## LNK Model Advantages Summary

### Biological Realism 🧠
- Matches experimental RGC recordings
- Implements contrast adaptation
- Captures temporal dynamics

### Enhanced Processing 🔄
- Dynamic center-surround interaction
- Adaptive gain control  
- Realistic firing patterns

### Research Quality 📊
- State-of-the-art RGC modeling
- Publication-ready results
- Validates against experimental data

## Getting Started Checklist

### Prerequisites ✅
- [ ] Excel file with LNK parameters (`SimulationParams.xlsx`)
- [ ] Understanding of your experimental setup
- [ ] Decision on LN vs LNK model (recommendation: LNK)

### Setup Steps ✅
1. [ ] Read [LNK vs LN Model Guide](LNK_vs_LN_Model_Guide.md) for model selection
2. [ ] Follow [LNK Usage Examples](LNK_Usage_Examples.md) for implementation
3. [ ] Consult [LNK Technical Implementation](LNK_Technical_Implementation.md) for advanced features
4. [ ] Use [LNK Integration Guide](LNK_Integration_Guide.md) as reference

### Validation ✅
- [ ] Run comparison between LN and LNK models
- [ ] Validate parameters against literature
- [ ] Check response realism for your stimuli

## Support and Troubleshooting

### Common Issues
1. **LNK parameters not found**: Check Excel sheet name and format
2. **Parameter length mismatch**: Ensure parameters match number of RGCs
3. **Slow performance**: Reduce batch size or use GPU acceleration
4. **Unrealistic responses**: Validate parameter ranges

### Debug Steps
1. Enable detailed logging
2. Use validation functions
3. Compare with LN model baseline
4. Check parameter configuration summary

## Future Extensions

The modular LNK implementation supports:
- Multiple adaptation timescales
- Spatial parameter variations  
- Cell-type specific modeling
- Custom adaptation functions

## References

- Coppola, D. M., et al. (1998). The distribution of oriented contours in the real world. *PNAS*
- Keat, J., et al. (2001). Predicting every spike: a model for the responses of visual neurons. *Neuron*
- Pillow, J. W., et al. (2008). Spatio-temporal correlations and visual signalling in a complete neuronal population. *Nature*

---

**For questions or issues**, refer to the appropriate documentation file or consult the code comments in `datasets/lnk_utils.py` and `datasets/sim_cricket.py`.
