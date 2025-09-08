# RGC2Prey — Retinal Ganglion Cell to Prey Capture Model

## Overview
This project implements a comprehensive retinal encoding framework that uses recorded ON/OFF alpha RGC responses to build Linear-Nonlinear (LN) and Linear-Nonlinear-Kinetics (LNK) retina encoders. The goal is to probe how different retinal encoding strategies affect downstream behavior in cricket hunting and prey-capture tasks using CNN+LSTM readout models.

## Key Features
- **Multi-Modal Retinal Encoding**: Support for both LN and LNK models with adaptation dynamics
- **Binocular Vision**: Complete binocular disparity implementation for moving stimuli
- **Comprehensive Testing**: Extensive test suite for all major components
- **Synthetic Data Generation**: Cricket image synthesis, moving dot/bar stimuli, and background generation
- **End-to-End Pipeline**: From stimulus generation to behavior prediction with full visualization

## Quick Start

### Basic Training
```bash
# Standard LN model training
python train_preycapture.py --experiment_name "ln_test" --num_epochs 10

# LNK model with adaptation dynamics
python train_preycapture.py --experiment_name "lnk_test" --use_lnk_model --lnk_sheet_name "LNK_params"

# End-to-end training pipeline
python train_preycapture_end2end.py --experiment_name "end2end_test"
```

### Visualization and Analysis
- **Model Predictions**: [`visual_preycapture_prediction.py`](visual_preycapture_prediction.py)
- **Receptive Field Analysis**: [`visual_preycapture_end2end_RFs.py`](visual_preycapture_end2end_RFs.py)  
- **End-to-End Predictions**: [`visual_preycapture_end2end_prediction.py`](visual_preycapture_end2end_prediction.py)

### Testing and Validation
```bash
# Run comprehensive test suite
cd test/
python test_moving_bar_comprehensive.py    # Test binocular moving bar generation
python test_cricket_image_generator.py     # Test cricket image synthesis
python test_rgc_rf_generator.py           # Test RGC receptive field generation
python lnk_verify.py                      # Verify LNK implementation
```

## Core Components and Data Flow

### 1. Stimulus Generation
- **Movie Generation**: [`datasets.sim_cricket.SynMovieGenerator`](datasets/sim_cricket.py) - Synthetic cricket movies for RGC probing
- **Moving Stimuli**: [`datasets.movie_generator.MovingBarMovieGenerator`](datasets/movie_generator.py) - Moving bars/dots with binocular disparity support
- **Background Generation**: [`utils.background_generator`](utils/background_generator.py) - Natural background synthesis
- **Cricket Images**: [`datasets/cricketimages/`](datasets/cricketimages/) and [`datasets/cricketimages2/`](datasets/cricketimages2/) - Cricket stimulus databases

### 2. Retinal Processing 
- **RGC Array Creation**: [`datasets.sim_cricket.RGCrfArray`](datasets/sim_cricket.py) - Spatial/temporal RF construction and grid mapping
- **LN/LNK Models**: 
  - Standard LN responses: vectorized convolution + rectification
  - LNK responses: [`datasets.simple_lnk.compute_lnk_response`](datasets/simple_lnk.py) - Adaptation dynamics implementation
  - Parameter management: [`datasets.lnk_utils`](datasets/lnk_utils.py) - Loading, validation, and configuration utilities

### 3. Dataset Pipeline
- **RGC Response Generation**: [`datasets.sim_cricket.Cricket2RGCs`](datasets/sim_cricket.py) - Maps movies to RGC outputs
- **Binocular Processing**: Full binocular disparity support with horizontal shifts
- **Data Handling**: [`utils.data_handling`](utils/data_handling.py) - Checkpoint management and data loading

### 4. Neural Network Models
- **CNN+LSTM Architecture**: [`models.rgc2behavior.CNN_LSTM_ObjectLocation`](models/rgc2behavior.py) - Standard object localization
- **RGC-Aware Model**: [`models.rgc2behavior.RGC_CNN_LSTM_ObjectLocation`](models/rgc2behavior.py) - RGC-specific processing
- **Feature Extraction**: [`models.rgc2behavior.ParallelCNNFeatureExtractor`](models/rgc2behavior.py) - Multi-channel CNN features

### 5. Training and Optimization
- **Main Training**: [`train_preycapture.py`](train_preycapture.py) - Standard training pipeline
- **End-to-End Training**: [`train_preycapture_end2end.py`](train_preycapture_end2end.py) - Comprehensive pipeline
- **Utilities**: [`utils.initialization`](utils/initialization.py) - Logging, reproducibility, and setup

## Major Scripts & Utilities

### Training Scripts
- **[`train_preycapture.py`](train_preycapture.py)** - Main training orchestration with RF creation, dataset instantiation, and model training
- **[`train_preycapture_end2end.py`](train_preycapture_end2end.py)** - End-to-end training pipeline

### Core Datasets & Models
- **[`datasets/sim_cricket.py`](datasets/sim_cricket.py)** - RGC array creation (`RGCrfArray`) and cricket-to-RGC dataset (`Cricket2RGCs`)
- **[`datasets/movie_generator.py`](datasets/movie_generator.py)** - Moving bar/dot movie generation with binocular disparity
- **[`datasets/lnk_utils.py`](datasets/lnk_utils.py)** - LNK parameter loading, validation, and configuration utilities
- **[`datasets/simple_lnk.py`](datasets/simple_lnk.py)** - Compact LNK implementation with adaptation dynamics
- **[`models/rgc2behavior.py`](models/rgc2behavior.py)** - CNN/LSTM architectures for behavior prediction

### Visualization & Analysis
- **[`visual_preycapture_prediction.py`](visual_preycapture_prediction.py)** - Model prediction visualization
- **[`visual_preycapture_end2end_RFs.py`](visual_preycapture_end2end_RFs.py)** - Receptive field extraction and visualization
- **[`visual_preycapture_end2end_prediction.py`](visual_preycapture_end2end_prediction.py)** - End-to-end prediction analysis

### Utility Scripts
- **[`generate_background.py`](generate_background.py)** - Background image generation
- **[`image_processing.py`](image_processing.py)** - Image processing utilities
- **[`single_contrast_image_generator.py`](single_contrast_image_generator.py)** - Contrast-specific image generation
- **[`complete_time_estimation.py`](complete_time_estimation.py)** - Training time estimation
- **[`demo_moving_dot.py`](demo_moving_dot.py)** - Moving dot demonstration

### Examples & Documentation
- **[`examples/lnk_example.py`](examples/lnk_example.py)** - LNK parameter creation and validation examples
- **[`docs/`](docs/)** - Comprehensive documentation including LNK guides and technical implementation details

## Configuration & Parameters

### Runtime Configuration
- **RF/LNK Parameter Excel**: Training scripts read Excel files referenced as `rf_params_file` (configurable in training scripts)
- **Model Selection**: 
  - Enable LNK: `--use_lnk_model` flag in training scripts
  - LNK parameters: `--lnk_sheet_name` (default: "LNK_params")
  - Adaptation mode: `--lnk_adapt_mode` (choices: divisive, subtractive)
- **Surround Control**: Use `set_s_scale` for LN-surround; LNK uses `w_xs` for center-surround dynamics

### Important Training Flags
```bash
--is_GPU                 # Enable GPU acceleration
--batch_size 64          # Training batch size
--num_epochs 100         # Number of training epochs  
--num_worker 4           # DataLoader workers
--do_not_train           # Skip training (data generation only)
--is_generate_movie      # Generate stimulus movies
--experiment_name "test" # Experiment identifier
```

### LNK Integration Points
- **Parameter Loading**: [`datasets.lnk_utils.load_lnk_parameters`](datasets/lnk_utils.py) - Canonical parameter loader
- **Dataset Integration**: 
  - `RGCrfArray` produces `lnk_params` when `use_lnk_override=True`
  - `Cricket2RGCs` uses `lnk_params` and calls appropriate LNK computation methods
- **Implementation**: [`datasets.sim_cricket.Cricket2RGCs._compute_rgc_time`](datasets/sim_cricket.py) - Main LNK computation entry point

## Testing Framework

### Comprehensive Test Suite
Located in [`test/`](test/) directory:

- **[`test_moving_bar_comprehensive.py`](test/test_moving_bar_comprehensive.py)** - Comprehensive moving bar generation testing with binocular disparity
- **[`test_cricket_image_generator.py`](test/test_cricket_image_generator.py)** - Cricket image synthesis validation  
- **[`test_rgc_rf_generator.py`](test/test_rgc_rf_generator.py)** - RGC receptive field generation testing
- **[`test_spot_coord_prediction.py`](test/test_spot_coord_prediction.py)** - Spot coordinate prediction with CNN-LSTM models
- **[`lnk_verify.py`](test/lnk_verify.py)** - LNK implementation verification
- **[`lnk_compare.py`](test/lnk_compare.py)** - LNK model comparison utilities
- **[`test_model_separation.py`](test/test_model_separation.py)** - Model architecture separation testing
- **[`test_load.py`](test/test_load.py) / [`test_save.py`](test/test_save.py)** - Data loading/saving validation

### Test Features
- **Matplotlib-Optional**: Tests work with or without matplotlib for headless environments
- **Text-Based Visualization**: Fallback ASCII visualizations when graphics unavailable
- **Comprehensive Coverage**: Tests cover data generation, model training, and visualization pipelines

## Experiments & Extensions

### Supported Features
- **Multi-Channel Processing**: ON/OFF channels with binocular inputs
- **Multiple Grid Support**: Various RGC array configurations and grid mappings
- **Flexible Filter Loading**: Separate center/surround filter configuration
- **Binocular Disparity**: Complete horizontal disparity implementation for moving stimuli

### Parameter Tuning
Experiment with different configurations via Excel parameter sheets:
- **LNK Parameters**: `tau`, `alpha_d`, `w_xs`, `sigma0` - see [`docs/LNK_Usage_Examples.md`](docs/LNK_Usage_Examples.md)
- **Model Comparison**: LN vs LNK performance - see [`docs/LNK_vs_LN_Model_Guide.md`](docs/LNK_vs_LN_Model_Guide.md)
- **Rapid Prototyping**: Use [`examples/lnk_example.py`](examples/lnk_example.py) for quick parameter testing

### Performance Considerations
- **LNK Computational Cost**: ~2-3× slower than LN due to iterative time-stepping per cell
- **GPU Acceleration**: Controlled via `--is_GPU` flag with automatic device selection
- **Memory Management**: Efficient DataLoader configuration with `--num_worker` setting

## Code Organization

### Key Implementation Locations
- **Training Orchestration**: [`train_preycapture.py`](train_preycapture.py) - Main training loop with timing and checkpointing
- **RGC Array Synthesis**: [`datasets.sim_cricket.RGCrfArray`](datasets/sim_cricket.py) - Filter generation and spatial mapping
- **Movie Generation**: [`datasets.sim_cricket.SynMovieGenerator`](datasets/sim_cricket.py) and [`datasets.movie_generator`](datasets/movie_generator.py)
- **LNK Implementation**: 
  - Parameter utilities: [`datasets/lnk_utils.py`](datasets/lnk_utils.py)
  - Core solver: [`datasets/simple_lnk.compute_lnk_response`](datasets/simple_lnk.py)
- **Model Architectures**: [`models/rgc2behavior`](models/rgc2behavior.py) - CNN/LSTM implementations
- **Visualization Utilities**: [`utils.utils`](utils/utils.py), [`utils.tools`](utils/tools.py) - Plotting and analysis helpers

## Getting Started - Step by Step

### 1. Environment Setup
```bash
# Ensure you have the required dependencies
pip install torch torchvision matplotlib opencv-python pandas numpy
```

### 2. Parameter Configuration
Create or verify your LNK parameter sheet (SimulationParams.xlsx):
- See examples in [`examples/lnk_example.py`](examples/lnk_example.py)
- Detailed parameter guides in [`docs/LNK_Usage_Examples.md`](docs/LNK_Usage_Examples.md)

### 3. Quick Test Run
```bash
# Small test run to verify setup
python train_preycapture.py --experiment_name test_ln --num_samples 10 --num_epochs 1 --batch_size 2

# Test with LNK model
python train_preycapture.py --experiment_name test_lnk --use_lnk_model --lnk_sheet_name LNK_params --num_samples 10 --num_epochs 1 --batch_size 2
```

### 4. Validation and Visualization
```bash
# Run comprehensive tests
cd test/
python test_moving_bar_comprehensive.py

# Visualize results
python ../visual_preycapture_prediction.py
python ../visual_preycapture_end2end_RFs.py
```

## Documentation

### Comprehensive Guides
- **[`docs/README.md`](docs/README.md)** - Project documentation index
- **[`docs/LNK_vs_LN_Model_Guide.md`](docs/LNK_vs_LN_Model_Guide.md)** - Model comparison and selection guide
- **[`docs/LNK_Integration_Guide.md`](docs/LNK_Integration_Guide.md)** - Integration and setup instructions
- **[`docs/LNK_Technical_Implementation.md`](docs/LNK_Technical_Implementation.md)** - Technical implementation details and equations
- **[`docs/LNK_Usage_Examples.md`](docs/LNK_Usage_Examples.md)** - Parameter configuration and usage examples

### Test Documentation
- **[`test/README.md`](test/README.md)** - Comprehensive testing guide and usage instructions

## Recent Updates

### New Features
- **Enhanced Binocular Support**: Complete binocular disparity implementation with horizontal-only shifts for moving stimuli
- **Comprehensive Test Suite**: Organized test framework with 10+ test scripts covering all major components
- **Improved Documentation**: Updated guides and examples for better user experience
- **Code Organization**: Better project structure with organized test directory and utility scripts

### Technical Improvements
- **MovingBarMovieGenerator**: Enhanced with proper binocular disparity calculations
- **Test Framework**: Matplotlib-optional testing with text-based fallbacks
- **File Organization**: Clean separation of test scripts, utilities, and core functionality
- **Error Handling**: Improved robustness in data loading and model initialization

## Project Structure
```
RGC2Prey/
├── README.md                           # This file
├── datasets/                           # Core data generation
│   ├── sim_cricket.py                 # RGC arrays and cricket datasets  
│   ├── movie_generator.py             # Moving stimulus generation
│   ├── lnk_utils.py                   # LNK parameter utilities
│   ├── simple_lnk.py                  # LNK implementation
│   └── cricketimages/                 # Cricket stimulus databases
├── models/                            # Neural network architectures
│   └── rgc2behavior.py                # CNN/LSTM models
├── utils/                             # Utility functions
│   ├── data_handling.py               # Data loading and checkpointing
│   ├── initialization.py              # Setup and logging
│   └── tools.py                       # Helper functions
├── test/                              # Comprehensive test suite
│   ├── README.md                      # Test documentation
│   ├── test_moving_bar_comprehensive.py  # Binocular moving bar tests
│   ├── test_cricket_image_generator.py   # Cricket image synthesis tests
│   ├── test_rgc_rf_generator.py          # RGC RF generation tests
│   └── lnk_verify.py                     # LNK verification tests
├── docs/                              # Documentation
│   ├── LNK_Technical_Implementation.md
│   ├── LNK_Usage_Examples.md
│   └── LNK_vs_LN_Model_Guide.md
├── examples/                          # Usage examples
│   └── lnk_example.py                 # LNK parameter examples
├── train_preycapture.py               # Main training script
├── train_preycapture_end2end.py       # End-to-end training
└── visual_preycapture_*.py            # Visualization scripts
```

---

For technical support or questions about implementation details, refer to the comprehensive documentation in the [`docs/`](docs/) directory or examine the test scripts in [`test/`](test/) for usage examples.
