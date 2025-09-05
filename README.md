# RGC2Prey — README

Short description
- Goal: Use recorded ON / OFF alpha RGC responses to build LN and LNK retina encoders, then probe how retinal encoding affects downstream behavior (cricket hunting / prey-capture) using a CNN+LSTM readout.

Quick start
- Inspect / run training (produces RGC datasets and trains the CNN+LSTM):
  - Main training script: [`train_preycapture.py`](train_preycapture.py)
  - Example: python train_preycapture.py --experiment_name "lnk_test" --use_lnk_model --lnk_sheet_name "LNK_params"
- Visual prediction tools / RF extraction:
  - Visual prediction script: [`visual_preycapture_prediction.py`](visual_preycapture_prediction.py)
  - End-to-end RF extraction & visualization: [`visual_preycapture_end2end_RFs.py`](visual_preycapture_end2end_RFs.py)
  - End-to-end prediction visualization: [`visual_preycapture_end2end_prediction.py`](visual_preycapture_end2end_prediction.py)

Core components and data flow (high-level)
1. Movie / stimulus generation
   - [`datasets.SynMovieGenerator`](datasets/sim_cricket.py) generates synthetic movies used to probe RGCs.
     - See [`datasets.sim_cricket.SynMovieGenerator`](datasets/sim_cricket.py)
2. RGC array creation and RFs
   - RF construction and management: [`datasets.RGCrfArray`](datasets/sim_cricket.py)
     - Returns spatial filters, temporal filters, grid mappings, and optional LNK parameters via [`datasets.sim_cricket.RGCrfArray.get_results`](datasets/sim_cricket.py)
3. Dataset: movies → RGC responses
   - Dataset wrapper: [`datasets.Cricket2RGCs`](datasets/sim_cricket.py)
     - It can compute standard LN responses (vectorized conv + rectification) or LNK responses (adaptation dynamics).
     - LNK integration points:
       - LNK parameter loader and helpers: [`datasets.lnk_utils.load_lnk_parameters`](datasets/lnk_utils.py) and related utilities (`create_cricket2rgcs_config`, `validate_lnk_config`, `get_lnk_config_summary`) — see [`datasets/lnk_utils.py`](datasets/lnk_utils.py)
       - A simpler LNK implementation also exists in [`datasets.simple_lnk.compute_lnk_response`](datasets/simple_lnk.py)
     - Implementation entrypoint inside dataset: [`datasets.sim_cricket.Cricket2RGCs._compute_rgc_time`](datasets/sim_cricket.py)
4. Downstream model (readout)
   - CNN + LSTM readout: [`models.rgc2behavior.CNN_LSTM_ObjectLocation`](models/rgc2behavior.py)
   - Alternate RGC-aware model: [`models.rgc2behavior.RGC_CNN_LSTM_ObjectLocation`](models/rgc2behavior.py)
   - CNN feature extractors and normalization layers are in [`models.rgc2behavior.ParallelCNNFeatureExtractor*` ](models/rgc2behavior.py)
5. Training loop and utilities
   - Main training orchestration: [`train_preycapture.py`](train_preycapture.py) — constructs RFs (RGCrfArray) → dataset (Cricket2RGCs) → DataLoader → model → optimizer → scheduler → checkpoint saving.
   - Checkpoint loader / saver: [`utils.data_handling.CheckpointLoader`](utils/data_handling.py) and [`utils.data_handling.save_checkpoint`](utils/data_handling.py)
   - Logging & reproducibility: [`utils.initialization.process_seed`](utils/initialization.py) and [`utils.initialization.initialize_logging`](utils/initialization.py)
   - Small helpers: [`utils.tools.MovieGenerator`, `utils.tools.timer`, utils.utils plotting helpers`](utils/tools.py) and [`utils.utils.plot_*` functions](utils/utils.py)

Major scripts & files (what they do; open them directly)
- [`train_preycapture.py`](train_preycapture.py)
  - Orchestrates RF creation, dataset instantiation, model definition, training loop, checkpointing and visualizations.
  - Reads RF parameter Excel at runtime (path set by `rf_params_file` inside the script).
- [`datasets/sim_cricket.py`](datasets/sim_cricket.py)
  - RGCrfArray: builds RF mosaics and returns filters and locations. See [`datasets.sim_cricket.RGCrfArray`](datasets/sim_cricket.py)
  - Cricket2RGCs: dataset that maps movies to RGC outputs. See [`datasets.sim_cricket.Cricket2RGCs`](datasets/sim_cricket.py)
- [`datasets/lnk_utils.py`](datasets/lnk_utils.py)
  - LNK parameter loading, validation, and helpers. See [`datasets.lnk_utils.load_lnk_parameters`](datasets/lnk_utils.py)
- [`datasets/simple_lnk.py`](datasets/simple_lnk.py)
  - A compact LNK implementation: [`datasets.simple_lnk.compute_lnk_response`](datasets/simple_lnk.py)
- [`models/rgc2behavior.py`](models/rgc2behavior.py)
  - CNN/LSTM architectures (`CNN_LSTM_ObjectLocation`, `RGC_CNN_LSTM_ObjectLocation`) and CNN feature extractor classes.
  - See [`models.rgc2behavior.CNN_LSTM_ObjectLocation`](models/rgc2behavior.py) and [`models.rgc2behavior.RGC_CNN_LSTM_ObjectLocation`](models/rgc2behavior.py)
- Visualization and analysis utilities
  - [`visual_preycapture_prediction.py`](visual_preycapture_prediction.py)
  - [`visual_preycapture_end2end_RFs.py`](visual_preycapture_end2end_RFs.py)
  - [`visual_preycapture_end2end_prediction.py`](visual_preycapture_end2end_prediction.py)
- Examples & tests
  - LNK example: [`examples/lnk_example.py`](examples/lnk_example.py) (shows parameter creation, loading and validation)
  - Tests: [`test/lnk_verify.py`](test/lnk_verify.py) and [`test/test_model_separation.py`](test/test_model_separation.py)

Configuration & parameters
- RF / LNK parameter Excel: the training scripts read an Excel file referenced as `rf_params_file` inside scripts (default in-code location; edit in [`train_preycapture.py`](train_preycapture.py) or pass flags).
- LNK vs LN:
  - Enable LNK with flag `--use_lnk_model` in [`train_preycapture.py`](train_preycapture.py).
  - LNK parameter sheet name controlled via `--lnk_sheet_name` (defaults to "LNK_params").
  - LNK adaptation mode controlled with `--lnk_adapt_mode` (choices: divisive, subtractive).
  - For LN-surround control use `set_s_scale` (passed to RGCrfArray). When LNK is used, LNK `w_xs` controls center-surround.
- Important runtime flags: `--is_GPU`, `--batch_size`, `--num_epochs`, `--num_worker`, `--do_not_train`, `--is_generate_movie`, etc. See [`train_preycapture.parse_args()`](train_preycapture.py).

How the LNK model is wired in
- Parameter loading: [`datasets.lnk_utils.load_lnk_parameters`](datasets/lnk_utils.py) is the canonical loader.
- Dataset usage:
  - RGCrfArray produces `lnk_params` when `use_lnk_override=True`.
  - Cricket2RGCs uses `lnk_params` (and optional separate surround filters) and either:
    - calls `compute_lnk_response()` from [`datasets/simple_lnk.py`](datasets/simple_lnk.py), or
    - internally runs the LNK logic inside [`datasets.sim_cricket.Cricket2RGCs._compute_rgc_time`](datasets/sim_cricket.py).
- Typical LNK equations / implementation details are documented in `/docs/LNK_Technical_Implementation.md` and supporting md docs under `/docs/`.

Notes on experiments & extensions
- The pipeline supports:
  - ON / OFF channels, binocular inputs, multiple grids, and separate center / surround filter loading.
  - Tuning experiments: vary `tau`, `alpha_d`, `w_xs`, `sigma0`, etc., via the Excel sheet; see [`docs/LNK_Usage_Examples.md`](docs/LNK_Usage_Examples.md) and [`docs/LNK_vs_LN_Model_Guide.md`](docs/LNK_vs_LN_Model_Guide.md).
  - Rapid prototyping via [`examples/lnk_example.py`](examples/lnk_example.py).
- Performance considerations:
  - LNK cost: iterative time-stepping per cell -> ~2–3× slower than LN (see docs).
  - GPU acceleration: controlled via `--is_GPU` and device selection in code (`train_preycapture.py`).

Where to look for specific code pieces
- Training loop and timing instruments: [`train_preycapture.py`](train_preycapture.py) (timers, accumulation, scheduler).
- RGC array / filter synthesis: [`datasets.sim_cricket.RGCrfArray`](datasets/sim_cricket.py)
- Movie generator: [`datasets.sim_cricket.SynMovieGenerator`](datasets/sim_cricket.py)
- LNK utilities and parameter processing: [`datasets/lnk_utils.py`](datasets/lnk_utils.py)
- Compact LNK solver: [`datasets/simple_lnk.compute_lnk_response`](datasets/simple_lnk.py)
- Models: [`models/rgc2behavior.CNN_LSTM_ObjectLocation`](models/rgc2behavior.py) and [`models/rgc2behavior.RGC_CNN_LSTM_ObjectLocation`](models/rgc2behavior.py)
- Plotting / helper utilities: [`utils.utils`](utils/utils.py), [`utils.tools`](utils/tools.py)

Recommended first steps to reproduce results
1. Create / verify LNK parameter sheet (SimulationParams.xlsx). See examples in [`examples/lnk_example.py`](examples/lnk_example.py) and docs in [`docs/LNK_Usage_Examples.md`](docs/LNK_Usage_Examples.md).
2. Run a small training / data-generation job:
   - Example: python train_preycapture.py --experiment_name test_ln --num_samples 10 --num_epochs 1 --batch_size 2
   - To enable LNK: add `--use_lnk_model --lnk_sheet_name LNK_params`
3. Use visualization scripts to inspect RFs and model predictions:
   - [`visual_preycapture_prediction.py`](visual_preycapture_prediction.py)
   - [`visual_preycapture_end2end_RFs.py`](visual_preycapture_end2end_RFs.py)

Useful docs (open these)
- General guides: [`docs/LNK_vs_LN_Model_Guide.md`](docs/LNK_vs_LN_Model_Guide.md)
- Integration & technical details: [`docs/LNK_Integration_Guide.md`](docs/LNK_Integration_Guide.md), [`docs/LNK_Technical_Implementation.md`](docs/LNK_Technical_Implementation.md)
- Usage examples: [`docs/LNK_Usage_Examples.md`](docs/LNK_Usage_Examples.md)
- Project docs index: [`docs/README.md`](docs/README.md)

Tests and verification
- Example test scripts:
  - LNK verification: [`test/lnk_verify.py`](test/lnk_verify.py)
  - Model separation tests: [`test/test_model_separation.py`](test/test_model_separation.py)
