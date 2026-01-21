# Simulation (Methods-style)

This document describes the end-to-end synthetic "cricket" simulation and the corresponding RGC-to-behavior training pipeline, with an emphasis on conceptual clarity and parameter accuracy for replication.

Core implementation files:
- Dataset + simulation: [datasets/sim_cricket.py](datasets/sim_cricket.py)
- Simplified LNK model: [datasets/simple_lnk.py](datasets/simple_lnk.py)
- Binocular disparity helpers: [utils/trajectory.py](utils/trajectory.py)
- Model (CNN+LSTM): [models/rgc2behavior.py](models/rgc2behavior.py)
- Training entrypoint / CLI surface: [train_preycapture.py](train_preycapture.py)
- Example training scripts: [examples/running_bash/train_preycapture_25101402.sh](examples/running_bash/train_preycapture_25101402.sh), [examples/running_bash/train_preycapture_25100502.sh ](examples/running_bash/train_preycapture_25100502.sh%20)


## 1) Overview of the pipeline

Each training sample is generated on-the-fly and follows the same conceptual stages:

1. **Stimulus synthesis** ("movie")
   - A naturalistic background image and a cricket image are composited frame-by-frame.
   - The cricket (object) and background follow independent stochastic trajectories.
   - The cricket is progressively rescaled over time; in binocular mode the scale is converted into a disparity signal.

2. **Retinal encoding** (LN or simplified LNK)
   - A population of RGC receptive fields (spatial + temporal filters) is instantiated on a hexagonal grid.
   - Each RGC produces a time series response to the stimulus.
   - Optional post-processing includes spike quantization, smoothing, additive noise, and rectification.

3. **Spatial pooling to a fixed grid**
   - The irregularly spaced RGC outputs are mapped onto a fixed-size 2D grid per time step.

4. **Behavioral decoding model**
   - A CNN converts each grid frame into a feature vector.
   - An LSTM integrates the temporal sequence and predicts the object position $(x,y)$ per frame.
   - Optionally, an auxiliary background-prediction head is trained jointly.


## 2) Coordinate systems and outputs

- **Object/background trajectories** are generated in a centered coordinate frame with bounds given by `--boundary_size` (default string "(220, 140)").
- **Target coordinate scaling**:
  - If `--is_norm_coords` is enabled, target trajectories are divided by
    $$\mathrm{norm\_path\_fac} = \left[\tfrac{\Delta x}{2}, \tfrac{\Delta y}{2}\right]$$
    where $\Delta x = x_{\max}-x_{\min}$ from `--xlim` and $\Delta y = y_{\max}-y_{\min}$ from `--ylim`.
- **Dataset return shapes** (conceptual):
  - `grid_seq`: a time sequence of fixed 2D grids with multiple channels, arranged as `[T_out, C, X, Y]` after an internal permutation.
  - `targets`: `[T_out, 2]` for object $(x,y)$.
  - `bg_info`: `[T_out, 2]` for background $(x,y)$ (used for auxiliary loss if enabled).


## 3) Stimulus synthesis (synthetic movies)

Implemented in `SynMovieGenerator` in [datasets/sim_cricket.py](datasets/sim_cricket.py).

### 3.1 Trajectories

- Two independent stochastic trajectories are generated:
  - **Object path** controlled by `--prob_stay_ob`, `--prob_mov_ob`, `--momentum_decay_ob`, `--velocity_randomness_ob`, `--angle_range_ob`.
  - **Background path** controlled by `--prob_stay_bg`, `--prob_mov_bg`, `--momentum_decay_bg`, `--velocity_randomness_bg`, `--angle_range_bg`.
- `--num_ext` prepends a static segment by repeating the first position for that many frames.

### 3.2 Size (scaling) schedule

- The cricket is rescaled over time using a **piecewise-constant schedule** derived from background motion:
  - A scale starts at `--start_scaling` and increases toward `--end_scaling`.
  - The number of increments equals the number of background steps where the background position changes.
- `--dynamic_scaling` perturbs both endpoints per sample:
  - `start_scaling += U(0, dynamic_scaling)` and `end_scaling -= U(0, dynamic_scaling)`, then ordered to keep `start_scaling <= end_scaling`.

### 3.3 Binocular disparity

If `--is_binocular` is enabled:

- The per-frame scaling values are mapped linearly to viewing distances between **21 cm** and **4 cm** (constants in [utils/trajectory.py](utils/trajectory.py)).
- Distances are converted to disparity (degrees) given interocular distance `--interocular_dist` (cm):
  $$\theta = 2\,\arctan\left(\frac{\mathrm{IOD}/2}{d}\right)$$
- Disparity is then converted to pixels using:
  - `deg2um=32.5`, `pix2um=4.375`, `scaling=0.54` (defaults in [utils/trajectory.py](utils/trajectory.py)).
- `--fix_disparity FLOAT` overrides the computed disparities and uses a constant disparity per frame.

### 3.4 Image compositing and contrast controls

- A background image ("bottom") and cricket image ("top") are overlaid frame-by-frame.
- The implementation overlays **only the green channel** of the top onto the bottom (alpha-masked), then returns the green channel as the stimulus.
- Optional appearance manipulations:
  - `--bottom_contrast`, `--top_contrast`: contrast scaling around 0.5.
  - `--mean_diff_offset`: adjusts the green-channel means of bottom vs top with a constrained split so the total offset equals `mean_diff_offset`.

### 3.5 Coordinate correction of the cricket image

- The generated object path can be corrected using a per-image offset loaded from a `.mat` summary file selected by `--coord_adj_type`.
- The correction is scaled by the per-frame scaling factor and applied with a sign given by `--coord_adj_dir`.
- `--is_reverse_xy` swaps the correction’s x/y fields.


## 4) RGC receptive fields and population geometry

Implemented by `RGCrfArray` in [datasets/sim_cricket.py](datasets/sim_cricket.py). Parameter values are loaded from an Excel workbook (default path is hard-coded in the training script).

### 4.1 RGC locations (hex grid)

- RGC centers are generated on a noisy hexagonal grid within the region defined by `--xlim` and `--ylim`.
- `--target_num_centers` controls the number of RGCs in the primary grid.
- A second grid can be generated with `--anti_alignment` controlling how far it is shifted relative to the first (0 = overlapping, 1 = maximally spaced).

### 4.2 Spatial filters

- Spatial filters are generated per RGC from a parameter table (`--sf_sheet_name`).
- **LN mode** uses a Difference-of-Gaussians-style parameterization (center + surround) where surround strength uses `s_scale`.
- Spatial filters can be constrained by:
  - `--sf_constraint_method circle`: apply a circular cutoff.
  - `--sf_mask_radius`: the nominal dendritic radius scale (pixels).
- `--set_surround_size_scalar` overrides how the surround *size* is derived:
  - In LN mode, surround sigmas are set to `sigma_center * sf_scalar * set_surround_size_scalar`.
  - In LNK override mode, the surround is generated by scaling the center sigmas by `--surround_sigma_ratio`, and the circular-mask cutoff uses `--set_surround_size_scalar` if provided (otherwise it follows `--surround_sigma_ratio`).

### 4.3 Temporal filters

- Temporal filters are generated from a table (`--tf_sheet_name`).
- `--temporal_filter_len` sets the length.
- `--set_biphasic_scale` (if set) ties the second lobe amplitude to the first.
- `--is_reversed_tf` negates the temporal filter.
- `--is_pixelized_tf` replaces the temporal filter with a delta (last sample = 1).

### 4.4 Parameter synchronization across cells

- `--syn_params` can enforce shared table indices across specified parameter groups (e.g., synchronize TF and SF sampling).
- When synchronization is enabled, NaN rows are avoided by re-sampling indices from valid rows.


## 5) Retinal encoding: LN vs simplified LNK

Implemented in `Cricket2RGCs._compute_rgc_time` in [datasets/sim_cricket.py](datasets/sim_cricket.py).

### 5.1 LN pathway

1. Spatial projection: dot product of movie frames with each RGC’s spatial filter.
2. Temporal filtering: grouped 1D convolution per RGC.
  - Note: in LN mode, “surround” is not computed as a separate convolution during encoding; it is already embedded in the single DoG spatial filter constructed in Section 4.2.
3. Optional post-processing:
   - `--fr2spikes` with `--quantize_scale`: Poisson sampling on a clipped rate.
   - `--smooth_data`: Gaussian smoothing in time.
   - `--add_noise`: additive Gaussian noise (see noise sampling below).
   - `--is_rectified`: rectification via either:
     - `--rectified_mode hard`: clamp below threshold.
     - `--rectified_mode softplus`: softplus centered at threshold, with `--rectified_softness`.

### 5.2 Simplified LNK pathway

When `--use_lnk_model` is enabled, the dataset uses `compute_lnk_response` from [datasets/simple_lnk.py](datasets/simple_lnk.py):

- Two spatiotemporal drive signals are computed:
  - center drive $x_c$ (center SF + TF)
  - surround drive $x_s$ (surround SF + TF)
- An adaptation state $a_t$ is updated with a first-order dynamic:
  $$a_{t+1} = a_t + dt\,\frac{\alpha_d\,\max(0, x_c-\theta) - a_t}{\tau}$$
- The instantaneous response uses **divisive normalization** with optional additive term:
  $$y_t = \frac{x_c + w_{xs} x_s + \epsilon_t}{\sigma_0 + \alpha a_t} + \beta a_t + b_{out}$$
  and an output softplus nonlinearity scaled by $g_{out}$.

Notes for replication:
- The simplified LNK implementation currently corresponds to a **divisive** form.
- The CLI flag `--lnk_adapt_mode` exists, but the *simplified* LNK path used by `Cricket2RGCs` does not branch on that setting; `subtractive` vs `divisive` behavior appears to be implemented in the legacy utilities in [datasets/lnk_utils.py](datasets/lnk_utils.py), not in the simplified path.

### 5.3 Noise sampling (LN and simplified LNK)

If `--add_noise` is enabled:
- If `--rgc_noise_std > 0`, that fixed standard deviation is used.
- Else if `--rgc_noise_std_max` is set, a **log-uniform** standard deviation is sampled per sample in $[\mathrm{max}/32,\,\mathrm{max}]$.


## 6) Mapping irregular RGC outputs to a fixed grid

After per-RGC responses are computed, the population is pooled into a fixed grid per frame. The mapping method is controlled by `--grid_generate_method`:

- `closest`: each grid pixel takes the nearest RGC value.
- `decay`: a distance-weighted decay (controlled by `--tau`).
- `circle`: a circular mask per RGC (controlled by `--mask_radius`).

The final grid resolution is set by `--grid_size_fac` relative to the coordinate extents (`--xlim`, `--ylim`).


## 7) Channel configurations (binocular, ON/OFF, two-grids)

Channel stacking is handled in `Cricket2RGCs.__getitem__` in [datasets/sim_cricket.py](datasets/sim_cricket.py).

- **Default** (monocular, one grid): $C=1$.
- **Binocular only** (`--is_binocular`): $C=2$ (one channel per eye), with a shared RGC grid.
- **ON/OFF** (`--is_both_ON_OFF`): $C=2$ (ON + OFF), with two independently generated grids/filters.
- **ON/OFF + binocular**: $C=4$ (ON-left, ON-right, OFF-left, OFF-right).
- **Two grids** (`--is_two_grids`): $C=2$ (grid-1 + grid-2), both with the same sign semantics.
- **Two grids + binocular**: paired mapping by default: each eye is processed by one grid, yielding $C=2$ (not 4). This is intentional in the current implementation.


## 8) Behavioral decoder model and training objective

Training is implemented in [train_preycapture.py](train_preycapture.py), model in [models/rgc2behavior.py](models/rgc2behavior.py).

### 8.1 Model inputs and shapes

- The decoder consumes the pooled RGC grids as a tensor shaped `[B, T_out, C, H, W]`.
- In this repo, the dataset permutes the mapped grids before returning them so that the last two axes are `[grid_width, grid_height]` (not `[grid_height, grid_width]`). Concretely:
  - `target_height = xlim[1]-xlim[0]`, `target_width = ylim[1]-ylim[0]`.
  - `grid_height = round(target_height * --grid_size_fac)`, `grid_width = round(target_width * --grid_size_fac)`.
  - `grid_seq` is returned as `[T_out, C, grid_width, grid_height]`, so the model is constructed with `input_height=grid_width`, `input_width=grid_height`.

Optional input normalization (applied per sample, before CNN):
- Enable with `--is_input_norm`.
- If `--is_channel_normalization` is set, normalization is per-channel (mean/std over time and space). Otherwise it normalizes the full sample jointly over time, channels, and space.

### 8.2 CNN feature extractor

`CNN_LSTM_ObjectLocation` uses a parallel CNN feature extractor family (`ParallelCNNFeatureExtractor*`) selected by `--cnn_extractor_version`.

Common pattern across versions:
- Input: one frame `[B, C, H, W]` where `C` equals the dataset channel count.
- Several parallel convolution “branches” (different effective receptive fields via stride/dilation), each with BatchNorm + ReLU (and sometimes pooling).
- Branch outputs are flattened and concatenated, then projected to a fixed feature vector of dimension `--cnn_feature_dim`.

Key parameters:
- `--conv_out_channels`: base channel count used inside the CNN branches.
- `--cnn_feature_dim`: output feature dimensionality per frame.
- `--cnn_extractor_version`: architecture variant (the experiments in the provided bash scripts use version 4).

### 8.3 LSTM temporal integration and readouts

- The per-frame CNN features are fed to an LSTM (batch-first) with:
  - hidden size `--lstm_hidden_size`
  - number of layers `--lstm_num_layers`
- The LSTM output is LayerNorm-normalized.
- Coordinate prediction head (per frame): a linear layer + ReLU, followed by a final linear layer to `--output_dim` (typically 2 for $(x,y)$).

Optional background auxiliary head (enabled when `--bg_info_cost_ratio != 0`):
- `--bg_processing_type one-proj`: a single linear projection from the main LSTM output to background $(x,y)$.
- `--bg_processing_type two-proj`: a 2-layer MLP (linear → ReLU → linear) from the main LSTM output.
- `--bg_processing_type lstm-proj`: a second LSTM processes the same CNN features; its output is used for background prediction and is also concatenated with the main LSTM output for object prediction.

### 8.4 Training objective and target preprocessing

- Output is a per-frame coordinate prediction $(x,y)$.
- The loss is MSE with an optional auxiliary background loss:
  $$\mathcal{L} = (1-\lambda)\,\mathrm{MSE}(\hat{p}, p) + \lambda\,\mathrm{MSE}(\hat{b}, b)$$
  where `--bg_info_cost_ratio` sets $\lambda$.

Background target preprocessing:
- If `--bg_info_type rloc`, background targets are transformed by a causal difference of short vs long moving averages (controlled by `--short_window_length`, `--long_window_length`).

### 8.5 Optimization, scheduling, and checkpointing

- Optimizer: Adam with learning rate `--learning_rate`.
- Scheduler (`--schedule_method`):
  - `RLRP`: ReduceLROnPlateau on the epoch-averaged training loss (factor `--schedule_factor`, patience fixed at 5 in code).
  - `CAWR`: CosineAnnealingWarmRestarts (eta-min `--min_lr`, with `T_0=5`, `T_mult=2` fixed in code).
- Gradient accumulation: `--accumulation_steps` batches are accumulated before an optimizer step (the code accumulates raw gradients; it does not divide the loss by `accumulation_steps`).
- Optional gradient clipping: enable `--is_gradient_clip` and set `--max_norm`.

Checkpointing:
- Checkpoints are saved every `--num_epoch_save` epochs to `.../{experiment_name}_checkpoint_epoch_{E}.pth`.
- The checkpoint includes: epoch index, model state, optimizer state, scheduler state, training losses, and the full CLI args (used for downstream evaluation).
- Training can resume from a checkpoint via `--load_checkpoint_epoch`.

### 8.6 Inference and performance analysis

Performance evaluation and visualization are implemented in [visual_preycapture_prediction.py](visual_preycapture_prediction.py).

High-level workflow:
- Load a saved checkpoint by experiment name + `--epoch_number`, then load the stored training args from the checkpoint.
- Reconstruct the RF bank and dataset using those args, with optional CLI overrides for test-time manipulations (e.g., noise level sweeps and fixed disparity).
- Run the trained model in `eval()` mode and compute per-batch MSE losses over a test loader; save `*_prediction_error.mat` containing both test losses and the training-loss trace from the checkpoint.
- Optionally generate videos (and/or per-frame images) overlaying ground truth and predicted trajectories.

Background target preprocessing:
- If `--bg_info_type rloc`, background targets are transformed by a causal difference of short vs long moving averages (controlled by `--short_window_length`, `--long_window_length`).

Optimization details:
- Adam optimizer (`--learning_rate`).
- Scheduler controlled by `--schedule_method` (e.g., CAWR uses cosine warm restarts with `--min_lr`).
- Gradient accumulation via `--accumulation_steps`.
- Optional gradient clipping via `--is_gradient_clip` and `--max_norm`.


## 9) Canonical example configurations (from provided bash scripts)

These scripts represent a commonly used experimental bundle:

- Shared key settings:
  - Dataset size: `--num_samples 2000`
  - Epochs: `--num_epochs 200`
  - Two grids + binocular: `--is_two_grids --is_binocular`
  - RGC noise: `--add_noise --rgc_noise_std 0.016`
  - Rectification: `--is_rectified --rectified_thr_ON 0.087` (OFF threshold defaults to ON)
  - RF scaling: `--sf_scalar 0.54 --sf_mask_radius 50 --mask_radius 17.7`
  - Surround resizing: `--set_surround_size_scalar 8.0 --set_s_scale -0.09`
  - Grid resolution: `--grid_size_fac 0.5 --target_num_centers 475`
  - CNN+LSTM: `--cnn_extractor_version 4 --cnn_feature_dim 256 --lstm_hidden_size 128 --lstm_num_layers 4`
  - Batch/accumulation: `--batch_size 4 --accumulation_steps 128`
  - Background images: `--bg_folder 'drift-grating-blend'`

Differences:
- [examples/running_bash/train_preycapture_25100502.sh ](examples/running_bash/train_preycapture_25100502.sh%20) additionally fixes disparity: `--fix_disparity 6.0`.

### 9.1 Experiment settings used in paper (training + inference)

This subsection summarizes the *typical* configuration used for the manuscript experiments in this repo. For exact reproducibility of any specific run, treat the checkpoint’s stored CLI args (saved by [train_preycapture.py](train_preycapture.py)) as the source of truth.

**Training settings (paper bundle)**

From the example scripts in [examples/running_bash/train_preycapture_25101402.sh](examples/running_bash/train_preycapture_25101402.sh) and [examples/running_bash/train_preycapture_25100502.sh ](examples/running_bash/train_preycapture_25100502.sh%20):

- Dataset generation: `--num_samples 2000`, `--num_epochs 200`, `--seed fixed`.
- Input channels: typically `--is_two_grids --is_binocular` (paired mapping; $C=2$).
- Noise + rectification: `--add_noise --rgc_noise_std 0.016`, `--is_rectified --rectified_thr_ON 0.087` (rectification is softplus by default via `--rectified_mode softplus`).
- RF/grid geometry: `--grid_size_fac 0.5`, `--target_num_centers 475`, `--mask_radius 17.7`, `--sf_scalar 0.54`, `--sf_mask_radius 50`.
- Surround manipulation (LN mode controls): `--set_surround_size_scalar 8.0 --set_s_scale -0.09`.
- Decoder architecture: `--cnn_extractor_version 4`, `--cnn_feature_dim 256`, `--lstm_hidden_size 128`, `--lstm_num_layers 4`.
- Optimization: `--batch_size 4`, `--accumulation_steps 128` (effective batch size = 512 sequences per optimizer step, ignoring any DataLoader-level randomness).

**Exact decoder architecture used by example `2025100502`**

This corresponds to [examples/running_bash/train_preycapture_25100502.sh ](examples/running_bash/train_preycapture_25100502.sh%20) and is implemented by `CNN_LSTM_ObjectLocation(CNNextractor_version=4)` in [models/rgc2behavior.py](models/rgc2behavior.py).

Inputs for this run:
- Channel count: `--is_two_grids --is_binocular` ⇒ `C=2` (paired mapping).
- With defaults `xlim=(-120,120)`, `ylim=(-90,90)` and `--grid_size_fac 0.5`:
  - `target_height = 240`, `target_width = 180`.
  - `grid_height = round(240 * 0.5) = 120`, `grid_width = round(180 * 0.5) = 90`.
  - The dataset returns frames as `[C, grid_width, grid_height]`, so each frame is `[2, 90, 120]` and the model is constructed with `input_height=90`, `input_width=120`.

Pre-CNN normalization:
- Enabled by `--is_input_norm --is_channel_normalization`.
- Uses channel-wise z-scoring per sample: for each channel, mean/std are computed over (time, height, width).

Framewise CNN feature extractor (ParallelCNNFeatureExtractor4; `--conv_out_channels 16`, `--cnn_feature_dim 256`):
- The CNN has 3 parallel branches (A1/A2/A3) starting directly from the input, each followed by 2 more convolution stages (B then C). All convs use BatchNorm2d + ReLU.

Branch A1 (local, high-resolution):
- A1: Conv2d(2→8, k=4, s=2, p=0, d=1) ⇒ output spatial size (90,120)→(44,59)
- B1: Conv2d(8→16, k=4, s=1, p=0) ⇒ (44,59)→(41,56)
- C1: Conv2d(16→16, k=3, s=1, p=0) ⇒ (41,56)→(39,54)
- Flatten: 16×39×54 = 33,696

Branch A2 (medium-scale via dilation/stride):
- A2: Conv2d(2→8, k=4, s=8, p=4, d=4) (effective k=13) ⇒ (90,120)→(11,15)
- B2: Conv2d(8→16, k=3, s=1, p=0) ⇒ (11,15)→(9,13)
- C2: Conv2d(16→16, k=3, s=1, p=0) ⇒ (9,13)→(7,11)
- Flatten: 16×7×11 = 1,232

Branch A3 (coarse-scale via dilation/stride):
- A3: Conv2d(2→8, k=4, s=16, p=8, d=8) (effective k=25) ⇒ (90,120)→(6,7)
- B3: Conv2d(8→16, k=2, s=1, p=0) ⇒ (6,7)→(5,6)
- C3: Conv2d(16→16, k=3, s=1, p=0) ⇒ (5,6)→(3,4)
- Flatten: 16×3×4 = 192

Concatenation and projection:
- Concatenate flattened branch vectors: 33,696 + 1,232 + 192 = 35,120 features
- FC: Linear(35,120→256)

Temporal integration and outputs (`--lstm_hidden_size 128 --lstm_num_layers 4`):
- LSTM: input_size=256, hidden_size=128, num_layers=4, dropout=0 (not set in code)
- LayerNorm(128)
- FC head: Linear(128→128) + ReLU, then Linear(128→2)

Auxiliary background head:
- Although `--bg_processing_type one-proj` is passed, this particular example sets `--bg_info_cost_ratio 0.0`, so the background head is effectively disabled for training loss (and the model returns `bg_predictions = coord_predictions`).

**Inference / performance analysis (paper workflow)**

Evaluation is performed by [visual_preycapture_prediction.py](visual_preycapture_prediction.py), which:

- Loads a checkpoint by experiment name and epoch (default `--epoch_number 200`) and restores the stored training args from the checkpoint.
- Reconstructs the stimulus generator and RGC simulator using those stored args (ensuring the evaluation data distribution matches the training configuration unless explicitly overridden).
- Computes test MSE on a held-out synthetic dataset (script default `num_sample=1000`) and saves MATLAB outputs:
  - `*_prediction_error.mat` (test losses + training loss trace from the checkpoint)
  - optional `*_prediction_error_with_path.mat` when saving trajectories.

Common evaluation manipulations used for figure/performance sweeps:
- **Noise sweeps** via `--noise_levels ...` (this overrides the dataset noise level at test time).
- **Fixed-disparity sweeps** via `--fix_disparity_degrees ...` (overrides binocular disparity at test time).
- **Background/object substitutions** via `--test_bg_folder` and `--test_ob_folder`.


## 10) Mapping the manuscript figure to code parameters

Your uploaded figure summarizes the exact conceptual stack implemented here: a synthetic movie generator → an RGC response simulator (SF + TF + noise + rectification) → a CNN+LSTM decoder that predicts $(x,y)$ over time.

### 10.1 Panel a (pipeline schematic)

This corresponds directly to:
- **Movie synthesis**: `SynMovieGenerator` ([datasets/sim_cricket.py](datasets/sim_cricket.py#L1003-L1185)).
- **RGC simulator**: `Cricket2RGCs._compute_rgc_time` ([datasets/sim_cricket.py](datasets/sim_cricket.py#L560-L706)).
  - Noise injection: `--add_noise` with `--rgc_noise_std` or `--rgc_noise_std_max`.
  - “softplus” nonlinearity: enable `--is_rectified` and set `--rectified_mode softplus`.
- **Decoder**: `CNN_LSTM_ObjectLocation` in [models/rgc2behavior.py](models/rgc2behavior.py) trained by [train_preycapture.py](train_preycapture.py).

### 10.2 Panel b (example frames, RGC grids, and predicted trajectories)

Conceptually, panel b shows:
- A sequence of stimulus frames.
- Two RGC activity maps per time point (labeled “sONa” and “sOFFa”). In this codebase, “two pathway maps” are produced by generating two RF populations and stacking their mapped grids.

Implementation options that match “two pathway maps”:
- **ON/OFF pathways**: use `--is_both_ON_OFF` (yields $C=2$ in monocular, $C=4$ in binocular).
- **Two same-sign grids**: use `--is_two_grids` (yields $C=2$ in monocular; in binocular it pairs one grid to one eye by default).

If you intend “ON vs OFF” specifically (as the labels suggest), `--is_both_ON_OFF` is the semantically direct match.

### 10.3 Panels c–d (nasal vs temporal subtypes)

Panels c–d compare “nasal” vs “temporal” versions of the same pathway (sONa or sOFFa).

In this repo, nasal/temporal subtype differences are represented by selecting different SF/TF parameter tables from the Excel workbook via:
- `--sf_sheet_name` / `--tf_sheet_name` (primary pathway)
- and optionally `--sf_sheet_name_additional` / `--tf_sheet_name_additional` (second pathway if using `--is_both_ON_OFF`).

The provided example scripts use temporal sheets (`SF_ON-T`, `TF_ON-T`). To reproduce nasal-vs-temporal comparisons, use the nasal sheet names present in your `SimulationParams.xlsx` (not tracked in this repo).

### 10.4 Panels e–h (density vs coverage manipulations)

The figure varies either **density** (number of cells) at fixed coverage, or **coverage** at fixed density.

In this implementation, the closest parameter mappings are:
- **Density** → `--target_num_centers` (and `--target_num_centers_additional` for the second pathway/grid).
  - Example: “0.5× density” ≈ halve `--target_num_centers`; “2× density” ≈ double it.
- **Coverage (spatial pooling footprint on the output grid)** → `--mask_radius` when using `--grid_generate_method circle`.
  - Larger `--mask_radius` makes each RGC contribute to more grid pixels in the pooling step.

Related-but-distinct parameters (often held constant when interpreting “coverage”):
- `--sf_mask_radius` affects the **spatial-filter cutoff** (dendritic mask) rather than the pooling footprint.
- `--grid_size_fac` changes the **output grid resolution**, not the number of RGCs.

### 10.5 Panels i–j (surround strength)

The “No surround” vs “4× surround” comparison maps to surround *weighting* in the LN pathway:
- **Surround weight** → `--set_s_scale` (LN mode; it overrides `row['s_scale']`).
  - “No surround” is approximated by setting `--set_s_scale 0`.
  - “4× surround” is approximated by multiplying your baseline `--set_s_scale` by 4.

Note: `--set_surround_size_scalar` changes surround *size* (sigma), not the surround weight, so it is not the direct control for “surround strength” in that panel.

### 10.6 Noise level sweeps

The noise-axis values shown in the figure (e.g., 0.016, 0.064, 0.256) correspond most directly to `--rgc_noise_std` (with `--add_noise` enabled).

If you want to match “noise sweeps” without launching multiple jobs manually, you can instead use `--rgc_noise_std_max` (and leave `--rgc_noise_std 0`) to sample a log-uniform noise level per sample; however, that produces a *mixture* of noise levels within a run rather than a clean per-condition curve.


## Appendix A) Parameter reference (CLI flags)

This appendix is intended for replication: it lists the main simulation-relevant flags and their defaults (as defined in [train_preycapture.py](train_preycapture.py)).

### A.1 Stimulus synthesis (`SynMovieGenerator`)

| Flag | Default | Meaning |
|---|---:|---|
| `--crop_size` | `(320, 240)` | Output frame size (width, height). |
| `--boundary_size` | `"(220, 140)"` | Movement bounds for trajectory generation (x_limit, y_limit). |
| `--max_steps` | `200` | Max trajectory length before trimming. |
| `--num_ext` | `50` | Number of initial repeated (static) frames. |
| `--prob_stay_ob` / `--prob_mov_ob` | `0.95` / `0.975` | Object stay→stay / move→move persistence. |
| `--prob_stay_bg` / `--prob_mov_bg` | `0.95` / `0.975` | Background stay→stay / move→move persistence. |
| `--initial_velocity` | `6` | Initial step velocity. |
| `--momentum_decay_ob` / `--momentum_decay_bg` | `0.95` / `0.9` | Momentum decay while moving. |
| `--velocity_randomness_ob` / `--velocity_randomness_bg` | `0.02` / `0.01` | Step-to-step velocity noise. |
| `--angle_range_ob` / `--angle_range_bg` | `0.5` / `0.25` | Random angular jitter range. |
| `--start_scaling` / `--end_scaling` | `1.0` / `2.0` | Rescaling endpoints for the cricket image. |
| `--dynamic_scaling` | `0.0` | Randomly perturbs scaling endpoints per sample. |
| `--is_binocular` | off | Generates two stimulus channels with disparity. |
| `--interocular_dist` | `1.0` | Interocular distance (cm) for disparity geometry. |
| `--fix_disparity` | `None` | Overrides computed disparities with a constant (degrees, pre-conversion). |
| `--bottom_contrast` / `--top_contrast` | `1.0` / `1.0` | Contrast multipliers for background/cricket green channel. |
| `--mean_diff_offset` | `0.0` | Mean offset budget between bottom and top green channels. |
| `--coord_adj_type` | `body` | Selects which coordinate-correction `.mat` summary to load. |
| `--coord_adj_dir` | `1.0` | Correction sign (and scalar). |
| `--is_reverse_xy` | off | Swaps correction x/y. |

### A.2 RGC dataset (`Cricket2RGCs`)

| Flag | Default | Meaning |
|---|---:|---|
| `--num_samples` | `20` | Number of generated samples per epoch (dataset length). |
| `--grid_size_fac` | `1` | Output grid resolution scaling factor. |
| `--is_norm_coords` | off | Normalizes target coordinates by half-extents from `xlim/ylim`. |
| `--fr2spikes` | off | Converts firing rates to Poisson spikes (LN pathway). |
| `--quantize_scale` | `1.0` | Spike quantization scaling. |
| `--smooth_data` | off | Gaussian smoothing over time (LN pathway). |
| `--add_noise` | off | Adds Gaussian noise to responses (LN and simplified LNK). |
| `--rgc_noise_std` | `0.0` | Fixed noise std if >0. |
| `--rgc_noise_std_max` | `None` | Enables per-sample log-uniform noise sampling if `rgc_noise_std==0`. |
| `--is_rectified` | off | Enables rectification after LN conv. |
| `--rectified_thr_ON` | `0.0` | Rectification threshold for ON. |
| `--rectified_thr_OFF` | `0.0` | Rectification threshold for OFF (defaults to ON if omitted). |
| `--rectified_mode` | `softplus` | `hard` clamp or `softplus` rectification. |
| `--rectified_softness` | `1.0` | Softplus transition width for ON. |
| `--rectified_softness_OFF` | `None` | Softplus transition width for OFF (defaults to ON). |
| `--is_direct_image` | off | Bypasses RF bank; applies temporal filtering directly to pixels. |
| `--is_both_ON_OFF` | off | Creates ON and OFF pathways (2 channels; 4 if binocular). |
| `--is_two_grids` | off | Creates two grids with same sign semantics (paired with eyes if binocular). |

### A.3 RGC RF array (`RGCrfArray`)

| Flag | Default | Meaning |
|---|---:|---|
| `--rgc_array_rf_size` | `(320, 240)` | Spatial filter support size. |
| `--xlim` / `--ylim` | `(-120,120)` / `(-90,90)` | Coordinate extents for grid generation. |
| `--target_num_centers` | `500` | Number of RGCs in the first grid. |
| `--target_num_centers_additional` | `None` | Overrides the second grid size if provided. |
| `--grid_generate_method` | `circle` | How RGCs map to fixed grid (`closest`, `decay`, `circle`). |
| `--tau` | `3` | Decay constant for `decay` mapping. |
| `--mask_radius` | `30` | Radius parameter for `circle` mapping. |
| `--sf_scalar` | `0.2` | Global scale factor applied to SF sigmas. |
| `--sf_constraint_method` | `circle` | SF masking constraint (`circle`, `threshold`, `None`). |
| `--sf_mask_radius` | `35` | Base mask radius for SF constraint. |
| `--anti_alignment` | `1` | Shift of second grid vs first (0..1). |
| `--grid_noise_level` | `0.3` | Noise level in grid generation. |
| `--sf_sheet_name` | `SF_params_modified` | Excel sheet for SF parameters. |
| `--tf_sheet_name` | `TF_params` | Excel sheet for TF parameters. |
| `--syn_params` | unset | Which parameter groups share indices (`tf`, `sf`, `lnk`). |
| `--set_s_scale` | empty | Overrides LN surround weight (LN mode). |
| `--set_surround_size_scalar` | `None` | Overrides surround size derivation (see Section 4.2). |

### A.4 Simplified LNK

| Flag | Default | Meaning |
|---|---:|---|
| `--use_lnk_model` | off | Uses simplified LNK response generation. |
| `--lnk_sheet_name` | `LNK_params` | Excel sheet name for LNK parameters. |
| `--surround_sigma_ratio` | `4.0` | Scales center sigma to generate surround in LNK override mode. |
| `--lnk_adapt_mode` | `divisive` | Logged by training script; not used by simplified LNK path. |

### A.5 Decoder and training

| Flag | Default | Meaning |
|---|---:|---|
| `--cnn_feature_dim` | `256` | CNN feature dimensionality per frame. |
| `--conv_out_channels` | `16` | Base CNN branch channels. |
| `--cnn_extractor_version` | `1` | CNN extractor variant used in `CNN_LSTM_ObjectLocation`. |
| `--lstm_hidden_size` | `64` | LSTM hidden size. |
| `--lstm_num_layers` | `3` | LSTM layer count. |
| `--output_dim` | `2` | Output dimension (typically $(x,y)$). |
| `--is_seq_reshape` | off | If enabled, reshapes `[B,T]` into one batch for CNN to avoid Python loop. |
| `--is_input_norm` | off | Enables sample-level input normalization before CNN. |
| `--is_channel_normalization` | off | If enabled, normalize each channel separately; else normalize the whole sample jointly. |
| `--bg_info_cost_ratio` | `0` | Auxiliary background loss weight $\lambda$ (0 disables background head usage for loss). |
| `--bg_processing_type` | `one-proj` | Background head type: `one-proj`, `two-proj`, or `lstm-proj`. |
| `--batch_size` | `4` | Training batch size. |
| `--accumulation_steps` | `1` | Gradient accumulation steps before an optimizer update. |
| `--num_epochs` | `10` | Training epochs. |
| `--learning_rate` | `0.001` | Adam learning rate. |
| `--schedule_method` | `RLRP` | Scheduler type: ReduceLROnPlateau (`RLRP`) or cosine warm restarts (`CAWR`). |
| `--schedule_factor` | `0.2` | LR reduction factor for `RLRP`. |
| `--min_lr` | `1e-6` | Minimum LR for `CAWR`. |
| `--is_gradient_clip` | off | Enables gradient clipping. |
| `--max_norm` | `5.0` | Clip threshold (norm). |
| `--num_epoch_save` | `5` | Checkpoint save interval (epochs). |
| `--seed` | `fixed` | Seed behavior (processed by `process_seed`). |
| `--bg_info_type` | `rloc` | Background target type; `rloc` applies moving-average differencing. |
| `--short_window_length` / `--long_window_length` | `3` / `10` | Background moving-average windows for `bg_info_type=rloc`. |


## Appendix B) Reproducibility notes

- The DataLoader RNG is seeded via `--seed` (processed by `process_seed`), and the dataset can also receive a deterministic base seed (`rnd_seed+5678`) so sample `idx` maps to a repeatable trajectory/movie.
- For strict replication of published conditions, record the full CLI, the exact Excel parameter workbook (and sheet names), and the random seed.
