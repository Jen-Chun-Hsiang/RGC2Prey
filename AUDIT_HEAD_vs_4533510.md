# Audit: HEAD vs 453351026fef4137a1942618e6d9a0ea35912763

This note summarizes **simulation-relevant differences** that can cause **different results even with identical CLI parameters and `--seed`**.

- Older reference commit: `453351026fef4137a1942618e6d9a0ea35912763` (2025-09-15)
- Current HEAD (at time of audit): `f5951d31261538029336a051508585942baabf43` (2026-02-14)

The user-reported commands are effectively identical (the newer one adds `--temporal_shift_frames 0`, which should be a no-op):

```bash
python $HOME/RGC2Prey/RGC2Prey/train_preycapture.py ... --seed 8901 ...
python $HOME/RGC2Prey/RGC2Prey/train_preycapture.py ... --seed 8901 ... --temporal_shift_frames 0
```

## Old vs new: what the actual setup is

This section is a **concrete “what is different in the run setup”** comparison between:

- Old: `453351026fef4137a1942618e6d9a0ea35912763`
- New: HEAD `f5951d31261538029336a051508585942baabf43`

### Training setup (`train_preycapture.py`)

**Old (4533510):**

- Calls `process_seed(args.seed)` but does **not** capture/propagate the returned seed.
- Creates `Cricket2RGCs(...)` **without** any `rnd_seed` argument (because the dataset did not support it).
- Builds `DataLoader(..., shuffle=True, worker_init_fn=worker_init_fn)` **without** a seeded `generator=...`.

**New (HEAD):**

- Stores the processed seed: `rnd_seed = process_seed(args.seed)` ([train_preycapture.py](train_preycapture.py#L219)).
- Passes deterministic dataset seeding into `Cricket2RGCs` using an **explicit offset**: `rnd_seed=rnd_seed+5678` ([train_preycapture.py](train_preycapture.py#L425), [train_preycapture.py](train_preycapture.py#L487), [train_preycapture.py](train_preycapture.py#L572)).
- Seeds the *shuffle order* explicitly via `dl_generator = torch.Generator().manual_seed(rnd_seed)` and `DataLoader(..., generator=dl_generator)` ([train_preycapture.py](train_preycapture.py#L579-L584)).
- Threads new RF/response modifiers into the RF generator and dataset:
  - `RGCrfArray(..., set_bias=..., set_biphasic_scale=..., temporal_shift_frames=...)` (see `RGCrfArray` in [datasets/sim_cricket.py](datasets/sim_cricket.py#L1368)).
  - `Cricket2RGCs(..., ln_contrast_gain=..., ln_contrast_gain_off=..., ln_contrast_gain_apply_to_lnk=...)` (see `Cricket2RGCs` in [datasets/sim_cricket.py](datasets/sim_cricket.py#L350)).

### Inference/test setup (`visual_preycapture_prediction.py`)

**Old (4533510):**

- Calls `process_seed(args.seed)` but does **not** capture/propagate the returned seed.
- Creates `Cricket2RGCs(...)` without any dataset `rnd_seed`.
- Uses `DataLoader(..., shuffle=True, worker_init_fn=worker_init_fn)` without a seeded `generator=...`.

**New (HEAD):**

- Stores the processed seed: `rnd_seed = process_seed(args.seed)` ([visual_preycapture_prediction.py](visual_preycapture_prediction.py#L389)).
- Passes `rnd_seed=rnd_seed` into `Cricket2RGCs` for deterministic per-sample generation ([visual_preycapture_prediction.py](visual_preycapture_prediction.py#L604)).
- Seeds test shuffling via `generator=torch.Generator().manual_seed(rnd_seed)` ([visual_preycapture_prediction.py](visual_preycapture_prediction.py#L619)).
- Uses a second, different generator seed (`rnd_seed + 1`) for the small analysis/visualization loader when it shuffles ([visual_preycapture_prediction.py](visual_preycapture_prediction.py#L697)).

### Simulator/RF setup (`datasets/sim_cricket.py`, `datasets/rgc_rf.py`)

**Old (4533510):**

- `Cricket2RGCs` has **no** `rnd_seed` support and therefore does *not* reseed per sample.
- `Cricket2RGCs` has **no** LN contrast gain knobs (`ln_contrast_gain*`) and does not scale the linear drive.
- `RGCrfArray` does **not** accept `set_bias`, `set_biphasic_scale`, or `temporal_shift_frames`.
- `gaussian_temporalfilter` signature is `def gaussian_temporalfilter(n, OptW)` (no biphasic scaling hook).
- `HexagonalGridGenerator.generate_second_grid()` has a bug where `offset_x`/`offset_y` become 1-tuples due to trailing commas.

**New (HEAD):**

- `Cricket2RGCs(..., rnd_seed=...)` exists and is used by both training and inference ([datasets/sim_cricket.py](datasets/sim_cricket.py#L350)).
- LN contrast gain is implemented as a multiplicative factor applied to the LN linear drive before Poisson/noise/rectification, and can optionally apply to LNK outputs ([datasets/sim_cricket.py](datasets/sim_cricket.py#L560)).
- `RGCrfArray(..., set_bias, set_biphasic_scale, temporal_shift_frames)` exists, enabling direct RF-shape modifications ([datasets/sim_cricket.py](datasets/sim_cricket.py#L1368)).
- `gaussian_temporalfilter(..., set_biphasic_scale=...)` exists and scales the second lobe amplitude ([datasets/rgc_rf.py](datasets/rgc_rf.py#L7-L31)).
- `HexagonalGridGenerator.generate_second_grid()` tuple bug is fixed, changing OFF/two-grid placements when those modes are enabled ([datasets/rgc_rf.py](datasets/rgc_rf.py#L501)).

If you want to verify the “old setup” directly, the fastest way is:

```bash
git show 453351026fef4137a1942618e6d9a0ea35912763:train_preycapture.py | sed -n '180,560p'
git show 453351026fef4137a1942618e6d9a0ea35912763:visual_preycapture_prediction.py | sed -n '230,520p'
git show 453351026fef4137a1942618e6d9a0ea35912763:datasets/sim_cricket.py | sed -n '330,520p'
git show 453351026fef4137a1942618e6d9a0ea35912763:datasets/rgc_rf.py | egrep -n "def gaussian_temporalfilter|generate_second_grid|offset_x|offset_y" | head
```

## Executive summary (what most likely changed your simulation)

### 1) Dataset-level deterministic seeding was introduced

In HEAD, `Cricket2RGCs` accepts a `rnd_seed` argument and implements **per-sample seeding** inside `__getitem__`.

- Current code: `sample_seed = self.rnd_seed + idx` and then seeds `random`, `numpy`, and `torch` for sample generation.
  - See: [datasets/sim_cricket.py](datasets/sim_cricket.py#L760-L915) (entry) and [datasets/sim_cricket.py](datasets/sim_cricket.py#L770)

In the older commit, `Cricket2RGCs` did **not** have `rnd_seed` and did **not** reset RNG state per sample. Samples were generated from the evolving global RNG stream.

**Impact:** even if you pass the same `--seed`, HEAD will generate a *different* movie/path/noise stream than the older commit because the sampling scheme changed.

### 2) The training script now passes `rnd_seed` into the dataset (and with an offset)

Training in HEAD calls `rnd_seed = process_seed(args.seed)` and then injects it into the dataset.

- Seed processing (HEAD): [train_preycapture.py](train_preycapture.py#L219)
- Dataset construction (HEAD) passes `rnd_seed=rnd_seed+5678` in multiple places:
  - [train_preycapture.py](train_preycapture.py#L425)
  - [train_preycapture.py](train_preycapture.py#L487)
  - [train_preycapture.py](train_preycapture.py#L572)

The older training script did **not** pass any dataset seed.

**Impact:** training data generation is now deterministically tied to the global seed + index (and additionally shifted by `+5678`). That will not match the older behavior.

### 3) DataLoader shuffle order is now explicitly seeded

In HEAD, DataLoader shuffling is driven by a `torch.Generator().manual_seed(rnd_seed)`.

- Training loader generator: [train_preycapture.py](train_preycapture.py#L579-L584)
- Inference test loader generator: [visual_preycapture_prediction.py](visual_preycapture_prediction.py#L619)

The older scripts used `shuffle=True` without an explicit generator (so order depends on global RNG / worker timing).

**Impact:** optimization trajectories and evaluation aggregates can diverge due to different batch/sample orders.

### 4) Inference uses a different dataset seed offset than training

- Training passes `rnd_seed + 5678` into `Cricket2RGCs`.
- Inference passes `rnd_seed` (no +5678).

Even within HEAD, **train vs test/visualization are not sampling the same synthetic trajectories** unless you intentionally align these offsets.

- Inference seed processing: [visual_preycapture_prediction.py](visual_preycapture_prediction.py#L389)
- Inference dataset uses `rnd_seed=rnd_seed`: [visual_preycapture_prediction.py](visual_preycapture_prediction.py#L604)

## Training process: changes to watch (train_preycapture.py)

### A) New arguments that can affect RFs / responses

HEAD adds/threads through several RF/response modifiers that can change simulated responses *without any RNG involvement*.

#### 1) `--temporal_shift_frames` (RF temporal latency/phase)

This shifts the **temporal kernel** used in the LN conv1d by an integer number of frames, with **zero padding** (no warping).

- Shift helper and convention: [datasets/sim_cricket.py](datasets/sim_cricket.py#L1769-L1810)
- Shift application + re-normalization: [datasets/sim_cricket.py](datasets/sim_cricket.py#L1820-L1866)

Mechanistically:

- The LN response is computed as `conv1d(sf_frame, tf)` (see below), so translating `tf` in time changes the effective **response latency** and the alignment between stimulus features and peaks/troughs in the RGC timecourse.
- Because shifting uses zero padding, a shift can also effectively **truncate** one end of the kernel, slightly changing its energy; the code then re-normalizes by `sum(abs(tf))` (unless the kernel becomes all zeros).

If `temporal_shift_frames == 0`, the shift helper returns the original kernel and this should be a no-op.

#### 2) `--set_biphasic_scale` (RF temporal biphasic shape)

This scales the second lobe amplitude ("amp2") when synthesizing Gaussian temporal filters.

- Temporal filter generator: [datasets/rgc_rf.py](datasets/rgc_rf.py#L7-L31)
- Passed into filter creation: [datasets/sim_cricket.py](datasets/sim_cricket.py#L1827-L1860)

Mechanistically:

- Changing `amp2` changes the relative strength of the positive/negative lobes (more biphasic vs more monophasic), which changes **transient vs sustained** dynamics and can change sign/overshoot behavior.
- Even though the code later normalizes `tf` by `sum(abs(tf))`, the *shape* (lobe ratio) still changes.

#### 3) `--set_bias` (RF spatial baseline offset)

This overrides the `bias` used in the spatial filter (Difference-of-Gaussians-like) parameter table before generating the spatial RF.

- Bias override site (LN pathway): [datasets/sim_cricket.py](datasets/sim_cricket.py#L1718-L1750)
- How `bias` enters the 2D filter: [datasets/rgc_rf.py](datasets/rgc_rf.py#L33-L62)
- Filter normalization + optional median subtraction: [datasets/sim_cricket.py](datasets/sim_cricket.py#L1752-L1796)

Mechanistically:

- **What it is before you set it (default behavior):** in both old and new versions, the spatial filter is parameterized by an Excel row (`row = sf_table.iloc[pid_i]`) and uses `row['bias']` from the sheet as the additive constant term in `gaussian2d`.
- **What changes when you pass `--set_bias X`:** in HEAD, the code overwrites the Excel value (`row['bias'] = X`) right before building `sf_params` for `gaussian_multi(...)` ([datasets/sim_cricket.py](datasets/sim_cricket.py#L1718-L1750)). That means *every* LN spatial filter uses the same constant bias `X`.
- `bias` is then **added everywhere** in the spatial filter (`z = c_scale*c + s_scale*s + bias`) ([datasets/rgc_rf.py](datasets/rgc_rf.py#L33-L62)), and the resulting RF is optionally median-subtracted + normalized by `sum(abs(rf))` ([datasets/sim_cricket.py](datasets/sim_cricket.py#L1752-L1796)).
- **When this does not apply:** in the LNK override branch, `base_params` hard-codes `bias=0` for center/surround generation, so `--set_bias` only affects the *standard LN* spatial-filter path.
- **Important for your use case:** if you do not pass `--set_bias` (default `None`), there is no bias-override logic applied and the behavior remains “use the Excel bias”. So `--set_bias` cannot explain version differences when you truly ran with identical defaults.

#### 4) LN contrast gain family (response scaling pre-stochasticity)

These options scale the **linear drive** of the LN model (and optionally LNK outputs), which changes response magnitude *and* the effective SNR.

- Gain parameters stored and per-channel ON/OFF defaults: [datasets/sim_cricket.py](datasets/sim_cricket.py#L383-L535)
- Applied to LN linear drive (after conv1d, before noise/Poisson/rectification): [datasets/sim_cricket.py](datasets/sim_cricket.py#L611-L706)
- Optionally applied to LNK output when `--ln_contrast_gain_apply_to_lnk` is enabled: [datasets/sim_cricket.py](datasets/sim_cricket.py#L582-L645)

Mechanistically:

- For LN: `rgc_time = conv1d(...)` then `rgc_time *= ln_gain` happens **before**:
  - Poisson spike conversion (`fr2spikes`),
  - additive noise (`add_noise`),
  - rectification (hard clamp or softplus).
  This means gain changes both mean response and how much time is spent above threshold; with Poisson it changes variance as well.
- `--ln_contrast_gain_off` allows different scaling of OFF vs ON channels; even if both channels exist, a mismatch changes ON/OFF balance feeding the downstream behavior model.
- If `use_lnk` is enabled, `--ln_contrast_gain_apply_to_lnk` can additionally scale the LNK output, making LNK-vs-LN comparisons sensitive to this flag.

Bottom line: even with identical seeds and stimulus paths, these arguments can reshape RFs (`set_bias`, `set_biphasic_scale`, `temporal_shift_frames`) and/or rescale responses (`ln_contrast_gain*`).

### B) RF temporal shift implementation (should be no-op when 0)

Temporal shifting is implemented in `RGCrfArray._create_temporal_filter`.

- Shift helper: [datasets/sim_cricket.py](datasets/sim_cricket.py#L1797)
- Shift application sites: [datasets/sim_cricket.py](datasets/sim_cricket.py#L1843-L1864)

If `temporal_shift_frames == 0`, the shift function returns the original kernel.

### C) Training’s dataset seed offset (`+5678`)

This is a deliberate divergence from the global seed.

- See: [train_preycapture.py](train_preycapture.py#L425) / [train_preycapture.py](train_preycapture.py#L487) / [train_preycapture.py](train_preycapture.py#L572)

If your expectation is “exact same synthetic dataset as older commit when CLI args match”, this offset prevents that.

## Inference / test process: changes to watch (visual_preycapture_prediction.py)

### A) Dataset seeding introduced in inference

Inference now does:

- `rnd_seed = process_seed(args.seed)` at [visual_preycapture_prediction.py](visual_preycapture_prediction.py#L389)
- passes `rnd_seed` into `Cricket2RGCs` at [visual_preycapture_prediction.py](visual_preycapture_prediction.py#L604)

Older inference code did not pass dataset seeds.

### B) Seeded shuffling for evaluation

- Test loader uses seeded generator: [visual_preycapture_prediction.py](visual_preycapture_prediction.py#L619)
- The analysis/visualization section uses a different generator seed (`rnd_seed + 1`) when shuffling: [visual_preycapture_prediction.py](visual_preycapture_prediction.py#L697)

This changes which samples get evaluated first/when and which samples are visualized.

## Simulator / dataset core: changes that alter generated data

### A) Per-sample deterministic seeding in Cricket2RGCs

Key logic (HEAD):

- Entry: [datasets/sim_cricket.py](datasets/sim_cricket.py#L760)
- Seed definition: [datasets/sim_cricket.py](datasets/sim_cricket.py#L770)

This resets RNG streams per sample and restores global RNG state after generation.

### B) Two-grid / OFF-grid generation bugfix (only affects runs using OFF / two grids)

The old `HexagonalGridGenerator.generate_second_grid` used trailing commas:

```python
offset_x = self.offset_x - shift_x,
offset_y = self.offset_y - shift_y,
```

Those commas make `offset_x` / `offset_y` 1-tuples, changing downstream computations.

- Fixed in HEAD: [datasets/rgc_rf.py](datasets/rgc_rf.py#L501)

If your experiments ever toggled `--is_both_ON_OFF` or `--is_two_grids`, this could materially change the generated RGC grid locations.

## Practical recommendations

1) If you want HEAD to behave closer to the old commit, consider a compatibility option to disable per-sample reseeding in `Cricket2RGCs` (i.e. run with `rnd_seed=None`).
2) Align dataset seeds between training and inference if you expect “same samples” across scripts.
3) If comparing results across versions, pin:
   - DataLoader `generator` seeding and `shuffle` choices
   - `num_workers` (and deterministic worker init)
   - GPU determinism settings (PyTorch can still be nondeterministic for some ops unless forced)

---

### Notes about tooling

This workspace doesn’t have `ripgrep` (`rg`) installed, so searches used `grep`.
