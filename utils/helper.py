# External utility for RGC time distribution
import numpy as np
import matplotlib.pyplot as plt

def estimate_rgc_signal_distribution(
    dataset,        # instance of Cricket2RGCs
    N=500,          # number of samples
    channel_idx=0,  # which channel to sample (0 = first channel)
    return_histogram=False,
    bins=50
):
    """
    Estimate mean and std of rgc_time (noise-free) over N samples.
    Only computes raw pathway responses (not grid-mapped).

    Returns:
      (μ, σ) or (μ, σ, (hist, edges)) if return_histogram.
    """
    all_vals = []
    # Backup original noise settings
    orig_noise_flag = dataset.add_noise
    orig_noise_std  = dataset.rgc_noise_std

    # Ensure noise is off
    dataset.add_noise = False
    dataset.rgc_noise_std = 0.0

    for _ in range(N):
        # Generate one synthetic movie
        syn_movie, *rest = dataset.movie_generator.generate()
        # Extract single eye/movie tensor of shape [T, H, W]
        if syn_movie.ndim == 4:
            # [T, eyes, H, W] -> pick first eye
            mv = syn_movie[:, 0]
        elif syn_movie.ndim == 3:
            # already [T, H, W]
            mv = syn_movie
        else:
            raise ValueError(f"Unexpected movie dims: {syn_movie.shape}")

        # Get channel config
        ch = dataset.channels[channel_idx]
        # Compute raw RGC time-series
        rgc_time = dataset._compute_rgc_time(
            mv,
            ch['sf'],
            ch['tf'],
            ch['rect_thr']
        )
        # Flatten and collect
        vals = rgc_time.flatten().detach().cpu().numpy()
        all_vals.append(vals)

    # Concatenate all samples
    data = np.concatenate(all_vals)
    μ, σ = data.mean(), data.std()

    # Restore noise settings
    dataset.add_noise = orig_noise_flag
    dataset.rgc_noise_std = orig_noise_std

    if return_histogram:
        hist, edges = np.histogram(data, bins=bins, density=True)
        return μ, σ, (hist, edges)
    return μ, σ


