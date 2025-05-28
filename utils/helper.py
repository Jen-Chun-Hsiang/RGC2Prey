import numpy as np

def estimate_rgc_signal_distribution(
    dataset,        # instance of Cricket2RGCs
    N=500,          # number of samples
    channel_idx=0,  # which channel to sample (0 = ON, etc.)
    return_histogram=False,
    bins=50
):
    """
    Estimate mean and std of rgc_time (noise-free) over N samples.
    Only runs if external flag is_rgc_distribution is True.

    Returns:
      μ, σ or (μ, σ, (hist, edges)) if return_histogram.
    """
    all_vals = []
    # Temporarily store original noise settings
    orig_noise_flag = dataset.add_noise
    orig_noise_std  = dataset.rgc_noise_std

    # Ensure noise is off
    dataset.add_noise = False
    dataset.rgc_noise_std = 0.0

    for _ in range(N):
        movie, *_ = dataset.movie_generator.generate()
        ch = dataset.channels[channel_idx]
        # compute rgc_time fully
        rgc_time = dataset._compute_rgc_time(
            movie,
            ch['sf'],
            ch['tf'],
            ch['rect_thr']
        )
        vals = rgc_time.flatten().detach().cpu().numpy()
        all_vals.append(vals)

    data = np.concatenate(all_vals)
    μ, σ = data.mean(), data.std()

    # restore original noise settings
    dataset.add_noise = orig_noise_flag
    dataset.rgc_noise_std = orig_noise_std

    if return_histogram:
        hist, edges = np.histogram(data, bins=bins, density=True)
        return μ, σ, (hist, edges)
    return μ, σ