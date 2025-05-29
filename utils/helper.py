import numpy as np

def estimate_rgc_signal_distribution(
    dataset,            # instance of Cricket2RGCs
    N=500,              # number of samples
    channel_idx=0,      # which channel to sample (0 = first channel)
    return_histogram=False,
    bins=50
):
    """
    Estimate mean and std of rgc_time (noise-free) over N samples,
    plus mean repeat-reliability (r) across N noisy replicates.

    Returns:
      (μ, σ, mean_r)
    or, if return_histogram:
      (μ, σ, (hist, edges), mean_r)
    """
    # Backup original noise settings
    orig_noise_flag = dataset.add_noise
    orig_noise_std  = dataset.rgc_noise_std

    all_vals = []
    r_list  = []

    for _ in range(N):
        # 1) generate a movie
        syn_movie, *rest = dataset.movie_generator.generate()
        mv = syn_movie[:, 0] if syn_movie.ndim == 4 else syn_movie

        ch = dataset.channels[channel_idx]

        # 2) Noise‐free RGC time‐series
        rgc_clean = dataset._compute_rgc_time(
            mv, ch['sf'], ch['tf'], ch['rect_thr']
        )
        vals = rgc_clean.flatten().detach().cpu().numpy()
        all_vals.append(vals)

        # 3) If original config had noise, do two noisy replicates
        if orig_noise_flag:
            rgc_noisy1 = dataset._compute_rgc_time(
                mv, ch['sf'], ch['tf'], ch['rect_thr']
            )
            rgc_noisy2 = dataset._compute_rgc_time(
                mv, ch['sf'], ch['tf'], ch['rect_thr']
            )

            f1 = rgc_noisy1.flatten().detach().cpu().numpy()
            f2 = rgc_noisy2.flatten().detach().cpu().numpy()
            # Pearson r
            r_list.append(np.corrcoef(f1, f2)[0, 1])


    # compute μ, σ and mean reliability
    data   = np.concatenate(all_vals)
    μ, σ    = data.mean(), data.std()
    mean_r = np.mean(r_list) if r_list else np.nan

    # Restore original noise settings
    dataset.add_noise = orig_noise_flag
    dataset.rgc_noise_std = orig_noise_std

    if return_histogram:
        hist, edges = np.histogram(data, bins=bins, density=True)
        return μ, σ, mean_r, (hist, edges)

    return μ, σ, mean_r
