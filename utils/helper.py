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
    corr_threshold = 0.75

    all_vals = []
    r_list  = []
    width_list = [] 

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

        corr = np.correlate(vals, vals, mode='full')
        corr_min, corr_max = corr.min(), corr.max()
        corr = (corr - corr_min) / (corr_max - corr_min)

        width_idx = np.where(corr >= corr_threshold)[0]
        if width_idx.size == 0:
            continue

        width = width_idx.max() - width_idx.min()
        width_list.append(width)

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
    mean_r = float(np.nanmean(r_list))
    mean_width = np.nanmean(width_list)

    n_trials = len(r_list)
    n_nan = int(np.sum(np.isnan(r_list)))
    pct_nan = n_nan / n_trials * 100
    
    n_nan = int(np.sum(np.isnan(width_list)))
    pct_nan_width = n_nan / N * 100

    # Restore original noise settings
    dataset.add_noise = orig_noise_flag
    dataset.rgc_noise_std = orig_noise_std

    if return_histogram:
        hist, edges = np.histogram(data, bins=bins, density=True)
        return μ, σ, mean_r, mean_width, (hist, edges), pct_nan, pct_nan_width

    return μ, σ, mean_r, mean_width
