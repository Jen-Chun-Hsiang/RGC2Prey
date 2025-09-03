import numpy as np
import os
import scipy.io as sio
import torch

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
    is_xcorr_width = False
    is_single_middle_rgc = True
    is_data_saved = False
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
            mv, ch['sf'], ch['sf_surround'], ch['tf'], ch['rect_thr'], ch['lnk_params']
        )
        vals = rgc_clean.flatten().detach().cpu().numpy()
        all_vals.append(vals)

        if is_xcorr_width:
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
                mv, ch['sf'], ch['sf_surround'], ch['tf'], ch['rect_thr'], ch['lnk_params']
            )
            rgc_noisy2 = dataset._compute_rgc_time(
                mv, ch['sf'], ch['sf_surround'], ch['tf'], ch['rect_thr'], ch['lnk_params']
            )
            
            if is_single_middle_rgc:
                n_chan   = rgc_noisy1.shape[0]      # number of filters
                mid_idx  = n_chan // 2              # integer division gives the “middle” index

                ts1 = rgc_noisy1[mid_idx, :]        # shape: (time_points,)
                ts2 = rgc_noisy2[mid_idx, :]

                f1 = ts1.detach().cpu().numpy()
                f2 = ts2.detach().cpu().numpy()
            else:
                f1 = rgc_noisy1.flatten().detach().cpu().numpy()
                f2 = rgc_noisy2.flatten().detach().cpu().numpy()

            # Pearson r
            r_list.append(np.corrcoef(f1, f2)[0, 1])


    # compute μ, σ and mean reliability
    data   = np.concatenate(all_vals)
    μ, σ    = data.mean(), data.std()
    mean_r = float(np.nanmean(r_list))
    n_trials = len(r_list)
    n_nan = int(np.sum(np.isnan(r_list)))
    pct_nan = n_nan / n_trials * 100

    if is_xcorr_width:
        mean_width = np.nanmean(width_list)
        n_nan = int(np.sum(np.isnan(width_list)))
        pct_nan_width = n_nan / N * 100
    else:
        mean_width = np.nan
        pct_nan_width = np.nan
        corr = np.nan

    if not is_data_saved:
        data = np.nan


    # Restore original noise settings
    dataset.add_noise = orig_noise_flag
    dataset.rgc_noise_std = orig_noise_std

    if return_histogram:
        hist, edges = np.histogram(data, bins=bins, density=True)
        return μ, σ, mean_r, mean_width, (hist, edges), pct_nan, pct_nan_width, data, corr

    return μ, σ, mean_r, mean_width


def save_sf_data_to_mat(
    multi_opt_sf,
    rf_save_folder,
    experiment_name,
    multi_opt_sf_off=None,
    index_range=(51, 69)
):
    """
    Save the full volumes and the selected slices (index_range) of `multi_opt_sf`
    (and, if provided, `multi_opt_sf_off`) into a single .mat file.

    Parameters
    ----------
    multi_opt_sf : torch.Tensor or np.ndarray
        A 3D array of shape (H, W, D). If torch.Tensor, must be convertible to NumPy.
    rf_save_folder : str
        Folder where the .mat file will be written.
    experiment_name : str
        Base name used for the .mat filename.
    multi_opt_sf_off : torch.Tensor or np.ndarray, optional
        A second 3D array (H, W, D). Only saved if not None.
    index_range : tuple(int, int), default (51, 69)
        Inclusive range of slice‐indices along the 3rd dimension to collect into
        `temp_sf_selected` (and `temp_sf_off_selected` if provided).
        For example, (51, 69) will grab slices 51,52,…,69 (0‐based indexing).
    """
    # 1) Prepare the dictionary
    mat_dict = {}

    # Helper to convert to NumPy if needed
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            return np.array(x)

    # 2) Full volume of multi_opt_sf
    mat_dict['multi_opt_sf'] = to_numpy(multi_opt_sf)

    # 3) If OFF‐channel is given, save it too
    if multi_opt_sf_off is not None:
        mat_dict['multi_opt_sf_off'] = to_numpy(multi_opt_sf_off)

    # 4) Collect selected 2D slices from multi_opt_sf
    start_idx, end_idx = index_range
    temp_sfs = []
    H, W, D = mat_dict['multi_opt_sf'].shape
    for i in range(start_idx, end_idx + 1):
        # double‐check i is within range
        if 0 <= i < D:
            temp_sfs.append(mat_dict['multi_opt_sf'][:, :, i])
    if temp_sfs:
        # shape = (num_slices, H, W)
        mat_dict['temp_sf_selected'] = np.stack(temp_sfs, axis=0)
    else:
        mat_dict['temp_sf_selected'] = np.empty((0, 0, 0))

    # 5) If OFF‐channel was provided, collect its slices too
    if multi_opt_sf_off is not None:
        temp_sfs_off = []
        H_off, W_off, D_off = mat_dict['multi_opt_sf_off'].shape
        for i in range(start_idx, end_idx + 1):
            if 0 <= i < D_off:
                temp_sfs_off.append(mat_dict['multi_opt_sf_off'][:, :, i])
        if temp_sfs_off:
            mat_dict['temp_sf_off_selected'] = np.stack(temp_sfs_off, axis=0)
        else:
            mat_dict['temp_sf_off_selected'] = np.empty((0, 0, 0))

    # 6) Write out the .mat
    os.makedirs(rf_save_folder, exist_ok=True)
    save_mat_path = os.path.join(
        rf_save_folder,
        f"{experiment_name}_sf_data.mat"
    )
    sio.savemat(save_mat_path, mat_dict)

