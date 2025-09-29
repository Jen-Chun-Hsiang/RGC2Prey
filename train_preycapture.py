import argparse
import torch
import pandas as pd
import time
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
from datetime import datetime
import torch.optim as optim
from scipy.io import savemat
import sys

from datasets.sim_cricket import RGCrfArray, SynMovieGenerator, Cricket2RGCs
from utils.utils import plot_tensor_and_save, plot_vector_and_save, plot_two_path_comparison, plot_coordinate_and_save
from models.rgc2behavior import CNN_LSTM_ObjectLocation
from utils.data_handling import save_checkpoint, none_or_float
from utils.tools import timer, MovieGenerator, save_distributions
from utils.utils import causal_moving_average
from utils.data_handling import CheckpointLoader
from utils.initialization import process_seed, initialize_logging, worker_init_fn
from utils.helper import estimate_rgc_signal_distribution, save_sf_data_to_mat
from typing import Dict, Optional, Union, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--config_name', type=str, default='not_yet_there', help='Config file name for data generation')
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')

    # Arguments for SynMovieGenerator
    parser.add_argument('--crop_size', type=tuple, default=(320, 240), help="Crop size as (width, height).")
    parser.add_argument('--boundary_size', type=str, default='(220, 140)', help="Boundary size as '(x_limit, y_limit)'.")
    parser.add_argument('--center_ratio', type=tuple, default=(0.2, 0.2), help="Center ratio for initial movement placement.")
    parser.add_argument('--max_steps', type=int, default=200, help="Maximum steps for movement.")
    parser.add_argument('--prob_stay_ob', type=float, default=0.95, help='Probability of step transition from stay to stay')
    parser.add_argument('--prob_mov_ob', type=float, default=0.975, help='Probability of step transition from moving to moving')
    parser.add_argument('--prob_stay_bg', type=float, default=0.95, help='Probability of step transition from stay to stay')
    parser.add_argument('--prob_mov_bg', type=float, default=0.975, help='Probability of step transition from moving to moving')
    parser.add_argument('--num_ext', type=int, default=50, help="Number of extended static frames.")
    parser.add_argument('--initial_velocity', type=float, default=6, help="Initial velocity for movement.")
    parser.add_argument('--momentum_decay_ob', type=float, default=0.95, help='Reduce speed in each run after moving for object')
    parser.add_argument('--momentum_decay_bg', type=float, default=0.9, help='Reduce speed in each run after moving for background')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Size of cricket image compare to its original size')
    parser.add_argument('--velocity_randomness_ob', type=float, default=0.02, help='Variation in speed change of each step')
    parser.add_argument('--velocity_randomness_bg', type=float, default=0.01, help='Variation in speed change of each step')
    parser.add_argument('--angle_range_ob', type=float, default=0.5, help='Variation in speed change of each step')
    parser.add_argument('--angle_range_bg', type=float, default=0.25, help='Variation in speed change of each step')
    parser.add_argument('--bg_folder', type=str, default='single-contrast', help='Image background folder name')
    parser.add_argument('--coord_adj_dir', type=float, default=1.0, help='Sign and value for coordinate correction for the cricket image')
    parser.add_argument('--coord_adj_type', type=str, default='body', help='Type of center points for coordinate adjustment (body/head)')
    parser.add_argument('--is_reverse_xy', action='store_true', help="Reverse x, y coordinates in cricket position correction")
    parser.add_argument('--start_scaling', type=float, default=1.0, help='Beginning scale factor of the cricket image')
    parser.add_argument('--end_scaling', type=float, default=2.0, help='Final scale factor of the cricket image')
    parser.add_argument('--dynamic_scaling', type=float, default=0.0, help='Final scale factor of the cricket image')
    parser.add_argument('--is_binocular', action='store_true', help='if generate two channels separately by binocular inputs')
    parser.add_argument('--interocular_dist', type=float, default=1.0, help='interocular distance between two eyes (in cm)')
    parser.add_argument('--is_reversed_OFF_sign', action='store_true', help='reversed off temporal filter to be the same as ON')
    parser.add_argument('--bottom_contrast', type=float, default=1.0, help='Contrast level of bottom, 1.0 original contrast')
    parser.add_argument('--top_contrast', type=float, default=1.0, help='Contrast level of top, 1.0 original contrast')
    parser.add_argument('--mean_diff_offset', type=float, default=0.0, help='control bias to mean diff_offset (bottom minius top)')
    parser.add_argument('--fix_disparity', metavar="FLOAT", type=none_or_float, default=None, nargs="?", help="Fix all disparities to FLOAT. Omit the flag or pass 'None' ")

    # Arguments for Cricket2RGCs (from movies to RGC array activities based on receptive field properties)
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples in the synthesized dataset")
    parser.add_argument('--is_norm_coords', action='store_true', help='normalize the coordinate as inputs')
    parser.add_argument('--fr2spikes', action='store_true', help='convert firing rate to spikes and keep positive (fr)')
    parser.add_argument('--quantize_scale', type=float, default=1.0, help="Firing rate to spike - quantization scaling.")
    parser.add_argument('--add_noise', action='store_true', help='Add noise to the RGC outputs')
    parser.add_argument('--rgc_noise_std', type=float, default=0.0, help="Level of noise added to the RGC outputs")
    parser.add_argument('--rgc_noise_std_max', type=float, default=None, help="Max level for sampling noise std (if set, sample uniform from [0, rgc_noise_std_max])")
    parser.add_argument('--smooth_data', action='store_true', help='Smooth data of RGC outputs, especially quantized one')
    parser.add_argument('--is_rectified', action='store_true', help='Rectify the RGC outputs')
    parser.add_argument('--is_direct_image', action='store_true', help='By passing RGC convolution')
    parser.add_argument('--is_two_grids', action='store_true', help='Make two grids instead of one, like ON and OFF, but same sign')
    parser.add_argument('--rectified_thr_ON', type=float, default=0.0, help='Threshold for ON rectification')
    parser.add_argument('--rectified_thr_OFF', type=float, default=0.0, help='Threshold for OFF rectification')
    # New rectification options matching Cricket2RGCs constructor
    parser.add_argument('--rectified_mode', type=str, default='softplus', choices=['hard', 'softplus'],
                        help='Rectification mode for RGC outputs (hard clamp_min or softplus)')
    parser.add_argument('--rectified_softness', type=float, default=1.0,
                        help='Softness parameter for softplus rectification (ON channel)')
    parser.add_argument('--rectified_softness_OFF', type=float, default=None,
                        help='Softness parameter for OFF channel; if omitted, uses ON softness')
    
    # Arguments for RGCrfArray
    parser.add_argument('--rgc_array_rf_size', type=tuple, default=(320, 240), help="Receptive field size (height, width).")
    parser.add_argument('--xlim', type=tuple, default=(-120, 120), help="x-axis limits for grid centers.")
    parser.add_argument('--ylim', type=tuple, default=(-90, 90), help="y-axis limits for grid centers.")
    parser.add_argument('--target_num_centers', type=int, default=500, help="Number of target centers to generate.")
    parser.add_argument('--target_num_centers_additional', type=int, default=None, help="Number of target centers for additional grid (overrides target_num_centers in get_additional_results).")
    parser.add_argument('--sf_scalar', type=float, default=0.2, help="Scaling factor for spatial frequency.")
    parser.add_argument('--grid_generate_method', type=str, default='circle', 
                        choices=['closest', 'decay', 'circle'], help="Method for grid generation.")
    parser.add_argument('--sf_constraint_method', type=str, default='circle', 
                        choices=['circle', 'threshold', 'None'], help="Method for grid generation.")
    parser.add_argument('--tau', type=float, default=3, help="Decay factor for 'decay' method.")
    parser.add_argument('--sf_mask_radius', type=float, default=35, help='RGC dendritic receptive field radius size in pixel')
    parser.add_argument('--mask_radius', type=float, default=30, help='RGC axonal in SC radius size in pixel')
    parser.add_argument('--rgc_rand_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--num_gauss_example', type=int, default=1, help="Number of Gaussian examples.")
    parser.add_argument('--temporal_filter_len', type=int, default=50, help="Number of time points for a temporal filter")
    parser.add_argument('--is_pixelized_tf', action='store_true', help="Flag for pixelized receptive field.")
    parser.add_argument('--grid_size_fac', type=float, default=1, help='Resize the grid size that transformed from RGC outputs')
    parser.add_argument('--set_s_scale', type=float, nargs='*', default=[], help='Set scale for surround weight of RF (LN model only; ignored in LNK model)')
    parser.add_argument('--set_s_scale_additional', type=float, nargs='*', default=[], help='Set scale for surround weight of RF for additional grid (LN model only; ignored in LNK model)')
    parser.add_argument('--is_rf_median_subtract', action='store_true', help="Flag for substract median of rf")
    parser.add_argument('--is_both_ON_OFF', action='store_true', help="Flag for including OFF cell")
    parser.add_argument('--sf_sheet_name', type=str, default='SF_params_modified', help='Excel sheet name for the spatial filter')
    parser.add_argument('--sf_sheet_name_additional', type=str, default=None, help='Excel sheet name for the spatial filter of additional grid')
    parser.add_argument('--tf_sheet_name', type=str, default='TF_params', help='Excel sheet name for the temporal filter')
    parser.add_argument('--tf_sheet_name_additional', type=str, default=None, help='Excel sheet name for the temporal filter of additional grid')
    parser.add_argument('--anti_alignment', type=float, default=1, help="value of anti-alignment from 0 (overlapping) to 1 (maximum spacing)")
    parser.add_argument('--grid_noise_level', type=float, default=0.3, help='Grid noise level (float)')
    parser.add_argument('--is_reversed_tf', action='store_true', help='Convert TF to the opposite contrast')
    parser.add_argument("--sf_id_list", type=int, nargs="+", default=None, help='select RF ids from sf_sheet_name --pid 2 7 12')
    parser.add_argument("--sf_id_list_additional", type=int, nargs="+", default=None, help='select RF ids from sf_sheet_name --pid 2 7 12')
    parser.add_argument('--syn_params', type=str, nargs='+', 
                       choices=['tf', 'sf', 'lnk'],
                       help='Parameters to synchronize across cells. Example: --syn_params tf sf lnk')
    parser.add_argument('--is_rescale_diffgaussian', action='store_true', help='Rescale the diffgaussian RF to have zero min and max to 1')
    parser.add_argument('--set_surround_size_scalar', type=float, default=None, help='Resize the rf surround size ')
    parser.add_argument('--set_bias', type=float, default=None, help='Set bias for all RGCs if provided (overrides Excel values)')
    parser.add_argument('--set_biphasic_scale', type=float, default=None, help='Set biphasic scale for temporal filter: amp2 = set_biphasic_scale * amp1')

    # Arguments for LNK model
    parser.add_argument('--use_lnk_model', action='store_true', help='Use LNK model instead of LN model for RGC responses. In LNK: w_xs controls surround interaction, s_scale is ignored.')
    parser.add_argument('--lnk_sheet_name', type=str, default='LNK_params', help='Excel sheet name for LNK parameters (includes w_xs for center-surround interaction)')
    parser.add_argument('--lnk_adapt_mode', type=str, default='divisive', choices=['divisive', 'subtractive'], 
                        help='LNK adaptation mode: divisive or subtractive')
    parser.add_argument('--use_separate_surround', action='store_true', help='Use separate center/surround filters for LNK')
    parser.add_argument('--sf_center_sheet_name', type=str, default=None, help='Sheet name for center spatial filters (if separate)')
    parser.add_argument('--sf_surround_sheet_name', type=str, default=None, help='Sheet name for surround spatial filters')
    parser.add_argument('--tf_center_sheet_name', type=str, default=None, help='Sheet name for center temporal filters (if separate)')
    parser.add_argument('--tf_surround_sheet_name', type=str, default=None, help='Sheet name for surround temporal filters')
    parser.add_argument('--surround_sigma_ratio', type=float, default=4.0, help='Ratio to scale center sigma for surround generation (default 4.0, matching MATLAB)')
    parser.add_argument('--surround_generation', type=str, default='auto', choices=['auto', 'sheet', 'both'],
                        help='Surround generation mode: auto (from center), sheet (from Excel), or both (auto if no sheet)')

    # Arguments for CNN_LSTM 
    parser.add_argument('--cnn_feature_dim', type=int, default=256, help="Number of CNN feature dimensions.")
    parser.add_argument('--lstm_hidden_size', type=int, default=64, help="Number of LSTM hiddne size.")
    parser.add_argument('--lstm_num_layers', type=int, default=3, help="Number of LSTM hiddne size.")
    parser.add_argument('--output_dim', type=int, default=2, help="Number of output dimension.")
    parser.add_argument('--conv_out_channels', type=int, default=16, help="Number of output channel in convultion layers.")
    parser.add_argument('--is_seq_reshape', action='store_true', help="Use reshape with sequence to remove for loop")
    parser.add_argument('--is_input_norm', action='store_true', help="Normalize inputs to the CNN.")
    parser.add_argument('--cnn_extractor_version', type=int, default=1, help="Versioin of CNN extractor")
    parser.add_argument('--bg_processing_type', type=str, default='one-proj', help='background processing for auxiliary cost. [one-proj|two-proj|lstm-proj]')
    parser.add_argument('--is_channel_normalization', action='store_true', help="Is perform channel normalization separately to the inputs")
    
    # Model training parameters
    parser.add_argument('--load_checkpoint_epoch', type=int, default=None, help='Epoch number of a load checkpint')
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for dataloader")
    parser.add_argument('--num_worker', type=int, default=0, help="Number of worker for dataloader")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of worker for dataloader")
    parser.add_argument('--seed', type=str, default='fixed', help=( "Seed type: 'fixed' for deterministic behavior, "
                                                                  "'random' for a random seed, or a numeric value for a custom seed."))
    parser.add_argument('--schedule_method', type=str, default='RLRP', help='Method used for scheduler')
    parser.add_argument('--schedule_factor', type=float, default=0.2, help='Scheduler reduction factor')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help="minimum learning rate for CAWR")
    parser.add_argument('--is_gradient_clip', action='store_true', help="Apply gradient clip to training process")
    parser.add_argument('--max_norm', type=float, default=5.0, help='Value for clipping by Norm')
    parser.add_argument('--do_not_train', action='store_true', help='debug for initialization')
    parser.add_argument('--is_GPU', action='store_true', help='Using GPUs for accelaration')
    parser.add_argument('--timer_tau', type=float, default=0.9, help='moving winder constant')
    parser.add_argument('--timer_sample_cicle', type=int, default=1, help='Sample circle for the timer')
    parser.add_argument('--exam_batch_idx', type=int, default=None, help='examine the timer and stop code in the middle')
    parser.add_argument('--num_epoch_save', type=int, default=5, help='Number of epoch to save a checkpoint')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of mini-batch to optimize (gradient accumulation)')
    parser.add_argument('--is_generate_movie', action='store_true', help="Generate a video to visualize the dataset")
    parser.add_argument('--bg_info_cost_ratio', type=float, default=0, help="background information ratio of its objective cost, compared to object prediction")
    parser.add_argument('--short_window_length', type=int, default=3, help='For background information extraction, short moving windom length')
    parser.add_argument('--long_window_length', type=int, default=10, help='For background information extraction, long moving windom length')
    parser.add_argument('--bg_info_type', type=str, default='rloc', help='Method for getting bg_info loc, rloc')

    return parser.parse_args()

def main():
    is_show_rgc_rf_individual = False #True
    is_show_rgc_tf = True #True
    is_show_movie_frames = False 
    is_show_pathes = False #True
    is_show_grids = False #True
    is_show_rgc_grid = True
    is_rgc_distribution = False

    args = parse_args()
    # Shared root to avoid repeating long absolute prefixes
    root_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver'

    bottom_img_folder = os.path.join(root_folder, 'CricketDataset', 'Images', 'cropped', args.bg_folder) + '/'  # grass
    top_img_folder = os.path.join(root_folder, 'CricketDataset', 'Images', 'cropped', 'cricket') + '/'
    syn_save_folder = os.path.join(root_folder, 'CricketDataset', 'Images', 'syn_img') + '/'
    rf_save_folder = os.path.join(root_folder, 'CricketDataset', 'RFs', 'RGCs') + '/'
    distribution_save_folder = os.path.join(root_folder, 'CricketDataset', 'Distribution', 'RGC_outputs') + '/'
    plot_save_folder = os.path.join(root_folder, 'RGC2Prey', 'Results', 'Figures', 'TFs') + '/'
    log_save_folder = os.path.join(root_folder, 'RGC2Prey', 'Results', 'Prints') + '/'
    savemodel_dir = os.path.join(root_folder, 'RGC2Prey', 'Results', 'CheckPoints') + '/'
    rf_params_file = os.path.join(root_folder, 'RGC2Prey', 'SimulationParams.xlsx')
    coord_mat_file = os.path.join(root_folder, 'RGC2Prey', f'selected_points_summary_{args.coord_adj_type}.mat')   # selected_points_summary.mat
    video_save_folder = os.path.join(root_folder, 'RGC2Prey', 'Results', 'Videos', 'RFs') + '/'
    mat_save_folder = os.path.join(root_folder, 'RGC2Prey', 'Results', 'Mats') + '/'

    initialize_logging(log_save_folder=log_save_folder, experiment_name=args.experiment_name)
    rnd_seed = process_seed(args.seed)
    
    if args.is_GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    logging.info( f"is_rescale_diffgaussian: {args.is_rescale_diffgaussian}")
    logging.info( f'set_s_scale: {args.set_s_scale}')
    
    # Log LNK configuration
    if args.use_lnk_model:
        logging.info( f"LNK model enabled with adaptation mode: {args.lnk_adapt_mode}")
        logging.info( f"LNK sheet name: {args.lnk_sheet_name}")
        if args.use_separate_surround:
            logging.info( f"Using separate center/surround filters")
        lnk_param_table = pd.read_excel(rf_params_file, sheet_name=args.lnk_sheet_name, usecols='C:L')
    else:
        lnk_param_table = None
        logging.info( f"Using standard LN model")

        # Simple parameter loading for RGCs
    # make sure the following usecol is provide to avoid full empty columns
    sf_param_table = pd.read_excel(rf_params_file, sheet_name=args.sf_sheet_name, usecols='C:L')
    tf_param_table = pd.read_excel(rf_params_file, sheet_name=args.tf_sheet_name, usecols='C:I')
    
    if not args.use_lnk_model and args.syn_params:
        syn_params = [p for p in args.syn_params if p != 'lnk']
    else:
        syn_params = args.syn_params
    
    # Initialize RGC array with LNK support
    rgc_array = RGCrfArray(
        sf_param_table, tf_param_table, rgc_array_rf_size=args.rgc_array_rf_size, xlim=args.xlim, ylim=args.ylim,
        target_num_centers=args.target_num_centers, sf_scalar=args.sf_scalar, grid_generate_method=args.grid_generate_method, 
        tau=args.tau, mask_radius=args.mask_radius, rgc_rand_seed=args.rgc_rand_seed, num_gauss_example=args.num_gauss_example, 
        sf_constraint_method=args.sf_constraint_method, temporal_filter_len=args.temporal_filter_len, grid_size_fac=args.grid_size_fac,
        sf_mask_radius=args.sf_mask_radius, is_pixelized_tf=args.is_pixelized_tf, set_s_scale=args.set_s_scale, 
        is_rf_median_subtract=args.is_rf_median_subtract, is_rescale_diffgaussian=args.is_rescale_diffgaussian, 
        grid_noise_level=args.grid_noise_level, is_reversed_tf=args.is_reversed_tf, sf_id_list=args.sf_id_list,
        use_lnk_override=args.use_lnk_model, lnk_param_table=lnk_param_table, syn_params=syn_params,
        set_surround_size_scalar=args.set_surround_size_scalar, set_bias=args.set_bias, set_biphasic_scale=args.set_biphasic_scale
    )
    
    logging.info( f"{args.experiment_name} processing...1")

    multi_opt_sf, multi_opt_sf_surround, tf, grid2value_mapping, map_func, rgc_locs, lnk_params = rgc_array.get_results()
    
    logging.info(f" rgc_locs overall min={np.nanmin(rgc_locs)}, max={np.nanmax(rgc_locs)}, shape={rgc_locs.shape}")
    logging.info( f"{args.experiment_name} processing...2")

    if args.is_both_ON_OFF or args.is_two_grids:
        if args.sf_sheet_name_additional and args.tf_sheet_name_additional:
            sf_param_table = pd.read_excel(rf_params_file, sheet_name=args.sf_sheet_name_additional, usecols='C:L')
            tf_param_table = pd.read_excel(rf_params_file, sheet_name=args.tf_sheet_name_additional, usecols='C:I')
        else:
            sf_param_table = None
            tf_param_table = None
        multi_opt_sf_off, multi_opt_sf_surround_off, tf_off, grid2value_mapping_off, map_func_off, rgc_locs_off, lnk_params_off = \
            rgc_array.get_additional_results(anti_alignment=args.anti_alignment, sf_id_list_additional=args.sf_id_list_additional,
                                             sf_param_table_override=sf_param_table, tf_param_table_override=tf_param_table,
                                             target_num_centers_override=args.target_num_centers_additional, 
                                             set_s_scale_override=args.set_s_scale_additional)  
        
        if args.is_binocular: 
            num_input_channel = 4
        else:
            num_input_channel = 2
    else:
        if args.is_binocular:
            num_input_channel = 2
        else:
            num_input_channel = 1
        multi_opt_sf_off, multi_opt_sf_surround_off, tf_off, grid2value_mapping_off, map_func_off, rgc_locs_off, lnk_params_off = None, None, None, None, None, None, None
    # Ensure mat save folder exists and save RGC locations with a timestamp so we can track
    # how RGC grids are generated across runs for consistency checks.
    try:
        os.makedirs(mat_save_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(mat_save_folder, f'{args.experiment_name}_rgc_locations_{timestamp}.mat')
        save_dict = {'rgc_locs': rgc_locs}
        if rgc_locs_off is not None:
            save_dict['rgc_locs_off'] = rgc_locs_off
        savemat(save_path, save_dict)
        logging.info(f"Saved RGC locations to {save_path}")
    except Exception as _e:
        logging.error(f"Failed to save RGC locations to mat file: {_e}")
    # Also save a lightweight compressed .npz file for quick repeat-checking by other scripts.
    try:
        repeat_check_dir = os.path.join(mat_save_folder, 'repeat_check', args.experiment_name)
        os.makedirs(repeat_check_dir, exist_ok=True)
        repeat_save_path = os.path.join(repeat_check_dir, f'{args.experiment_name}_rgc_locations.npz')
        if 'rgc_locs_off' in locals() and rgc_locs_off is not None:
            np.savez_compressed(repeat_save_path, rgc_locs=rgc_locs, rgc_locs_off=rgc_locs_off)
        else:
            np.savez_compressed(repeat_save_path, rgc_locs=rgc_locs)
        logging.info(f"Saved lightweight RGC repeat-check data to {repeat_save_path}")
    except Exception as _e:
        logging.error(f"Failed to save lightweight repeat-check data: {_e}")
    
    if is_show_rgc_grid:
        plot_coordinate_and_save(rgc_locs, rgc_locs_off, plot_save_folder, file_name=f'{args.experiment_name}_rgc_grids.png')
        
    logging.info( f"{args.experiment_name} processing...3")
    # Check results of RGC array synthesis
    if is_show_rgc_rf_individual:
        for i in range(multi_opt_sf.shape[2]): 
            if i > 50 and i < 70:
                temp_sf = multi_opt_sf[:, :, i].copy()
                temp_sf = torch.from_numpy(temp_sf).float()
                plot_tensor_and_save(temp_sf, rf_save_folder, f'{args.experiment_name}_receptive_field_check_{i + 1}.png')
            

        if args.is_both_ON_OFF:
            for i in range(multi_opt_sf_off.shape[2]): 
                if i > 50 and i < 70:
                    temp_sf = multi_opt_sf_off[:, :, i].copy()
                    temp_sf = torch.from_numpy(temp_sf).float()
                    plot_tensor_and_save(temp_sf, rf_save_folder, f'{args.experiment_name}_receptive_field_check_OFF_{i + 1}.png')
             
            save_sf_data_to_mat(multi_opt_sf=multi_opt_sf, multi_opt_sf_off=multi_opt_sf_off, rf_save_folder=mat_save_folder,
                experiment_name=args.experiment_name, index_range=(51, 69)
            )
        else:
            save_sf_data_to_mat(multi_opt_sf=multi_opt_sf,rf_save_folder=mat_save_folder, experiment_name=args.experiment_name,
                index_range=(51, 69)
            )
        
        raise ValueError(f"check data range...")

    logging.info( f"{args.experiment_name} processing...4")
    if is_show_rgc_tf:
        # Delegate plotting of 1D or multi-filter temporal filters to utils.plot_temporal_filters
        try:
            from utils.utils import plot_temporal_filters
            plot_temporal_filters(tf, plot_save_folder, base_file_name=f'{args.experiment_name}_temporal_filter')
        except Exception:
            # Fallback: try the original simple plot
            plot_vector_and_save(tf, plot_save_folder, file_name=f'{args.experiment_name}_temporal_filter.png')

        sys.exit(0)

    movie_generator = SynMovieGenerator(top_img_folder, bottom_img_folder,
        crop_size=args.crop_size, boundary_size=args.boundary_size, center_ratio=args.center_ratio, max_steps=args.max_steps,
        prob_stay_ob=args.prob_stay_ob, prob_mov_ob=args.prob_mov_ob, prob_stay_bg=args.prob_stay_bg, prob_mov_bg=args.prob_mov_bg, 
        num_ext=args.num_ext, initial_velocity=args.initial_velocity, momentum_decay_ob=args.momentum_decay_ob, 
        momentum_decay_bg=args.momentum_decay_bg, scale_factor=args.scale_factor, velocity_randomness_ob = args.velocity_randomness_ob, 
        velocity_randomness_bg=args.velocity_randomness_bg, angle_range_ob=args.angle_range_ob, angle_range_bg=args.angle_range_bg, 
        coord_mat_file=coord_mat_file, correction_direction=args.coord_adj_dir, is_reverse_xy=args.is_reverse_xy, 
        start_scaling=args.start_scaling, end_scaling=args.end_scaling, dynamic_scaling=args.dynamic_scaling, is_binocular=args.is_binocular,
        interocular_dist=args.interocular_dist, bottom_contrast=args.bottom_contrast, top_contrast=args.top_contrast, 
        mean_diff_offset=args.mean_diff_offset, fix_disparity=args.fix_disparity
    )
    logging.info( f"{args.experiment_name} processing...5")
    xlim, ylim = args.xlim, args.ylim
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]

    if is_rgc_distribution:
        # Create simple Cricket2RGCs dataset for distribution analysis
        train_dataset = Cricket2RGCs(
            num_samples=args.num_samples,
            multi_opt_sf=multi_opt_sf,
            tf=tf,
            map_func=map_func,
            grid2value_mapping=grid2value_mapping,
            target_width=target_width,
            target_height=target_height,
            movie_generator=movie_generator,
            grid_size_fac=args.grid_size_fac,
            is_norm_coords=args.is_norm_coords,
            is_syn_mov_shown=True,
            fr2spikes=args.fr2spikes,
            is_both_ON_OFF=args.is_both_ON_OFF,
            quantize_scale=args.quantize_scale,
            add_noise=args.add_noise,
            rgc_noise_std=args.rgc_noise_std,
            rgc_noise_std_max=args.rgc_noise_std_max,
            smooth_data=args.smooth_data,
            is_rectified=args.is_rectified,
            is_direct_image=args.is_direct_image,
            is_reversed_OFF_sign=args.is_reversed_OFF_sign,
            is_two_grids=args.is_two_grids,
            rectified_thr_ON=args.rectified_thr_ON,
            rectified_thr_OFF=args.rectified_thr_OFF,
            rectified_mode=args.rectified_mode,
            rectified_softness=args.rectified_softness,
            rectified_softness_OFF=args.rectified_softness_OFF,
            multi_opt_sf_off=multi_opt_sf_off,
            tf_off=tf_off,
            map_func_off=map_func_off,
            grid2value_mapping_off=grid2value_mapping_off,
            # Simple LNK parameters
            use_lnk=args.use_lnk_model,
            lnk_params=lnk_params,
            lnk_params_off=lnk_params_off,
            surround_sigma_ratio=args.surround_sigma_ratio,
            surround_sf=multi_opt_sf_surround,
            surround_sf_off=multi_opt_sf_surround_off,
            # Random seed for consistent data generation
            rnd_seed=rnd_seed+5678
        )
        mu, sigma, mean_r, mean_width, (hist, edges), pct_nan, pct_nan_width, data, corr = estimate_rgc_signal_distribution(
            train_dataset,
            N=1000,
            channel_idx=0,
            return_histogram=True,
            bins=100    
        )
        logging.info(f"Signal μ={mu:.3f}, σ={sigma:.3f}, mean_r={mean_r:.3f}, mean_width={mean_width:.3f}")

        save_path = os.path.join(mat_save_folder, f'{args.experiment_name}_rgc_time_distribution_singleRGC.mat')
        savemat(save_path, {'mu': mu, 'sigma': sigma, 'mean_r': mean_r, 'hist': hist, 'edges': edges, 'pct_nan':pct_nan, 
                            'rgc_noise_std':args.rgc_noise_std, 'pct_nan_width': pct_nan_width, 'mean_width': mean_width,
                            'data': data, 'corr': corr})
    
        sys.exit(0)

    # Create training dataset with simplified LNK support  
    train_dataset = Cricket2RGCs(
        num_samples=args.num_samples,
        multi_opt_sf=multi_opt_sf,
        tf=tf,
        map_func=map_func,
        grid2value_mapping=grid2value_mapping,
        target_width=target_width,
        target_height=target_height,
        movie_generator=movie_generator,
        grid_size_fac=args.grid_size_fac,
        is_norm_coords=args.is_norm_coords,
        is_syn_mov_shown=True,
        fr2spikes=args.fr2spikes,
        is_both_ON_OFF=args.is_both_ON_OFF,
        quantize_scale=args.quantize_scale,
        add_noise=args.add_noise,
        rgc_noise_std=args.rgc_noise_std,
        rgc_noise_std_max=args.rgc_noise_std_max,
        smooth_data=args.smooth_data,
        is_rectified=args.is_rectified,
        is_direct_image=args.is_direct_image,
        is_reversed_OFF_sign=args.is_reversed_OFF_sign,
        is_two_grids=args.is_two_grids,
        rectified_thr_ON=args.rectified_thr_ON,
        rectified_thr_OFF=args.rectified_thr_OFF,
        rectified_mode=args.rectified_mode,
        rectified_softness=args.rectified_softness,
        rectified_softness_OFF=args.rectified_softness_OFF,
        multi_opt_sf_off=multi_opt_sf_off,
        tf_off=tf_off,
        map_func_off=map_func_off,
        grid2value_mapping_off=grid2value_mapping_off,
        # Simple LNK parameters
        use_lnk=args.use_lnk_model,
        lnk_params=lnk_params,
        lnk_params_off=lnk_params_off,
        surround_sigma_ratio=args.surround_sigma_ratio,
        surround_sf=multi_opt_sf_surround,
        surround_sf_off=multi_opt_sf_surround_off,
        # Random seed for consistent data generation
        rnd_seed=rnd_seed+5678
    )
    
    logging.info( f"{args.experiment_name} processing...6")
    # Visualize a few points
    # for i in range(20):
    #     logging.info(f'iteration: {i}')
    #     sequence, path, path_bg, syn_movie, scaling_factors, _, _, _ = train_dataset[i]
    # sys.exit(0)

    if is_show_movie_frames:
        for i in range(syn_movie.shape[0]):
            Timg = syn_movie[i, :, :]
            plot_tensor_and_save(Timg, syn_save_folder, f'{args.experiment_name}_synthesized_movement_doublecheck_{i + 1}.png')  
    if is_show_pathes:
        plot_two_path_comparison(path, path_bg, plot_save_folder, file_name=f'{args.experiment_name}_movement_pathes.png')
    if is_show_grids:
        for i in range(sequence.shape[0]):
            Timg = sequence[i, 0, :, :].squeeze()
            plot_tensor_and_save(Timg, syn_save_folder, f'{args.experiment_name}_RGCgrid_activity_doublecheck_{i + 1}.png')
            if args.is_both_ON_OFF:
                Timg = sequence[i, 1, :, :].squeeze()
                plot_tensor_and_save(Timg, syn_save_folder, f'{args.experiment_name}_RGCgrid_activity_doublecheck_OFF_{i + 1}.png')
        if is_show_pathes:
            plot_two_path_comparison(path, path_bg, plot_save_folder, file_name=f'{args.experiment_name}_dataset_path.png')
    
    logging.info( f"{args.experiment_name} processing...7")
    if args.is_generate_movie:
        frame_width = 640
        frame_height = 480
        fps = 20
        path = path.squeeze()*train_dataset.norm_path_fac
        path_bg = path_bg.squeeze()*train_dataset.norm_path_fac
        syn_movie = syn_movie.squeeze().numpy()
        sequence = sequence.squeeze().numpy()
        scaling_factors = scaling_factors.squeeze()
        predicted_path = None
        data_movie = MovieGenerator(frame_width, frame_height, fps, video_save_folder, bls_tag=f'{args.experiment_name}',
                                grid_generate_method=args.grid_generate_method)
        data_movie.generate_movie(sequence, syn_movie, path, path_bg, predicted_path, scaling_factors, video_id=1)
    
    # Create final training dataset (no movie visualization)
    train_dataset = Cricket2RGCs(
        num_samples=args.num_samples,
        multi_opt_sf=multi_opt_sf,
        tf=tf,
        map_func=map_func,
        grid2value_mapping=grid2value_mapping,
        target_width=target_width,
        target_height=target_height,
        movie_generator=movie_generator,
        grid_size_fac=args.grid_size_fac,
        is_norm_coords=args.is_norm_coords,
        is_syn_mov_shown=False,
        fr2spikes=args.fr2spikes,
        is_both_ON_OFF=args.is_both_ON_OFF,
        quantize_scale=args.quantize_scale,
        add_noise=args.add_noise,
        rgc_noise_std=args.rgc_noise_std,
        rgc_noise_std_max=args.rgc_noise_std_max,
        smooth_data=args.smooth_data,
        is_rectified=args.is_rectified,
        is_direct_image=args.is_direct_image,
        is_reversed_OFF_sign=args.is_reversed_OFF_sign,
        is_two_grids=args.is_two_grids,
        rectified_thr_ON=args.rectified_thr_ON,
        rectified_thr_OFF=args.rectified_thr_OFF,
        rectified_mode=args.rectified_mode,
        rectified_softness=args.rectified_softness,
        rectified_softness_OFF=args.rectified_softness_OFF,
        multi_opt_sf_off=multi_opt_sf_off,
        tf_off=tf_off,
        map_func_off=map_func_off,
        grid2value_mapping_off=grid2value_mapping_off,
        # Simple LNK parameters
        use_lnk=args.use_lnk_model,
        lnk_params=lnk_params,
        surround_sigma_ratio=args.surround_sigma_ratio,
        lnk_params_off=lnk_params_off,
        surround_sf=multi_opt_sf_surround,
        surround_sf_off=multi_opt_sf_surround_off,
        # Random seed for consistent data generation
        rnd_seed=rnd_seed+5678
    )

    logging.info( f"{args.experiment_name} processing...8")
    dl_generator = torch.Generator().manual_seed(rnd_seed) 
    if args.num_worker==0:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  worker_init_fn=worker_init_fn, generator=dl_generator)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_worker, pin_memory=True, persistent_workers=False, 
                                        worker_init_fn=worker_init_fn, generator=dl_generator)

    logging.info( f"{args.experiment_name} processing...9")
    # Sample Training Loop
    grid_width = int(np.round(target_width*args.grid_size_fac))
    grid_height = int(np.round(target_height*args.grid_size_fac))
    model = CNN_LSTM_ObjectLocation(cnn_feature_dim=args.cnn_feature_dim, lstm_hidden_size=args.lstm_hidden_size,
                                     lstm_num_layers=args.lstm_num_layers, output_dim=args.output_dim,
                                    input_height=grid_width, input_width=grid_height, conv_out_channels=args.conv_out_channels,
                                    is_input_norm=args.is_input_norm, is_seq_reshape=args.is_seq_reshape, CNNextractor_version=args.cnn_extractor_version,
                                    num_input_channel=num_input_channel, bg_info_cost_ratio=args.bg_info_cost_ratio, bg_processing_type=args.bg_processing_type,
                                    is_channel_normalization=args.is_channel_normalization)
    
    logging.info( f"{args.experiment_name} processing...10")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    if args.schedule_method.lower() == 'rlrp':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.schedule_factor, patience=5)
    elif args.schedule_method.lower() == 'cawr':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=args.min_lr)

    if args.load_checkpoint_epoch:
        checkpoint_filename = f'{args.experiment_name}_checkpoint_epoch_{args.load_checkpoint_epoch}'
        checkpoint_filename = os.path.join(savemodel_dir, f'{checkpoint_filename}.pth')
        checkpoint_loader = CheckpointLoader(checkpoint_filename)
        model, optimizer, scheduler = checkpoint_loader.load_checkpoint(model, optimizer, scheduler)
        start_epoch = checkpoint_loader.load_epoch()
        training_losses = checkpoint_loader.load_training_losses()
    else:
        start_epoch = 0
        training_losses = []  # To store the loss at each epoch


    logging.info( f"{args.experiment_name} processing...11")
    if args.do_not_train:
        # Set the number of initial batches to process
        n = 20  # Change this value to the desired number of batches
        # Loop through the data loader and process the first n batches
        model.eval()  # Set the model to evaluation mode

        plot_file_name = f'{args.experiment_name}_value_distribution_n{n}.png'

        save_distributions(train_loader, n, folder_name=distribution_save_folder, file_name=plot_file_name, logging=None)


        # with torch.no_grad():  # Disable gradient computation
        #     for batch_idx, (sequences, targets, _) in enumerate(train_loader):
        #         if batch_idx >= n:
        #             break  # Exit after processing n batches

        #         # Print inputs and outputs for the current batch
        #         print(f"Batch {batch_idx + 1} Inputs:")
        #         print(f'sequence min {torch.min(sequences)}')
        #         print(f'sequence max {torch.max(sequences)}')
        #         print(f'targets min {torch.min(targets)}')
        #         print(f'targets max {torch.max(targets)}')

        #         outputs = model(sequences)
        #         print(f"\nBatch {batch_idx + 1} Outputs:")
        #         print(f'output min {torch.min(outputs)}')
        #         print(f'output max {torch.max(outputs)}')
        #         print("\n" + "-" * 50 + "\n")

        

    else:
        num_epochs = args.num_epochs
        timer_data_loading = {'min': None, 'max': None, 'moving_avg': None, 'counter': 0}
        timer_data_transfer = {'min': None, 'max': None, 'moving_avg': None, 'counter': 0}
        timer_data_processing = {'min': None, 'max': None, 'moving_avg': None, 'counter': 0}
        timer_data_backpropagate = {'min': None, 'max': None, 'moving_avg': None, 'counter': 0}
        for epoch in range(start_epoch, num_epochs):
            model.train()
            optimizer.zero_grad()
            epoch_loss = 0.0
            
            start_time = time.time()
            data_iterator = iter(train_loader)
            for batch_idx in range(len(train_loader)):
                with timer(timer_data_loading, tau=args.timer_tau, n=args.timer_sample_cicle):
                    sequences, targets, bg_info = next(data_iterator)

                with timer(timer_data_transfer, tau=args.timer_tau, n=args.timer_sample_cicle):
                    sequences, targets, bg_info = sequences.to(device), targets.to(device), bg_info.to(device)

                if args.bg_info_type == 'rloc':
                    bg_info = causal_moving_average(bg_info, args.short_window_length) - \
                            causal_moving_average(bg_info, args.long_window_length)
                # Forward pass
                with timer(timer_data_processing, tau=args.timer_tau, n=args.timer_sample_cicle):
                    outputs, bg_pred = model(sequences)
                
                # Compute loss
                with timer(timer_data_backpropagate, tau=args.timer_tau, n=args.timer_sample_cicle):
                    loss = (1-args.bg_info_cost_ratio) * criterion(outputs, targets) + args.bg_info_cost_ratio * criterion(bg_pred, bg_info)
                    loss.backward()

                    # Apply gradient clipping
                    if args.is_gradient_clip:
                        max_norm = args.max_norm  # Max gradient norm
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                    if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                
                    epoch_loss += loss.item()

                # Print timer data at specific batch index
                if args.exam_batch_idx is not None:
                    if batch_idx*args.batch_size > args.exam_batch_idx:
                        logging.debug(f"Batch {batch_idx}:")
                        logging.debug(f"Data Loading: {timer_data_loading}")
                        logging.debug(f"Data Transfer: {timer_data_transfer}")
                        logging.debug(f"Processing: {timer_data_processing}")
                        logging.debug(f"Backpropagation: {timer_data_backpropagate}")
                        break

            if args.exam_batch_idx is not None:
                break
            
            # Average loss for the epoch
            avg_train_loss = epoch_loss / len(train_loader)

            # Simple sanity check for epoch loss: None, NaN, Inf, or extremely large values
            if avg_train_loss is None or not np.isfinite(avg_train_loss) or np.isnan(avg_train_loss) or abs(avg_train_loss) > 1e12:
                logging.error(f"Abnormal epoch loss detected at epoch {epoch}: {avg_train_loss}")
                # Try to save a checkpoint for debugging
                try:
                    checkpoint_filename = f'{args.experiment_name}_abnormal_epoch_loss_epoch_{epoch + 1}.pth'
                    save_checkpoint(epoch, model, optimizer, training_losses=training_losses, scheduler=scheduler, args=args,
                                    file_path=os.path.join(savemodel_dir, checkpoint_filename))
                    logging.info(f"Saved checkpoint to {os.path.join(savemodel_dir, checkpoint_filename)} before aborting.")
                except Exception as e:
                    logging.error(f"Failed to save checkpoint before aborting: {e}")

                raise ValueError(f"Abnormal epoch loss detected: {avg_train_loss} at epoch {epoch}")

            training_losses.append(avg_train_loss)
            # Scheduler step
            if args.schedule_method.lower() == 'rlrp':
                scheduler.step(avg_train_loss)
            elif args.schedule_method.lower() == 'cawr':
                scheduler.step(epoch + (epoch / num_epochs))

            elapsed_time = time.time()  - start_time
            logging.info( f"{args.experiment_name} Epoch [{epoch + 1}/{num_epochs}], Elapsed time: {elapsed_time:.2f} seconds \n"
                            f"\tLoss: {avg_train_loss:.4f} \n")
            
            if (epoch + 1) % args.num_epoch_save == 0:  # Example: Save every 10 epochs
                checkpoint_filename = f'{args.experiment_name}_checkpoint_epoch_{epoch + 1}.pth'
                save_checkpoint(epoch, model, optimizer, training_losses=training_losses, scheduler=scheduler, args=args,  
                                    file_path=os.path.join(savemodel_dir, checkpoint_filename))

if __name__ == '__main__':
    main()