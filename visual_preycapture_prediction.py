import argparse
import torch
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import multiprocessing as mp
from datetime import datetime
from torch.utils.data import DataLoader
from scipy.io import savemat
from datasets.sim_cricket import SynMovieGenerator, Cricket2RGCs, RGCrfArray
from models.rgc2behavior import CNN_LSTM_ObjectLocation
from utils.utils import plot_two_path_comparison, plot_coordinate_and_save
from utils.data_handling import CheckpointLoader
from utils.tools import MovieGenerator
from utils.file_name import make_file_name
from utils.initialization import process_seed, initialize_logging, worker_init_fn


def _generate_movie_job(job):
    """Worker function to generate a single movie in a separate process.
    
    Args:
        job (dict): Dictionary containing all necessary data and parameters
                   for movie generation (must be picklable).
    """
    try:
        # Import here to avoid issues with multiprocessing
        from utils.tools import MovieGenerator
        
        data_movie = MovieGenerator(
            job['frame_width'], job['frame_height'], job['fps'], 
            job['video_save_folder'], bls_tag=f"{job['file_name']}_{job['epoch_number']}",
            grid_generate_method=job['grid_generate_method']
        )
        
        data_movie.generate_movie(
            job['inputs'], job['syn_movie'], job['true_path'], job['bg_path'],
            job['predicted_path'], job['scaling_factors'], video_id=job['video_id'],
            weighted_coords=job.get('weighted_coords', None),
            save_frames=job.get('save_frames', False),
            frame_save_root=job.get('frame_save_root', None),
            truth_marker_style=job.get('truth_marker_style', None),
            prediction_marker_style=job.get('prediction_marker_style', None),
            center_marker_style=job.get('center_marker_style', None),
            enable_truth_marker=job.get('enable_truth_marker', False),
            enable_prediction_marker=job.get('enable_prediction_marker', False),
            enable_center_marker=job.get('enable_center_marker', False),
            input_channel_index=job.get('input_channel_index', 0)
        )
        
        return f"Movie {job['video_id']} generated successfully"
        
    except Exception as e:
        import logging
        logging.exception(f"Movie generation failed for job {job.get('video_id', 'unknown')}: {e}")
        return f"Movie {job.get('video_id', 'unknown')} failed: {str(e)}"


def parse_args():
    
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--experiment_names', type=str, nargs='+', required=True, help="List of experiment names")
    parser.add_argument('--noise_levels', type=float, nargs='+', default=None, help="List of noise levels as numbers")
    parser.add_argument('--fix_disparity_degrees', type=float, nargs='+', default=None, help="List of fixed disparity degrees as numbers")
    parser.add_argument('--test_bg_folder', type=str, default=None, help="Background folder for testing")
    parser.add_argument('--test_ob_folder', type=str, default=None, help="Object folder for testing")
    parser.add_argument('--boundary_size', type=str, default=None, help="Boundary size as '(x_limit, y_limit)'.")
    parser.add_argument('--epoch_number', type=int, default=200, help="Epoch number to check")
    parser.add_argument('--num_worker', type=int, default=None,
                        help="Number of DataLoader workers and max parallel movie workers (overrides checkpoint setting)")
    parser.add_argument('--save_movie_frames', action='store_true', help="Save per-frame images for movies and RGC outputs")
    parser.add_argument('--frame_save_root', type=str, default=None, help="Root directory for saved frame images")
    parser.add_argument('--frame_truth_marker', type=str, default='o', help="Marker symbol for ground truth positions")
    parser.add_argument('--frame_truth_color', type=str, default='royalblue', help="Marker color for ground truth positions")
    parser.add_argument('--frame_truth_size', type=float, default=60.0, help="Marker size for ground truth positions")
    parser.add_argument('--frame_pred_marker', type=str, default='x', help="Marker symbol for predicted positions")
    parser.add_argument('--frame_pred_color', type=str, default='darkorange', help="Marker color for predicted positions")
    parser.add_argument('--frame_pred_size', type=float, default=60.0, help="Marker size for predicted positions")
    parser.add_argument('--frame_center_marker', type=str, default='+', help="Marker symbol for center RF positions")
    parser.add_argument('--frame_center_color', type=str, default='crimson', help="Marker color for center RF positions")
    parser.add_argument('--frame_center_size', type=float, default=60.0, help="Marker size for center RF positions")
    parser.add_argument('--add_truth_marker', action='store_true', help="Include ground truth marker in saved frames")
    parser.add_argument('--add_pred_marker', action='store_true', help="Include prediction marker in saved frames")
    parser.add_argument('--add_center_marker', action='store_true', help="Include center RF marker in saved frames")
    parser.add_argument('--visual_sample_ids', type=int, nargs='+', default=None,
                        help="0-based indices from Section 2 analysis to visualize in Section 3")
    parser.add_argument('--movie_eye', type=str, default='left',
                        help="Which eye/grid to use when generating movies and frames: 'left'/'right' or '0'/'1' or 'first'/'second'.")
    parser.add_argument('--movie_input_channel', type=str, default='first',
                        help="Which input channel/grid to visualize in movies: 'first'/'second' or numeric index (0-based).")

    return parser.parse_args()


def main():
    args = parse_args()
    noise_levels = args.noise_levels if args.noise_levels else [None]
    for experiment_name in args.experiment_names:
        if args.fix_disparity_degrees:
                for disp in args.fix_disparity_degrees:
                    for noise_level in noise_levels:
                        run_experiment(
                            experiment_name=experiment_name,
                            noise_level=noise_level,
                            fix_disparity_degree=disp,
                            test_bg_folder=args.test_bg_folder,
                            test_ob_folder=args.test_ob_folder,
                            boundary_size=args.boundary_size,
                            cli_num_worker=args.num_worker,
                            epoch_number=args.epoch_number,
                            save_movie_frames=args.save_movie_frames,
                            frame_save_root=args.frame_save_root,
                            frame_truth_marker=args.frame_truth_marker,
                            frame_truth_color=args.frame_truth_color,
                            frame_truth_size=args.frame_truth_size,
                            frame_pred_marker=args.frame_pred_marker,
                            frame_pred_color=args.frame_pred_color,
                            frame_pred_size=args.frame_pred_size,
                            frame_center_marker=args.frame_center_marker,
                            frame_center_color=args.frame_center_color,
                            frame_center_size=args.frame_center_size,
                            add_truth_marker=args.add_truth_marker,
                            add_pred_marker=args.add_pred_marker,
                            add_center_marker=args.add_center_marker,
                            visual_sample_ids=args.visual_sample_ids,
                            movie_eye=args.movie_eye,
                            movie_input_channel=args.movie_input_channel
                        )
        else:
            for noise_level in noise_levels:
                run_experiment(
                    experiment_name=experiment_name,
                    noise_level=noise_level,
                    # omit disparity argument or pass None
                    test_bg_folder=args.test_bg_folder,
                    test_ob_folder=args.test_ob_folder,
                    boundary_size=args.boundary_size,
                    cli_num_worker=args.num_worker,
                    epoch_number=args.epoch_number,
                    save_movie_frames=args.save_movie_frames,
                    frame_save_root=args.frame_save_root,
                    frame_truth_marker=args.frame_truth_marker,
                    frame_truth_color=args.frame_truth_color,
                    frame_truth_size=args.frame_truth_size,
                    frame_pred_marker=args.frame_pred_marker,
                    frame_pred_color=args.frame_pred_color,
                    frame_pred_size=args.frame_pred_size,
                    frame_center_marker=args.frame_center_marker,
                    frame_center_color=args.frame_center_color,
                    frame_center_size=args.frame_center_size,
                    add_truth_marker=args.add_truth_marker,
                    add_pred_marker=args.add_pred_marker,
                    add_center_marker=args.add_center_marker,
                    visual_sample_ids=args.visual_sample_ids,
                    movie_eye=args.movie_eye,
                    movie_input_channel=args.movie_input_channel
                )


def run_experiment(experiment_name, noise_level=None, fix_disparity_degree=None, test_bg_folder=None, test_ob_folder=None,
                   boundary_size=None, epoch_number=200, cli_num_worker=None, save_movie_frames=False,
                   frame_save_root=None, frame_truth_marker='o', frame_truth_color='royalblue', frame_truth_size=60.0,
                   frame_pred_marker='x', frame_pred_color='darkorange', frame_pred_size=60.0,
                   frame_center_marker='+', frame_center_color='crimson', frame_center_size=60.0,
                   add_truth_marker=False, add_pred_marker=False, add_center_marker=False, visual_sample_ids=None,
                   movie_eye='left', movie_input_channel='first'):
    num_display = 3
    frame_width = 640
    frame_height = 480
    fps = 20
    num_sample = 1000
    is_making_video = True
    is_add_noise = False
    is_plot_centerFR = True
    is_show_rgc_grid = True
    is_save_movie_sequence_to_mat = False
    checkpoint_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/CheckPoints/'
    
    rf_params_file = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/SimulationParams.xlsx'
    test_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Figures/'
    plot_save_folder =  '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Figures/TFs/'
    # coord_mat_file = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/selected_points_summary.mat'
    
    video_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Videos/'
    mat_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Mats/'
    rgc_array_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/RGC_arrays/'
    log_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Prints/'
    frame_img_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Frames/'

    if save_movie_frames:
        frame_save_root_final = frame_save_root if frame_save_root is not None else os.path.join(frame_img_folder, 'frames')
        frame_save_msg = f"Frame saving enabled. Frames will be stored under {frame_save_root_final}"
        
        enable_truth_marker = bool(add_truth_marker)
        enable_pred_marker = bool(add_pred_marker)
        enable_center_marker = bool(add_center_marker)
        if not (enable_truth_marker or enable_pred_marker or enable_center_marker):
            frame_save_msg = f"Frame saving requested without marker flags; frames will be saved without markers."
    else:
        frame_save_root_final = None
        enable_truth_marker = False
        enable_pred_marker = False
        enable_center_marker = False
        if add_truth_marker or add_pred_marker or add_center_marker:
            frame_save_msg = "Marker flags provided but --save_movie_frames was not set; markers will be ignored."

    truth_marker_style = None
    if enable_truth_marker:
        truth_marker_style = {
            'marker': frame_truth_marker,
            'color': frame_truth_color,
            'size': frame_truth_size,
            'label': 'Ground Truth'
        }

    pred_marker_style = None
    if enable_pred_marker:
        pred_marker_style = {
            'marker': frame_pred_marker,
            'color': frame_pred_color,
            'size': frame_pred_size,
            'label': 'Prediction'
        }

    center_marker_style = None
    if enable_center_marker:
        center_marker_style = {
            'marker': frame_center_marker,
            'color': frame_center_color,
            'size': frame_center_size,
            'label': 'Center RF'
        }

    if test_ob_folder is None:
        test_ob_folder = 'cricket'
    elif test_ob_folder == 'white-spot':
        coord_mat_file = None

    if noise_level is not None:
        is_add_noise = True
    
    file_name = make_file_name(experiment_name, test_ob_folder, test_bg_folder, noise_level=noise_level, fix_disparity_degree=fix_disparity_degree)
    initialize_logging(log_save_folder=log_save_folder, experiment_name=file_name)

    logging.info(frame_save_msg)
    logging.info(f"{file_name} processing...-1 noise:{noise_level} type:{type(noise_level)}")
    
    # Load checkpoint
    if experiment_name.startswith('1229'):
        checkpoint_filename = os.path.join(checkpoint_path, f'{experiment_name}_cricket_location_prediction_checkpoint_epoch_{epoch_number}.pth')
    else:
        checkpoint_filename = os.path.join(checkpoint_path, f'{experiment_name}_checkpoint_epoch_{epoch_number}.pth')
    checkpoint_loader = CheckpointLoader(checkpoint_filename)
    args = checkpoint_loader.load_args()
    # Merge CLI override for num_worker if provided (do not overwrite other checkpoint args)
    if cli_num_worker is not None:
        try:
            args.num_worker = int(cli_num_worker)
            logging.info(f"Overriding checkpoint num_worker with CLI value: {args.num_worker}")
        except Exception:
            logging.warning(f"Invalid cli_num_worker provided: {cli_num_worker}. Using checkpoint or defaults.")
    training_losses = checkpoint_loader.load_training_losses()

    
    if test_bg_folder is None:
        test_bg_folder = args.bg_folder
    top_img_folder    = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/{test_ob_folder}/'
    bottom_img_folder = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/{test_bg_folder}/'  #grass

    if test_ob_folder == 'cricket':
        coord_mat_file = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/selected_points_summary_{args.coord_adj_type}.mat'   #selected_points_summary.mat
    
    if boundary_size is None:
        boundary_size = args.boundary_size

    logging.info(f"{file_name} processing...0")
    if args.is_GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    if not hasattr(args, 'mask_radius'):
        args.mask_radius = None
    if not hasattr(args, 'rgc_rand_seed'):
        args.rgc_rand_seed = 42
    if not hasattr(args, 'is_norm_coords'):
        args.is_norm_coords = False
    if not hasattr(args, 'is_input_norm'):
        args.is_input_norm = False
    if not hasattr(args, 'seed'):
        args.seed = '42'
    if not hasattr(args, 'is_direct_image'):
        args.is_direct_image = False
    if not hasattr(args, 'sf_sheet_name'):
        args.sf_sheet_name = 'SF_params_modified'
    if not hasattr(args, 'tf_sheet_name'):
        args.tf_sheet_name = 'TF_params'
    if not hasattr(args, 'bg_info_cost_ratio'):
        args.bg_info_cost_ratio = 0
    if not hasattr(args, 'bg_processing_type'):
        args.bg_processing_type = 'one-proj'
    if not hasattr(args, 'grid_noise_level'):
        args.grid_noise_level = 0.3
    if not hasattr(args, 'is_channel_normalization'):
        args.is_channel_normalization = False
    if not hasattr(args, 'is_binocular'):
        args.is_binocular = False
    if not hasattr(args, 'interocular_dist'):
        args.interocular_dist = 1.0
    if not hasattr(args, 'is_reversed_tf'):
        args.is_reversed_tf = False
    if not hasattr(args, 'is_reversed_OFF_sign'):
        args.is_reversed_OFF_sign = False
    # Rectification threshold defaults (handle legacy misspelled attributes)
    if not hasattr(args, 'rectified_thr_ON'):
        if hasattr(args, 'rectifed_thr_ON'):
            # legacy misspelled attribute exists, copy it
            args.rectified_thr_ON = args.rectifed_thr_ON
        else:
            args.rectified_thr_ON = 0.0
    if not hasattr(args, 'rectified_thr_OFF'):
        if hasattr(args, 'rectifed_thr_OFF'):
            # legacy misspelled attribute exists, copy it
            args.rectified_thr_OFF = args.rectifed_thr_OFF
        else:
            args.rectified_thr_OFF = 0.0
    # Ensure rectification mode/softness attributes exist (may come from checkpoint)
    if not hasattr(args, 'rectified_mode'):
        args.rectified_mode = 'softplus'
    if not hasattr(args, 'rectified_softness'):
        args.rectified_softness = 1.0
    if not hasattr(args, 'rectified_softness_OFF'):
        args.rectified_softness_OFF = None
    if not hasattr(args, 'bottom_contrast'):
        args.bottom_contrast = 1.0
    if not hasattr(args, 'top_contrast'):
        args.top_contrast = 1.0
    if not hasattr(args, 'mean_diff_offset'):
        args.mean_diff_offset = 0.0
    if not hasattr(args, 'syn_tf_sf'):
        args.syn_tf_sf = False
    if not hasattr(args, 'is_rescale_diffgaussian'):
        args.is_rescale_diffgaussian = True
    if not hasattr(args, 'set_surround_size_scalar'):
        args.set_surround_size_scalar = None
    
    # Add LNK model attributes
    if not hasattr(args, 'use_lnk_model'):
        args.use_lnk_model = False
    if not hasattr(args, 'lnk_sheet_name'):
        args.lnk_sheet_name = 'LNK_params'
    if not hasattr(args, 'lnk_adapt_mode'):
        args.lnk_adapt_mode = 'divisive'
    if not hasattr(args, 'surround_sigma_ratio'):
        args.surround_sigma_ratio = 4.0
    if not hasattr(args, 'is_two_grids'):
        args.is_two_grids = False
    if not hasattr(args, 'syn_params'):
        args.syn_params = None
    if not hasattr(args, 'sf_id_list'):
        args.sf_id_list = None
    if not hasattr(args, 'sf_id_list_additional'):
        args.sf_id_list_additional = None
    if not hasattr(args, 'anti_alignment'):
        args.anti_alignment = 1.0
    if not hasattr(args, 'rgc_noise_std_max'):
        args.rgc_noise_std_max = None
    if not hasattr(args, 'set_bias'):
        args.set_bias = None
    if not hasattr(args, 'set_biphasic_scale'):
        args.set_biphasic_scale = None
    if not hasattr(args, 'sf_sheet_name_additional'):
        args.sf_sheet_name_additional = None
    if not hasattr(args, 'tf_sheet_name_additional'):
        args.tf_sheet_name_additional = None
    if not hasattr(args, 'target_num_centers_additional'):
        args.target_num_centers_additional = None
    if not hasattr(args, 'set_s_scale_additional'):
        args.set_s_scale_additional = []

    rnd_seed = process_seed(args.seed)

    # Determine which eye/grid index to use when generating movies/frames.
    # Acceptable values: 'left'/'first'/0 => 0; 'right'/'second'/1 => 1. Defaults to 0.
    try:
        raw_eye = movie_eye
        if isinstance(raw_eye, str):
            raw_eye_l = raw_eye.lower()
            if raw_eye_l in ('left', 'first', '0'):
                movie_eye_index = 0
            elif raw_eye_l in ('right', 'second', '1'):
                movie_eye_index = 1
            else:
                movie_eye_index = int(raw_eye)
        else:
            movie_eye_index = int(raw_eye)
    except Exception:
        movie_eye_index = 0

    
    try:
        raw_input_ch = movie_input_channel
        if isinstance(raw_input_ch, str):
            raw_input_l = raw_input_ch.lower()
            if raw_input_l in ('first', 'left', '0'):
                movie_input_channel_index = 0
            elif raw_input_l in ('second', 'right', '1'):
                movie_input_channel_index = 1
            else:
                movie_input_channel_index = int(raw_input_ch)
        else:
            movie_input_channel_index = int(raw_input_ch)
    except Exception:
        movie_input_channel_index = 0

    logging.info(f"{file_name} processing...1 seed:{args.seed}, {rnd_seed}")
    
    # Simple parameter loading for RGCs
    # make sure the following usecol is provide to avoid full empty columns
    sf_param_table = pd.read_excel(rf_params_file, sheet_name=args.sf_sheet_name, usecols='C:L')
    tf_param_table = pd.read_excel(rf_params_file, sheet_name=args.tf_sheet_name, usecols='C:I')
    
    if args.use_lnk_model:
        lnk_param_table = pd.read_excel(rf_params_file, sheet_name=args.lnk_sheet_name, usecols='C:L')
    else:
        lnk_param_table = None
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
    logging.info(f"{file_name} processing...1.5")
    
    multi_opt_sf, multi_opt_sf_surround, tf, grid2value_mapping, map_func, rgc_locs, lnk_params = rgc_array.get_results()

    # Load LNK parameters if needed - they are now returned directly from get_results()
    if args.use_lnk_model:
        logging.info(f"Using LNK model with adaptation mode: {args.lnk_adapt_mode}")
        if lnk_params is not None:
            logging.info(f"Loaded LNK parameters for {multi_opt_sf.shape[2]} cells")
        else:
            logging.warning("No LNK parameters available, using LN model")

    if args.is_both_ON_OFF or args.is_two_grids:
        grid_centers = None
        is_plot_centerFR = False
        # Load additional parameter tables if specified
        if args.sf_sheet_name_additional and args.tf_sheet_name_additional:
            sf_param_table_additional = pd.read_excel(rf_params_file, sheet_name=args.sf_sheet_name_additional, usecols='C:L')
            tf_param_table_additional = pd.read_excel(rf_params_file, sheet_name=args.tf_sheet_name_additional, usecols='C:I')
        else:
            sf_param_table_additional = None
            tf_param_table_additional = None
        multi_opt_sf_off, multi_opt_sf_surround_off, tf_off, grid2value_mapping_off, map_func_off, rgc_locs_off, lnk_params_off = \
            rgc_array.get_additional_results(anti_alignment=args.anti_alignment, sf_id_list_additional=args.sf_id_list_additional,
                                             sf_param_table_override=sf_param_table_additional, tf_param_table_override=tf_param_table_additional,
                                             target_num_centers_override=args.target_num_centers_additional,
                                             set_s_scale_override=args.set_s_scale_additional)  
        if args.is_both_ON_OFF and args.is_binocular:
            num_input_channel = 4
        elif args.is_two_grids and args.is_binocular:
            num_input_channel = 2
        else:
            num_input_channel = 2
            
    else:
        if args.is_binocular:
            num_input_channel = 2
        else:
            num_input_channel = 1
        grid_centers = rgc_locs
        multi_opt_sf_off, multi_opt_sf_surround_off, tf_off, grid2value_mapping_off, map_func_off, rgc_locs_off, lnk_params_off = None, None, None, None, None, None, None
    # Ensure mat save folder exists and save RGC locations with a timestamp so we can track
    # how RGC grids are generated across runs for consistency checks.
    try:
        os.makedirs(mat_save_folder, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(mat_save_folder, f'{file_name}_rgc_locations_in_visualization_{timestamp}.mat')
        save_dict = {'rgc_locs': rgc_locs}
        if 'rgc_locs_off' in locals() and rgc_locs_off is not None:
            save_dict['rgc_locs_off'] = rgc_locs_off
        savemat(save_path, save_dict)
        logging.info(f"Saved RGC locations to {save_path}")
    except Exception as _e:
        logging.error(f"Failed to save RGC locations to mat file: {_e}")
    # Quick comparison: load the training-produced lightweight .npz by experiment_name
    try:
        ref_path = os.path.join(mat_save_folder, 'repeat_check', experiment_name, f'{experiment_name}_rgc_locations.npz')
        if not os.path.exists(ref_path):
            logging.info(f'No repeat-check file found for experiment {experiment_name}; skipping quick check.')
        else:
            # Load the exact keys written by the training script (np.savez_compressed with keys 'rgc_locs' and optional 'rgc_locs_off')
            with np.load(ref_path, allow_pickle=True) as ref:
                # The training script saves 'rgc_locs' and optionally 'rgc_locs_off'
                ref_locs = ref['rgc_locs']
                ref_locs_off = ref['rgc_locs_off'] if 'rgc_locs_off' in ref.files else None

            # Compare main ON grid
            main_ok = np.allclose(ref_locs, rgc_locs, rtol=1e-6, atol=1e-8, equal_nan=True)

            # Compare OFF grid only if the reference contains it. Follow training semantics:
            # - if ref has OFF but current run does not -> FAIL
            # - if neither has OFF -> OK
            # - if both have OFF -> compare with same tolerance
            off_ok = True
            if ref_locs_off is not None:
                if 'rgc_locs_off' not in locals() or rgc_locs_off is None:
                    off_ok = False
                else:
                    off_ok = np.allclose(ref_locs_off, rgc_locs_off, rtol=1e-6, atol=1e-8, equal_nan=True)

            if main_ok and off_ok:
                logging.info(f'Quick repeat-check PASSED against {ref_path}')
            else:
                logging.info(f'Quick repeat-check FAILED against {ref_path} (main_ok={main_ok}, off_ok={off_ok})')
    except Exception as e:
        logging.info(f'Error during quick repeat-check: {e}')

    if is_show_rgc_grid:
        plot_coordinate_and_save(rgc_locs, rgc_locs_off, plot_save_folder, file_name=f'{args.experiment_name}_rgc_grids_test.png')
 
    # raise ValueError(f"Temporal close exam processing...")
    logging.info(f"{file_name} processing...2")

    movie_generator = SynMovieGenerator(top_img_folder, bottom_img_folder,
        crop_size=args.crop_size, boundary_size=boundary_size, center_ratio=args.center_ratio, max_steps=args.max_steps,
        prob_stay_ob=args.prob_stay_ob, prob_mov_ob=args.prob_mov_ob, prob_stay_bg=args.prob_stay_bg, prob_mov_bg=args.prob_mov_bg, 
        num_ext=args.num_ext, initial_velocity=args.initial_velocity, momentum_decay_ob=args.momentum_decay_ob, 
        momentum_decay_bg=args.momentum_decay_bg, scale_factor=args.scale_factor, velocity_randomness_ob = args.velocity_randomness_ob, 
        velocity_randomness_bg=args.velocity_randomness_bg, angle_range_ob=args.angle_range_ob, angle_range_bg=args.angle_range_bg, 
        coord_mat_file=coord_mat_file, correction_direction=args.coord_adj_dir, is_reverse_xy=args.is_reverse_xy, 
        start_scaling=args.start_scaling, end_scaling=args.end_scaling, dynamic_scaling=args.dynamic_scaling, is_binocular=args.is_binocular,
        interocular_dist=args.interocular_dist, bottom_contrast=args.bottom_contrast, top_contrast=args.top_contrast, 
        mean_diff_offset=args.mean_diff_offset, fix_disparity=fix_disparity_degree
    )

    logging.info(f"{file_name} processing...3")
    xlim, ylim = args.xlim, args.ylim
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]
    
    grid_width = int(np.round(target_width*args.grid_size_fac))
    grid_height = int(np.round(target_height*args.grid_size_fac))
    
    model = CNN_LSTM_ObjectLocation(cnn_feature_dim=args.cnn_feature_dim, lstm_hidden_size=args.lstm_hidden_size,
                                     lstm_num_layers=args.lstm_num_layers, output_dim=args.output_dim,
                                    input_height=grid_width, input_width=grid_height, conv_out_channels=args.conv_out_channels,
                                    is_input_norm=args.is_input_norm, is_seq_reshape=args.is_seq_reshape, CNNextractor_version=args.cnn_extractor_version,
                                    num_input_channel=num_input_channel, bg_info_cost_ratio=args.bg_info_cost_ratio, bg_processing_type=args.bg_processing_type,
                                    is_channel_normalization=args.is_channel_normalization)
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer, _ = checkpoint_loader.load_checkpoint(model, optimizer)
    criterion = nn.MSELoss()
    model.to(device)
    model.eval()

    logging.info(f"{file_name} processing...4")

    # [Section 1] Create test dataset and loader
    test_dataset = Cricket2RGCs(num_samples=num_sample, multi_opt_sf=multi_opt_sf, tf=tf, map_func=map_func,
                                grid2value_mapping=grid2value_mapping, multi_opt_sf_off=multi_opt_sf_off, tf_off=tf_off, 
                                map_func_off=map_func_off, grid2value_mapping_off=grid2value_mapping_off, target_width=target_width, 
                                target_height=target_height, movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                is_norm_coords=args.is_norm_coords, is_syn_mov_shown=False, fr2spikes=args.fr2spikes, 
                                is_both_ON_OFF=args.is_both_ON_OFF, quantize_scale=args.quantize_scale, 
                                add_noise=is_add_noise, rgc_noise_std=noise_level, rgc_noise_std_max=args.rgc_noise_std_max, smooth_data=args.smooth_data, 
                                is_rectified=args.is_rectified, is_direct_image=args.is_direct_image, is_reversed_OFF_sign=args.is_reversed_OFF_sign,
                                rectified_thr_ON=args.rectified_thr_ON, rectified_thr_OFF=args.rectified_thr_OFF,
                                rectified_mode=args.rectified_mode, rectified_softness=args.rectified_softness,
                                rectified_softness_OFF=args.rectified_softness_OFF,
                                is_two_grids=args.is_two_grids,
                                # LNK parameters
                                use_lnk=args.use_lnk_model,
                                lnk_params=lnk_params,
                                lnk_params_off=lnk_params_off,
                                surround_sigma_ratio=args.surround_sigma_ratio,
                                surround_sf=multi_opt_sf_surround,
                                surround_sf_off=multi_opt_sf_surround_off,
                                # Random seed for consistent data generation
                                rnd_seed=rnd_seed)

    # Ensure num_workers is a non-negative integer (DataLoader accepts 0 for inline workers)
    dl_num_workers = getattr(args, 'num_worker', None)
    if dl_num_workers is None:
        dl_num_workers = 0
    else:
        try:
            dl_num_workers = max(0, int(dl_num_workers))
        except Exception:
            dl_num_workers = 0

    dl_generator = torch.Generator().manual_seed(rnd_seed)         # for larger-batch test loader 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=dl_num_workers, pin_memory=True, persistent_workers=False, 
                             worker_init_fn=worker_init_fn, generator=dl_generator)

    logging.info(f"{file_name} processing...5")
    test_losses = [] 

    for batch_idx, (inputs, true_path, bg_info) in enumerate(test_loader):
        inputs, true_path, bg_info = inputs.to(device), true_path.to(device), bg_info.to(device)
        with torch.no_grad():
            predicted_path, _ = model(inputs)
            loss = criterion(predicted_path, true_path)
        test_losses.append(loss.item())

    logging.info(f"{file_name} processing...6")

    test_losses = np.array(test_losses)
    training_losses = np.array(training_losses)
    save_path = os.path.join(mat_save_folder, f'{file_name}_{epoch_number}_prediction_error.mat')
    savemat(save_path, {'test_losses': test_losses, 'training_losses': training_losses})
    
    # [Section 2] Create a small test dataset and loader for visualization and movie generation
    reuse_section2_samples = bool(visual_sample_ids and len(visual_sample_ids) > 0)

    if reuse_section2_samples:
        analysis_dataset = Cricket2RGCs(num_samples=int(num_sample*0.1), multi_opt_sf=multi_opt_sf, tf=tf, map_func=map_func,
                                    grid2value_mapping=grid2value_mapping, multi_opt_sf_off=multi_opt_sf_off, tf_off=tf_off, 
                                    map_func_off=map_func_off, grid2value_mapping_off=grid2value_mapping_off, target_width=target_width, 
                                    target_height=target_height, movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                    is_norm_coords=args.is_norm_coords, is_syn_mov_shown=True, fr2spikes=args.fr2spikes, 
                                    is_both_ON_OFF=args.is_both_ON_OFF, quantize_scale=args.quantize_scale, 
                                    add_noise=is_add_noise, rgc_noise_std=noise_level, rgc_noise_std_max=args.rgc_noise_std_max, smooth_data=args.smooth_data, 
                                    is_rectified=args.is_rectified, is_direct_image=args.is_direct_image, grid_coords=grid_centers,
                                    is_reversed_OFF_sign=args.is_reversed_OFF_sign, rectified_thr_ON=args.rectified_thr_ON, 
                                    rectified_thr_OFF=args.rectified_thr_OFF,
                                    rectified_mode=args.rectified_mode, rectified_softness=args.rectified_softness,
                                    rectified_softness_OFF=args.rectified_softness_OFF,
                                    is_two_grids=args.is_two_grids,
                                    # LNK parameters
                                    use_lnk=args.use_lnk_model,
                                    lnk_params=lnk_params,
                                    lnk_params_off=lnk_params_off,
                                    surround_sigma_ratio=args.surround_sigma_ratio,
                                    surround_sf=multi_opt_sf_surround,
                                    surround_sf_off=multi_opt_sf_surround_off,
                                    # Random seed for consistent data generation
                                    rnd_seed=rnd_seed)
        analysis_loader = DataLoader(analysis_dataset, batch_size=1, shuffle=False, 
                                     worker_init_fn=worker_init_fn)
    else:
        analysis_dataset = Cricket2RGCs(num_samples=int(num_sample*0.1), multi_opt_sf=multi_opt_sf, tf=tf, map_func=map_func,
                                    grid2value_mapping=grid2value_mapping, multi_opt_sf_off=multi_opt_sf_off, tf_off=tf_off, 
                                    map_func_off=map_func_off, grid2value_mapping_off=grid2value_mapping_off, target_width=target_width, 
                                    target_height=target_height, movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                    is_norm_coords=args.is_norm_coords, is_syn_mov_shown=True, fr2spikes=args.fr2spikes, 
                                    is_both_ON_OFF=args.is_both_ON_OFF, quantize_scale=args.quantize_scale, 
                                    add_noise=is_add_noise, rgc_noise_std=noise_level, rgc_noise_std_max=args.rgc_noise_std_max, smooth_data=args.smooth_data, 
                                    is_rectified=args.is_rectified, is_direct_image=args.is_direct_image, grid_coords=grid_centers,
                                    is_reversed_OFF_sign=args.is_reversed_OFF_sign, rectified_thr_ON=args.rectified_thr_ON, 
                                    rectified_thr_OFF=args.rectified_thr_OFF,
                                    rectified_mode=args.rectified_mode, rectified_softness=args.rectified_softness,
                                    rectified_softness_OFF=args.rectified_softness_OFF,
                                    is_two_grids=args.is_two_grids,
                                    # LNK parameters
                                    use_lnk=args.use_lnk_model,
                                    lnk_params=lnk_params,
                                    lnk_params_off=lnk_params_off,
                                    surround_sigma_ratio=args.surround_sigma_ratio,
                                    surround_sf=multi_opt_sf_surround,
                                    surround_sf_off=multi_opt_sf_surround_off,
                                    # Random seed for consistent data generation
                                    rnd_seed=rnd_seed)
        dl_generator_small = torch.Generator().manual_seed(rnd_seed + 1)
        analysis_loader = DataLoader(analysis_dataset, batch_size=1, shuffle=True, 
                                     worker_init_fn=worker_init_fn, generator=dl_generator_small)

    logging.info(f"{file_name} processing...7")
    test_losses = [] 
    all_paths = []
    all_paths_pred = []
    all_paths_bg = []
    all_path_cm = []
    all_scaling_factors = []
    all_bg_file = []
    all_id_numbers = []
    section2_samples = []

    for batch_idx, (inputs, true_path, path_bg, syn_movie, scaling_factors, bg_image_name, image_id, weighted_coords) in enumerate(analysis_loader):
        # temporalily check
        if is_save_movie_sequence_to_mat:
            sequence = inputs.cpu().numpy()
            savemat(os.path.join(mat_save_folder, 'sequence_evaluation.mat'), {'sequence': sequence})
            raise ValueError(f"check mat data range...")
        true_path_np = true_path.squeeze(0).cpu().numpy()
        path_bg_np = path_bg.squeeze(0).cpu().numpy()
        if isinstance(scaling_factors, torch.Tensor):
            scaling_factors_np = scaling_factors.squeeze(0).cpu().numpy()
        else:
            scaling_factors_np = np.array(scaling_factors).squeeze()
        if isinstance(weighted_coords, torch.Tensor):
            weighted_coords_np = weighted_coords.squeeze(0).cpu().numpy()
        else:
            weighted_coords_np = np.array(weighted_coords)

        path = true_path_np.reshape(1, -1)  # Ensure row vector
        path_bg = path_bg_np.reshape(1, -1)  # Ensure row vector
        path_cm = weighted_coords_np.reshape(1, -1)
        scaling_factors_flat = scaling_factors_np.reshape(1, -1)  # Ensure row vector

        if isinstance(bg_image_name, (list, tuple)):
            bg_image_name_value = bg_image_name[0]
        elif isinstance(bg_image_name, np.ndarray):
            bg_image_name_value = bg_image_name.reshape(-1)[0]
        else:
            bg_image_name_value = bg_image_name
        bg_image_name_value = str(bg_image_name_value)

        if isinstance(image_id, torch.Tensor):
            image_id_value = int(image_id.reshape(-1)[0].item())
        else:
            image_id_value = int(np.array(image_id).reshape(-1)[0])

        all_paths.append(path)
        all_paths_bg.append(path_bg)
        all_path_cm.append(path_cm)
        all_scaling_factors.append(scaling_factors_flat)
        all_bg_file.append(bg_image_name_value)
        all_id_numbers.append(image_id_value)

        inputs_device = inputs.to(device)
        true_path_device = true_path.to(device).float()
        with torch.no_grad():
            predicted_path, _ = model(inputs_device)
            loss = criterion(predicted_path, true_path_device)
        test_losses.append(loss.item())
        predicted_path_np = predicted_path.cpu().numpy()
        all_paths_pred.append(predicted_path_np)

        if reuse_section2_samples:
            section2_samples.append({
                'analysis_index': batch_idx,
                'dataset_index': batch_idx,
                'bg_image_name': bg_image_name_value,
                'image_id': image_id_value,
                'loss': loss.item()
            })

    logging.info(f"{file_name} processing...8")
    
    # Concatenate along rows
    all_paths = np.vstack(all_paths)
    all_paths_bg = np.vstack(all_paths_bg)
    all_path_cm = np.vstack(all_path_cm)
    all_scaling_factors = np.vstack(all_scaling_factors)
    all_bg_file = np.array(all_bg_file, dtype=object)  # Keep as string array
    all_id_numbers = np.array(all_id_numbers, dtype=int)  # Convert to int array
    all_paths_pred = np.array(all_paths_pred)

    test_losses = np.array(test_losses)
    training_losses = np.array(training_losses)
    save_path = os.path.join(mat_save_folder, f'{file_name}_{epoch_number}_prediction_error_with_path.mat')
    mat_payload = {'test_losses': test_losses, 'training_losses': training_losses, 'all_paths': all_paths,
                   'all_paths_bg': all_paths_bg, 'all_scaling_factors': all_scaling_factors, 'all_bg_file': all_bg_file,
                   'all_id_numbers': all_id_numbers, 'all_paths_pred': all_paths_pred, 'all_path_cm': all_path_cm}
    if reuse_section2_samples:
        mat_payload['analysis_indices'] = np.arange(all_paths.shape[0])
    savemat(save_path, mat_payload)

    logging.info(f"{file_name} processing...9")

    # [Section 3] Visualization and movie generation
    model.to('cpu')

    selected_sample_records = []
    requested_indices = None
    if visual_sample_ids and len(visual_sample_ids) > 0:
        requested_indices = list(visual_sample_ids)
    else:
        logging.info(f"{file_name}: no visual_sample_ids provided; Section 3 will generate fresh samples and Section 2 remains unchanged")

    if requested_indices:
        if not section2_samples:
            logging.warning(f"{file_name}: requested visualization indices but Section 2 recorded no samples; generating new samples instead.")
        else:
            normalised_indices = []
            for raw_idx in requested_indices:
                idx = raw_idx
                if idx < 0:
                    idx = len(section2_samples) + idx
                if 0 <= idx < len(section2_samples):
                    normalised_indices.append(idx)
                else:
                    logging.warning(f"{file_name}: requested visualization index {raw_idx} is out of range (available 0-{len(section2_samples)-1}).")
            seen = set()
            ordered_unique = []
            for idx in normalised_indices:
                if idx not in seen:
                    ordered_unique.append(idx)
                    seen.add(idx)
            if analysis_dataset is None:
                logging.warning(f"{file_name}: analysis dataset unavailable; cannot reuse Section 2 samples.")
            else:
                for idx in ordered_unique:
                    record = section2_samples[idx]
                    selected_sample_records.append({
                        'analysis_index': idx,
                        'dataset_index': record['dataset_index'],
                        'loss': record['loss'],
                        'bg_image_name': record['bg_image_name'],
                        'image_id': record['image_id']
                    })
                if selected_sample_records:
                    logging.info(f"{file_name}: using Section 2 sample indices {ordered_unique} for visualization and movie generation")
                elif requested_indices:
                    logging.warning(f"{file_name}: none of the requested indices {requested_indices} were valid; falling back to fresh samples.")

    video_jobs = []

    if selected_sample_records:
        logging.info(f"{file_name} processing...10 (Section 3 using pre-selected samples)")
    else:
        logging.info(f"{file_name} processing...10")

    if selected_sample_records:
        def _ensure_xy(array_like):
            arr = np.asarray(array_like)
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)
            return arr

        for record in selected_sample_records:
            dataset_idx = record['dataset_index']
            sample = analysis_dataset[dataset_idx]
            inputs_single, true_path_single, bg_path_single, syn_movie_single, scaling_factors_single, bg_image_name_single, image_id_single, weighted_coords_single = sample

            
            logging.info(f"inputs_single dimension: {inputs_single.shape} ")
            logging.info(f"syn_movie_single dimension: {syn_movie_single.shape} ")

            if isinstance(inputs_single, torch.Tensor):
                inputs_tensor = inputs_single.unsqueeze(0).float()
                inputs_np = inputs_single.cpu().numpy()
            else:
                inputs_tensor = torch.tensor(inputs_single, dtype=torch.float32).unsqueeze(0)
                inputs_np = np.asarray(inputs_single)

            if inputs_np.ndim == 4:
                inputs_np = inputs_np[:, movie_input_channel_index, :, :]
            
            if isinstance(true_path_single, torch.Tensor):
                true_path_single = true_path_single.cpu().numpy()
            if isinstance(bg_path_single, torch.Tensor):
                bg_path_single = bg_path_single.cpu().numpy()
            if isinstance(syn_movie_single, torch.Tensor):
                syn_movie_np = syn_movie_single.squeeze().cpu().numpy()
            else:
                syn_movie_np = np.asarray(syn_movie_single).squeeze()
            # If the synthesized movie contains an eye/grid dimension, select the requested one.
            try:
                # find any axis that has size==2 (common binocular axis)
                axes_with_two = [i for i, s in enumerate(syn_movie_np.shape) if s == 2]
                if axes_with_two:
                    axis = axes_with_two[0]
                    syn_movie_np = np.take(syn_movie_np, indices=movie_eye_index, axis=axis)
            except Exception:
                # fallback: leave syn_movie_np unchanged
                pass
            if isinstance(scaling_factors_single, torch.Tensor):
                scaling_factors_np = scaling_factors_single.squeeze().cpu().numpy()
            else:
                scaling_factors_np = np.asarray(scaling_factors_single).squeeze()
            if isinstance(weighted_coords_single, torch.Tensor):
                weighted_coords_raw = weighted_coords_single.cpu().numpy()
            else:
                weighted_coords_raw = np.asarray(weighted_coords_single)

            with torch.no_grad():
                predicted_path_tensor, _ = model(inputs_tensor)

            true_path_arr = _ensure_xy(true_path_single)
            bg_path_arr = _ensure_xy(bg_path_single)
            predicted_path_arr = _ensure_xy(predicted_path_tensor.squeeze(0).cpu().numpy())

            if is_plot_centerFR:
                weighted_coords_use = _ensure_xy(weighted_coords_raw) if weighted_coords_raw.size > 1 else None
            else:
                weighted_coords_use = None

            syn_movie_np = np.squeeze(syn_movie_np)

            sequence_length = len(true_path_arr)

            video_id = record['analysis_index']

            if is_making_video:
                video_jobs.append({
                    'inputs': inputs_np,
                    'syn_movie': syn_movie_np,
                    'true_path': true_path_arr,
                    'bg_path': bg_path_arr,
                    'predicted_path': predicted_path_arr,
                    'scaling_factors': scaling_factors_np,
                    'video_id': video_id,
                    'weighted_coords': weighted_coords_use,
                    'frame_width': frame_width,
                    'frame_height': frame_height,
                    'fps': fps,
                    'video_save_folder': video_save_folder,
                    'file_name': file_name,
                    'epoch_number': epoch_number,
                    'grid_generate_method': args.grid_generate_method,
                    'save_frames': save_movie_frames,
                    'frame_save_root': frame_save_root_final,
                    'truth_marker_style': truth_marker_style,
                    'prediction_marker_style': pred_marker_style,
                    'center_marker_style': center_marker_style,
                    'enable_truth_marker': enable_truth_marker,
                    'enable_prediction_marker': enable_pred_marker,
                    'enable_center_marker': enable_center_marker,
                    'input_channel_index': movie_input_channel_index
                })

            x1, y1 = true_path_arr[:, 0], true_path_arr[:, 1]
            x2, y2 = predicted_path_arr[:, 0], predicted_path_arr[:, 1]
            x3, y3 = bg_path_arr[:, 0], bg_path_arr[:, 1]
            label_1 = 'Truth'
            label_2 = 'Prediction'
            label_3 = 'Background'

            plt.figure(figsize=(12, 12))

            plt.subplot(2, 2, 1)
            plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss per Epoch")
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(x1, y1, label=label_1, color="darkblue", linestyle="-", linewidth=2)
            plt.plot(x2, y2, label=label_2, color="maroon", linestyle="--", linewidth=2)

            plt.subplot(2, 2, 3)
            plt.plot(range(sequence_length), x1, label=label_1, color='darkblue')
            plt.plot(range(sequence_length), x2, label=label_2, color='maroon')
            plt.plot(range(sequence_length), x3, label=label_3, color='seagreen')
            plt.xlabel("Time step")
            plt.ylabel("X-coordinate")
            plt.title("X-Coordinate Trace over Time")
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(range(sequence_length), y1, label=label_1, color='darkblue')
            plt.plot(range(sequence_length), y2, label=label_2, color='maroon')
            plt.plot(range(sequence_length), y3, label=label_3, color='seagreen')
            plt.xlabel("Time step")
            plt.ylabel("Y-coordinate")
            plt.title("Y-Coordinate Trace over Time")
            plt.legend()

            save_path = os.path.join(test_save_folder, f'{file_name}_{epoch_number}_prediction_plot_sample_{video_id + 1}.png')
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    else:
        visual_dataset = Cricket2RGCs(num_samples=num_display, multi_opt_sf=multi_opt_sf, tf=tf, map_func=map_func,
                                    grid2value_mapping=grid2value_mapping, multi_opt_sf_off=multi_opt_sf_off, tf_off=tf_off, 
                                    map_func_off=map_func_off, grid2value_mapping_off=grid2value_mapping_off, target_width=target_width, 
                                    target_height=target_height, movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                    is_norm_coords=args.is_norm_coords, is_syn_mov_shown=True, fr2spikes=args.fr2spikes, 
                                    is_both_ON_OFF=args.is_both_ON_OFF, quantize_scale=args.quantize_scale, 
                                    add_noise=is_add_noise, rgc_noise_std=noise_level, rgc_noise_std_max=args.rgc_noise_std_max, smooth_data=args.smooth_data, 
                                    is_rectified=args.is_rectified, is_direct_image=args.is_direct_image, grid_coords=grid_centers,
                                    is_reversed_OFF_sign=args.is_reversed_OFF_sign, rectified_thr_ON=args.rectified_thr_ON, 
                                    rectified_thr_OFF=args.rectified_thr_OFF,
                                    rectified_mode=args.rectified_mode, rectified_softness=args.rectified_softness,
                                    rectified_softness_OFF=args.rectified_softness_OFF,
                                    is_two_grids=args.is_two_grids,
                                    # LNK parameters
                                    use_lnk=args.use_lnk_model,
                                    lnk_params=lnk_params,
                                    lnk_params_off=lnk_params_off,
                                    surround_sigma_ratio=args.surround_sigma_ratio,
                                    surround_sf=multi_opt_sf_surround,
                                    surround_sf_off=multi_opt_sf_surround_off,
                                    # Random seed for consistent data generation
                                    rnd_seed=rnd_seed)
        visual_loader = DataLoader(visual_dataset, batch_size=1, shuffle=False, worker_init_fn=worker_init_fn)

        for batch_idx, (inputs, true_path, bg_path, syn_movie, scaling_factors, bg_image_name, image_id, weighted_coords) in enumerate(visual_loader):
            true_path = true_path.squeeze(0).cpu().numpy()
            bg_path = bg_path.squeeze(0).cpu().numpy()
            if is_plot_centerFR:
                weighted_coords = weighted_coords.squeeze(0).cpu().numpy()
            else:
                weighted_coords = None

            with torch.no_grad():
                predicted_path, _ = model(inputs)
                predicted_path = predicted_path.squeeze().cpu().numpy()

            sequence_length = len(true_path)

            if is_making_video:
                syn_movie_np = syn_movie.squeeze().cpu().numpy()
                inputs_np = inputs.squeeze().cpu().numpy()
                scaling_factors_np = scaling_factors.squeeze().cpu().numpy()
                # If the synthesized movie contains an eye/grid dimension, select the requested one.
                try:
                    axes_with_two = [i for i, s in enumerate(syn_movie_np.shape) if s == 2]
                    if axes_with_two:
                        axis = axes_with_two[0]
                        syn_movie_np = np.take(syn_movie_np, indices=movie_eye_index, axis=axis)
                except Exception:
                    pass
                video_jobs.append({
                    'inputs': inputs_np,
                    'syn_movie': syn_movie_np,
                    'true_path': true_path,
                    'bg_path': bg_path,
                    'predicted_path': predicted_path,
                    'scaling_factors': scaling_factors_np,
                    'video_id': batch_idx,
                    'weighted_coords': weighted_coords,
                    'frame_width': frame_width,
                    'frame_height': frame_height,
                    'fps': fps,
                    'video_save_folder': video_save_folder,
                    'file_name': file_name,
                    'epoch_number': epoch_number,
                    'grid_generate_method': args.grid_generate_method,
                    'save_frames': save_movie_frames,
                    'frame_save_root': frame_save_root_final,
                    'truth_marker_style': truth_marker_style,
                    'prediction_marker_style': pred_marker_style,
                    'center_marker_style': center_marker_style,
                    'enable_truth_marker': enable_truth_marker,
                    'enable_prediction_marker': enable_pred_marker,
                    'enable_center_marker': enable_center_marker,
                    'input_channel_index': movie_input_channel_index
                })

            x1, y1 = true_path[:, 0], true_path[:, 1]
            x2, y2 = predicted_path[:, 0], predicted_path[:, 1]
            x3, y3 = bg_path[:, 0], bg_path[:, 1]
            label_1 = 'Truth'
            label_2 = 'Prediction'
            label_3 = 'Background'

            plt.figure(figsize=(12, 12))

            plt.subplot(2, 2, 1)
            plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss per Epoch")
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(x1, y1, label=label_1, color="darkblue", linestyle="-", linewidth=2)
            plt.plot(x2, y2, label=label_2, color="maroon", linestyle="--", linewidth=2)

            plt.subplot(2, 2, 3)
            plt.plot(range(sequence_length), x1, label=label_1, color='darkblue')
            plt.plot(range(sequence_length), x2, label=label_2, color='maroon')
            plt.plot(range(sequence_length), x3, label=label_3, color='seagreen')
            plt.xlabel("Time step")
            plt.ylabel("X-coordinate")
            plt.title("X-Coordinate Trace over Time")
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(range(sequence_length), y1, label=label_1, color='darkblue')
            plt.plot(range(sequence_length), y2, label=label_2, color='maroon')
            plt.plot(range(sequence_length), y3, label=label_3, color='seagreen')
            plt.xlabel("Time step")
            plt.ylabel("Y-coordinate")
            plt.title("Y-Coordinate Trace over Time")
            plt.legend()

            save_path = os.path.join(test_save_folder, f'{file_name}_{epoch_number}_prediction_plot_sample_{batch_idx + 1}.png')
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

    # Generate movies in parallel after all data processing is complete
    if is_making_video and len(video_jobs) > 0:
        # Use up to 24 workers (respecting available CPUs) for parallel movie generation
        # Movie generation is CPU-intensive (video encoding), so we can use all available cores
        # Determine pool size: prefer explicit CLI/checkpoint setting if provided and > 0
        pool_worker_setting = getattr(args, 'num_worker', None)
        if pool_worker_setting is not None:
            try:
                pool_worker_setting = int(pool_worker_setting)
            except Exception:
                pool_worker_setting = None

        if pool_worker_setting and pool_worker_setting > 0:
            num_workers = min(pool_worker_setting, mp.cpu_count(), len(video_jobs))
        else:
            num_workers = min(24, mp.cpu_count(), len(video_jobs))
        logging.info(f"{file_name}: generating {len(video_jobs)} videos using {num_workers} parallel workers")
        
        try:
            # Use spawn method explicitly for better cross-platform compatibility
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(_generate_movie_job, video_jobs)
            
            # Log results
            for result in results:
                logging.info(result)
                
        except Exception as e:
            logging.error(f"Parallel movie generation failed: {e}")
            # Fallback to sequential processing
            logging.info("Falling back to sequential movie generation...")
            for job in video_jobs:
                result = _generate_movie_job(job)
                logging.info(result)

if __name__ == "__main__":
    main()
