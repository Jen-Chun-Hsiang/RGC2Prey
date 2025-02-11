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

from datasets.sim_cricket import RGCrfArray, SynMovieGenerator, Cricket2RGCs
from utils.utils import plot_tensor_and_save, plot_vector_and_save, plot_two_path_comparison
from models.rgc2behavior import CNN_LSTM_ObjectLocation
from utils.data_handling import save_checkpoint
from utils.tools import timer, MovieGenerator, save_distributions
from utils.utils import causal_moving_average
from utils.initialization import process_seed, initialize_logging, worker_init_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--config_name', type=str, default='not_yet_there', help='Config file name for data generation')
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')

    # Default values
    default_crop_size = (320, 240)
    default_boundary_size = '(220, 140)'
    default_center_ratio = (0.2, 0.2)
    default_rg_array_rf_size = (320, 240)
    default_xlim = (-120, 120)
    default_ylim = (-90, 90)
    default_cricket_size_range = (40, 100)
    default_initial_velocity = 6
    default_max_steps = 200
    default_num_ext = 50
    default_sf_scalar = 0.2
    default_tau = 3
    default_grid_generate_method = 'circle'
    default_sf_constraint_method = 'circle'
    default_target_width = 640
    default_target_height = 480

    # Arguments for SynMovieGenerator
    parser.add_argument('--crop_size', type=tuple, default=default_crop_size, help="Crop size as (width, height).")
    parser.add_argument('--boundary_size', type=str, default=default_boundary_size, help="Boundary size as '(x_limit, y_limit)'.")
    parser.add_argument('--center_ratio', type=tuple, default=default_center_ratio, help="Center ratio for initial movement placement.")
    parser.add_argument('--max_steps', type=int, default=default_max_steps, help="Maximum steps for movement.")
    parser.add_argument('--prob_stay_ob', type=float, default=0.95, help='Probability of step transition from stay to stay')
    parser.add_argument('--prob_mov_ob', type=float, default=0.975, help='Probability of step transition from moving to moving')
    parser.add_argument('--prob_stay_bg', type=float, default=0.95, help='Probability of step transition from stay to stay')
    parser.add_argument('--prob_mov_bg', type=float, default=0.975, help='Probability of step transition from moving to moving')
    parser.add_argument('--num_ext', type=int, default=default_num_ext, help="Number of extended static frames.")
    parser.add_argument('--initial_velocity', type=float, default=default_initial_velocity, help="Initial velocity for movement.")
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

    # Arguments for Cricket2RGCs (from movies to RGC array activities based on receptive field properties)
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples in the synthesized dataset")
    parser.add_argument('--is_norm_coords', action='store_true', help='normalize the coordinate as inputs')
    parser.add_argument('--fr2spikes', action='store_true', help='convert firing rate to spikes and keep positive (fr)')
    parser.add_argument('--quantize_scale', type=float, default=1.0, help="Firing rate to spike - quantization scaling.")
    parser.add_argument('--add_noise', action='store_true', help='Add noise to the RGC outputs')
    parser.add_argument('--rgc_noise_std', type=float, default=0.0, help="Level of noise added to the RGC outputs")
    parser.add_argument('--smooth_data', action='store_true', help='Smooth data of RGC outputs, especially quantized one')
    parser.add_argument('--is_rectified', action='store_true', help='Rectify the RGC outputs')
    parser.add_argument('--is_direct_image', action='store_true', help='By passing RGC convolution')
    
    # Arguments for RGCrfArray
    parser.add_argument('--rgc_array_rf_size', type=tuple, default=default_rg_array_rf_size, help="Receptive field size (height, width).")
    parser.add_argument('--xlim', type=tuple, default=default_xlim, help="x-axis limits for grid centers.")
    parser.add_argument('--ylim', type=tuple, default=default_ylim, help="y-axis limits for grid centers.")
    parser.add_argument('--target_num_centers', type=int, default=500, help="Number of target centers to generate.")
    parser.add_argument('--sf_scalar', type=float, default=default_sf_scalar, help="Scaling factor for spatial frequency.")
    parser.add_argument('--grid_generate_method', type=str, default=default_grid_generate_method, 
                        choices=['closest', 'decay', 'circle'], help="Method for grid generation.")
    parser.add_argument('--sf_constraint_method', type=str, default=default_sf_constraint_method, 
                        choices=['circle', 'threshold', 'None'], help="Method for grid generation.")
    parser.add_argument('--tau', type=float, default=default_tau, help="Decay factor for 'decay' method.")
    parser.add_argument('--sf_mask_radius', type=float, default=35, help='RGC dendritic receptive field radius size in pixel')
    parser.add_argument('--mask_radius', type=float, default=30, help='RGC axonal in SC radius size in pixel')
    parser.add_argument('--rgc_rand_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--num_gauss_example', type=int, default=1, help="Number of Gaussian examples.")
    parser.add_argument('--temporal_filter_len', type=int, default=50, help="Number of time points for a temporal filter")
    parser.add_argument('--is_pixelized_tf', action='store_true', help="Flag for pixelized receptive field.")
    parser.add_argument('--grid_size_fac', type=float, default=1, help='Resize the grid size that transformed from RGC outputs')
    parser.add_argument('--set_s_scale', type=float, nargs='*', default=[], help='Set scale fo surround weight of RF')
    parser.add_argument('--is_rf_median_subtract', action='store_true', help="Flag for substract median of rf")
    parser.add_argument('--is_both_ON_OFF', action='store_true', help="Flag for including OFF cell")
    parser.add_argument('--tf_sheet_name', type=str, default='TF_params', help='Excel sheet name for the temporal filter')

    # Arguments for CNN_LSTM 
    parser.add_argument('--cnn_feature_dim', type=int, default=256, help="Number of CNN feature dimensions.")
    parser.add_argument('--lstm_hidden_size', type=int, default=64, help="Number of LSTM hiddne size.")
    parser.add_argument('--lstm_num_layers', type=int, default=3, help="Number of LSTM hiddne size.")
    parser.add_argument('--output_dim', type=int, default=2, help="Number of output dimension.")
    parser.add_argument('--conv_out_channels', type=int, default=16, help="Number of output channel in convultion layers.")
    parser.add_argument('--is_seq_reshape', action='store_true', help="Use reshape with sequence to remove for loop")
    parser.add_argument('--is_input_norm', action='store_true', help="Normalize inputs to the CNN.")
    parser.add_argument('--cnn_extractor_version', type=int, default=1, help="Versioin of CNN extractor")

    # Model training parameters
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

    return parser.parse_args()

def main():
    is_show_rgc_rf_individual = False #True
    is_show_rgc_tf = False #True
    is_show_movie_frames = False 
    is_show_pathes = False #True
    is_show_grids = False #True

    args = parse_args()
    bottom_img_folder = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/{args.bg_folder}/'  #grass
    top_img_folder    = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/cricket/'
    syn_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/syn_img/'
    rf_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/RFs/RGCs/'
    distribution_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Distribution/RGC_outputs/'
    plot_save_folder =  '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Figures/TFs/'
    log_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Prints/'
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/CheckPoints/'
    rf_params_file = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/SimulationParams.xlsx'
    coord_mat_file = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/selected_points_summary_{args.coord_adj_type}.mat'   #selected_points_summary.mat
    video_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Videos/RFs/'

    initialize_logging(log_save_folder=log_save_folder, experiment_name=args.experiment_name)
    process_seed(args.seed)
    
    if args.is_GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    sf_param_table = pd.read_excel(rf_params_file, sheet_name='SF_params_modified', usecols='A:L')
    tf_param_table = pd.read_excel(rf_params_file, sheet_name=args.tf_sheet_name, usecols='A:I')
    rgc_array = RGCrfArray(
        sf_param_table, tf_param_table, rgc_array_rf_size=args.rgc_array_rf_size, xlim=args.xlim, ylim=args.ylim,
        target_num_centers=args.target_num_centers, sf_scalar=args.sf_scalar, grid_generate_method=args.grid_generate_method, 
        tau=args.tau, mask_radius=args.mask_radius, rgc_rand_seed=args.rgc_rand_seed, num_gauss_example=args.num_gauss_example, 
        sf_constraint_method=args.sf_constraint_method, temporal_filter_len=args.temporal_filter_len, grid_size_fac=args.grid_size_fac,
        sf_mask_radius=args.sf_mask_radius, is_pixelized_tf=args.is_pixelized_tf, set_s_scale=args.set_s_scale, 
        is_rf_median_subtract=args.is_rf_median_subtract
    )
    logging.info( f"{args.experiment_name} processing...1")
    multi_opt_sf, tf, grid2value_mapping, map_func, rgc_locs = rgc_array.get_results()

    logging.info( f"{args.experiment_name} processing...2")

    if args.is_both_ON_OFF:
        num_input_channel = 2
        # sf_param_table = pd.read_excel(rf_params_file, sheet_name='SF_params_OFF', usecols='A:L')
        rgc_array = RGCrfArray(
            sf_param_table, tf_param_table, rgc_array_rf_size=args.rgc_array_rf_size, xlim=args.xlim, ylim=args.ylim,
            target_num_centers=args.target_num_centers, sf_scalar=args.sf_scalar, grid_generate_method=args.grid_generate_method, 
            tau=args.tau, mask_radius=args.mask_radius, rgc_rand_seed=args.rgc_rand_seed+1, num_gauss_example=args.num_gauss_example, 
            sf_constraint_method=args.sf_constraint_method, temporal_filter_len=args.temporal_filter_len, grid_size_fac=args.grid_size_fac,
            sf_mask_radius=args.sf_mask_radius, is_pixelized_tf=args.is_pixelized_tf, set_s_scale=args.set_s_scale, 
            is_rf_median_subtract=args.is_rf_median_subtract
        )
        multi_opt_sf_off, tf_off, grid2value_mapping_off, map_func_off, rgc_locs = rgc_array.get_results()
    else:
        num_input_channel = 1
        multi_opt_sf_off, tf_off, grid2value_mapping_off, map_func_off, rgc_locs = None, None, None, None, None

    logging.info( f"{args.experiment_name} processing...3")
    # Check results of RGC array synthesis
    if is_show_rgc_rf_individual:
        for i in range(multi_opt_sf.shape[2]): 
            temp_sf = multi_opt_sf[:, :, i].copy()
            temp_sf = torch.from_numpy(temp_sf).float()
            plot_tensor_and_save(temp_sf, rf_save_folder, f'{args.experiment_name}_receptive_field_check_{i + 1}.png')
            if i == 3:
                break

        if args.is_both_ON_OFF:
            for i in range(multi_opt_sf_off.shape[2]): 
                temp_sf = multi_opt_sf_off[:, :, i].copy()
                temp_sf = torch.from_numpy(temp_sf).float()
                plot_tensor_and_save(temp_sf, rf_save_folder, f'{args.experiment_name}_receptive_field_check_OFF_{i + 1}.png')
                if i == 3:
                    break

    logging.info( f"{args.experiment_name} processing...4")
    if is_show_rgc_tf:
        plot_vector_and_save(tf, plot_save_folder, file_name=f'{args.experiment_name}_temporal_filter.png')

    movie_generator = SynMovieGenerator(top_img_folder, bottom_img_folder,
        crop_size=args.crop_size, boundary_size=args.boundary_size, center_ratio=args.center_ratio, max_steps=args.max_steps,
        prob_stay_ob=args.prob_stay_ob, prob_mov_ob=args.prob_mov_ob, prob_stay_bg=args.prob_stay_bg, prob_mov_bg=args.prob_mov_bg, 
        num_ext=args.num_ext, initial_velocity=args.initial_velocity, momentum_decay_ob=args.momentum_decay_ob, 
        momentum_decay_bg=args.momentum_decay_bg, scale_factor=args.scale_factor, velocity_randomness_ob = args.velocity_randomness_ob, 
        velocity_randomness_bg=args.velocity_randomness_bg, angle_range_ob=args.angle_range_ob, angle_range_bg=args.angle_range_bg, 
        coord_mat_file=coord_mat_file, correction_direction=args.coord_adj_dir, is_reverse_xy=args.is_reverse_xy, 
        start_scaling=args.start_scaling, end_scaling=args.end_scaling, dynamic_scaling=args.dynamic_scaling
    )
    logging.info( f"{args.experiment_name} processing...5")
    xlim, ylim = args.xlim, args.ylim
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]
    train_dataset = Cricket2RGCs(num_samples=args.num_samples, multi_opt_sf=multi_opt_sf, tf=tf, map_func=map_func,
                                grid2value_mapping=grid2value_mapping, multi_opt_sf_off=multi_opt_sf_off, tf_off=tf_off, 
                                map_func_off=map_func_off, grid2value_mapping_off=grid2value_mapping_off, target_width=target_width, 
                                target_height=target_height, movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                is_norm_coords=args.is_norm_coords, is_syn_mov_shown=True, fr2spikes=args.fr2spikes,
                                is_both_ON_OFF=args.is_both_ON_OFF, quantize_scale=args.quantize_scale, 
                                add_noise=args.add_noise, rgc_noise_std=args.rgc_noise_std, smooth_data=args.smooth_data,
                                is_rectified=args.is_rectified, is_direct_image=args.is_direct_image)
    
    
    logging.info( f"{args.experiment_name} processing...6")
    # Visualize one data points
    sequence, path, path_bg, syn_movie, scaling_factors, _, _, _ = train_dataset[0]
    # raise ValueError(f"Temporal close exam processing...")
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
    
    train_dataset = Cricket2RGCs(num_samples=args.num_samples, multi_opt_sf=multi_opt_sf, tf=tf, map_func=map_func,
                                grid2value_mapping=grid2value_mapping, multi_opt_sf_off=multi_opt_sf_off, tf_off=tf_off, 
                                map_func_off=map_func_off, grid2value_mapping_off=grid2value_mapping_off, target_width=target_width, 
                                target_height=target_height, movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                is_norm_coords=args.is_norm_coords, is_syn_mov_shown=False, fr2spikes=args.fr2spikes,
                                is_both_ON_OFF=args.is_both_ON_OFF, quantize_scale=args.quantize_scale,
                                add_noise=args.add_noise, rgc_noise_std=args.rgc_noise_std, smooth_data=args.smooth_data,
                                is_rectified=args.is_rectified, is_direct_image=args.is_direct_image)

    logging.info( f"{args.experiment_name} processing...8")
    if args.num_worker==0:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_worker, pin_memory=True, persistent_workers=False, worker_init_fn=worker_init_fn)
        

    logging.info( f"{args.experiment_name} processing...9")
    # Sample Training Loop
    grid_width = int(np.round(target_width*args.grid_size_fac))
    grid_height = int(np.round(target_height*args.grid_size_fac))
    model = CNN_LSTM_ObjectLocation(cnn_feature_dim=args.cnn_feature_dim, lstm_hidden_size=args.lstm_hidden_size,
                                     lstm_num_layers=args.lstm_num_layers, output_dim=args.output_dim,
                                    input_height=grid_width, input_width=grid_height, conv_out_channels=args.conv_out_channels,
                                    is_input_norm=args.is_input_norm, is_seq_reshape=args.is_seq_reshape, CNNextractor_version=args.cnn_extractor_version,
                                    num_input_channel=num_input_channel, bg_info_cost_ratio=args.bg_info_cost_ratio)
    
    logging.info( f"{args.experiment_name} processing...10")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    if args.schedule_method.lower() == 'rlrp':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.schedule_factor, patience=5)
    elif args.schedule_method.lower() == 'cawr':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=args.min_lr)

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
        training_losses = []  # To store the loss at each epoch
        num_epochs = args.num_epochs
        timer_data_loading = {'min': None, 'max': None, 'moving_avg': None, 'counter': 0}
        timer_data_transfer = {'min': None, 'max': None, 'moving_avg': None, 'counter': 0}
        timer_data_processing = {'min': None, 'max': None, 'moving_avg': None, 'counter': 0}
        timer_data_backpropagate = {'min': None, 'max': None, 'moving_avg': None, 'counter': 0}
        for epoch in range(num_epochs):
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
                        print(f"Batch {batch_idx}:")
                        print(f"Data Loading: {timer_data_loading}")
                        print(f"Data Transfer: {timer_data_transfer}")
                        print(f"Processing: {timer_data_processing}")
                        print(f"Backpropagation: {timer_data_backpropagate}")
                        break

            if args.exam_batch_idx is not None:
                break
            
            # Average loss for the epoch
            avg_train_loss = epoch_loss / len(train_loader)
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