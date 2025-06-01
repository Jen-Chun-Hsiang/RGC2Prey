import argparse
import os
import time
import torch
import logging
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.initialization import process_seed, initialize_logging, worker_init_fn
from datasets.sim_cricket import SynMovieGenerator, CricketMovie
from models.rgc2behavior import RGC_CNN_LSTM_ObjectLocation
from utils.data_handling import CheckpointLoader
from utils.data_handling import save_checkpoint
from utils.tools import timer
from utils.utils import causal_moving_average

def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--config_name', type=str, default='not_yet_there', help='Config file name for data generation')
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')

        # Default values
    default_crop_size = (320, 240)
    default_boundary_size = '(220, 140)'
    default_center_ratio = (0.2, 0.2)
    default_max_steps = 200
    default_num_ext = 50
    default_initial_velocity = 6
    default_xlim = (-120, 120)
    default_ylim = (-90, 90)

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
    parser.add_argument('--is_binocular', action='store_true', help='if generate two channels separately by binocular inputs')
    parser.add_argument('--interocular_dist', type=float, default=1.0, help='interocular distance between two eyes (in cm)')
    parser.add_argument('--is_reversed_OFF_sign', action='store_true', help='reversed off temporal filter to be the same as ON')
    parser.add_argument('--bottom_contrast', type=float, default=1.0, help='Contrast level of bottom, 1.0 original contrast')
    parser.add_argument('--top_contrast', type=float, default=1.0, help='Contrast level of top, 1.0 original contrast')
    parser.add_argument('--mean_diff_offset', type=float, default=0.0, help='control bias to mean diff_offset (bottom minius top)')


    # Arguments for CricketMovie (from movies to RGC array activities based on receptive field properties)
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples in the synthesized dataset")
    parser.add_argument('--is_norm_coords', action='store_true', help='normalize the coordinate as inputs')
    parser.add_argument('--grid_size_fac', type=float, default=1, help='Resize the grid size that transformed from RGC outputs')

    parser.add_argument('--xlim', type=tuple, default=default_xlim, help="x-axis limits for grid centers.")
    parser.add_argument('--ylim', type=tuple, default=default_ylim, help="y-axis limits for grid centers.")

    # Arguments for CNN_LSTM 
    parser.add_argument('--cnn_feature_dim', type=int, default=256, help="Number of CNN feature dimensions.")
    parser.add_argument('--lstm_hidden_size', type=int, default=64, help="Number of LSTM hiddne size.")
    parser.add_argument('--lstm_num_layers', type=int, default=3, help="Number of LSTM hiddne size.")
    parser.add_argument('--output_dim', type=int, default=2, help="Number of output dimension.")
    parser.add_argument('--RGC_time_length', type=int, default=50, help='Length of RGC time window')
    parser.add_argument('--conv_out_channels', type=int, default=16, help="Number of output channel in convultion layers.")
    parser.add_argument('--is_seq_reshape', action='store_true', help="Use reshape with sequence to remove for loop")
    parser.add_argument('--is_input_norm', action='store_true', help="Normalize inputs to the CNN.")
    parser.add_argument('--cnn_extractor_version', type=int, default=1, help="Versioin of CNN extractor")
    parser.add_argument('--is_channel_normalization', action='store_true', help="Is perform channel normalization separately to the inputs")
    parser.add_argument('--temporal_noise_level', type=float, default=0.2, help='noise level for temporal process with movie, for smooth temporal filter')
    parser.add_argument('--num_RGC', type=int, default=2, help='Number of RGC in RGC_CNN module')

    # Model training parameters
    parser.add_argument('--seed', type=str, default='fixed', help=( "Seed type: 'fixed' for deterministic behavior, "
                                                                  "'random' for a random seed, or a numeric value for a custom seed."))
    parser.add_argument('--is_GPU', action='store_true', help='Using GPUs for accelaration')
    parser.add_argument('--load_checkpoint_epoch', type=int, default=None, help='Epoch number of a load checkpint')
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for dataloader")
    parser.add_argument('--num_worker', type=int, default=0, help="Number of worker for dataloader")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of worker for dataloader")
    parser.add_argument('--schedule_method', type=str, default='CAWR', help='Method used for scheduler')
    parser.add_argument('--schedule_factor', type=float, default=0.2, help='Scheduler reduction factor')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help="minimum learning rate for CAWR")
    parser.add_argument('--is_gradient_clip', action='store_true', help="Apply gradient clip to training process")
    parser.add_argument('--bg_info_cost_ratio', type=float, default=0.0, help="background information ratio of its objective cost, compared to object prediction")
    parser.add_argument('--exam_batch_idx', type=int, default=None, help='examine the timer and stop code in the middle')
    parser.add_argument('--num_epoch_save', type=int, default=5, help='Number of epoch to save a checkpoint')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of mini-batch to optimize (gradient accumulation)')
    parser.add_argument('--timer_tau', type=float, default=0.9, help='moving winder constant')
    parser.add_argument('--timer_sample_cicle', type=int, default=1, help='Sample circle for the timer')
    parser.add_argument('--add_noise', action='store_true', help='Add noise to the RGC outputs')
    parser.add_argument('--rgc_noise_std', type=float, default=0.0, help="Level of noise added to the RGC outputs")

    return parser.parse_args()

def main():

    args = parse_args()
    bottom_img_folder = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/{args.bg_folder}/'  #grass
    top_img_folder    = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/cricket/'
    syn_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/syn_img/'
    log_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Prints/'
    coord_mat_file = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/selected_points_summary_{args.coord_adj_type}.mat' 
    savemodel_dir = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/CheckPoints/'

    initialize_logging(log_save_folder=log_save_folder, experiment_name=args.experiment_name)
    process_seed(args.seed)

    if args.is_GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    if args.is_binocular:
        num_input_channel = 2
    else:
        num_input_channel = 1
    logging.info(f'num_input_channel: {num_input_channel}. 1:monocular, 2:binocular \n')

    movie_generator = SynMovieGenerator(top_img_folder, bottom_img_folder,
        crop_size=args.crop_size, boundary_size=args.boundary_size, center_ratio=args.center_ratio, max_steps=args.max_steps,
        prob_stay_ob=args.prob_stay_ob, prob_mov_ob=args.prob_mov_ob, prob_stay_bg=args.prob_stay_bg, prob_mov_bg=args.prob_mov_bg, 
        num_ext=args.num_ext, initial_velocity=args.initial_velocity, momentum_decay_ob=args.momentum_decay_ob, 
        momentum_decay_bg=args.momentum_decay_bg, scale_factor=args.scale_factor, velocity_randomness_ob = args.velocity_randomness_ob, 
        velocity_randomness_bg=args.velocity_randomness_bg, angle_range_ob=args.angle_range_ob, angle_range_bg=args.angle_range_bg, 
        coord_mat_file=coord_mat_file, correction_direction=args.coord_adj_dir, is_reverse_xy=args.is_reverse_xy, 
        start_scaling=args.start_scaling, end_scaling=args.end_scaling, dynamic_scaling=args.dynamic_scaling, is_binocular=args.is_binocular,
        interocular_dist=args.interocular_dist, bottom_contrast=args.bottom_contrast, top_contrast=args.top_contrast, 
        mean_diff_offset=args.mean_diff_offset
    )

    xlim, ylim = args.xlim, args.ylim
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]
    train_dataset = CricketMovie(num_samples=args.num_samples, target_width=target_width, target_height=target_height, 
                                 movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                is_norm_coords=args.is_norm_coords, is_syn_mov_shown=False)
    
    if args.num_worker==0:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_worker, pin_memory=True, persistent_workers=False, worker_init_fn=worker_init_fn)
        
    grid_width = int(np.round(target_width*args.grid_size_fac))
    grid_height = int(np.round(target_height*args.grid_size_fac))
    model = RGC_CNN_LSTM_ObjectLocation(cnn_feature_dim=args.cnn_feature_dim, lstm_hidden_size=args.lstm_hidden_size,
                                     lstm_num_layers=args.lstm_num_layers, output_dim=args.output_dim,
                                    input_height=grid_width, input_width=grid_height, input_depth=args.RGC_time_length, 
                                    conv_out_channels=args.conv_out_channels, is_input_norm=args.is_input_norm, 
                                    is_seq_reshape=args.is_seq_reshape, CNNextractor_version=args.cnn_extractor_version,
                                    temporal_noise_level=args.temporal_noise_level, num_RGC=args.num_RGC,
                                    num_input_channel=num_input_channel, is_channel_normalization=args.is_channel_normalization)
        
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

            # if args.bg_info_type == 'rloc':
            #     bg_info = causal_moving_average(bg_info, args.short_window_length) - \
            #             causal_moving_average(bg_info, args.long_window_length)

            # Forward pass
            with timer(timer_data_processing, tau=args.timer_tau, n=args.timer_sample_cicle):
                if args.add_noise:
                    outputs, bg_pred = model(sequences, args.rgc_noise_std)
                else:
                    outputs, bg_pred = model(sequences)
            
            # Compute loss
            with timer(timer_data_backpropagate, tau=args.timer_tau, n=args.timer_sample_cicle):
                loss = (1-args.bg_info_cost_ratio) * criterion(outputs, targets[:, -outputs.size(1):, :]) \
                        + args.bg_info_cost_ratio * criterion(bg_pred, bg_info[:, -bg_pred.size(1):, :])
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