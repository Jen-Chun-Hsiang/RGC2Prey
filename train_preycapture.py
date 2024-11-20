import argparse
import torch
import pandas as pd
import time
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
from datetime import datetime

from datasets.sim_cricket import RGCrfArray, SynMovieGenerator, Cricket2RGCs
from utils.utils import plot_tensor_and_save, plot_vector_and_save, plot_two_path_comparison
from models.rgc2behavior import CNN_LSTM_ObjectLocation

def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--config_name', type=str, default='not_yet_there', help='Config file name for data generation')
    parser.add_argument('--experiment_name', type=str, default='new_experiment', help='Experiment name')

    # Default values
    default_crop_size = (320, 240)
    default_boundary_size = (220, 140)
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
    default_grid_generate_method = 'decay'
    default_target_width = 640
    default_target_height = 480

    # Arguments for SynMovieGenerator
    parser.add_argument('--crop_size', type=tuple, default=default_crop_size, help="Crop size as (width, height).")
    parser.add_argument('--boundary_size', type=tuple, default=default_boundary_size, help="Boundary size as (x_limit, y_limit).")
    parser.add_argument('--center_ratio', type=tuple, default=default_center_ratio, help="Center ratio for initial movement placement.")
    parser.add_argument('--max_steps', type=int, default=default_max_steps, help="Maximum steps for movement.")
    parser.add_argument('--prob_stay', type=float, default=0.95, help='Probability of step transition from stay to stay')
    parser.add_argument('--prob_mov', type=float, default=0.975, help='Probability of step transition from moving to moving')
    parser.add_argument('--num_ext', type=int, default=default_num_ext, help="Number of extended static frames.")
    parser.add_argument('--initial_velocity', type=float, default=default_initial_velocity, help="Initial velocity for movement.")
    parser.add_argument('--momentum_decay_ob', type=float, default=0.95, help='Reduce speed in each run after moving for object')
    parser.add_argument('--momentum_decay_bg', type=float, default=0.9, help='Reduce speed in each run after moving for background')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Size of cricket image compare to its original size')
    parser.add_argument('--velocity_randomness_ob', type=float, default=0.02, help='Variation in speed change of each step')
    parser.add_argument('--velocity_randomness_bg', type=float, default=0.01, help='Variation in speed change of each step')
    parser.add_argument('--angle_range_ob', type=float, default=0.5, help='Variation in speed change of each step')
    parser.add_argument('--angle_range_bg', type=float, default=0.25, help='Variation in speed change of each step')

    # Arguments for Cricket2RGCs (from movies to RGC array activities based on receptive field properties)
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples in the synthesized dataset")

    # Arguments for RGCrfArray
    parser.add_argument('--rgc_array_rf_size', type=tuple, default=default_rg_array_rf_size, help="Receptive field size (height, width).")
    parser.add_argument('--xlim', type=tuple, default=default_xlim, help="x-axis limits for grid centers.")
    parser.add_argument('--ylim', type=tuple, default=default_ylim, help="y-axis limits for grid centers.")
    parser.add_argument('--target_num_centers', type=int, default=500, help="Number of target centers to generate.")
    parser.add_argument('--sf_scalar', type=float, default=default_sf_scalar, help="Scaling factor for spatial frequency.")
    parser.add_argument('--grid_generate_method', type=str, default=default_grid_generate_method, choices=['closest', 'decay'], help="Method for grid generation.")
    parser.add_argument('--tau', type=float, default=default_tau, help="Decay factor for 'decay' method.")
    parser.add_argument('--rand_seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--num_gauss_example', type=int, default=1, help="Number of Gaussian examples.")
    parser.add_argument('--temporal_filter_len', type=int, default=50, help="Number of time points for a temporal filter")
    parser.add_argument('--is_pixelized_rf', action='store_true', help="Flag for pixelized receptive field.")

    # Arguments for CNN_LSTM 
    parser.add_argument('--cnn_feature_dim', type=int, default=256, help="Number of CNN feature dimensions.")
    parser.add_argument('--lstm_hidden_size', type=int, default=64, help="Number of LSTM hiddne size.")
    parser.add_argument('--lstm_num_layers', type=int, default=3, help="Number of LSTM hiddne size.")
    parser.add_argument('--output_dim', type=int, default=3, help="Number of output dimension.")
    parser.add_argument('--conv1_out_channels', type=int, default=16, help="Number of output channel in convultion 1st layer.")
    parser.add_argument('--conv2_out_channels', type=int, default=32, help="Number of output channel in convultion 2nd layer.")
    parser.add_argument('--fc_out_features', type=int, default=32, help="Number of output features in convultion layers.")

    # Model training parameters
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for dataloader")
    parser.add_argument('--num_worker', type=int, default=0, help="Number of worker for dataloader")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of worker for dataloader")

    return parser.parse_args()

def main():
    is_show_rgc_rf_individual = False
    is_show_rgc_tf = False
    is_show_movie_frames = False
    is_show_pathes = False
    is_show_grids = False
    bottom_img_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/grass/'
    top_img_folder    = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/cricket/'
    syn_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/syn_img/'
    plot_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Figures/'
    log_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Logs/'
    rf_params_file = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/SimulationParams.xlsx'

    args = parse_args()
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Construct the full path for the log file
    file_name = f'{args.experiment_name}_cricket_location_prediction'
    log_filename = os.path.join(log_save_folder, f'{file_name}_training_log_{timestr}.txt')

    # Setup logging
    logging.basicConfig(filename=log_filename,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')

    
    sf_param_table = pd.read_excel(rf_params_file, sheet_name='SF_params', usecols='A:L')
    tf_param_table = pd.read_excel(rf_params_file, sheet_name='TF_params', usecols='A:I')
    rgc_array = RGCrfArray(
        sf_param_table, tf_param_table, rgc_array_rf_size=args.rgc_array_rf_size, xlim=args.xlim, ylim=args.ylim,
        target_num_centers=args.target_num_centers, sf_scalar=args.sf_scalar, grid_generate_method=args.grid_generate_method, 
        tau=args.tau,rand_seed=args.rand_seed, num_gauss_example=args.num_gauss_example, is_pixelized_rf=args.is_pixelized_rf,
        temporal_filter_len=args.temporal_filter_len
    )
    multi_opt_sf, tf, grid2value_mapping, map_func = rgc_array.get_results()

    # Check results of RGC array synthesis
    if is_show_rgc_rf_individual:
        for i in range(multi_opt_sf.shape[2]): 
            temp_sf = multi_opt_sf[:, :, i].copy()
            temp_sf = torch.from_numpy(temp_sf).float()
            plot_tensor_and_save(temp_sf, syn_save_folder, f'{args.experiment_name}_receptive_field_check_{i + 1}.png')

    if is_show_rgc_tf:
        plot_vector_and_save(tf, plot_save_folder, file_name=f'{args.experiment_name}_temporal_filter.png')

    
    movie_generator = SynMovieGenerator(top_img_folder, bottom_img_folder,
        crop_size=args.crop_size, boundary_size=args.boundary_size, center_ratio=args.center_ratio, max_steps=args.max_steps,
        prob_stay=args.prob_stay, prob_mov=args.prob_mov, num_ext=args.num_ext, initial_velocity=args.initial_velocity, 
        momentum_decay_ob=args.momentum_decay_ob, momentum_decay_bg=args.momentum_decay_bg, scale_factor=args.scale_factor,
        velocity_randomness_ob = args.velocity_randomness_ob, velocity_randomness_bg=args.velocity_randomness_bg,
        angle_range_ob=args.angle_range_ob, angle_range_bg=args.angle_range_bg
    )
    syn_movie, path, path_bg = movie_generator.generate()
    print(f'syn_movie shape: {syn_movie.shape}')
    if is_show_movie_frames:
        for i in range(syn_movie.shape[2]):
            Timg = syn_movie[:, :, i]
            plot_tensor_and_save(Timg, syn_save_folder, f'{args.experiment_name}_synthesized_movement_doublecheck_{i + 1}.png')
    
    if is_show_pathes:
        plot_two_path_comparison(path, path_bg, plot_save_folder, file_name=f'{args.experiment_name}_movement_pathes.png')

    xlim, ylim = args.xlim, args.ylim
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]
    train_dataset = Cricket2RGCs(num_samples=args.num_samples, multi_opt_sf=multi_opt_sf, tf=tf, map_func=map_func,
                                grid2value_mapping=grid2value_mapping, target_width=target_width, target_height=target_height,
                                movie_generator=movie_generator)
    
    if args.num_worker==0:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_worker, pin_memory=True, persistent_workers=False)
    
    start_time = time.perf_counter()
    data = next(iter(train_loader))
    elapsed_time = time.perf_counter() - start_time  # Calculate elapsed time
    print(f"[{elapsed_time:.2f}s] Main Process - Batch 0")
    sequence, path, path_bg = data
    print(f'sequence shape: {sequence.shape}')
    print(f'path shape: {path.shape}')
    if is_show_grids:
        sequence = sequence[0]
        for i in range(sequence.shape[2]):
            Timg = syn_movie[i, :, :]
            plot_tensor_and_save(Timg, syn_save_folder, f'{args.experiment_name}_RGCgrid_activity_doublecheck_{i + 1}.png')
        

    # Sample Training Loop
    model = CNN_LSTM_ObjectLocation(cnn_feature_dim=args.cnn_feature_dim, lstm_hidden_size=args.lstm_hidden_size,
                                     lstm_num_layers=args.lstm_num_layers, output_dim=args.output_dim,
                                    input_height=target_width, input_width=target_height, conv1_out_channels=args.conv1_out_channels, 
                                    conv2_out_channels=args.conv2_out_channels, fc_out_features=args.fc_out_features)
    # model = CNNFeatureExtractor(input_height=24, input_width=32, conv1_out_channels=16, conv2_out_channels=32, fc_out_features=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    epoch_losses = []  # To store the loss at each epoch
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        start_time = time.time()
        for sequences, targets, _ in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences)
            
            # Compute loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
    
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        elapsed_time = time.time()  - start_time
        logging.info( f"{file_name} Epoch [{epoch + 1}/{num_epochs}], Elapsed time: {elapsed_time:.2f} seconds \n"
                        f"\tLoss: {avg_epoch_loss:.4f} \n")

if __name__ == '__main__':
    main()