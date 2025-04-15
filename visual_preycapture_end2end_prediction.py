import argparse
import torch
import os
import logging
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import savemat
from torch.utils.data import DataLoader

from models.rgc2behavior import RGC_CNN_LSTM_ObjectLocation
from datasets.sim_cricket import SynMovieGenerator, CricketMovie
from utils.data_handling import CheckpointLoader
from utils.tools import MovieGenerator
from utils.initialization import process_seed, initialize_logging, worker_init_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--experiment_names', type=str, nargs='+', required=True, help="List of experiment names")
    parser.add_argument('--noise_levels', type=float, nargs='+', default=None, help="List of noise levels as numbers")
    parser.add_argument('--test_bg_folder', type=str, default=None, help="Background folder for testing")
    parser.add_argument('--test_ob_folder', type=str, default=None, help="Object folder for testing")
    parser.add_argument('--boundary_size', type=str, default=None, help="Boundary size as '(x_limit, y_limit)'.")
    parser.add_argument('--epoch_number', type=int, default=200, help="Epoch number to check")

    return parser.parse_args()


def main():
    args = parse_args()

    # Iterate through experiment names and run them
    for experiment_name in args.experiment_names:
        noise_levels = args.noise_levels if args.noise_levels else [None]
        for noise_level in noise_levels:
            run_experiment(experiment_name, noise_level, args.test_bg_folder, args.test_ob_folder, args.boundary_size, args.epoch_number)


def run_experiment(experiment_name, noise_level=None, test_bg_folder=None, test_ob_folder=None, boundary_size=None, epoch_number=200):
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
    log_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Prints/'

    if test_ob_folder is None:
        test_ob_folder = 'cricket'
    elif test_ob_folder == 'white-spot':
        coord_mat_file = None

    if noise_level is not None:
        file_name = f'{experiment_name}_{test_ob_folder}_{test_bg_folder}_noise{noise_level}_cricket_location_prediction'
        is_add_noise = True
    else:
        file_name = f'{experiment_name}_{test_ob_folder}_{test_bg_folder}_cricket_location_prediction'
    initialize_logging(log_save_folder=log_save_folder, experiment_name=file_name)

    checkpoint_filename = os.path.join(checkpoint_path, f'{experiment_name}_checkpoint_epoch_{epoch_number}.pth')
    checkpoint_loader = CheckpointLoader(checkpoint_filename)
    args = checkpoint_loader.load_args()
    training_losses = checkpoint_loader.load_training_losses()

    if test_bg_folder is None:
        test_bg_folder = args.bg_folder
    top_img_folder    = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/{test_ob_folder}/'
    bottom_img_folder = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/{test_bg_folder}/'  #grass

    if test_ob_folder == 'cricket':
        coord_mat_file = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/selected_points_summary_{args.coord_adj_type}.mat'   #selected_points_summary.mat
    
    if boundary_size is None:
        boundary_size = args.boundary_size

    if args.is_GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    process_seed(args.seed)

    if args.is_binocular:
        num_input_channel = 2
    else:
        num_input_channel = 1

    movie_generator = SynMovieGenerator(top_img_folder, bottom_img_folder,
        crop_size=args.crop_size, boundary_size=args.boundary_size, center_ratio=args.center_ratio, max_steps=args.max_steps,
        prob_stay_ob=args.prob_stay_ob, prob_mov_ob=args.prob_mov_ob, prob_stay_bg=args.prob_stay_bg, prob_mov_bg=args.prob_mov_bg, 
        num_ext=args.num_ext, initial_velocity=args.initial_velocity, momentum_decay_ob=args.momentum_decay_ob, 
        momentum_decay_bg=args.momentum_decay_bg, scale_factor=args.scale_factor, velocity_randomness_ob = args.velocity_randomness_ob, 
        velocity_randomness_bg=args.velocity_randomness_bg, angle_range_ob=args.angle_range_ob, angle_range_bg=args.angle_range_bg, 
        coord_mat_file=coord_mat_file, correction_direction=args.coord_adj_dir, is_reverse_xy=args.is_reverse_xy, 
        start_scaling=args.start_scaling, end_scaling=args.end_scaling, dynamic_scaling=args.dynamic_scaling, is_binocular=args.is_binocular,
        interocular_dist=args.interocular_dist
    )

    xlim, ylim = args.xlim, args.ylim
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]
    grid_width = int(np.round(target_width*args.grid_size_fac))
    grid_height = int(np.round(target_height*args.grid_size_fac))

    model = RGC_CNN_LSTM_ObjectLocation(cnn_feature_dim=args.cnn_feature_dim, lstm_hidden_size=args.lstm_hidden_size,
                                     lstm_num_layers=args.lstm_num_layers, output_dim=args.output_dim,
                                    input_height=grid_width, input_width=grid_height, input_depth=args.RGC_time_length, 
                                    conv_out_channels=args.conv_out_channels, is_input_norm=args.is_input_norm, 
                                    is_seq_reshape=args.is_seq_reshape, CNNextractor_version=args.cnn_extractor_version,
                                    temporal_noise_level=args.temporal_noise_level, num_RGC=args.num_RGC,
                                    num_input_channel=num_input_channel, is_channel_normalization=args.is_channel_normalization)
    logging.info(f'RGC output size: {model.rgc_output_size} \n')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model, optimizer, _ = checkpoint_loader.load_checkpoint(model, optimizer)
    criterion = nn.MSELoss()
    model.to(device)
    model.eval()

    # (1) Prediction Test
    test_dataset = CricketMovie(num_samples=args.num_samples, target_width=target_width, target_height=target_height, 
                                 movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                is_norm_coords=args.is_norm_coords, is_syn_mov_shown=False)
    
    if args.num_worker==0:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    else:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_worker, pin_memory=True, persistent_workers=False, worker_init_fn=worker_init_fn)
    
    test_losses = [] 

    for batch_idx, (inputs, true_path, bg_info) in enumerate(test_loader):
        inputs, true_path, bg_info = inputs.to(device), true_path.to(device), bg_info.to(device)
        with torch.no_grad():
            predicted_path, _ = model(inputs, noise_level)
            loss = criterion(predicted_path, true_path)
        test_losses.append(loss.item())
    
    test_losses = np.array(test_losses)
    training_losses = np.array(training_losses)
    save_path = os.path.join(mat_save_folder, f'{file_name}_{epoch_number}_prediction_error.mat')
    savemat(save_path, {'test_losses': test_losses, 'training_losses': training_losses})
    
    # (2) Prediction Path Data (less datapoint but more details)
    test_dataset = CricketMovie(num_samples=int(num_sample*0.1), target_width=target_width, target_height=target_height, 
                                 movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                is_norm_coords=args.is_norm_coords, is_syn_mov_shown=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, worker_init_fn=worker_init_fn)
    
    test_losses = [] 
    all_paths = []
    all_paths_pred = []
    all_paths_bg = []
    all_path_cm = []
    all_scaling_factors = []
    all_bg_file = []
    all_id_numbers = []

    for batch_idx, (inputs, true_path, path_bg, _, scaling_factors, bg_image_name, image_id, weighted_coords) in enumerate(test_loader):
        # temporalily check
        if is_save_movie_sequence_to_mat:
            sequence = inputs.cpu().numpy()
            savemat(os.path.join(mat_save_folder, 'sequence_evaluation.mat'), {'sequence': sequence})
            raise ValueError(f"check mat data range...")
        path = true_path.reshape(1, -1)  # Ensure row vector
        path_bg = path_bg.reshape(1, -1)  # Ensure row vector
        path_cm = weighted_coords.reshape(1, -1)
        scaling_factors = scaling_factors.reshape(1, -1)  # Ensure row vector

        all_paths.append(path)
        all_paths_bg.append(path_bg)
        all_path_cm.append(path_cm)
        all_scaling_factors.append(scaling_factors)
        all_bg_file.append(bg_image_name)
        all_id_numbers.append(image_id)

        true_path = torch.tensor(true_path, dtype=torch.float32)  # convert to torch due to different processing
        inputs, true_path = inputs.to(device), true_path.to(device)
        with torch.no_grad():
            predicted_path, _ = model(inputs)
            loss = criterion(predicted_path, true_path)
        test_losses.append(loss.item())
        all_paths_pred.append(predicted_path.cpu().numpy())
    
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
    savemat(save_path, {'test_losses': test_losses, 'training_losses': training_losses, 'all_paths': all_paths,
                        'all_paths_bg': all_paths_bg, 'all_scaling_factors': all_scaling_factors, 'all_bg_file': all_bg_file,
                        'all_id_numbers': all_id_numbers, 'all_paths_pred': all_paths_pred, 'all_path_cm':all_path_cm})

    # (3) Draw movies
    model.to('cpu')
    test_dataset = CricketMovie(num_samples=num_display, target_width=target_width, target_height=target_height, 
                                 movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                is_norm_coords=args.is_norm_coords, is_syn_mov_shown=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, worker_init_fn=worker_init_fn)
    
    
    
    for batch_idx, (inputs, true_path, bg_path, syn_movie, scaling_factors, bg_image_name, image_id, weighted_coords) in enumerate(test_loader):
        # inputs = inputs.to(args.device)
        true_path = true_path.squeeze(0).cpu().numpy()
        bg_path = bg_path.squeeze(0).cpu().numpy()
        if is_plot_centerFR:
            weighted_coords = weighted_coords.squeeze(0).cpu().numpy()
        else:
            weighted_coords = None

        with torch.no_grad():
            predicted_path, _ = model(inputs)
            predicted_path = predicted_path.squeeze().cpu().numpy()

        # Extract x and y coordinates
        sequence_length = len(true_path)

        if is_making_video:
            syn_movie = syn_movie.squeeze().cpu().numpy()
            inputs = inputs.squeeze().cpu().numpy()
            scaling_factors = scaling_factors.squeeze().cpu().numpy()
            data_movie = MovieGenerator(frame_width, frame_height, fps, video_save_folder, bls_tag=f'{file_name}_{epoch_number}',
                                    grid_generate_method=args.grid_generate_method)
                                
            data_movie.generate_movie(inputs, syn_movie, true_path, bg_path, predicted_path, scaling_factors, video_id=batch_idx, 
                                      weighted_coords=weighted_coords)

        x1, y1 = true_path[:, 0], true_path[:, 1]
        x2, y2 = predicted_path[:, 0], predicted_path[:, 1]
        x3, y3 = bg_path[:, 0], bg_path[:, 1]
        label_1 = 'Truth'
        label_2 = 'Prediction'
        label_3 = 'Background'

        # Plot the loss over epochs
        plt.figure(figsize=(12, 12))

        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss per Epoch")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(x1, y1, label=label_1, color="darkblue", linestyle="-", linewidth=2)
        plt.plot(x2, y2, label=label_2, color="maroon", linestyle="--", linewidth=2)
        
        # X-coordinate over time
        plt.subplot(2, 2, 3)
        plt.plot(range(sequence_length), x1, label=label_1, color='darkblue')
        plt.plot(range(sequence_length), x2, label=label_2, color='maroon')
        plt.plot(range(sequence_length), x3, label=label_3, color='seagreen')
        plt.xlabel("Time step")
        plt.ylabel("X-coordinate")
        plt.title("X-Coordinate Trace over Time")
        plt.legend()

        # X-coordinate over time
        plt.subplot(2, 2, 4)
        plt.plot(range(sequence_length), y1, label=label_1, color='darkblue')
        plt.plot(range(sequence_length), y2, label=label_2, color='maroon')
        plt.plot(range(sequence_length), y3, label=label_3, color='seagreen')
        plt.xlabel("Time step")
        plt.ylabel("Y-coordinate")
        plt.title("Y-Coordinate Trace over Time")
        plt.legend()

        # Save the plot
        save_path = os.path.join(test_save_folder, f'{file_name}_{epoch_number}_prediction_plot_sample_{batch_idx + 1}.png')
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()