import argparse
import torch
import os
import logging

from utils.data_handling import CheckpointLoader
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


if __name__ == "__main__":
    main()