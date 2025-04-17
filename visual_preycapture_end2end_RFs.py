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
    parser.add_argument('--epoch_number', type=int, default=200, help="Epoch number to check")

    return parser.parse_args()


def main():
    args = parse_args()

    # Iterate through experiment names and run them
    for experiment_name in args.experiment_names:
        run_experiment(experiment_name, args.epoch_number)


def run_experiment(experiment_name, epoch_number=200):
    num_display = 3
    frame_width = 640
    frame_height = 480
    fps = 20
    num_sample = 1000
    is_making_video = True
    is_add_noise = False
    is_show_rgc_grid = True
    is_save_movie_sequence_to_mat = False
    grid_generate_method = 'rgc_cnn'
    checkpoint_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/CheckPoints/'
    
    rf_params_file = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/SimulationParams.xlsx'
    test_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Figures/'
    plot_save_folder =  '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Figures/TFs/'
    # coord_mat_file = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/selected_points_summary.mat'
    
    video_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Videos/'
    mat_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Mats/'
    log_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Prints/'

    
    file_name = f'{experiment_name}_model_RF'
    initialize_logging(log_save_folder=log_save_folder, experiment_name=file_name)

    checkpoint_filename = os.path.join(checkpoint_path, f'{experiment_name}_checkpoint_epoch_{epoch_number}.pth')
    checkpoint_loader = CheckpointLoader(checkpoint_filename)
    args = checkpoint_loader.load_args()

    if args.is_GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

    process_seed(args.seed)

    if args.is_binocular:
        num_input_channel = 2
    else:
        num_input_channel = 1

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

    rgc_net: RGC_ANN = model.rgc
    rgc_net.eval()   # turn off any dropout / batchnorm

    # 1. Grab the raw conv_temporal weights and move to NumPy:
    #    shape is (out_channels, in_channels, D, kH, kW)
    wt = rgc_net.conv_temporal.weight.detach().cpu().numpy()
    out_ch, in_ch, D, kH, kW = wt.shape

    # 2. Prepare x‐axis (1…D) and a figure
    x = np.arange(1, D+1)
    plt.figure(figsize=(6,4))

    # 3. For each output‐channel, extract the depth‐vector at (in_ch=0, h=0, w=0)
    for oc in range(out_ch):
        v = wt[oc, 0, :, 0, 0]    # shape (D,)
        plt.plot(x, v, label=f"ch{oc}")

    # 4. Label and show
    plt.xlabel("Depth index (D)")
    plt.ylabel("Weight value")
    plt.title("conv_temporal weights per output channel")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(test_save_folder, f'{file_name}_{epoch_number}_temporal_filter.png')
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()