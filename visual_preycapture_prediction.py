import argparse
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.sim_cricket import SynMovieGenerator, Cricket2RGCs, RGCrfArray
from models.rgc2behavior import CNN_LSTM_ObjectLocation
from utils.utils import plot_two_path_comparison
from utils.data_handling import CheckpointLoader


def main():
    experiment_name = 1124202401
    epoch_number = 4
    num_display = 2
    checkpoint_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/CheckPoints/'
    bottom_img_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/grass/'
    top_img_folder    = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/cricket/'
    rf_params_file = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/SimulationParams.xlsx'
    test_save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Figures/'

    file_name = f'{experiment_name}_cricket_location_prediction'
    checkpoint_filename = os.path.join(checkpoint_path, f'{file_name}_checkpoint_epoch_{epoch_number}.pth')
    
    # Load checkpoint
    checkpoint_loader = CheckpointLoader(checkpoint_filename)
    args = checkpoint_loader.load_args()

    #
    sf_param_table = pd.read_excel(rf_params_file, sheet_name='SF_params', usecols='A:L')
    tf_param_table = pd.read_excel(rf_params_file, sheet_name='TF_params', usecols='A:I')
    rgc_array = RGCrfArray(
        sf_param_table, tf_param_table, rgc_array_rf_size=args.rgc_array_rf_size, xlim=args.xlim, ylim=args.ylim,
        target_num_centers=args.target_num_centers, sf_scalar=args.sf_scalar, grid_generate_method=args.grid_generate_method, 
        tau=args.tau,rand_seed=args.rand_seed, num_gauss_example=args.num_gauss_example, is_pixelized_rf=args.is_pixelized_rf,
        temporal_filter_len=args.temporal_filter_len, grid_size_fac=args.grid_size_fac
    )
    multi_opt_sf, tf, grid2value_mapping, map_func = rgc_array.get_results()

    movie_generator = SynMovieGenerator(top_img_folder, bottom_img_folder,
        crop_size=args.crop_size, boundary_size=args.boundary_size, center_ratio=args.center_ratio, max_steps=args.max_steps,
        prob_stay=args.prob_stay, prob_mov=args.prob_mov, num_ext=args.num_ext, initial_velocity=args.initial_velocity, 
        momentum_decay_ob=args.momentum_decay_ob, momentum_decay_bg=args.momentum_decay_bg, scale_factor=args.scale_factor,
        velocity_randomness_ob = args.velocity_randomness_ob, velocity_randomness_bg=args.velocity_randomness_bg,
        angle_range_ob=args.angle_range_ob, angle_range_bg=args.angle_range_bg
    )

    xlim, ylim = args.xlim, args.ylim
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]
    train_dataset = Cricket2RGCs(num_samples=args.num_samples, multi_opt_sf=multi_opt_sf, tf=tf, map_func=map_func,
                                grid2value_mapping=grid2value_mapping, target_width=target_width, target_height=target_height,
                                movie_generator=movie_generator, grid_size_fac=args.grid_size_fac)
    test_loader = DataLoader(train_dataset, batch_size=num_display, shuffle=True)
    #
    grid_width = int(np.round(target_width*args.grid_size_fac))
    grid_height = int(np.round(target_height*args.grid_size_fac))
    model = CNN_LSTM_ObjectLocation(cnn_feature_dim=args.cnn_feature_dim, lstm_hidden_size=args.lstm_hidden_size,
                                     lstm_num_layers=args.lstm_num_layers, output_dim=args.output_dim,
                                    input_height=grid_width, input_width=grid_height, conv_out_channels=args.conv_out_channels)
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer, _ = checkpoint_loader.load_checkpoint(model, optimizer)

    model.to(args.device)
    model.eval()
    # Test model on samples
    for batch_idx, (inputs, true_path, _) in enumerate(test_loader):
        inputs = inputs.to(args.device)
        true_path = true_path.squeeze(0).cpu().numpy()

        with torch.no_grad():
            predicted_path = model(inputs).cpu().numpy()

        # Plot the true path and the predicted path
        plt.figure(figsize=(10, 6))
        plt.title(f"Sample {batch_idx + 1}: True Path vs Predicted Path")
        plt.plot(true_path[:, 0], true_path[:, 1], label='True Path', color='blue')
        plt.plot(predicted_path[:, 0], predicted_path[:, 1], label='Predicted Path', color='red')
        plt.legend()
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid()

        # Save the plot
        save_path = os.path.join(test_save_folder, f'prediction_plot_sample_{batch_idx + 1}.png')
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    main()