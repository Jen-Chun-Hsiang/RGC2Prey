import argparse
import torch
import pandas as pd
from models.simpleSC import RGC2SCNet

from datasets.sim_cricket import RGCrfArray, SynMovieGenerator
from utils.utils import plot_tensor_and_save, plot_vector_and_save

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

    # Arguments for Cricket2RGCs
    parser.add_argument('--num_samples', type=int, default=20, help="Number of samples.")
    parser.add_argument('--num_frames', type=int, default=50, help="Number of frames in the dataset.")
    parser.add_argument('--crop_size', type=tuple, default=default_crop_size, help="Crop size as (width, height).")
    parser.add_argument('--boundary_size', type=tuple, default=default_boundary_size, help="Boundary size as (x_limit, y_limit).")
    parser.add_argument('--center_ratio', type=tuple, default=default_center_ratio, help="Center ratio for initial movement placement.")
    parser.add_argument('--max_steps', type=int, default=default_max_steps, help="Maximum steps for movement.")
    parser.add_argument('--num_ext', type=int, default=default_num_ext, help="Number of extended static frames.")
    parser.add_argument('--initial_velocity', type=float, default=default_initial_velocity, help="Initial velocity for movement.")
    parser.add_argument('--multi_opt_sf', type=float, default=1.0, help="Multiple optimization scale factor.")
    parser.add_argument('--tf', type=str, default='default_tf_params', help="Temporal filter parameters.")
    parser.add_argument('--map_func', type=str, default='default_map_func', help="Mapping function for grid values.")
    parser.add_argument('--grid2value_mapping', type=str, default='default_grid2value_mapping', help="Grid-to-value mapping.")
    parser.add_argument('--target_width', type=int, default=default_target_width, help="Target width for transformation.")
    parser.add_argument('--target_height', type=int, default=default_target_height, help="Target height for transformation.")

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
    parser.add_argument('--is_pixelized_rf', action='store_true', help="Flag for pixelized receptive field.")



    return parser.parse_args()

def main():
    is_show_rgc_rf_individual = True
    is_show_rgc_tf = True
    syn_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/syn_img/'
    plot_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Figures/'
    rf_params_file = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/SimulationParams.xlsx'

    args = parse_args()
    sf_param_table = pd.read_excel(rf_params_file, sheet_name='SF_params', usecols='A:L')
    tf_param_table = pd.read_excel(rf_params_file, sheet_name='TF_params', usecols='A:I')
    rgc_array = RGCrfArray(
        sf_param_table, tf_param_table, args.rgc_array_rf_size, args.xlim, args.ylim,
        args.target_num_centers, args.sf_scalar, args.grid_generate_method, args.tau,
        args.rand_seed, args.num_gauss_example, args.is_pixelized_rf
    )
    multi_opt_sf, tf, grid2value_mapping, map_func = rgc_array.get_results

    # Check results of RGC array synthesis
    if is_show_rgc_rf_individual:
        for i in range(multi_opt_sf.shape[2]): 
            temp_sf = multi_opt_sf[:, :, i].copy()
            temp_sf = torch.from_numpy(temp_sf).float()
            plot_tensor_and_save(temp_sf, syn_save_folder, f'{args.experiment_name}_receptive_field_check_{i + 1}.png')

    if is_show_rgc_tf:
        plot_vector_and_save(tf, plot_save_folder, file_name=f'{args.experiment_name}_temporal_filter.png')

    
    # movie_generator = SynMovieGenerator(
    #     args.num_frames, args.crop_size, args.boundary_size, args.center_ratio, args.max_steps,
    #     args.num_ext, args.initial_velocity, 'bottom_img_folder_placeholder', 'top_img_folder_placeholder'
    # )

if __name__ == '__main__':
    main()