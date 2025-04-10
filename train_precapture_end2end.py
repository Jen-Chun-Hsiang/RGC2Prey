import argparse
import torch

from utils.initialization import process_seed, initialize_logging, worker_init_fn
from datasets.sim_cricket import SynMovieGenerator, CricketMovie

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

    # Model training parameters
    parser.add_argument('--seed', type=str, default='fixed', help=( "Seed type: 'fixed' for deterministic behavior, "
                                                                  "'random' for a random seed, or a numeric value for a custom seed."))
    parser.add_argument('--is_GPU', action='store_true', help='Using GPUs for accelaration')
    return parser.parse_args()

def main():

    args = parse_args()
    bottom_img_folder = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/{args.bg_folder}/'  #grass
    top_img_folder    = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/cricket/'
    syn_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/syn_img/'
    log_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Prints/'
    coord_mat_file = f'/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/selected_points_summary_{args.coord_adj_type}.mat' 

    initialize_logging(log_save_folder=log_save_folder, experiment_name=args.experiment_name)
    process_seed(args.seed)

    if args.is_GPU:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'

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
    train_dataset = CricketMovie(num_samples=args.num_samples, target_width=target_width, target_height=target_height, 
                                 movie_generator=movie_generator, grid_size_fac=args.grid_size_fac, 
                                is_norm_coords=args.is_norm_coords, is_syn_mov_shown=True)

if __name__ == '__main__':
    main()