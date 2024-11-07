
import torch
import numpy as np
from datasets.rgc_rf import create_hexagonal_centers, precompute_grid_centers, get_closest_indices, map_to_fixed_grid_closest
from datasets.rgc_rf import compute_distance_decay_matrix, map_to_fixed_grid_decay
from utils.utils import plot_position_and_save, plot_map_and_save


if __name__ == "__main__":
    plot_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Figures/'
    file_name = 'rgc_rf_position_plot.png'
    xlim = (-120, 120)
    ylim = (-90, 90)
    target_num_centers = 140
    tau = 10
    grid_generate_method = 'decay'  #'closest', 'decay'
    points = create_hexagonal_centers(xlim, ylim, target_num_centers=50, rand_seed=42)
    plot_position_and_save(points, plot_save_folder, file_name=file_name)

    number_samples = len(points)
    values = np.random.uniform(0, 1, size=(number_samples, 1))
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]

    grid_centers = precompute_grid_centers(target_height, target_width, x_min=xlim[0], x_max=xlim[1],
                                            y_min=ylim[0], y_max=ylim[1])
    if grid_generate_method is 'closest':
        closest_points = get_closest_indices(grid_centers, points)
        grid_values = map_to_fixed_grid_closest(values, closest_points, target_width, target_height)
    elif grid_generate_method is'decay':
        decay_matrix = compute_distance_decay_matrix(grid_centers, points, tau)
        grid_values = map_to_fixed_grid_decay(values, decay_matrix, target_width, target_height)

    file_name = f'rgc_rf_gridmap_{grid_generate_method}_plot.png'
    plot_map_and_save(grid_values, plot_save_folder, file_name=file_name)