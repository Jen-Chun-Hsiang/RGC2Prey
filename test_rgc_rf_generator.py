
import torch
import numpy as np
from datasets.rgc_rf import create_hexagonal_centers, precompute_grid_centers, get_closest_indices, map_to_fixed_grid_closest
from utils.utils import plot_position_and_save, plot_map_and_save


if __name__ == "__main__":
    plot_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Figures/'
    file_name = 'rgc_rf_position_plot.png'
    xlim = (-120, 120)
    ylim = (-90, 90)
    target_num_centers = 140
    points = create_hexagonal_centers(xlim, ylim, target_num_centers=50, rand_seed=42)
    plot_position_and_save(points, plot_save_folder, file_name=file_name)

    number_samples = len(points)
    values = np.random.uniform(0, 1, size=(number_samples, 1))
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]

    grid_centers = precompute_grid_centers(target_height, target_width, x_min=0, x_max=1, y_min=0, y_max=1)
    closest_points = get_closest_indices(grid_centers, points)
    grid_values = map_to_fixed_grid_closest(values, closest_points, target_width, target_height)
    file_name = 'rgc_rf_gridmap__plot.png'

    plot_map_and_save(grid_values, plot_save_folder, file_name=file_name)