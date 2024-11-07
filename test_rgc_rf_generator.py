
import torch
import numpy as np
from datasets.rgc_rf import create_hexagonal_centers, precompute_grid_centers, get_closest_indices, map_to_fixed_grid_closest
from datasets.rgc_rf import compute_distance_decay_matrix, map_to_fixed_grid_decay
from utils.utils import plot_position_and_save, plot_map_and_save


if __name__ == "__main__":
    task_id = 1
    plot_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Figures/'
    file_name = 'rgc_rf_position_plot.png'
    xlim = (-120, 120)
    ylim = (-90, 90)
    target_num_centers = 140
    num_step = 20
    tau = 30
    grid_generate_method = 'decay'  #'closest', 'decay'
    points = create_hexagonal_centers(xlim, ylim, target_num_centers=50, rand_seed=42)
    
    number_samples = len(points)
    target_height = xlim[1]-xlim[0]
    target_width = ylim[1]-ylim[0]
    grid_centers = precompute_grid_centers(target_height, target_width, x_min=xlim[0], x_max=xlim[1],
                                            y_min=ylim[0], y_max=ylim[1])
    if grid_generate_method is 'closest':
        closest_points = get_closest_indices(grid_centers, points)
    elif grid_generate_method is'decay':
        decay_matrix = compute_distance_decay_matrix(grid_centers, points, tau)

    if grid_generate_method == 'closest':
        grid2value_mapping = get_closest_indices(grid_centers, points)
        map_func = map_to_fixed_grid_closest
    elif grid_generate_method == 'decay':
        grid2value_mapping = compute_distance_decay_matrix(grid_centers, points, tau)
        map_func = map_to_fixed_grid_decay
    else:
        raise ValueError("Invalid grid_generate_method. Use 'closest' or 'decay'.")

    if task_id == 1:
        values = np.random.uniform(0, 1, size=(number_samples, 1))
        plot_position_and_save(points, values=values, output_folder=plot_save_folder, file_name=file_name)

        '''
        if grid_generate_method is 'closest':
            closest_points = get_closest_indices(grid_centers, points)
            grid_values = map_to_fixed_grid_closest(values, closest_points, target_width, target_height)
        elif grid_generate_method is'decay':
            decay_matrix = compute_distance_decay_matrix(grid_centers, points, tau)
            grid_values = map_to_fixed_grid_decay(values, decay_matrix, target_width, target_height)
        '''
        # Call the selected mapping function
        grid_values = map_func(values, grid2value_mapping, target_width, target_height)

        file_name = f'rgc_rf_gridmap_{grid_generate_method}_plot.png'
        plot_map_and_save(grid_values, plot_save_folder, file_name=file_name)

    elif task_id == 2:
        print('Not yet')