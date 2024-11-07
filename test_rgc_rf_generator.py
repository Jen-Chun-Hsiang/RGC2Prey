
import torch
import numpy as np
from datasets.rgc_rf import create_hexagonal_centers
from utils.utils import plot_position_and_save


if __name__ == "__main__":
    plot_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Figures/'
    file_name = 'rgc_rf_position_plot.png'
    xlim = (-120, 120)
    ylim = (-90, 90)
    target_num_centers = 140
    points = create_hexagonal_centers(xlim, ylim, target_num_centers=50, rand_seed=42)
    plot_position_and_save(points, plot_save_folder, file_name=file_name)