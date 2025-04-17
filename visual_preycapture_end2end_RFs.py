import argparse
import torch
import os
import logging
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
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

def total_variation_3d(x):
    diff_d = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    diff_h = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    diff_w = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return (diff_d.abs().mean() +
            diff_h.abs().mean() +
            diff_w.abs().mean())

def jitter_image_3d(x, max_jitter=(1, 5, 5)):
    shift_d = random.randint(-max_jitter[0], max_jitter[0])
    shift_h = random.randint(-max_jitter[1], max_jitter[1])
    shift_w = random.randint(-max_jitter[2], max_jitter[2])
    return torch.roll(x,
                     shifts=(shift_d, shift_h, shift_w),
                     dims=(2, 3, 4))

def normalize_image_3d(x, eps=1e-8):
    return x / (x.norm() + eps)


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
    wt_param = rgc_net.conv_temporal.weight
    print("Has conv_temporal?", hasattr(rgc_net, "conv_temporal"))
    print("Parameter object:", wt_param)

    wt = wt_param.detach().cpu().numpy()
    print("Weight shape (out_ch, in_ch, D, kH, kW):", wt.shape)
    out_ch, in_ch, kH, kW, D = wt.shape

    assert wt.size != 0, "Weight tensor is empty!"
    assert D > 0, "Temporal dimension is zero!"
    assert kH == 1 and kW == 1, f"Unexpected spatial kernel dims: {(kH,kW)}"

    print(f'D: {D}')
    print(f'out_ch: {out_ch}')
    # 2. Prepare x‐axis (1…D) and a figure
    x = np.arange(1, D+1)
    plt.figure(figsize=(6,4))

    # 3. For each output‐channel, extract the depth‐vector at (in_ch=0, h=0, w=0)
    for oc in range(out_ch):
        v = wt[oc, 0, 0, 0, :]    # shape (D,)
        plt.plot(x, v, label=f"ch{oc}")
        print(f"Channel {oc:2d}: length={v.shape[0]}, "
          f"min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}, "
          f"all_zero={np.allclose(v, 0)}")

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

    # Activation
    target_layer = rgc_net.conv2   # or whichever sub‐layer in RGC_ANN
    num_kernels = target_layer.out_channels

    optimized_volumes = []

    for k in range(num_kernels):
        activations = {}
        hook_handle = target_layer.register_forward_hook(
            lambda module, inp, out: activations.setdefault("out", out)
        )

        B, C, H, W, D = 1, 3, 240, 180, 50
        input_raw = torch.randn(B, C, H, W, D, device=device)

        # Permute to (B, C, D, H, W) for 3D ops
        input_img = input_raw.permute(0, 1, 4, 2, 3).contiguous().requires_grad_(True)

        # 4) Multi‐scale settings (note the comma after the first tuple!)
        scales = [
            (5, 32, 24),    # very coarse
            (10, 60, 45),
            (25, 120, 90),
            (50, 240, 180)  # final target
        ]
        num_iterations = 100

        # 5) Initialize random input at coarsest scale
        # input_img = torch.randn(1, 3, *scales[0], device=device, requires_grad=True)

        for scale in scales:
            # Resize if needed
            if input_img.shape[2:] != scale:
                input_img = F.interpolate(
                    input_img, size=scale, mode='trilinear', align_corners=False
                ).requires_grad_(True)

            optimizer = optim.Adam([input_img], lr=0.05)
            print(f"Optimizing at scale (D,H,W) = {scale}")

            for it in range(1, num_iterations+1):
                optimizer.zero_grad()

                # a) Jitter
                img_j = jitter_image_3d(input_img)

                # b) Forward through RGC
                rgc_net(img_j)

                # c) Pull out the hooked activation
                act_map = activations["out"][0]   # shape (out_ch, H', W')
                C_out, H_out, W_out = act_map.shape

                # c) Choose the center cell
                mid_row = H_out // 2
                mid_col = W_out // 2
                
                loss_activation = -act_map[k, mid_row, mid_col]

                # d) Regularization
                loss_l2 = 1e-4 * input_img.norm()
                loss_tv = 1e-4 * total_variation_3d(input_img)
                loss = loss_activation + loss_l2 + loss_tv

                loss.backward()
                optimizer.step()

                # e) Normalize & clamp
                with torch.no_grad():
                    input_img.copy_(normalize_image_3d(input_img))
                    input_img.clamp_(-1.5, 1.5)

                if it % 25 == 0:
                    print(f"  iter {it}/{num_iterations}  loss {loss.item():.4f}")
        
        hook_handle.remove()
        optimized_volumes.append(input_img.detach().cpu().squeeze())  # (C_in, D, H, W)


    # 9) Visualize the middle frame of the final optimized volume
    fig, axes = plt.subplots(2, num_kernels, figsize=(4 * num_kernels, 8))
    for k, vol in enumerate(optimized_volumes):
        # vol shape: (C_in, D, H, W); assume C_in=1 for simplicity
        vol = vol[0]               # (D, H, W)
        mid_idx = vol.shape[0] // 2

        mid_frame = vol[mid_idx]   # (H, W)
        std_map   = vol.std(dim=0) # (H, W)

        axes[0, k].imshow(mid_frame, cmap='gray')
        axes[0, k].set_title(f"Kernel {k} Mid Frame")
        axes[0, k].axis('off')

        axes[1, k].imshow(std_map, cmap='gray')
        axes[1, k].set_title(f"Kernel {k} STD")
        axes[1, k].axis('off')

    plt.tight_layout()

    save_path = os.path.join(test_save_folder, f'{file_name}_{epoch_number}_optimized_stimuli4conv2.png')
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()