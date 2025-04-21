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

def jitter_image_3d(x, max_jitter=None):
    """
    Jitter by up to max_jitter along each axis.
    If max_jitter is None, we compute it as:
      d: ±1 frame, 
      h: ±max(1, H//20) pixels, 
      w: ±max(1, W//20) pixels
    """
    _, _, D, H, W = x.shape
    if max_jitter is None:
        jd = 1
        jh = max(1, H // 20)
        jw = max(1, W // 20)
    else:
        jd, jh, jw = max_jitter

    sd = random.randint(-jd, jd)
    sh = random.randint(-jh, jh)
    sw = random.randint(-jw, jw)
    return torch.roll(x, shifts=(sd, sh, sw), dims=(2, 3, 4))

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

    if not hasattr(args, 'min_lr'):
        args.min_lr = 1e-6

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
    # print("Has conv_temporal?", hasattr(rgc_net, "conv_temporal"))
    # print("Parameter object:", wt_param)

    wt = wt_param.detach().cpu().numpy()
    logging.info( f"Weight shape (out_ch, in_ch, kH, kW, D): {wt.shape}")
    out_ch, in_ch, kH, kW, D = wt.shape

    assert wt.size != 0, "Weight tensor is empty!"
    assert D > 0, "Temporal dimension is zero!"
    assert kH == 1 and kW == 1, f"Unexpected spatial kernel dims: {(kH,kW)}"

    logging.info( f"D: {D} out_ch: {out_ch} ")
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

    B, C_in, H_full, W_full, D = 1, 2, 120, 90, 50

    # 4) Multi‐scale settings (note the comma after the first tuple!)
    spatial_scales = [
        (3, 8, 6),
        (6, 16,  12),
        (12, 32,  24),    # small spatial
        (24, 60,  45), 
        (D, H_full, W_full), # full spatial (50, 120, 90)
    ]

    triangle_fracs = [
        (0.25, 0.50),  # top center
        (0.75, 0.25),  # bottom left
        (0.75, 0.75),  # bottom right
    ]

    iters_per_scale = 128

    for k in range(num_kernels):
        loc_vols = []
        for (frac_r, frac_c) in triangle_fracs:
            activations = {}

            def hook_fn(module, inp, out):
                activations["out"] = out
            
            hook = target_layer.register_forward_hook(hook_fn)
            # hook = target_layer.register_forward_hook(
            #     lambda m, inp, out, d=activations: d.setdefault("out", out)
            # )

            d0, h0, w0 = spatial_scales[0]
            small_img = torch.randn(B, C_in, d0, h0, w0, device=device)
            small_img = small_img.detach().requires_grad_(True)

            # 5) Initialize random input at coarsest scale
            # input_img = torch.randn(1, 3, *scales[0], device=device, requires_grad=True)

            for (d_s, h_s, w_s) in spatial_scales:
                # a) Upsample small_img to its own target (D,h_s,w_s)
                if small_img.shape[2:] != (d_s, h_s, w_s):
                    small_img = F.interpolate(
                        small_img,
                        size=(d_s, h_s, w_s),
                        mode='trilinear',
                        align_corners=False
                    ).detach().requires_grad_(True)

                optimizer = optim.Adam([small_img], lr=0.05)
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=args.min_lr)


                print(f'small_img shape: {small_img.shape}')

                for i in range(1, iters_per_scale+1):
                    optimizer.zero_grad()

                    # b) Upsample to the fixed model size
                    up = F.interpolate(
                        small_img,
                        size=(D, H_full, W_full),
                        mode='trilinear',
                        align_corners=False
                    )

                    # c) Jitter (now automatically scales to H_full,W_full)
                    up_j = jitter_image_3d(up)
                    up_j = up_j.permute(0, 1, 3, 4, 2) 

                    if i == 1:
                        print(f'up_j shape: {up_j.shape}')
                    # d) Forward and grab activation map
                    rgc_net(up_j)
                    assert activations["out"].shape[1] == 2, \
                        f"Expected 2 channels, got {activations['out'].shape[1]}"
                    act_map = activations["out"][0]       # (C_out, H_out, W_out)
                    _, H_out, W_out = act_map.shape

                    # compute the *one* target location for this run
                    r_loc = int(frac_r * H_out)
                    c_loc = int(frac_c * W_out)

                    # mid_r, mid_c = H_out//2, W_out//2

                    # e) Loss = –center activation of kernel k
                    loss_act = -act_map[k, r_loc, c_loc]  
                    # + regularization on *small_img*
                    loss_norm = 1e-4 * small_img.norm() 
                    loss_var = 1e-4 * total_variation_3d(small_img)
                    loss = loss_act + loss_norm + loss_var

                    loss.backward()
                    optimizer.step()

                    scheduler.step(i + (i / (iters_per_scale+1)))

                    # f) Normalize & clamp small_img
                    with torch.no_grad():
                        small_img.copy_(normalize_image_3d(small_img))
                        small_img.clamp_(-1.5, 1.5)

                    if i % 25 == 0:
                        logging.info(f" scale {(h_s,w_s)} iter {i}/{iters_per_scale} loss {loss.item():.4f}")
                        logging.info(f" loss_act:{loss_act.item():.4f} | loss_norm:{loss_norm.item():.4f} | loss_var:{loss_var.item():.4f}")

            hook.remove()
            loc_vols.append(small_img.detach().cpu().squeeze())

        optimized_volumes.append(loc_vols)  # (C_in, D, H, W)



    # 9) Visualize the middle frame of the final optimized volume
    for k, loc_vols in enumerate(optimized_volumes):
        fig, axes = plt.subplots(2, len(triangle_fracs),
                                figsize=(4 * len(triangle_fracs), 8))
        for j, vol in enumerate(loc_vols):
            vol = vol[0]                # (D, H, W)
            mid_idx   = vol.shape[0] // 2
            mid_frame = vol[mid_idx]    # (H, W)
            std_map   = vol.std(dim=0)  # (H, W)

            # top row: mid frame
            ax = axes[0, j]
            ax.imshow(mid_frame, cmap='gray')
            ax.set_title(f"Kernel {k} — Loc {j} (mid slice)")
            ax.axis('off')

            # bottom row: std map
            ax = axes[1, j]
            ax.imshow(std_map, cmap='gray')
            ax.set_title(f"Kernel {k} — Loc {j} (std)")
            ax.axis('off')

        plt.tight_layout()

        save_path = os.path.join(test_save_folder, f'{file_name}_{epoch_number}_k{k}_optimized_stimuli4conv2.png')
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()