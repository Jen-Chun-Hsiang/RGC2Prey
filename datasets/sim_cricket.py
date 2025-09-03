import random
import ast
from PIL import Image, ImageEnhance
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F
import logging

from datasets.rgc_rf import map_to_fixed_grid_decay_batch, gaussian_multi, gaussian_temporalfilter, get_closest_indices, compute_distance_decay_matrix
from datasets.rgc_rf import map_to_fixed_grid_closest_batch, precompute_grid_centers, compute_circular_mask_matrix
from datasets.rgc_rf import map_to_fixed_grid_circle_batch, HexagonalGridGenerator
from datasets.simple_lnk import sample_lnk_parameters
from utils.utils import get_random_file_path, get_image_number, load_mat_to_dataframe, get_filename_without_extension
from utils.tools import gaussian_smooth_1d
from utils.trajectory import disparity_from_scaling_factor, convert_deg_to_pix, adjust_trajectories, plot_trajectories
from datasets.simple_lnk import compute_lnk_response

def create_multiple_temporal_filters(base_tf, num_rgcs, variation_std=0.0):
    """
    Create multiple temporal filters for RGC cells.
    
    Args:
        base_tf: numpy array [T] - base temporal filter
        num_rgcs: int - number of RGC cells
        variation_std: float - standard deviation for filter variation (0 = identical filters)
    
    Returns:
        tf_multi: numpy array [num_rgcs, T] - temporal filters for each RGC
    """
    tf_length = len(base_tf)
    tf_multi = np.tile(base_tf, (num_rgcs, 1))
    
    if variation_std > 0:
        # Add Gaussian noise to create filter variations
        noise = np.random.normal(0, variation_std, (num_rgcs, tf_length))
        tf_multi += noise
        
        # Optionally normalize to maintain similar response magnitudes
        for i in range(num_rgcs):
            tf_multi[i] = tf_multi[i] / np.sum(np.abs(tf_multi[i])) * np.sum(np.abs(base_tf))
    
    return tf_multi


def jitter_position(position, jitter_range):
    """
    Apply jitter to a given position within a defined range.

    Parameters:
    - position: tuple (x, y) representing the original position.
    - jitter_range: tuple (jitter_x, jitter_y) specifying how much to jitter in x and y directions.

    Returns:
    - new_position: tuple (new_x, new_y) with the jittered position.
    """
    x, y = position
    jitter_x, jitter_y = jitter_range

    # Apply random jitter within the specified range
    new_x = x + random.randint(-jitter_x, jitter_x)
    new_y = y + random.randint(-jitter_y, jitter_y)

    return np.array([new_x, new_y])


def scale_image(image, scale_factor):
    """
    Scales the image by the given scale factor.

    Parameters:
    - image: PIL Image to be scaled.
    - scale_factor: float, scale factor to resize the image (e.g., 0.5 for half size, 2 for double size).

    Returns:
    - scaled_image: PIL Image that is scaled by the scale factor.
    """
    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    # Use Image.Resampling.LANCZOS instead of Image.ANTIALIAS
    scaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
    # scaled_image = image.resize(new_size, Image.ANTIALIAS)
    return scaled_image


def crop_image(image, crop_size, crop_pos):
    """
    Crops the image to the fixed size from the center.

    Parameters:
    - image: PIL Image to be cropped.
    - crop_size: tuple (width, height), size of the crop.

    Returns:
    - cropped_image: PIL Image cropped to the specified size.
    """
    width, height = image.size
    crop_width, crop_height = crop_size

    left = (width - crop_width) // 2 - crop_pos[0]
    top = (height - crop_height) // 2 - crop_pos[1]
    right = left + crop_width
    bottom = top + crop_height

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def calculate_scaling_factors(bottom_img_positions, start_scaling=1.0, end_scaling=2.0):
    """
    Calculate scaling factors based on shifts in bottom_img_positions.

    Parameters:
    - bottom_img_positions (np.ndarray): Array of shape (batch_size, 2) representing bottom image positions.

    Returns:
    - scaling_factors (list): List of scaling factors for each batch step.
    """
    batch_size = len(bottom_img_positions)

    # Calculate shifts between consecutive positions
    shifts = [
        np.linalg.norm(bottom_img_positions[i] - bottom_img_positions[i - 1])
        for i in range(1, batch_size)
    ]
    
    # Count steps where the position shifts
    total_shifting_steps = sum(1 for shift in shifts if shift > 0)

    # Scaling increment for each shifting step
    scaling_increment = (end_scaling - start_scaling) / max(1, total_shifting_steps)

    # Initialize scaling factors
    scaling_factors = [start_scaling]  # Start with 1.0
    current_scaling_factor = start_scaling

    for i in range(1, batch_size):
        if shifts[i - 1] > 0:  # Check if there is a position shift
            current_scaling_factor += scaling_increment
        scaling_factors.append(current_scaling_factor)

    return scaling_factors


def overlay_images_with_jitter_and_scaling(bottom_img_path, top_img_path, top_img_pos, bottom_img_pos,
                                           bottom_img_jitter_range, top_img_jitter_range,
                                           top_img_scale_range, crop_size, alpha=1.0):
    """
    Overlays a top image (with transparency) on top of a bottom image with independent jittering and scaling.

    Parameters:
    - bottom_img_path: str, path to the bottom image (background image).
    - top_img_path: str, path to the top image (transparent overlay).
    - top_img_pos: numpy array, (x, y) base position where the top image will be placed.
    - bottom_img_pos: numpy array, (x, y) base position where the bottom image will be placed.
    - bottom_img_jitter_range: numpy array, (jitter_x, jitter_y) range of jitter for bottom image.
    - top_img_jitter_range: numpy array, (jitter_x, jitter_y) range of jitter for top image.
    - top_img_scale_range: numpy array, (min_scale, max_scale) range to randomly scale the top image.
    - crop_size: numpy array, (width, height) of the final fixed-size crop of the resulting image.
    - alpha: float, transparency level of the top image (0 to 1, where 1 is fully opaque).

    Returns:
    - final_img: PIL Image that is cropped and contains the overlay of the top image with jittering and scaling.
    """

    # Open the images
    bottom_img = Image.open(bottom_img_path).convert("RGBA")
    top_img = Image.open(top_img_path).convert("RGBA")

    # print(f'top image size: {top_img.size}')
    # print(f'bottom image size: {bottom_img.size}')

    # Scale the top image within the provided range
    scale_factor = random.uniform(*top_img_scale_range)
    top_img = scale_image(top_img, scale_factor)

    # Get dimensions
    bottom_w, bottom_h = bottom_img.size
    top_w, top_h = top_img.size

    # Apply jitter, defaulting to centered positions
    jittered_bottom_pos = jitter_position(bottom_img_pos, bottom_img_jitter_range)
    jittered_top_pos = jitter_position(top_img_pos, top_img_jitter_range)

    # Convert images to PyTorch tensors
    bottom_tensor = T.ToTensor()(bottom_img)
    top_tensor = T.ToTensor()(top_img)

    # Create a blank canvas tensor to overlay the top image on the bottom image
    final_tensor = bottom_tensor.clone()

    fill_h = (bottom_h - top_h) // 2
    fill_w = (bottom_w - top_w) // 2

    jittered_top_pos -= jittered_bottom_pos

    # Overlay the top image at the jittered position
    for c in range(3):  # Iterate over RGB channels
        final_tensor[c, fill_h + jittered_top_pos[1]:fill_h + jittered_top_pos[1] + top_h,
                     fill_w + jittered_top_pos[0]:fill_w + jittered_top_pos[0] + top_w] = (
            top_tensor[c, :, :] * top_tensor[3, :, :] * alpha +
            final_tensor[c, fill_h + jittered_top_pos[1]:fill_h + jittered_top_pos[1] + top_h,
                         fill_w + jittered_top_pos[0]:fill_w + jittered_top_pos[0] + top_w] * (1 - top_tensor[3, :, :] * alpha)
        )

    # Convert back to PIL image and crop to desired size from the center
    final_img = T.ToPILImage()(final_tensor)
    cropped_img = crop_image(final_img, crop_size, jittered_bottom_pos)
    cropped_img = T.ToTensor()(cropped_img)

    return cropped_img


def synthesize_image_with_params(bottom_img_path, top_img_path, top_img_pos, bottom_img_pos,
                                 scale_factor, crop_size, alpha=1.0):
    # Open the images
    bottom_img = Image.open(bottom_img_path).convert("RGBA")
    top_img = Image.open(top_img_path).convert("RGBA")

    # Scale the top image using the provided scale factor
    top_img = scale_image(top_img, scale_factor)

    # Get dimensions
    bottom_w, bottom_h = bottom_img.size
    top_w, top_h = top_img.size

    # Convert images to PyTorch tensors
    bottom_tensor = T.ToTensor()(bottom_img)
    top_tensor = T.ToTensor()(top_img)

    # Create a blank canvas tensor to overlay the top image on the bottom image
    final_tensor = bottom_tensor.clone()

    fill_h = (bottom_h - top_h) // 2
    fill_w = (bottom_w - top_w) // 2

    top_img_pos -= bottom_img_pos

    # Overlay the top image at the jittered position
    for c in range(3):  # Iterate over RGB channels
        final_tensor[c, fill_h + top_img_pos[1]:fill_h + top_img_pos[1] + top_h,
                     fill_w + top_img_pos[0]:fill_w + top_img_pos[0] + top_w] = (
            top_tensor[c, :, :] * top_tensor[3, :, :] * alpha +
            final_tensor[c, fill_h + top_img_pos[1]:fill_h + top_img_pos[1] + top_h,
                         fill_w + top_img_pos[0]:fill_w + top_img_pos[0] + top_w] * (1 - top_tensor[3, :, :] * alpha)
        )

    # Convert back to PIL image and crop to desired size from the center
    final_img = T.ToPILImage()(final_tensor)
    cropped_img = crop_image(final_img, crop_size, bottom_img_pos)
    cropped_img = T.ToTensor()(cropped_img)

    return cropped_img


def random_movement(boundary_size, center_ratio, max_steps, prob_stay, prob_mov, initial_velocity=1.0, momentum_decay=0.95, velocity_randomness=0.02,
                    angle_range=0.5):
    """
    Simulates random movement within a 2D boundary centered at (0, 0), preserving momentum with slight randomness.

    Parameters:
    - boundary_size: (width, height) tuple defining the total size of the boundary (e.g., 100 means -50 to 50).
    - center_ratio: ratio (e.g., 0.2 for 20%) to define the starting center region.
    - max_steps: maximum steps allowed in the sequence.
    - a: probability of staying if currently staying.
    - b: probability of continuing to move if currently moving.
    - initial_velocity: initial velocity magnitude.
    - momentum_decay: factor by which velocity reduces per step (closer to 1 means slower decay).
    - velocity_randomness: max random factor to adjust the velocity (for slight increase/decrease).

    Returns:
    - path: List of (x, y) positions for each step until out of boundary or max_steps.
    """

    # Calculate half-widths and half-heights based on boundary size
    boundary_size = np.array(boundary_size)
    center_ratio = np.array(center_ratio)
    half_boundary = boundary_size / 2

    # Calculate center region bounds based on center_ratio
    center_region = half_boundary * center_ratio

     # Set initial position within the center region, with (0, 0) as the center
    x = np.random.uniform(-center_region[0], center_region[0])
    y = np.random.uniform(-center_region[1], center_region[1])
    
    # Initialize path and velocity history
    path = np.zeros((max_steps + 1, 2))  # Preallocate for max_steps + initial position
    path[0] = [x, y]
    velocity_history = np.zeros(max_steps)

    # State and movement parameters
    moving = False
    velocity = initial_velocity
    angle = np.random.uniform(0, 2 * np.pi)

    step_count = 1  # Counter for the number of steps taken

    for i in range(max_steps):
        if moving:
            if np.random.rand() > prob_mov:
                moving = False  # Switch to staying
                velocity_history[i] = 0
            else:
                # Adjust velocity by momentum decay and add a small random factor
                velocity = max(velocity * momentum_decay + np.random.uniform(-velocity_randomness, velocity_randomness), 0.01)
                
                # Slightly adjust direction to keep it mostly stable
                angle += np.random.uniform(-angle_range, angle_range)

                # Calculate new position with momentum applied
                new_x = x + velocity * np.cos(angle)
                new_y = y + velocity * np.sin(angle)

                # Check boundaries and reset if out of bounds
                if -half_boundary[0] <= new_x <= half_boundary[0] and -half_boundary[1] <= new_y <= half_boundary[1]:
                    x, y = new_x, new_y
                else:
                    # Stay in the same position and reset velocity to 0
                    velocity = 0
                    moving = False
                velocity_history[i] = velocity

        else:
            angle = np.random.uniform(0, 2 * np.pi)
            velocity_history[i] = 0
            if np.random.rand() > prob_stay:
                moving = True  # Switch to moving
                # Preserve momentum by reusing the last velocity and angle
                velocity = initial_velocity * momentum_decay + np.random.uniform(-velocity_randomness, velocity_randomness)

        path[step_count] = [x, y]
        step_count += 1
    
    # Trim path and velocity_history to actual steps taken
    path = path[:step_count]
    velocity_history = velocity_history[:step_count]

    return path, velocity_history


class Cricket2RGCs(Dataset):
    def __init__(
        self,
        num_samples,
        multi_opt_sf,
        tf,
        map_func,
        grid2value_mapping,
        target_width,
        target_height,
        movie_generator,
        grid_size_fac=1,
        is_norm_coords=False,
        is_syn_mov_shown=False,
        fr2spikes=False,
        multi_opt_sf_off=None,
        tf_off=None,
        map_func_off=None,
        grid2value_mapping_off=None,
        is_both_ON_OFF=False,
        quantize_scale=1,
        add_noise=False,
        rgc_noise_std=0.0,
        smooth_data=False,
        smooth_kernel_size=10,  # was 20
        sampleing_rate=100,
        smooth_sigma=0.025,   # was 0.05
        is_rectified=True,
        is_direct_image=False,
        grid_coords=None,
        is_reversed_OFF_sign=False,
        is_two_grids=False,
        rectified_thr_ON=0.0,
        rectified_thr_OFF=0.0,
        # Simple LNK parameters
        use_lnk=False,
        lnk_params=None,
        lnk_params_off=None,
        surround_sigma_ratio=4.0,
        surround_sf=None,  
        surround_sf_off=None,
    ):
        # Core attributes
        self.num_samples = num_samples
        self.movie_generator = movie_generator
        self.map_func = map_func
        self.grid2value_mapping = grid2value_mapping
        self.target_width = target_width
        self.target_height = target_height
        self.grid_size_fac = grid_size_fac
        self.grid_width = int(round(target_width * grid_size_fac))
        self.grid_height = int(round(target_height * grid_size_fac))

        # Flags and options
        self.is_syn_mov_shown = is_syn_mov_shown
        self.fr2spikes = fr2spikes
        self.quantize_scale = quantize_scale
        self.add_noise = add_noise
        self.rgc_noise_std = rgc_noise_std
        self.smooth_data = smooth_data
        self.smooth_kernel_size = smooth_kernel_size
        self.sampleing_rate = sampleing_rate
        self.smooth_sigma = smooth_sigma
        self.is_rectified = is_rectified
        self.is_direct_image = is_direct_image
        self.is_both_ON_OFF = is_both_ON_OFF
        self.is_two_grids = is_two_grids
        self.is_reversed_OFF_sign = is_reversed_OFF_sign
        self.rectified_thr_ON=rectified_thr_ON
        self.rectified_thr_OFF=rectified_thr_OFF

        # Normalization for path coordinates
        self.norm_path_fac = (
            np.array([target_height, target_width]) / 2
        ) if is_norm_coords else 1

        # Build channel configurations
        # Always include ON channel
        sf_tensor = torch.from_numpy(multi_opt_sf).float()
        num_rgcs = sf_tensor.shape[2]  # Get number of RGC cells
        
        # Handle temporal filter - if single tf, replicate for all RGCs
        if tf.ndim == 1:
            # Single temporal filter for all RGCs - replicate
            tf_multi = torch.from_numpy(tf.copy()).float().unsqueeze(0).repeat(num_rgcs, 1)
        elif tf.ndim == 2:
            # Multiple temporal filters - handle both orientations
            if tf.shape[0] == num_rgcs:
                # Shape: [num_rgcs, time_points] - correct orientation
                tf_multi = torch.from_numpy(tf.copy()).float()
            elif tf.shape[1] == num_rgcs:
                # Shape: [time_points, num_rgcs] - needs transpose
                tf_multi = torch.from_numpy(tf.T.copy()).float()
            else:
                raise ValueError(f"tf shape {tf.shape} incompatible with {num_rgcs} RGCs. "
                               f"Expected either ({num_rgcs}, time_points) or (time_points, {num_rgcs})")
        else:
            raise ValueError(f"tf must be 1D or 2D array, got {tf.ndim}D with shape {tf.shape}")
        
        # Store LNK parameters
        self.use_lnk = use_lnk
        self.surround_sigma_ratio = surround_sigma_ratio

        # Handle surround filters for LNK model
        if surround_sf is None and use_lnk:
            raise ValueError("Surround spatial filters are required for LNK model.")

        self.channels = [
            {
                'sf': sf_tensor,
                'sf_surround': torch.from_numpy(surround_sf).float(),
                'tf': tf_multi.view(num_rgcs, 1, -1),  # [N, 1, T] for conv1d
                'map_func': map_func,
                'grid2value': grid2value_mapping,
                'rect_thr': self.rectified_thr_ON,
                'lnk_params': lnk_params,
            }
        ]
        
        # Add OFF channel config if required
        if is_both_ON_OFF or is_two_grids:
            sf_off_tensor = torch.from_numpy(multi_opt_sf_off).float()
            num_rgcs_off = sf_off_tensor.shape[2]
            
            # Handle OFF temporal filter
            if tf_off.ndim == 1:
                tf_off_multi = torch.from_numpy(tf_off.copy()).float().unsqueeze(0).repeat(num_rgcs_off, 1)
            elif tf_off.ndim == 2:
                # Multiple temporal filters - handle both orientations
                if tf_off.shape[0] == num_rgcs_off:
                    # Shape: [num_rgcs, time_points] - correct orientation
                    tf_off_multi = torch.from_numpy(tf_off.copy()).float()
                elif tf_off.shape[1] == num_rgcs_off:
                    # Shape: [time_points, num_rgcs] - needs transpose
                    tf_off_multi = torch.from_numpy(tf_off.T.copy()).float()
                else:
                    raise ValueError(f"tf_off shape {tf_off.shape} incompatible with {num_rgcs_off} RGCs. "
                                   f"Expected either ({num_rgcs_off}, time_points) or (time_points, {num_rgcs_off})")
            else:
                raise ValueError(f"tf_off must be 1D or 2D array, got {tf_off.ndim}D with shape {tf_off.shape}")
            
            # Sign handling for OFF pathway
            if not (is_two_grids or is_reversed_OFF_sign):
                tf_off_multi = -tf_off_multi
                
            off_channel = {
                'sf': sf_off_tensor,
                'sf_surround': torch.from_numpy(surround_sf_off).float(),
                'tf': tf_off_multi.view(num_rgcs_off, 1, -1),
                'map_func': map_func_off,
                'grid2value': grid2value_mapping_off,
                'rect_thr': self.rectified_thr_OFF,
                'lnk_params': lnk_params_off
            }
            
            self.channels.append(off_channel)

        # Optional grid-coordinates for weighted output
        self.grid_coords = (
            torch.tensor(grid_coords, dtype=torch.float32)
            if grid_coords is not None else None
        )

    def __len__(self):
        return self.num_samples

    def _broadcast_param(self, p, N, device, dtype):
        """
        Utility: make sure a scalar or 1D array/tensor becomes a torch tensor [N] on device/dtype.
        """
        if p is None:
            return None
        if isinstance(p, (int, float)):
            return torch.full((N,), float(p), device=device, dtype=dtype)
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p)
        if isinstance(p, torch.Tensor):
            p = p.to(device=device, dtype=dtype)
        if p.ndim == 0:
            return p.view(1).repeat(N)
        if p.ndim == 1 and p.shape[0] == 1:
            return p.repeat(N)
        assert p.shape[0] == N, f"param length {p.shape[0]} != N {N}"
        return p

    def _compute_rgc_time(self, movie, sf, sf_surround, tf, rect_thr,
                            lnk_params=None):
        """
        Compute RGC response using either LN or simplified LNK model.
        
        Args:
            movie: [T, H, W]
            sf: [W, H, N] - spatial filters 
            tf: [N, 1, T_filter] - temporal filters
            rect_thr: float - rectification threshold
            sf_surround: [W, H, N] - surround spatial filters
            tf_surround: [N, 1, T_filter] - surround temporal filters
            lnk_params: unused (kept for compatibility)
        Returns:
            rgc_time: [N, T_out] - RGC responses over time
        """
        
        # Check if using simple LNK model
        if self.use_lnk and lnk_params is not None:
            
            return compute_lnk_response(
                movie=movie,
                center_sf=sf,
                center_tf=tf,
                surround_sf=sf_surround,
                surround_tf=tf,
                lnk_params=lnk_params,
                device=movie.device,
                dtype=movie.dtype,
                rgc_noise_std=self.rgc_noise_std if self.add_noise else 0.0,
            )
        
        # Original LN model (backward compatible)
        sf_frame = torch.einsum('whn,thw->nt', sf, movie)  # [N, T]
        sf_frame = sf_frame.unsqueeze(0)                   # [1, N, T]
        rgc_time = F.conv1d(sf_frame, tf, stride=1, padding=0, groups=sf_frame.shape[1]).squeeze(0)  # [N, T_out]

        # Post-processing
        if self.fr2spikes:
            rgc_time = torch.poisson(torch.clamp_min(rgc_time * self.quantize_scale, 0)) / self.quantize_scale
        if self.smooth_data:
            rgc_time = gaussian_smooth_1d(
                rgc_time,
                kernel_size=self.smooth_kernel_size,
                sampleing_rate=self.sampleing_rate,
                sigma=self.smooth_sigma,
            )
        if self.add_noise:
            rgc_time += torch.randn_like(rgc_time) * self.rgc_noise_std
        if self.is_rectified:
            rgc_time = torch.clamp_min(rgc_time, rect_thr)
            
        return rgc_time

    def _process_direct_image(self, syn_movie):
        """
        Process synthetic movie directly using temporal filtering.
        
        Args:
            syn_movie: [T, H, W] - input movie
            
        Returns:
            Interpolated RGC responses: [T_out, grid_height, grid_width]
        """
        # syn_movie: [T, H, W]
        T, H, W = syn_movie.shape
        
        # Flatten spatial dimensions and permute for conv1d
        flat = syn_movie.permute(1, 2, 0).reshape(-1, T).unsqueeze(0)  # [1, H*W, T]
        
        # Get temporal filter - use first channel's first temporal filter for all pixels
        # Note: this assumes all pixels use the same temporal filter
        tf_single = self.channels[0]['tf'][0:1]  # [1, 1, T_filter]
        
        # Replicate temporal filter for all spatial locations
        tf_rep = tf_single.repeat(flat.shape[1], 1, 1)  # [H*W, 1, T_filter]
        
        # Apply temporal filtering using grouped conv1d
        conv = F.conv1d(flat, tf_rep, stride=1, padding=0, groups=flat.shape[1]).squeeze()
        
        # Reshape back to spatial format
        reshaped = conv.view(H, W, -1).permute(2, 1, 0).unsqueeze(0)
        
        # Interpolate to target grid size
        return F.interpolate(
            reshaped,
            size=(self.grid_height, self.grid_width),
            mode='bilinear',
            align_corners=False
        ).squeeze()

    def __getitem__(self, idx):
        syn_movie, path, path_bg, *rest = self.movie_generator.generate()

        # Determine binocular vs monocular
        is_binocular = (syn_movie.ndim == 4 and syn_movie.shape[1] == 2)
        if is_binocular:
            movies = [syn_movie[:, i] for i in range(2)]
        else:
            mono = syn_movie[:, 0] if syn_movie.ndim == 4 else syn_movie
            movies = [mono]
            syn_movie = syn_movie[:, 0, :, :] 

        def _compute_for_channel(mv, ch):
            """Helper to compute RGC response - simplified"""
            return self._compute_rgc_time(
                mv,
                ch['sf'],
                ch['sf_surround'],
                ch['tf'], 
                ch['rect_thr'],
                ch['lnk_params']
            )

        grid_values_list = []

        # Direct-image branch
        if self.is_direct_image:
            grid_values_list.append(self._process_direct_image(movies[0]))

        else:
            # 1) Both ON/OFF + binocular => 4 outputs (2 pathways × 2 eyes)
            if self.is_both_ON_OFF and is_binocular:
                for ch in self.channels:
                    for mv in movies:
                        rgc_time = _compute_for_channel(mv, ch)  
                        grid_values_list.append(
                            ch['map_func'](
                                rgc_time,
                                ch['grid2value'],
                                self.grid_width,
                                self.grid_height
                            )
                        )

            # 2) Two grids + binocular => 4 outputs (2 grids × 2 eyes)
            elif self.is_two_grids and is_binocular:
                for ch in self.channels:
                    for mv in movies:
                        rgc_time = _compute_for_channel(mv, ch)
                        grid_values_list.append(
                            ch['map_func'](
                                rgc_time,
                                ch['grid2value'],
                                self.grid_width,
                                self.grid_height
                            )
                        )

            # 3) Binocular only => 2 outputs
            elif is_binocular:
                ch = self.channels[0]
                for mv in movies:
                    rgc_time = _compute_for_channel(mv, ch)
                    grid_values_list.append(
                        ch['map_func'](
                            rgc_time,
                            ch['grid2value'],
                            self.grid_width,
                            self.grid_height
                        )
                    )

            # 4) ON/OFF only => 2 outputs
            elif self.is_both_ON_OFF:
                for ch in self.channels:
                    rgc_time = _compute_for_channel(movies[0], ch)
                    grid_values_list.append(
                        ch['map_func'](
                            rgc_time,
                            ch['grid2value'],
                            self.grid_width,
                            self.grid_height
                        )
                    )

            # 5) Default single pathway => 1 output
            else:
                ch = self.channels[0]
                rgc_time = _compute_for_channel(movies[0], ch)
                grid_values_list.append(
                    ch['map_func'](
                        rgc_time,
                        ch['grid2value'],
                        self.grid_width,
                        self.grid_height
                    )
                )

        # Stack channels
        grid_seq = torch.stack(grid_values_list, dim=1)

        # Weighted coords if needed
        if self.grid_coords is not None:
            weighted_sum = torch.einsum('nt,nc->tc', rgc_time, self.grid_coords)
            sum_rates = torch.sum(rgc_time, dim=0, keepdim=True).clamp(min=1e-6).view(-1, 1)
            weighted_coords = (weighted_sum / sum_rates).detach().numpy()
        else:
            weighted_coords = np.array(0)

        # Normalize paths
        path = path[-rgc_time.shape[1]:] / self.norm_path_fac
        path_bg = path_bg[-rgc_time.shape[1]:] / self.norm_path_fac

        # print(f'grid_seq shape: {grid_seq.shape}')
        # Permute for final shape
        grid_seq = grid_seq.permute(0, 1, 3, 2)
        # if self.is_both_ON_OFF or self.is_two_grids or is_binocular:
        #     grid_seq = grid_seq.permute(0, 1, 3, 2)
        # else:
        #     grid_seq = grid_seq.permute(0, 2, 1).unsqueeze(1)

        out_path = torch.tensor(path, dtype=torch.float32)
        out_bg = torch.tensor(path_bg, dtype=torch.float32)

        # Return with or without synthetic movie
        if self.is_syn_mov_shown:
            return grid_seq, path, path_bg, syn_movie, *rest, weighted_coords
        return grid_seq, out_path, out_bg
    

class SynMovieGenerator:
    def __init__(self, top_img_folder, bottom_img_folder, crop_size, boundary_size, center_ratio, max_steps=200, prob_stay_ob=0.95, 
                 prob_mov_ob=0.975, prob_stay_bg=0.95, prob_mov_bg=0.975,num_ext=50, initial_velocity=6, momentum_decay_ob=0.95, 
                 momentum_decay_bg=0.5, scale_factor=1.0, velocity_randomness_ob=0.02, velocity_randomness_bg=0.01, angle_range_ob=0.5, 
                 angle_range_bg=0.25, coord_mat_file=None, correction_direction=1, is_reverse_xy=False, start_scaling=1, 
                 end_scaling=2, dynamic_scaling=0, is_binocular=False, interocular_dist=1, bottom_contrast=1.0, top_contrast=1.0,
                 mean_diff_offset=0.0, fix_disparity=None):
        """
        Initializes the SynMovieGenerator with configuration parameters.

        Parameters:
        - crop_size (tuple): Dimensions of the crop (width, height).
        - boundary_size (tuple): Boundary limits for movement.
        - center_ratio (tuple): Center ratios for initial movement placement.
        - max_steps (int): Maximum number of steps for movement paths.
        - num_ext (int): Number of extended static frames at the beginning.
        - initial_velocity (float): Initial velocity for movement.
        - bottom_img_folder (str): Path to the folder containing bottom images.
        - top_img_folder (str): Path to the folder containing top images.
        - scale_factor (float): Scale factor for image synthesis.
        """
        self.bottom_img_folder = bottom_img_folder
        self.top_img_folder = top_img_folder
        self.crop_size = crop_size
        self.boundary_size = tuple(ast.literal_eval(boundary_size)) if isinstance(boundary_size, str) else boundary_size
        # print(f"boundary_size type: {type(boundary_size)}")
        # print(f"boundary_size: {boundary_size}")
        self.center_ratio = center_ratio
        self.max_steps = max_steps
        self.prob_stay_ob = prob_stay_ob
        self.prob_mov_ob = prob_mov_ob
        self.prob_stay_bg = prob_stay_bg
        self.prob_mov_bg = prob_mov_bg
        self.num_ext = num_ext
        self.initial_velocity = initial_velocity
        self.momentum_decay_ob = momentum_decay_ob
        self.momentum_decay_bg = momentum_decay_bg
        self.scale_factor = scale_factor
        self.velocity_randomness_ob = velocity_randomness_ob
        self.velocity_randomness_bg = velocity_randomness_bg
        self.angle_range_ob = angle_range_ob
        self.angle_range_bg = angle_range_bg
        self.coord_dic = self._get_coord_dic(coord_mat_file)
        self.correction_direction = correction_direction
        self.is_reverse_xy = is_reverse_xy
        self.start_scaling = start_scaling 
        self.end_scaling = end_scaling 
        self.dynamic_scaling = dynamic_scaling
        self.is_binocular = is_binocular
        self.interocular_dist = interocular_dist  # in cm
        self.bottom_contrast       = bottom_contrast
        self.top_contrast          = top_contrast
        self.mean_diff_offset = mean_diff_offset
        self.fix_disparity = fix_disparity

    def _modify_scaling(self):
        start_scaling = self.start_scaling
        end_scaling = self.end_scaling

        if self.dynamic_scaling != 0:
            # Modify start_scaling and end_scaling with random adjustments
            start_scaling += random.uniform(0, self.dynamic_scaling)
            end_scaling -= random.uniform(0, self.dynamic_scaling)

            # Ensure start_scaling <= end_scaling
            start_scaling, end_scaling = min(start_scaling, end_scaling), max(start_scaling, end_scaling)

        return start_scaling, end_scaling
    
    def _get_coord_dic(self, coord_mat_file):
        index_column_name = 'image_id'
        if coord_mat_file is None:
            return None  # Or handle appropriately
        
        df = load_mat_to_dataframe(coord_mat_file)  # Specify 'custom_variable_name' if different
        if index_column_name not in df.columns:
            raise ValueError("Expected 'image_id' column in the input data.")
        
        data_dict = df.set_index(index_column_name).to_dict(orient='index')
        return data_dict

    def generate(self):
        """
        Generates a synthetic movie, path, and path_bg.

        Returns:
        - syn_movie (np.ndarray): 3D array of shape (height, width, time_steps).
        - path (np.ndarray): 2D array of object positions (time_steps, 2).
        - path_bg (np.ndarray): 2D array of background positions (time_steps, 2).
        """
        path, velocity = random_movement(self.boundary_size, self.center_ratio, self.max_steps, prob_stay=self.prob_stay_ob, 
                                  prob_mov=self.prob_mov_ob,initial_velocity=self.initial_velocity, momentum_decay=self.momentum_decay_ob,
                                  velocity_randomness=self.velocity_randomness_ob, angle_range=self.angle_range_ob)
        path_bg, velocity_bg = random_movement(self.boundary_size, self.center_ratio, self.max_steps, prob_stay=self.prob_stay_bg, 
                                  prob_mov=self.prob_mov_bg, initial_velocity=self.initial_velocity, momentum_decay=self.momentum_decay_bg,
                                  velocity_randomness=self.velocity_randomness_bg, angle_range=self.angle_range_bg)

        # Determine the minimum length for consistency
        min_length = min(len(path), len(path_bg))
        path = path[:min_length]
        velocity = velocity[:min_length]
        path_bg = path_bg[:min_length]
        velocity_bg = velocity_bg[:min_length]

        # Extend static frames at the beginning
        path = np.vstack((np.repeat(path[0:1, :], self.num_ext, axis=0), path))
        path_bg = np.vstack((np.repeat(path_bg[0:1, :], self.num_ext, axis=0), path_bg))

        bottom_img_path = get_random_file_path(self.bottom_img_folder)
        top_img_path = get_random_file_path(self.top_img_folder)

        # Convert paths to numpy arrays
        # top_img_positions = path.round().astype(int)
        bottom_img_positions = path_bg.round().astype(int)

        start_scaling, end_scaling = self._modify_scaling()
        scaling_factors = calculate_scaling_factors(bottom_img_positions, start_scaling=start_scaling, end_scaling=end_scaling)
        bounds = (-self.boundary_size[0]/2, self.boundary_size[0]/2, -self.boundary_size[1]/2, self.boundary_size[1]/2)

        if self.is_binocular:
            disparity, _ = disparity_from_scaling_factor(
                scaling_factors=scaling_factors,
                start_distance=21,
                end_distance=4,
                iod_cm=self.interocular_dist, 
                fix_disparity=self.fix_disparity
            )
            disparity = convert_deg_to_pix(disparity)
            top_img_disparity_positions = path.copy()
            top_img_disparity_positions[:, 0] += disparity
            
            top_img_positions_shifted, top_img_disparity_positions_shifted = adjust_trajectories(bounds, path.copy(), top_img_disparity_positions)
            # plot_save_folder =  '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/Figures/temps/'
            # plot_trajectories(bounds, top_img_positions, top_img_disparity_positions, top_img_positions_shifted, 
            #                  top_img_disparity_positions_shifted, plot_save_folder, filename="trajectory_plot.png")
            path = (top_img_positions_shifted + top_img_disparity_positions_shifted) / 2
        else:
            top_img_positions_shifted, top_img_disparity_positions_shifted = adjust_trajectories(bounds, path.copy())
            path = top_img_positions_shifted
        
        
        syn_movie = synthesize_image_with_params_batch(
                bottom_img_path, top_img_path, top_img_positions_shifted, bottom_img_positions,
                scaling_factors, self.crop_size, alpha=1.0, top_img_positions_shifted=top_img_disparity_positions_shifted,
                bottom_contrast=self.bottom_contrast, top_contrast=self.top_contrast, mean_diff_offset=self.mean_diff_offset
            )
        # Correct for the cricket head position
        bg_image_name = get_filename_without_extension(bottom_img_path)
        image_id = get_image_number(top_img_path)
        scaling_factors = np.array(scaling_factors) 

        if self.coord_dic is not None:
            if self.is_reverse_xy:
                coord_correction = np.array([self.coord_dic[image_id]['coord_y'], self.coord_dic[image_id]['coord_x']])
            else:
                coord_correction = np.array([self.coord_dic[image_id]['coord_x'], self.coord_dic[image_id]['coord_y']])
            
            scaled_coord_corrections = coord_correction[np.newaxis, :] * scaling_factors[:, np.newaxis]
            path = path - self.correction_direction*scaled_coord_corrections

        return syn_movie, path, path_bg, scaling_factors, bg_image_name, image_id
    


def synthesize_image_with_params_batch(bottom_img_path, top_img_path, top_img_positions, bottom_img_positions,
                                       scaling_factors, crop_size, alpha=1.0, top_img_positions_shifted=None,
                                       bottom_contrast=1.0, top_contrast=1.0, mean_diff_offset=0.0):
    """
    Synthesizes a batch of images by overlaying the top image on the bottom image at specific positions.

    Parameters:
    - bottom_img_path (str): Path to the bottom image file.
    - top_img_path (str): Path to the top image file.
    - top_img_positions (np.ndarray): Array of shape (batch_size, 2) with positions for the top image.
    - bottom_img_positions (np.ndarray): Array of shape (batch_size, 2) with positions for the bottom image.
    - scale_factor (float): Scaling factor for the top image.
    - crop_size (tuple): Desired output size (width, height).
    - alpha (float): Alpha blending factor for overlaying.

    Returns:
    - syn_images (torch.Tensor): Tensor of shape (batch_size, 3, crop_size[1], crop_size[0]) containing the synthesized images.
    """

    # Open and process the images
    bottom_img = Image.open(bottom_img_path).convert("RGBA")
    bottom_tensor = T.ToTensor()(bottom_img)
    original_top_img = Image.open(top_img_path).convert("RGBA")

    batch_size = len(top_img_positions)
    bottom_w, bottom_h = bottom_img.size

    channels = 2 if top_img_positions_shifted is not None else 1
    syn_images = torch.zeros((batch_size, channels, crop_size[1], crop_size[0]))


    # Variables to cache the last scaling factor and scaled image
    last_scaling_factor = None
    cached_scaled_image = None

    # flags for no-op
    do_mean  = abs(mean_diff_offset) > 1e-9
    do_contr = (bottom_contrast != 1.0) or (top_contrast != 1.0)

    for i in range(batch_size):
        current_scaling_factor = scaling_factors[i]

        # Check if the scaling factor is the same as the previous step
        if current_scaling_factor != last_scaling_factor:
            # Scale the top image
            scaled_top_img = scale_image(original_top_img, current_scaling_factor)
            cached_scaled_image = T.ToTensor()(scaled_top_img)
            last_scaling_factor = current_scaling_factor

        top_tensor = cached_scaled_image
        top_w, top_h = scaled_top_img.size

        # clone for per-frame ops
        b = bottom_tensor.clone()
        t = top_tensor.clone()

        # 2) optionally apply mean-difference offset & contrast squeeze
        if do_mean or do_contr:
            if do_mean:
                m_b = b[1].mean()
                m_t = t[1].mean()
                Δ_total = mean_diff_offset
                d_b, d_t = abs(m_b-0.5), abs(m_t-0.5)
                r_b = d_b/(d_b+d_t) if (d_b+d_t)>0 else 0.5
                B_b =  r_b * Δ_total
                B_t = -(1-r_b) * Δ_total
                b[1] = (b[1] + B_b).clamp(0, 1)
                t[1] = (t[1] + B_t).clamp(0, 1)
            if do_contr:
                b[1] = (0.5 + (b[1]-0.5)*bottom_contrast).clamp(0, 1)
                t[1] = (0.5 + (t[1]-0.5)*top_contrast).clamp(0, 1)

        fill_h = (bottom_h - top_h) // 2
        fill_w = (bottom_w - top_w) // 2

        def overlay_and_crop(canvas, top_ten, relative_pos, crop_pos):
            

            # Compute relative position
            relative_pos = relative_pos - crop_pos
            relative_pos = relative_pos.astype(int)
            x_offset = fill_w + relative_pos[0]
            y_offset = fill_h + relative_pos[1]

            # Overlay the top image using green channel only (index 1)
            canvas[1,  # Green channel
                         y_offset:y_offset + top_h,
                         x_offset:x_offset + top_w] = (
                top_ten[1, :, :] * top_ten[3:4, :, :] * alpha +  # Green channel overlay
                canvas[1,
                             y_offset:y_offset + top_h,
                             x_offset:x_offset + top_w] * (1 - top_ten[3:4, :, :] * alpha)
            )

            # Convert back to PIL image and crop to desired size from the center
            final_img = T.ToPILImage()(canvas)
            cropped_img = crop_image(final_img, crop_size, crop_pos)
            if cropped_img.mode == "RGBA":
                cropped_img = cropped_img.convert("RGB")
            cropped_img = 2 * T.ToTensor()(cropped_img) - 1

            # Return green channel [1, width, height]
            return cropped_img[1:2]
        
        syn_images[i, 0] = overlay_and_crop(b, t, top_img_positions[i], bottom_img_positions[i])

        if channels == 2:
            syn_images[i, 1] = overlay_and_crop(b, t, top_img_positions_shifted[i], bottom_img_positions[i])

    return syn_images

    

class RGCrfArray:
    def __init__(self, sf_param_table, tf_param_table, rgc_array_rf_size, xlim, ylim, target_num_centers, sf_scalar,
                 grid_generate_method, tau=None, mask_radius=None, rgc_rand_seed=42, num_gauss_example=1, sf_mask_radius=35, 
                 sf_pixel_thr=99.7, sf_constraint_method=None, temporal_filter_len=50, grid_size_fac=0.5, is_pixelized_tf=False, 
                 set_s_scale=[], is_rf_median_subtract=True, is_rescale_diffgaussian=True, is_both_ON_OFF = False,
                 grid_noise_level=0.3, is_reversed_tf = False, sf_id_list=None, syn_tf_sf=False, 
                 use_lnk_override=False,
                 surround_sigma_ratio=4.0,  # Add this parameter
                 # New synchronization parameters
                 syn_params=None,
                 lnk_param_table=None):
        """
        Enhanced constructor with flexible parameter synchronization.
        
        Args:
            ...existing args...
            syn_params: List of parameters to synchronize, e.g., ['tf', 'sf', 'lnk'] or ['tf', 'sf']
            lnk_param_table: DataFrame with LNK parameters for synchronization
        """
        self.sf_param_table = sf_param_table
        self.tf_param_table = tf_param_table
        self.use_lnk_override = use_lnk_override  # Flag to override s_scale when using LNK model
        self.rgc_array_rf_size = rgc_array_rf_size
        self.sf_scalar = sf_scalar
        self.temporal_filter_len = temporal_filter_len
        self.grid_generate_method = grid_generate_method
        self.tau = tau
        self.sf_mask_radius = sf_mask_radius
        self.sf_constraint_method = sf_constraint_method
        self.mask_radius = mask_radius
        self.rgc_rand_seed = rgc_rand_seed
        self.num_gauss_example = num_gauss_example
        self.target_num_centers = target_num_centers
        self.sf_pixel_thr = sf_pixel_thr
        self.grid_size_fac = grid_size_fac
        self.is_pixelized_tf = is_pixelized_tf
        self.set_s_scale = set_s_scale
        self.is_rf_median_subtract = is_rf_median_subtract
        self.is_rescale_diffgaussian=is_rescale_diffgaussian
        self.is_both_ON_OFF = is_both_ON_OFF
        self.target_height = xlim[1] - xlim[0]
        self.target_width = ylim[1] - ylim[0]
        self.grid_noise_level = grid_noise_level
        self.grid_centers = precompute_grid_centers(self.target_height, self.target_width, x_min=xlim[0], x_max=xlim[1],
                                            y_min=ylim[0], y_max=ylim[1], grid_size_fac=grid_size_fac)
        self.grid_generator = HexagonalGridGenerator(xlim, ylim, target_num_centers=self.target_num_centers, rand_seed=self.rgc_rand_seed, 
                                                     noise_level=self.grid_noise_level)
        self.is_reversed_tf = is_reversed_tf
        self.sf_id_list = sf_id_list
        self.syn_tf_sf = syn_tf_sf
        self.surround_sigma_ratio = surround_sigma_ratio  # Add this line
        
        # Handle new synchronization options
        self.lnk_param_table = lnk_param_table
        self.syn_params = syn_params if syn_params is not None else []
        # Validate synchronization requirements
        if self.syn_params:
            self._validate_sync_params()
        
        logging.info( f"   subprocessing...1.1")

    def _validate_sync_params(self):
        """Validate that all tables needed for synchronization have compatible lengths."""
        table_lengths = {}
        
        if 'sf' in self.syn_params:
            table_lengths['sf'] = len(self.sf_param_table)
        if 'tf' in self.syn_params:
            table_lengths['tf'] = len(self.tf_param_table)
        if 'lnk' in self.syn_params and self.lnk_param_table is not None:
            table_lengths['lnk'] = len(self.lnk_param_table)
        
        if len(set(table_lengths.values())) > 1:
            raise ValueError(f"Parameter tables have different lengths: {table_lengths}")
        else:
            self.sync_table_length = next(iter(table_lengths.values()))

    def get_results(self):
        """
        Always returns exactly 5 values for consistent unpacking.
        LNK parameters should be handled separately via the use_lnk_override flag.
        """
        points = self.grid_generator.generate_first_grid()
        idx_dict = self._generate_indices(points)

        lnk_params = sample_lnk_parameters(self.lnk_param_table, idx_dict.get('lnk'))

        # Create filters
        multi_opt_sf_center, multi_opt_sf_surround = self._create_multi_opt_sf(points, self.sf_id_list, idx_list=idx_dict.get('sf'))
        tf = self._create_temporal_filter(idx_list=idx_dict.get('tf'))
        grid2value_mapping, map_func = self._get_grid_mapping(points)

        return multi_opt_sf_center, multi_opt_sf_surround, tf, grid2value_mapping, map_func, points, lnk_params

    def _generate_indices(self, points):
        """Generate indices for parameter sampling with flexible synchronization."""
        num_points = len(points)
        idx_dict = {}

        def get_non_nan_indices(df):
            return df[~df.isna().any(axis=1)].index.to_numpy()

        if not self.syn_params:
            # No synchronization - generate independent indices
            if 'sf' not in idx_dict:
                idx_dict['sf'] = np.random.choice(len(self.sf_param_table), num_points)
            if 'tf' not in idx_dict:
                idx_dict['tf'] = np.random.choice(len(self.tf_param_table), num_points)
            if 'lnk' not in idx_dict:
                idx_dict['lnk'] = np.random.choice(len(self.lnk_param_table), num_points)
        else:
            # Get non-NaN indices for each table
            non_nan_indices = {}
            nan_indices = {}
            if 'sf' in self.syn_params:
                non_nan_indices['sf'] = get_non_nan_indices(self.sf_param_table)
                nan_indices['sf'] = self.sf_param_table[self.sf_param_table.isna().any(axis=1)].index.to_numpy()
            if 'tf' in self.syn_params:
                non_nan_indices['tf'] = get_non_nan_indices(self.tf_param_table)
                nan_indices['tf'] = self.tf_param_table[self.tf_param_table.isna().any(axis=1)].index.to_numpy()
            if 'lnk' in self.syn_params and self.lnk_param_table is not None:
                non_nan_indices['lnk'] = get_non_nan_indices(self.lnk_param_table)
                nan_indices['lnk'] = self.lnk_param_table[self.lnk_param_table.isna().any(axis=1)].index.to_numpy()

            # Sample base indices from the full range
            base_indices = np.random.choice(self.sync_table_length, num_points)

            # For each param, rewire indices that point to NaN rows
            for param in self.syn_params:
                idxs = base_indices.copy()
                nan_rows = nan_indices[param]
                valid_rows = non_nan_indices[param]
                # Find which indices are NaN
                for i, idx in enumerate(idxs):
                    if idx in nan_rows:
                        # Resample from valid rows for this table
                        idxs[i] = np.random.choice(valid_rows)
                idx_dict[param] = idxs
            
            # Add non-synchronized parameters
            if 'sf' not in self.syn_params:
                idx_dict['sf'] = np.random.choice(len(self.sf_param_table), num_points)
            if 'tf' not in self.syn_params:
                idx_dict['tf'] = np.random.choice(len(self.tf_param_table), num_points)
            if 'lnk' not in self.syn_params:
                idx_dict['lnk'] = np.random.choice(len(self.lnk_param_table), num_points)

        return idx_dict
    
    def get_additional_results(self, anti_alignment=1, sf_id_list_additional=None):
        """
        Always returns exactly 5 values for consistent unpacking.
        LNK parameters should be handled separately via the use_lnk_override flag.
        """
        points = self.grid_generator.generate_second_grid(anti_alignment=anti_alignment)
        idx_dict = self._generate_indices(points)

        lnk_params = sample_lnk_parameters(self.lnk_param_table, idx_dict.get('lnk'))
        
        # Create filters
        multi_opt_sf_center, multi_opt_sf_surround = self._create_multi_opt_sf(points, sf_id_list_additional, idx_list=idx_dict.get('sf'))
        tf = self._create_temporal_filter(idx_list=idx_dict.get('tf'))
        grid2value_mapping, map_func = self._get_grid_mapping(points)

        return multi_opt_sf_center, multi_opt_sf_surround, tf, grid2value_mapping, map_func, points, lnk_params

    def _get_grid_mapping(self, points):
        # Generate grid2value mapping and map function
        if self.grid_generate_method == 'closest':
            # Ensure `get_closest_indices` works with NumPy or PyTorch and output is a PyTorch tensor
            closest_indices = get_closest_indices(self.grid_centers, points)
            if isinstance(closest_indices, np.ndarray):
                closest_indices = torch.tensor(closest_indices, dtype=torch.long)
            grid2value_mapping = closest_indices
            map_func = map_to_fixed_grid_closest_batch

        elif self.grid_generate_method == 'decay':
            # Ensure `compute_distance_decay_matrix` works with NumPy or PyTorch and output is a PyTorch tensor
            decay_matrix = compute_distance_decay_matrix(self.grid_centers, points, self.tau)
            if isinstance(decay_matrix, np.ndarray):
                decay_matrix = torch.from_numpy(decay_matrix).float()
            grid2value_mapping = decay_matrix
            map_func = map_to_fixed_grid_decay_batch

        elif self.grid_generate_method == 'circle':
            # Compute the circular mask matrix
            mask_matrix = compute_circular_mask_matrix(self.grid_centers, points, self.mask_radius)

            # Convert to PyTorch tensor if needed
            if isinstance(mask_matrix, np.ndarray):
                mask_matrix = torch.from_numpy(mask_matrix).float()
            grid2value_mapping = mask_matrix
            map_func = map_to_fixed_grid_circle_batch  # Define the appropriate mapping function

        else:
            raise ValueError("Invalid grid_generate_method. Use 'closest' or 'decay'.")

        return grid2value_mapping, map_func

    def _create_multi_opt_sf(self, points, pids=None, idx_list=None):
        """Create multi-optical spatial filters"""
        multi_opt_sf = np.zeros((self.rgc_array_rf_size[0], self.rgc_array_rf_size[1], len(points)))
        multi_opt_sf_surround = None
        
        if self.use_lnk_override:
            multi_opt_sf_surround = np.zeros_like(multi_opt_sf)
        
        num_sim_data = len(self.sf_param_table)
        pid_list = None if pids is None else list(pids)
        
        for i, point in enumerate(points):
            # Select parameter index
            if idx_list is not None:
                pid_i = idx_list[i]
            elif pid_list is None:
                pid_i = np.random.randint(0, num_sim_data)
            else:
                pid_i = np.random.choice(pid_list)
                
            row = self.sf_param_table.iloc[pid_i]
            
            if self.use_lnk_override:
                # LNK model: simplified center and surround generation
                base_params = np.array([
                    point[1], point[0],  # position
                    row['sigma_x'] * self.sf_scalar,
                    row['sigma_y'] * self.sf_scalar,
                    row['theta'],
                    0, 1.0, 0, 0, 0  # bias=0, c_scale=1, no DOG surround
                ])
                
                # Generate and normalize center
                opt_sf_center = self._generate_and_normalize_sf(base_params, point)
                multi_opt_sf[:, :, i] = opt_sf_center
                
                # Generate surround with scaled sigmas
                surround_params = base_params.copy()
                surround_params[2] *= self.surround_sigma_ratio  # scale sigma_x
                surround_params[3] *= self.surround_sigma_ratio  # scale sigma_y
                opt_sf_surround = self._generate_and_normalize_sf(surround_params, point, 
                                                                  radius_scale=self.surround_sigma_ratio)
                multi_opt_sf_surround[:, :, i] = opt_sf_surround
                
                if i == 0:  # Log only once to avoid spam
                    logging.debug("LNK model: using separate center and surround SF generation")
                
            else:
                # Standard LN model
                s_scale = self._get_s_scale(row)
                sf_params = np.array([
                    point[1], point[0], 
                    row['sigma_x'] * self.sf_scalar, row['sigma_y'] * self.sf_scalar,
                    row['theta'], row['bias'], row['c_scale'], 
                    row['s_sigma_x'] * self.sf_scalar, row['s_sigma_y'] * self.sf_scalar, 
                    s_scale
                ])
                opt_sf = self._generate_and_normalize_sf(sf_params, point)
                multi_opt_sf[:, :, i] = opt_sf
        
        return multi_opt_sf, multi_opt_sf_surround

    def _generate_and_normalize_sf(self, sf_params, point, radius_scale=1.0):
        """Helper method to generate and normalize spatial filter"""
        opt_sf = gaussian_multi(sf_params, self.rgc_array_rf_size, 
                               self.num_gauss_example, self.is_rescale_diffgaussian)
        
        if self.is_rf_median_subtract:
            opt_sf -= np.median(opt_sf)
        
        # Normalize by absolute sum
        opt_sf = opt_sf / np.sum(np.abs(opt_sf))
        
        # Apply spatial constraints if specified
        if self.sf_constraint_method == 'circle':
            opt_sf = self._apply_circular_mask(opt_sf, point, self.sf_mask_radius * radius_scale)
        elif self.sf_constraint_method == 'threshold':
            threshold_value = np.percentile(opt_sf, self.sf_pixel_thr)
            opt_sf = np.where(opt_sf > threshold_value, 1, 0)
        
        return opt_sf

    def _apply_circular_mask(self, opt_sf, point, radius):
        """Apply circular mask to spatial filter"""
        rows, cols = np.ogrid[-opt_sf.shape[0]//2:opt_sf.shape[0]//2, 
                              -opt_sf.shape[1]//2:opt_sf.shape[1]//2]
        distance_from_center = np.sqrt((rows - point[0])**2 + (cols - point[1])**2)
        circular_mask = distance_from_center <= radius
        return np.where(circular_mask, opt_sf, 0)

    def _get_s_scale(self, row):
        """Get s_scale value based on configuration"""
        return row['s_scale'] if not self.set_s_scale else self.set_s_scale[0]

    def _create_temporal_filter(self, idx_list=None):
        if self.is_pixelized_tf:
            tf = np.zeros(self.temporal_filter_len)
            tf[-1] = 1 
        else:
            num_sim_data = len(self.tf_param_table)
            if idx_list is not None:
                # Return array of filters
                tf_arr = []
                for pid in idx_list:
                    row = self.tf_param_table.iloc[pid]
                    tf_params = np.array([row['sigma1'], row['sigma2'], row['mean1'], row['mean2'], row['amp1'], row['amp2'], row['offset']])
                    tf = gaussian_temporalfilter(self.temporal_filter_len, tf_params)
                    tf = tf-tf[0]
                    tf = tf / np.sum(np.abs(tf))
                    if self.is_reversed_tf:
                        tf = -tf
                    tf_arr.append(tf)
                return np.stack(tf_arr, axis=-1)
            else:
                pid = np.random.randint(0, num_sim_data)
                row = self.tf_param_table.iloc[pid]
                tf_params = np.array([row['sigma1'], row['sigma2'], row['mean1'], row['mean2'], row['amp1'], row['amp2'], row['offset']])
                tf = gaussian_temporalfilter(self.temporal_filter_len, tf_params)
                tf = tf-tf[0]
                tf = tf / np.sum(np.abs(tf))
                if self.is_reversed_tf:
                    tf = -tf
        return tf
    

class CricketMovie(Dataset):
    def __init__(self, num_samples, target_width, target_height, movie_generator, 
                 grid_size_fac=1, is_norm_coords=False, is_syn_mov_shown=False):
        """
        Args:
            num_samples (int): Total number of samples in the dataset.
            target_width (int): The target width used for further processing.
            target_height (int): The target height used for further processing.
            movie_generator: A pre-initialized movie generator that returns
                             (syn_movie, path, path_bg, scaling_factors, bg_image_name, image_id).
            grid_size_fac (float): Scaling factor for grid dimensions.
            is_norm_coords (bool): Whether to use normalization for path coordinates.
            is_syn_mov_shown (bool): Flag to determine the return structure.
        """
        self.num_samples = num_samples
        self.target_width = target_width
        self.target_height = target_height
        self.grid_size_fac = grid_size_fac
        self.grid_width = int(np.round(self.target_width * grid_size_fac))
        self.grid_height = int(np.round(self.target_height * grid_size_fac))
        self.movie_generator = movie_generator
        self.is_syn_mov_shown = is_syn_mov_shown
        
        if is_norm_coords:
            self.norm_path_fac = np.array([self.target_height, self.target_width]) / 2.0  
        else:
            self.norm_path_fac = 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate the raw movie and associated variables.
        syn_movie, path, path_bg, scaling_factors, bg_image_name, image_id = self.movie_generator.generate()
        if syn_movie.shape[1] != 2:
            syn_movie = syn_movie[:, 0:1, :, :]
        
        syn_movie = F.interpolate(syn_movie, size=(self.grid_height, self.grid_width), mode='area')
        # Create movie_sequence by reordering dimensions to mimic grid_values_sequence output.
        # Original default branch permutation: (time, 2nd dim, 1st dim) + extra dimension.
        # If syn_movie has shape (T, C, H, W) then movie_sequence becomes of shape (T, 1, W, H)
        
        # movie_sequence = syn_movie.permute(0, 1, 3, 2).unsqueeze(1)

        movie_sequence = syn_movie
        
        # Determine the number of time steps based on syn_movie.
        time_steps = syn_movie.shape[0]
        
        # Process the path and background path.
        # Use the last `time_steps` rows of path and path_bg and scale by norm_path_fac.
        path = path[-time_steps:, :] / self.norm_path_fac
        path_bg = path_bg[-time_steps:, :] / self.norm_path_fac

        # Return outputs preserving the original structure except that weighted_coords is removed.
        if self.is_syn_mov_shown:
            return movie_sequence, path, path_bg, syn_movie, scaling_factors, bg_image_name, image_id
        else:
            return movie_sequence, torch.tensor(path, dtype=torch.float32), torch.tensor(path_bg, dtype=torch.float32)


def test_multi_temporal_filters():
    """
    Test function to demonstrate the new multi-temporal filter functionality.
    This shows how to use different temporal filters for different RGC cells.
    """
    print("Testing multi-temporal filter implementation...")
    
    # Create dummy data
    T, H, W = 20, 32, 32
    num_rgcs = 5
    
    # Create dummy movie
    movie = torch.randn(T, H, W)
    
    # Create dummy spatial filters [W, H, N]
    sf = torch.randn(W, H, num_rgcs)
    
    # Test 1: Single temporal filter for all RGCs (backward compatibility)
    tf_single = np.random.randn(10)  # Single temporal filter
    tf_multi_single = create_multiple_temporal_filters(tf_single, num_rgcs, variation_std=0.0)
    tf_tensor_single = torch.from_numpy(tf_multi_single).float().view(num_rgcs, 1, -1)
    
    print(f"Single TF replicated - Shape: {tf_tensor_single.shape}")
    
    # Test 2: Different temporal filters for each RGC
    tf_multi_varied = create_multiple_temporal_filters(tf_single, num_rgcs, variation_std=0.1)
    tf_tensor_varied = torch.from_numpy(tf_multi_varied).float().view(num_rgcs, 1, -1)
    
    print(f"Varied TFs - Shape: {tf_tensor_varied.shape}")
    
    # Test the convolution operation
    sf_frame = torch.einsum('whn,thw->nt', sf, movie)  # [N, T]
    sf_frame = sf_frame.unsqueeze(0)  # [1, N, T]
    
    # Apply temporal filtering
    result_single = F.conv1d(sf_frame, tf_tensor_single, stride=1, padding=0, groups=num_rgcs)
    result_varied = F.conv1d(sf_frame, tf_tensor_varied, stride=1, padding=0, groups=num_rgcs)
    
    print(f"Result single TF - Shape: {result_single.shape}")
    print(f"Result varied TF - Shape: {result_varied.shape}")
    
    # Check that different filters produce different results
    are_different = not torch.allclose(result_single, result_varied, atol=1e-6)
    print(f"Single vs varied TF produce different results: {are_different}")
    
    print("Multi-temporal filter test completed successfully!")


if __name__ == "__main__":
    test_multi_temporal_filters()