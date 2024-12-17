import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F

from datasets.rgc_rf import map_to_fixed_grid_decay_batch, gaussian_multi, gaussian_temporalfilter, get_closest_indices, compute_distance_decay_matrix
from datasets.rgc_rf import map_to_fixed_grid_closest_batch, create_hexagonal_centers, precompute_grid_centers, compute_circular_mask_matrix
from datasets.rgc_rf import map_to_fixed_grid_circle_batch
from utils.utils import get_random_file_path, get_image_number, load_mat_to_dataframe



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
    def __init__(self, num_samples, multi_opt_sf, tf, map_func, grid2value_mapping, target_width, target_height,
                 movie_generator, grid_size_fac=1, is_norm_coords=False, is_syn_mov_shown=False, fr2spikes=False):
        self.num_samples = num_samples
        self.multi_opt_sf = torch.from_numpy(multi_opt_sf).float()
        self.tf = torch.from_numpy(tf.copy()).float().view(1, 1, -1)
        self.map_func = map_func
        self.grid2value_mapping = grid2value_mapping
        self.target_width = target_width
        self.target_height = target_height
        self.grid_size_fac = grid_size_fac
        self.grid_width = int(np.round(self.target_width*grid_size_fac))
        self.grid_height = int(np.round(self.target_height*grid_size_fac))
        # Accept pre-initialized movie generator
        self.movie_generator = movie_generator
        if is_norm_coords:
            self.norm_path_fac = np.array([self.target_height, self.target_width]) / 2  
        else:
            self.norm_path_fac = 1
        self.is_syn_mov_shown = is_syn_mov_shown  # cannot be used for training
        self.fr2spikes = fr2spikes


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        syn_movie, path, path_bg, scaling_factors = self.movie_generator.generate()
        sf_frame = torch.einsum('whn,thw->nt', self.multi_opt_sf, syn_movie)
        sf_frame = sf_frame.unsqueeze(0) 
        tf = np.repeat(self.tf, sf_frame.shape[1], axis=0)
        rgc_time = F.conv1d(sf_frame, tf, stride=1, padding=0, groups=sf_frame.shape[1]).squeeze()

        if self.fr2spikes:
            rgc_time = torch.poisson(torch.clamp_min(rgc_time, 0))

        grid_values_sequence = self.map_func(
            rgc_time,  # Shape: (time_steps', num_points)
            self.grid2value_mapping,  # Shape: (num_points, target_width * target_height)
            self.grid_width,
            self.grid_height
        ) 
        
        path = path[-rgc_time.shape[1]:, :]/self.norm_path_fac
        path_bg = path_bg[-rgc_time.shape[1]:, :]/self.norm_path_fac    

        if self.is_syn_mov_shown:
            return grid_values_sequence.permute(0, 2, 1).unsqueeze(1), path, path_bg, syn_movie, scaling_factors
    
        else:
            return grid_values_sequence.permute(0, 2, 1).unsqueeze(1), torch.tensor(path, dtype=torch.float32), torch.tensor(path_bg, dtype=torch.float32)
    

class SynMovieGenerator:
    def __init__(self, top_img_folder, bottom_img_folder, crop_size, boundary_size, center_ratio, max_steps=200, prob_stay_ob=0.95, 
                 prob_mov_ob=0.975, prob_stay_bg=0.95, prob_mov_bg=0.975,num_ext=50, initial_velocity=6, momentum_decay_ob=0.95, 
                 momentum_decay_bg=0.5, scale_factor=1.0, velocity_randomness_ob=0.02, velocity_randomness_bg=0.01, angle_range_ob=0.5, angle_range_bg=0.25, coord_mat_file=None, 
                correction_direction=1, is_reverse_xy=False, start_scaling=1, end_scaling=2, dynamic_scaling=0):
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
        self.boundary_size = boundary_size
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
        top_img_positions = path.round().astype(int)
        bottom_img_positions = path_bg.round().astype(int)

        start_scaling, end_scaling = self._modify_scaling()
        scaling_factors = calculate_scaling_factors(bottom_img_positions, start_scaling=start_scaling, end_scaling=end_scaling)
        # Generate the batch of images
        syn_movie = synthesize_image_with_params_batch(
            bottom_img_path, top_img_path, top_img_positions, bottom_img_positions,
            scaling_factors, self.crop_size, alpha=1.0
        )
        
        # Correct for the cricket head position
        image_id = get_image_number(top_img_path)

        if self.is_reverse_xy:
            coord_correction = np.array([self.coord_dic[image_id]['coord_y'], self.coord_dic[image_id]['coord_x']])
        else:
            coord_correction = np.array([self.coord_dic[image_id]['coord_x'], self.coord_dic[image_id]['coord_y']])
        scaling_factors = np.array(scaling_factors) 
        scaled_coord_corrections = coord_correction[np.newaxis, :] * scaling_factors[:, np.newaxis]
        path = path - self.correction_direction*scaled_coord_corrections

        return syn_movie[:, 1, :, :], path, path_bg, scaling_factors
    


def synthesize_image_with_params_batch(bottom_img_path, top_img_path, top_img_positions, bottom_img_positions,
                                       scaling_factors, crop_size, alpha=1.0):
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
    syn_images = torch.zeros((batch_size, 3, crop_size[1], crop_size[0]))

    # Variables to cache the last scaling factor and scaled image
    last_scaling_factor = None
    cached_scaled_image = None

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
        fill_h = (bottom_h - top_h) // 2
        fill_w = (bottom_w - top_w) // 2

        # Compute relative position
        relative_pos = top_img_positions[i] - bottom_img_positions[i]

        # Clone the bottom image tensor
        final_tensor = bottom_tensor.clone()

        # Overlay the top image (vectorized for all RGB channels)
        final_tensor[:3,
                     fill_h + relative_pos[1]:fill_h + relative_pos[1] + top_h,
                     fill_w + relative_pos[0]:fill_w + relative_pos[0] + top_w] = (
            top_tensor[:3, :, :] * top_tensor[3:4, :, :] * alpha +
            final_tensor[:3,
                         fill_h + relative_pos[1]:fill_h + relative_pos[1] + top_h,
                         fill_w + relative_pos[0]:fill_w + relative_pos[0] + top_w] * (1 - top_tensor[3:4, :, :] * alpha)
        )

        # Convert to PIL and crop
        final_img = T.ToPILImage()(final_tensor)
        cropped_img = crop_image(final_img, crop_size, bottom_img_positions[i])
        if cropped_img.mode == "RGBA":
            cropped_img = cropped_img.convert("RGB")
        syn_images[i] = T.ToTensor()(cropped_img)

    return syn_images


class RGCrfArray:
    def __init__(self, sf_param_table, tf_param_table, rgc_array_rf_size, xlim, ylim, target_num_centers, sf_scalar,
                 grid_generate_method, tau=None, mask_radius=None, rand_seed=42, num_gauss_example=1, sf_mask_radius=35, 
                 sf_pixel_thr=99.7, sf_constraint_method=None, temporal_filter_len=50, grid_size_fac=0.5, is_pixelized_tf=False, 
                 set_s_scale=[], is_rf_median_subtract=True, is_rescale_diffgaussian=True):
        """
        Args:
            sf_param_table (DataFrame): Table of spatial frequency parameters.
            tf_param_table (DataFrame): Table of temporal filter parameters.
            rgc_array_rf_size (tuple): Size of the receptive field (height, width).
            xlim (tuple): x-axis limits for grid centers.
            ylim (tuple): y-axis limits for grid centers.
            target_num_centers (int): Number of target centers to generate.
            sf_scalar (float): Scaling factor for spatial frequency parameters.
            grid_generate_method (str): Method for grid generation ('closest' or 'decay').
            tau (float, optional): Decay factor for the 'decay' method.
            rand_seed (int): Random seed for reproducibility.
        """
        self.sf_param_table = sf_param_table
        self.tf_param_table = tf_param_table
        self.rgc_array_rf_size = rgc_array_rf_size
        self.sf_scalar = sf_scalar
        self.temporal_filter_len = temporal_filter_len
        self.grid_generate_method = grid_generate_method
        self.tau = tau
        self.sf_mask_radius = sf_mask_radius
        self.sf_constraint_method = sf_constraint_method
        self.mask_radius = mask_radius
        self.rand_seed = rand_seed
        self.num_gauss_example = num_gauss_example
        self.target_num_centers = target_num_centers
        self.sf_pixel_thr = sf_pixel_thr
        self.grid_size_fac = grid_size_fac
        self.is_pixelized_tf = is_pixelized_tf
        self.set_s_scale = set_s_scale
        self.is_rf_median_subtract = is_rf_median_subtract
        self.is_rescale_diffgaussian=is_rescale_diffgaussian

        # Set random seed
        self.np_rng = np.random.default_rng(self.rand_seed)
        self.rng = random.Random(self.rand_seed)
        torch.manual_seed(self.rand_seed)
        

        # Generate points and grid centers
        self.points = create_hexagonal_centers(xlim, ylim, target_num_centers=self.target_num_centers, rand_seed=self.rand_seed)
        self.target_height = xlim[1] - xlim[0]
        self.target_width = ylim[1] - ylim[0]
        self.grid_centers = precompute_grid_centers(self.target_height, self.target_width, x_min=xlim[0], x_max=xlim[1],
                                            y_min=ylim[0], y_max=ylim[1], grid_size_fac=grid_size_fac)

        # Generate grid2value mapping and map function
        if grid_generate_method == 'closest':
            # Ensure `get_closest_indices` works with NumPy or PyTorch and output is a PyTorch tensor
            closest_indices = get_closest_indices(self.grid_centers, self.points)
            if isinstance(closest_indices, np.ndarray):
                closest_indices = torch.tensor(closest_indices, dtype=torch.long)
            self.grid2value_mapping = closest_indices
            self.map_func = map_to_fixed_grid_closest_batch

        elif grid_generate_method == 'decay':
            # Ensure `compute_distance_decay_matrix` works with NumPy or PyTorch and output is a PyTorch tensor
            decay_matrix = compute_distance_decay_matrix(self.grid_centers, self.points, self.tau)
            if isinstance(decay_matrix, np.ndarray):
                decay_matrix = torch.from_numpy(decay_matrix).float()
            self.grid2value_mapping = decay_matrix
            self.map_func = map_to_fixed_grid_decay_batch

        elif grid_generate_method == 'circle':
            # Compute the circular mask matrix
            mask_matrix = compute_circular_mask_matrix(self.grid_centers, self.points, self.mask_radius)

            # Convert to PyTorch tensor if needed
            if isinstance(mask_matrix, np.ndarray):
                mask_matrix = torch.from_numpy(mask_matrix).float()

            self.grid2value_mapping = mask_matrix
            self.map_func = map_to_fixed_grid_circle_batch  # Define the appropriate mapping function

        else:
            raise ValueError("Invalid grid_generate_method. Use 'closest' or 'decay'.")


        # Generate multi_opt_sf and tf arrays
        self.multi_opt_sf = self._create_multi_opt_sf()
        self.tf = self._create_temporal_filter()


    def _create_multi_opt_sf(self):
        # Create multi-optical spatial filters
        multi_opt_sf = np.zeros((self.rgc_array_rf_size[0], self.rgc_array_rf_size[1], len(self.points)))
        num_sim_data = len(self.sf_param_table)
        pid = self.rng.randint(0, num_sim_data - 1)
        row = self.sf_param_table.iloc[pid]
        s_scale = row['s_scale'] if not self.set_s_scale else self.set_s_scale[0]
        # print(f's_scale: {s_scale}')
        # print(f's_scale type: {type(s_scale)}')
        for i, point in enumerate(self.points):
            sf_params = np.array([
                point[1], point[0], row['sigma_x'] * self.sf_scalar, row['sigma_y'] * self.sf_scalar,
                row['theta'], row['bias'], row['c_scale'], row['s_sigma_x'] * self.sf_scalar,
                row['s_sigma_y'] * self.sf_scalar, s_scale
            ])
            opt_sf = gaussian_multi(sf_params, self.rgc_array_rf_size, self.num_gauss_example, self.is_rescale_diffgaussian)
            if self.is_rf_median_subtract:
                opt_sf -= np.median(opt_sf)  
            opt_sf = opt_sf / np.sum(np.abs(opt_sf))

            if self.sf_constraint_method == 'circle':
                rows, cols = np.ogrid[-opt_sf.shape[0] // 2:opt_sf.shape[0] // 2, 
                       -opt_sf.shape[1] // 2:opt_sf.shape[1] // 2]
                distance_from_center = np.sqrt((rows - point[0])**2 + (cols - point[1])**2)
                circular_mask = distance_from_center <= self.sf_mask_radius
                opt_sf = np.where(circular_mask, opt_sf, 0)

            if self.sf_constraint_method == 'threshold':
                threshold_value = np.percentile(opt_sf, self.sf_pixel_thr)
                opt_sf = np.where(opt_sf > threshold_value, 1, 0)

            multi_opt_sf[:, :, i] = opt_sf
            
        return multi_opt_sf


    def _create_temporal_filter(self):
        
        if self.is_pixelized_tf:
            tf = np.zeros(self.temporal_filter_len)
            tf[-1] = 1 
        else:
            num_sim_data = len(self.tf_param_table)
            pid = self.rng.randint(0, num_sim_data - 1)
            row = self.tf_param_table.iloc[pid]
            tf_params = np.array([row['sigma1'], row['sigma2'], row['mean1'], row['mean2'], row['amp1'], row['amp2'], row['offset']])
            tf = gaussian_temporalfilter(self.temporal_filter_len, tf_params)
            tf = tf-tf[0]
            tf = tf / np.sum(np.abs(tf))
        return tf

    def get_results(self):
        return self.multi_opt_sf, self.tf, self.grid2value_mapping, self.map_func

