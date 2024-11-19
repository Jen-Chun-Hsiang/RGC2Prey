import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F

from rgc_rf import map_to_fixed_grid_decay_batch, gaussian_multi, gaussian_temporalfilter, get_closest_indices, compute_distance_decay_matrix
from rgc_rf import map_to_fixed_grid_closest, map_to_fixed_grid_decay, create_hexagonal_centers, precompute_grid_centers
from utils.utils import get_random_file_path



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
                x += velocity * np.cos(angle)
                y += velocity * np.sin(angle)
                velocity_history[i] = velocity

        else:
            angle = np.random.uniform(0, 2 * np.pi)
            velocity_history[i] = 0
            if np.random.rand() > prob_stay:
                moving = True  # Switch to moving
                # Preserve momentum by reusing the last velocity and angle
                velocity = initial_velocity * momentum_decay + np.random.uniform(-velocity_randomness, velocity_randomness)

        # Check boundaries (with center at (0, 0))
        if not (-half_boundary[0] <= x <= half_boundary[0] and -half_boundary[1] <= y <= half_boundary[1]):
            break  # Stop if out of bounds

        path[step_count] = [x, y]
        step_count += 1
    
    # Trim path and velocity_history to actual steps taken
    path = path[:step_count]
    velocity_history = velocity_history[:step_count]

    return path, velocity_history


class Cricket2RGCs(Dataset):
    def __init__(self, num_samples, num_frames, crop_size, boundary_size, center_ratio, max_steps, num_ext, initial_velocity,
                 bottom_img_folder, top_img_folder, multi_opt_sf, tf, map_func, grid2value_mapping, target_width, target_height,
                 movie_generator):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.boundary_size = boundary_size
        self.center_ratio = center_ratio
        self.max_steps = max_steps
        self.num_ext = num_ext
        self.initial_velocity = initial_velocity
        self.bottom_img_folder = bottom_img_folder
        self.top_img_folder = top_img_folder
        self.multi_opt_sf = multi_opt_sf
        self.tf = tf.view(1, 1, -1)
        self.map_func = map_func
        self.grid2value_mapping = grid2value_mapping
        self.target_width = target_width
        self.target_height = target_height
        # Accept pre-initialized movie generator
        self.movie_generator = movie_generator

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        syn_movie, path, path_bg = self.generator.generate()
        syn_movie_batch = torch.tensor(syn_movie.transpose(2, 0, 1))  # Shape: (time_steps, height, width)
        sf_frame_batch = torch.einsum('whn,thw->tn', self.multi_opt_sf, syn_movie_batch)
        sf_frame_batch = sf_frame_batch.unsqueeze(0).transpose(1, 2)  # Shape: (1, num_points, time_steps)
        rgc_time = F.conv1d(sf_frame_batch, self.tf, stride=1, padding=0)  # Shape: (1, num_points, time_steps')
        rgc_time = rgc_time.squeeze(0).transpose(0, 1)  # Shape: (time_steps', num_points)

        grid_values_sequence = map_to_fixed_grid_decay_batch(
            rgc_time,  # Shape: (time_steps', num_points)
            self.grid2value_mapping,  # Shape: (num_points, target_width * target_height)
            self.target_width,
            self.target_height
        ) 

        return grid_values_sequence, torch.tensor(path, dtype=torch.float32), torch.tensor(path_bg, dtype=torch.float32)
    

class SynMovieGenerator:
    def __init__(self, num_frames, crop_size, boundary_size, center_ratio, max_steps, num_ext, initial_velocity,
                 bottom_img_folder, top_img_folder, scale_factor=1.0):
        """
        Initializes the SynMovieGenerator with configuration parameters.

        Parameters:
        - num_frames (int): Number of frames in the synthetic movie.
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
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.boundary_size = boundary_size
        self.center_ratio = center_ratio
        self.max_steps = max_steps
        self.num_ext = num_ext
        self.initial_velocity = initial_velocity
        self.bottom_img_folder = bottom_img_folder
        self.top_img_folder = top_img_folder
        self.scale_factor = scale_factor

    def generate(self):
        """
        Generates a synthetic movie, path, and path_bg.

        Returns:
        - syn_movie (np.ndarray): 3D array of shape (height, width, time_steps).
        - path (np.ndarray): 2D array of object positions (time_steps, 2).
        - path_bg (np.ndarray): 2D array of background positions (time_steps, 2).
        """
        path, _ = random_movement(self.boundary_size, self.center_ratio, self.max_steps, prob_stay=0.95, prob_mov=0.975,
                                  initial_velocity=self.initial_velocity, momentum_decay=0.95, velocity_randomness=0.02)
        path_bg, _ = random_movement(self.boundary_size, self.center_ratio, self.max_steps, prob_stay=0.98, prob_mov=0.98,
                                     initial_velocity=self.initial_velocity, momentum_decay=0.9, velocity_randomness=0.01)

        # Extend static frames at the beginning
        path = np.vstack((np.repeat(path[0:1, :], self.num_ext, axis=0), path))
        path_bg = np.vstack((np.repeat(path_bg[0:1, :], self.num_ext, axis=0), path_bg))

        bottom_img_path = get_random_file_path(self.bottom_img_folder)
        top_img_path = get_random_file_path(self.top_img_folder)

        # Convert paths to numpy arrays
        top_img_positions = path.round().astype(int)
        bottom_img_positions = path_bg.round().astype(int)

        # Generate the batch of images
        syn_movie = synthesize_image_with_params_batch(
            bottom_img_path, top_img_path, top_img_positions, bottom_img_positions,
            self.scale_factor, self.crop_size, alpha=1.0
        )

        syn_movie = syn_movie[:, 1, :, :]  # Extract green channel from all image

        return syn_movie, path, path_bg
    


def synthesize_image_with_params_batch(bottom_img_path, top_img_path, top_img_positions, bottom_img_positions,
                                       scale_factor, crop_size, alpha=1.0):
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
    top_img = Image.open(top_img_path).convert("RGBA")
    top_img = scale_image(top_img, scale_factor)

    # Convert to PyTorch tensors
    bottom_tensor = T.ToTensor()(bottom_img)
    top_tensor = T.ToTensor()(top_img)

    # Get dimensions
    bottom_h, bottom_w = bottom_img.size
    top_h, top_w = top_img.size
    batch_size = len(top_img_positions)

    # Prepare a batch tensor for the final output
    syn_images = torch.zeros((batch_size, 3, crop_size[1], crop_size[0]))

    for i in range(batch_size):
        # Compute relative position
        relative_pos = top_img_positions[i] - bottom_img_positions[i]
        fill_h = (bottom_h - top_h) // 2
        fill_w = (bottom_w - top_w) // 2

        # Clone the bottom image tensor
        final_tensor = bottom_tensor.clone()

        # Overlay the top image
        for c in range(3):  # Iterate over RGB channels
            final_tensor[c, fill_h + relative_pos[1]:fill_h + relative_pos[1] + top_h,
                         fill_w + relative_pos[0]:fill_w + relative_pos[0] + top_w] = (
                top_tensor[c, :, :] * top_tensor[3, :, :] * alpha +
                final_tensor[c, fill_h + relative_pos[1]:fill_h + relative_pos[1] + top_h,
                             fill_w + relative_pos[0]:fill_w + relative_pos[0] + top_w] * (1 - top_tensor[3, :, :] * alpha)
            )

        # Convert to PIL and crop
        final_img = T.ToPILImage()(final_tensor)
        cropped_img = crop_image(final_img, crop_size, bottom_img_positions[i])
        syn_images[i] = T.ToTensor()(cropped_img)

    return syn_images


class RGCrfArray:
    def __init__(self, sf_param_table, tf_param_table, rgc_array_rf_size, xlim, ylim, target_num_centers, sf_scalar,
                 grid_generate_method, tau=None, rand_seed=42, num_gauss_example=1, is_pixelized_rf=False, sf_pixel_thr=99.7):
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
        self.grid_generate_method = grid_generate_method
        self.tau = tau
        self.rand_seed = rand_seed
        self.num_gauss_example = num_gauss_example
        self.target_num_centers = target_num_centers
        self.is_pixelized_rf = is_pixelized_rf
        self.sf_pixel_thr = sf_pixel_thr

        # Set random seed
        self.np_rng = np.random.default_rng(self.rand_seed)
        self.rng = random.Random(self.rand_seed)
        torch.manual_seed(self.rand_seed)
        

        # Generate points and grid centers
        self.points = create_hexagonal_centers(xlim, ylim, target_num_centers=self.target_num_centers, rand_seed=self.rand_seed)
        self.target_height = xlim[1] - xlim[0]
        self.target_width = ylim[1] - ylim[0]
        self.grid_centers = precompute_grid_centers(self.target_height, self.target_width, x_min=xlim[0], x_max=xlim[1],
                                            y_min=ylim[0], y_max=ylim[1])

        # Generate grid2value mapping and map function
        if grid_generate_method == 'closest':
            self.grid2value_mapping = get_closest_indices(self.grid_centers, self.points)
            self.map_func = map_to_fixed_grid_closest
        elif grid_generate_method == 'decay':
            self.grid2value_mapping = compute_distance_decay_matrix(self.grid_centers, self.points, self.tau)
            self.map_func = map_to_fixed_grid_decay
        else:
            raise ValueError("Invalid grid_generate_method. Use 'closest' or 'decay'.")

        # Generate multi_opt_sf and tf arrays
        self.multi_opt_sf = self._create_multi_opt_sf()
        self.tf = self._create_temporal_filter(len(self.tf_param_table))


    def _create_multi_opt_sf(self):
        # Create multi-optical spatial filters
        multi_opt_sf = np.zeros((self.rgc_array_rf_size[0], self.rgc_array_rf_size[1], len(self.points)))
        num_sim_data = len(self.sf_param_table)
        pid = self.rng.randint(0, num_sim_data - 1)
        row = self.sf_param_table.iloc[pid]
        for i, point in enumerate(self.points):
            sf_params = np.array([
                point[1], point[0], row['sigma_x'] * self.sf_scalar, row['sigma_y'] * self.sf_scalar,
                row['theta'], row['bias'], row['c_scale'], row['s_sigma_x'] * self.sf_scalar,
                row['s_sigma_y'] * self.sf_scalar, row['s_scale']
            ])
            opt_sf = gaussian_multi(sf_params, self.rgc_array_rf_size, self.num_gauss_example)
            opt_sf -= np.median(opt_sf)  

            if self.is_pixelized_rf:
                threshold_value = np.percentile(opt_sf, self.sf_pixel_thr)
                opt_sf = np.where(opt_sf > threshold_value, 1, 0)
            multi_opt_sf[:, :, i] = opt_sf
        return multi_opt_sf


    def _create_temporal_filter(self, temporal_filter_len):
        if self.is_pixelized_rf:
            tf = np.zeros(temporal_filter_len)
            tf[-1] = 1 
        else:
            num_sim_data = len(self.tf_param_table)
            pid = self.rng.randint(0, num_sim_data - 1)
            row = self.tf_param_table.iloc[pid]
            tf_params = np.array([row['sigma1'], row['sigma2'], row['mean1'], row['mean2'], row['amp1'], row['amp2'], row['offset']])
            tf = gaussian_temporalfilter(temporal_filter_len, tf_params)
            tf = tf-tf[0]
        return tf

    def get_results(self):
        return self.multi_opt_sf, self.tf, self.grid2value_mapping, self.map_func

