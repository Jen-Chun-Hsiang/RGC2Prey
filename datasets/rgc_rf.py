import numpy as np
import torch
from scipy.stats import norm


def gaussmf_python(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def gaussian_temporalfilter(n, OptW):
    # Generate x as 1:n (inclusive)
    x = np.arange(1, n + 1)
    
    # Extract parameters from OptW
    sigma1, sigma2 = OptW[0], OptW[1]
    mean1, mean2 = OptW[2], OptW[3]
    amplitude1, amplitude2 = OptW[4], OptW[5]
    offset = OptW[6]
    
    # Compute the temporal filter
    # tf = (norm.pdf(x, loc=mean1, scale=sigma1) * amplitude1 -
    #       norm.pdf(x, loc=mean2, scale=sigma2) * amplitude2) + offset
    tf = (gaussmf_python(x, mean1, sigma1) * amplitude1 -
          gaussmf_python(x, mean2, sigma2) * amplitude2) + offset
    
    return tf

def gaussian2d(x, y, params, is_rescale_diffgaussian=True):
    # Ensure params is a numpy array for element-wise operations
    x_mean, y_mean, sigma_x, sigma_y, theta, bias, c_scale, s_sigma_x, s_sigma_y, s_scale = params

    # Compute rotated coordinates
    x_rot = (x - x_mean) * np.cos(theta) + (y - y_mean) * np.sin(theta)
    y_rot = -(x - x_mean) * np.sin(theta) + (y - y_mean) * np.cos(theta)

    if is_rescale_diffgaussian:
        # Calculate c Gaussian
        c_gaussian = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))
        c_sum = np.sum(np.abs(c_scale * c_gaussian))  # Sum of absolute values scaled by c_scale

        # Calculate s Gaussian
        s_gaussian = np.exp(-(x_rot**2 / (2 * s_sigma_x**2) + y_rot**2 / (2 * s_sigma_y**2)))
        s_sum = np.sum(np.abs(s_scale * s_gaussian))  # Sum of absolute values scaled by s_scale

        # Adjust s_scale to ensure it matches the ratio relative to c_scale
        if s_sum != 0:  # Avoid division by zero
            s_scale = s_scale * (c_sum / s_sum)

        # Recompute the Gaussian function with corrected s_scale
        z = (c_scale * c_gaussian +
            s_scale * s_gaussian + bias)
    else:
        # Calculate the Gaussian function
        z = (c_scale * np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))) +
            s_scale * np.exp(-(x_rot**2 / (2 * s_sigma_x**2) + y_rot**2 / (2 * s_sigma_y**2))) + bias)
    return z


def gaussian_multi(params, image_size, num_gauss, is_rescale_diffgaussian):
    # Create a meshgrid for the image dimensions based on image_size
    height, width = image_size
    grid_x = np.linspace(-height//2, height//2, height)
    grid_y = np.linspace(-width//2, width//2, width)
    X, Y = np.meshgrid(grid_y, grid_x)

    # Reshape params into the number of Gaussians, assuming each row represents one Gaussian's parameters
    params = np.reshape(np.array(params), (-1, num_gauss)).T

    # Initialize the Gaussian model with the first set of parameters
    gaussian_model = gaussian2d(X, Y, params[0], is_rescale_diffgaussian)
    for i in range(1, num_gauss):
        gaussian_model += gaussian2d(X, Y, params[i], is_rescale_diffgaussian)

    return gaussian_model


    grid_x = np.linspace(x_min, x_max, target_height)
    grid_y = np.linspace(y_min, y_max, target_width)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing='ij')
    grid_centers = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)


def create_hexagonal_centers(xlim, ylim, target_num_centers, max_iterations=100, noise_level=0.3, rand_seed=None, num_positions=None, position_indices=None):
    
    # Save the current random state
    current_state = np.random.get_state()

    if rand_seed is not None:
        np.random.seed(rand_seed)

    x_min, x_max = xlim
    y_min, y_max = ylim
    x_range = x_max - x_min
    y_range = y_max - y_min

    approximate_area = x_range * y_range
    approximate_cell_area = approximate_area / target_num_centers
    side_length = np.sqrt(approximate_cell_area / (3 * np.sqrt(3) / 2))

    dx = side_length * np.sqrt(3)
    dy = side_length * 1.5

    cols = int(np.ceil(x_range / dx))
    rows = int(np.ceil(y_range / dy))

    def generate_points_with_noise(dx, dy, offset_x, offset_y, noise_level):
        points = []
        for row in range(rows):
            for col in range(cols):
                x = col * dx + x_min - offset_x
                y = row * dy + y_min - offset_y
                if row % 2 == 1:
                    x += dx / 2

                x += (np.random.rand() - 0.5) * 2 * noise_level * dx
                y += (np.random.rand() - 0.5) * 2 * noise_level * dy

                if x_min <= x < x_max and y_min <= y < y_max:
                    points.append((x, y))
        return np.array(points)

    offset_x = (cols * dx - x_range) / 2
    offset_y = (rows * dy - y_range) / 2
    points = generate_points_with_noise(dx, dy, offset_x, offset_y, noise_level)

    for _ in range(max_iterations):
        if len(points) > target_num_centers:
            dx *= 1.01
            dy *= 1.01
        else:
            dx *= 0.99
            dy *= 0.99

        cols = int(np.ceil(x_range / dx))
        rows = int(np.ceil(y_range / dy))
        offset_x = (cols * dx - x_range) / 2
        offset_y = (rows * dy - y_range) / 2
        points = generate_points_with_noise(dx, dy, offset_x, offset_y, noise_level)

        if abs(len(points) - target_num_centers) <= target_num_centers * 0.05:
            break

    if position_indices is not None:
        points = points[position_indices[position_indices < len(points)]]
    elif num_positions is not None and num_positions < len(points):
        indices = np.random.choice(len(points), num_positions, replace=False)
        points = points[indices]

    # Restore the previous random state
    np.random.set_state(current_state)

    return points


def get_closest_indices(grid_centers, coords):
    """
    Finds the closest coordinate index for each grid center.

    Parameters:
    - grid_centers (np.ndarray): Array of shape (M, 2), where M is the number of grid centers.
    - coords (np.ndarray): Array of shape (N, 2), where N is the number of coordinates.

    Returns:
    - closest_points (np.ndarray): Array of shape (M,) with the index of the closest coordinate
      for each grid center.
    """
    # Calculate the pairwise distances between each grid center and each coordinate
    distances = np.linalg.norm(grid_centers[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    
    # Find the index of the closest coordinate for each grid center
    closest_points = distances.argmin(axis=1)
    
    return closest_points


def map_to_fixed_grid_closest(values, closest_points, target_width, target_height):
    """
    Maps each grid cell to the value of the closest coordinate using precomputed indices.

    Parameters:
    - values (np.ndarray): Array of shape (N,) containing values for each coordinate.
    - closest_points (np.ndarray): Array of shape (M,) with the index of the closest coordinate
      for each grid center.
    - target_width (int): Width of the target grid.
    - target_height (int): Height of the target grid.

    Returns:
    - grid_values (np.ndarray): 2D array of shape (target_height, target_width) with values
      mapped based on the closest coordinates.
    """
    # Assign the value of the closest coordinate to each grid cell
    grid_values = values[closest_points].reshape(target_height, target_width)
    
    return grid_values

def map_to_fixed_grid_closest_batch(values, closest_points, target_width, target_height):
    """
    Maps a batch of grid cells to the values of the closest coordinates using precomputed indices.

    Parameters:
    - values (np.ndarray or torch.Tensor): Array of shape (T, N), where T is the batch size
      (time steps) and N is the number of input points (coordinates).
    - closest_points (np.ndarray): Array of shape (M,) with the index of the closest coordinate
      for each grid center. Assumes the same closest points apply to all batches.
    - target_width (int): Width of the target grid.
    - target_height (int): Height of the target grid.

    Returns:
    - grid_values_batch (np.ndarray or torch.Tensor): A 3D array of shape (T, target_height, target_width),
      with values mapped to each grid cell based on the closest coordinates for the entire batch.
    """
    # Index the closest values for each batch
    # values: (T, N), closest_points: (M,)
    print(f'values shape: {values.shape}')
    print(f'closest_points shape: {closest_points.shape}')
    grid_values = values[closest_points, :].T  # Shape: (T, M)

    # Reshape to (T, target_height, target_width)
    grid_values_batch = grid_values.view(-1, target_height, target_width)  # Shape: (T, H, W)

    return grid_values_batch


def generate_random_samples(number_samples, x_range=(0, 1), y_range=(0, 1)):
    """
    Generates random (x, y) coordinates and values within specified ranges.

    Parameters:
    - number_samples (int): Number of random samples to generate.
    - x_range (tuple): Range (min, max) for x coordinates.
    - y_range (tuple): Range (min, max) for y coordinates.

    Returns:
    - coords (np.ndarray): Array of shape (number_samples, 2) with random coordinates.
    - values (np.ndarray): Array of shape (number_samples,) with random values.
    """
    x_coords = np.random.rand(number_samples) * (x_range[1] - x_range[0]) + x_range[0]
    y_coords = np.random.rand(number_samples) * (y_range[1] - y_range[0]) + y_range[0]
    coords = np.stack((x_coords, y_coords), axis=1)
    values = np.random.rand(number_samples)
    return coords, values


def precompute_grid_centers(target_height, target_width, x_min=0, x_max=1, y_min=0, y_max=1, grid_size_fac=1):
    """
    Precomputes grid centers for mapping.

    Parameters:
    - target_height (int): Height of the grid.
    - target_width (int): Width of the grid.

    Returns:
    - grid_centers (np.ndarray): Array of shape (target_height * target_width, 2)
      containing the grid center coordinates.
    """
    grid_x = np.linspace(x_min, x_max, int(np.round(target_height*grid_size_fac)))
    grid_y = np.linspace(y_min, y_max, int(np.round(target_width*grid_size_fac)))
    grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing='ij')
    grid_centers = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
    return grid_centers



def compute_distance_decay_matrix(grid_centers, coords, tau):
    """
    Computes a 2D matrix where each element represents the distance decay between a 
    coordinate and a grid center based on the decay constant tau.

    Parameters:
    - grid_centers (np.ndarray): Array of shape (M, 2), where M is the number of grid centers.
    - coords (np.ndarray): Array of shape (N, 2), where N is the number of coordinates.
    - tau (float): The decay constant for the distance decay function.

    Returns:
    - decay_matrix (np.ndarray): A 2D array of shape (N, M), where each element 
      represents the distance decay from a coordinate to a grid center.
    """
    # Compute pairwise distances between each coordinate and each grid center
    distances = np.linalg.norm(coords[:, np.newaxis, :] - grid_centers[np.newaxis, :, :], axis=2)

    # Apply distance decay function
    decay_matrix = np.exp(-distances / tau)

    return decay_matrix


def map_to_fixed_grid_decay(values, decay_matrix, target_width, target_height):
    """
    Maps each grid cell to a value weighted by the decay matrix.

    Parameters:
    - values (np.ndarray): Array of shape (N,) containing values for each coordinate.
    - decay_matrix (np.ndarray): 2D array of shape (N, M), where each element represents
      the decay factor between a coordinate and a grid center.
    - target_width (int): Width of the target grid.
    - target_height (int): Height of the target grid.

    Returns:
    - grid_values (np.ndarray): A 2D array of shape (target_height, target_width) 
      with values mapped to each grid cell based on decay-weighted values.
    """
    # Reshape `values` if necessary to make it a 1D array
    values = values.ravel()  # Flatten to (49,)

    # Expand values to align with the decay matrix for broadcasting
    weighted_values = np.dot(values, decay_matrix)  # shape (M,)

    # Reshape to the target grid shape
    grid_values = weighted_values.reshape(target_height, target_width)

    return grid_values


def map_to_fixed_grid_decay_batch(values, decay_matrix, target_width, target_height):
    """
    Maps a batch of value arrays to a fixed grid using decay weights.

    Parameters:
    - values (np.ndarray or torch.Tensor): Array of shape (T, N), where T is the batch size
      (time steps) and N is the number of input points (coordinates).
    - decay_matrix (np.ndarray or torch.Tensor): 2D array of shape (N, M), where each element
      represents the decay factor between a coordinate and a grid center.
    - target_width (int): Width of the target grid.
    - target_height (int): Height of the target grid.

    Returns:
    - grid_values_batch (torch.Tensor): A 3D tensor of shape (T, target_height, target_width),
      with values mapped to each grid cell based on decay-weighted values for the entire batch.
    """
    # if isinstance(values, np.ndarray):
    #     values = torch.tensor(values, dtype=torch.float32)
    # if isinstance(decay_matrix, np.ndarray):
    #     decay_matrix = torch.tensor(decay_matrix, dtype=torch.float32)
    
    # Perform batch matrix multiplication to compute weighted values for all time steps
    # values: (T, N), decay_matrix: (N, M)
    weighted_values = torch.matmul(values.T, decay_matrix)  # shape: (T, M)

    # Reshape the result to (T, target_height, target_width)
    grid_values_batch = weighted_values.view(-1, target_height, target_width)  # Shape: (T, H, W)

    return grid_values_batch


def compute_circular_mask_matrix(grid_centers, coords, mask_radius):
    """
    Computes a binary matrix where each element represents whether a grid center 
    is within a circular region centered at a given coordinate.

    Parameters:
    - grid_centers (np.ndarray): Array of shape (M, 2), where M is the number of grid centers.
    - coords (np.ndarray): Array of shape (N, 2), where N is the number of coordinates.
    - mask_radius (float): The radius of the circular mask.

    Returns:
    - mask_matrix (np.ndarray): A binary 2D array of shape (N, M), where each element 
      is 1 if the grid center is within the circular region of the corresponding coordinate, 0 otherwise.
    """
    # Compute pairwise distances between each coordinate and each grid center
    distances = np.linalg.norm(coords[:, np.newaxis, :] - grid_centers[np.newaxis, :, :], axis=2)

    # Apply the circular mask condition
    mask_matrix = (distances <= mask_radius).astype(float)

    return mask_matrix


def map_to_fixed_grid_circle_batch(values, mask_matrix, target_width, target_height):
    """
    Maps a batch of value arrays to a fixed grid using circular masks.

    Parameters:
    - values (np.ndarray or torch.Tensor): Array of shape (T, N), where T is the batch size
      (time steps) and N is the number of input points (coordinates).
    - mask_matrix (np.ndarray or torch.Tensor): 2D array of shape (N, M), where each element
      indicates whether a grid center is within the circular mask of a coordinate.
    - target_width (int): Width of the target grid.
    - target_height (int): Height of the target grid.

    Returns:
    - grid_values_batch (torch.Tensor): A 3D tensor of shape (T, target_height, target_width),
      with values mapped to each grid cell based on the circular mask for the entire batch.
    """
    

    # Normalize the mask to ensure each grid cell receives appropriate contributions
    # Avoid division by zero using a small epsilon
    # print(f'mask_matrix shape: {mask_matrix.shape}')
    epsilon = 1e-8
    normalized_mask = mask_matrix / (torch.sum(mask_matrix, dim=0, keepdim=True) + epsilon)

    # Perform batch matrix multiplication to compute masked values for all time steps
    # values: (T, N), normalized_mask: (N, M)
    masked_values = torch.matmul(values.T, normalized_mask)  # Shape: (T, M)

    # Reshape the result to (T, target_height, target_width)
    grid_values_batch = masked_values.view(-1, target_height, target_width)  # Shape: (T, H, W)

    return grid_values_batch


class HexagonalGridGenerator:
    def __init__(self, xlim, ylim, target_num_centers, max_iterations=200, noise_level=0.3, rand_seed=None, num_positions=None,
                position_indices=None):
        self.xlim = xlim
        self.ylim = ylim
        self.target_num_centers = target_num_centers
        self.max_iterations = max_iterations
        self.noise_level = noise_level
        self.rand_seed = rand_seed
        self.dx = None
        self.dy = None
        self.offset_x = None
        self.offset_y = None
        self.is_points1_generated = False
        self.current_state = np.random.get_state()
        self._initialize_grid()
        self.num_positions = num_positions
        self.position_indices = position_indices
    
    def _initialize_grid(self):
        if self.rand_seed is not None:
            np.random.seed(self.rand_seed)
        
        x_min, x_max = self.xlim
        y_min, y_max = self.ylim
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        approximate_area = x_range * y_range
        approximate_cell_area = approximate_area / self.target_num_centers
        side_length = np.sqrt(approximate_cell_area / (3 * np.sqrt(3) / 2))
        
        self.dx = side_length * np.sqrt(3)
        self.dy = side_length * 1.5
    
    def _generate_points_with_noise(self, dx, dy, offset_x, offset_y):
        x_min, x_max = self.xlim
        y_min, y_max = self.ylim
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        cols = int(np.ceil(x_range / dx))
        rows = int(np.ceil(y_range / dy))
        points = []
        
        for row in range(rows):
            for col in range(cols):
                x = col * dx + x_min - offset_x
                y = row * dy + y_min - offset_y
                if row % 2 == 1:
                    x += dx / 2
                x += (np.random.rand() - 0.5) * 2 * self.noise_level * dx
                y += (np.random.rand() - 0.5) * 2 * self.noise_level * dy
                if x_min <= x < x_max and y_min <= y < y_max:
                    points.append((x, y))
        return np.array(points)
    
    def sub_select_points(self, pts):
        if self.position_indices is not None:
            valid_indices = self.position_indices[self.position_indices < len(pts)]
            return pts[valid_indices]
        elif self.num_positions is not None and self.num_positions < len(pts):
            chosen = np.random.choice(len(pts), self.num_positions, replace=False)
            return pts[chosen]
        else:
            return pts
    
    def generate_first_grid(self):
        x_min, x_max = self.xlim
        y_min, y_max = self.ylim
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        for _ in range(self.max_iterations):
            cols = int(np.ceil(x_range / self.dx))
            rows = int(np.ceil(y_range / self.dy))
            self.offset_x = (cols * self.dx - x_range) / 2
            self.offset_y = (rows * self.dy - y_range) / 2
            points = self._generate_points_with_noise(self.dx, self.dy, self.offset_x, self.offset_y)
            if abs(len(points) - self.target_num_centers) <= self.target_num_centers * 0.05:
                break
            if len(points) > self.target_num_centers:
                self.dx *= 1.01
                self.dy *= 1.01
            else:
                self.dx *= 0.99
                self.dy *= 0.99
        
        points = self.sub_select_points(points)
        self.is_points1_generated = True
        np.random.set_state(self.current_state)
        return points
    
    def generate_second_grid(self, anti_alignment=0.0):
        if not self.is_points1_generated:
            raise ValueError("First grid has not been generated. Call generate_first_grid() first.")
        shift_x = anti_alignment * self.dx
        shift_y = anti_alignment * self.dy
        offset_x = self.offset_x - shift_x,  # note we subtract shift in offset
        offset_y = self.offset_y - shift_y,
        points = self._generate_points_with_noise(self.dx, self.dy, offset_x, offset_y)
        points = self.sub_select_points(points)
        np.random.set_state(self.current_state)
        return points
