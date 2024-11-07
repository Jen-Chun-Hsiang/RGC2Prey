import numpy as np
import torch

def gaussian2d(x, y, params):
    x_mean, y_mean, sigma_x, sigma_y, theta, bias, scale, s_sigma_x, s_sigma_y, s_scale = params

    x_rot = (x - x_mean) * np.cos(theta) + (y - y_mean) * np.sin(theta)
    y_rot = -(x - x_mean) * np.sin(theta) + (y - y_mean) * np.cos(theta)

    z = (scale * np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2))) +
         s_scale * np.exp(-(x_rot**2 / (2 * s_sigma_x**2) + y_rot**2 / (2 * s_sigma_y**2))) + bias)
    return z


def gaussian_multi(params, image, num_gauss):
    X, Y = np.meshgrid(np.arange(1, image.shape[1] + 1), np.arange(1, image.shape[0] + 1))
    params = np.reshape(params, (-1, num_gauss)).T

    gaussian_model = gaussian2d(X, Y, params[0])
    for i in range(1, num_gauss):
        gaussian_model += gaussian2d(X, Y, params[i])

    return gaussian_model  # Optionally, compute error or difference from image if required


def create_hexagonal_centers(xlim, ylim, target_num_centers, max_iterations=100, noise_level=0.3, rand_seed=None, num_positions=None, position_indices=None):
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

    return points


def precompute_grid_centers(target_height, target_width, x_min=0, x_max=1, y_min=0, y_max=1):
    """
    Precomputes grid centers and distance indices for mapping.
    """
    # Define the grid range based on the input limits
    grid_x = torch.linspace(x_min, x_max, target_width)
    grid_y = torch.linspace(y_min, y_max, target_height)

    # Generate grid centers using meshgrid
    grid_centers = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='ij'), dim=-1).reshape(-1, 2)

    return grid_centers

def get_closest_indices(grid_centers, coords):
    """
    Finds the closest coordinate index for each grid center.
    """
    # Convert numpy arrays to torch tensors
    coords = torch.from_numpy(coords).float()
    
    distances = torch.cdist(grid_centers, coords)
    closest_points = distances.argmin(dim=1)
    return closest_points

def map_to_fixed_grid_closest(values, closest_points, target_width, target_height):
    """
    Maps each grid cell to the value of the closest coordinate using precomputed indices.
    """
    values = torch.from_numpy(values).float()
    # Assign the value of the closest coordinate to each grid cell
    grid_values = values[closest_points].view(target_width, target_height)
    return grid_values.numpy()

