import numpy as np
import math
import os
import matplotlib.pyplot as plt

def adjust_trajectories(bounds, trajectory_1, trajectory_2=None):
    """
    Adjust trajectories to fit within bounds by rescaling or randomly shifting.
    
    Parameters:
    - bounds: Tuple (x_min, x_max, y_min, y_max) defining the boundary of the area.
    - trajectory_1: NumPy array of shape (n, 2) for the first trajectory.
    - trajectory_2: NumPy array of shape (m, 2) for the second trajectory (optional).
    
    Returns:
    - adjusted_trajectory_1: Adjusted trajectory_1 as a NumPy array.
    - adjusted_trajectory_2: Adjusted trajectory_2 as a NumPy array, or None if not provided.
    """
    if trajectory_1 is None or not isinstance(trajectory_1, np.ndarray):
        raise ValueError("trajectory_1 must be a NumPy array.")
    
    # Combine trajectories for joint processing if trajectory_2 exists
    combined_trajectories = trajectory_1 if trajectory_2 is None else np.vstack((trajectory_1, trajectory_2))

    # Determine min, max bounds of the trajectories
    x_min_traj, y_min_traj = np.min(combined_trajectories, axis=0)
    x_max_traj, y_max_traj = np.max(combined_trajectories, axis=0)
    
    # Compute width and height of the trajectory scope
    traj_width = x_max_traj - x_min_traj
    traj_height = y_max_traj - y_min_traj
    
    # Compute width and height of the bounds
    bound_width = bounds[1] - bounds[0]
    bound_height = bounds[3] - bounds[2]
    
    # Flags to determine if rescaling is required in each dimension
    rescale_x = traj_width > bound_width
    rescale_y = traj_height > bound_height

    # Rescaling factor (if needed)
    scale_factor_x = bound_width / traj_width if rescale_x else 1.0
    scale_factor_y = bound_height / traj_height if rescale_y else 1.0
    scale_factor = min(scale_factor_x, scale_factor_y)

    # Rescale and shift to the lower-left corner of the bounds
    def rescale_and_shift(trajectory):
        return ((trajectory - np.array([x_min_traj, y_min_traj])) * scale_factor) + np.array([bounds[0], bounds[2]])

    adjusted_trajectory_1 = rescale_and_shift(trajectory_1)
    adjusted_trajectory_2 = rescale_and_shift(trajectory_2) if trajectory_2 is not None else None

    # Compute tolerance for shifting after rescaling
    new_width = traj_width * scale_factor
    new_height = traj_height * scale_factor

    x_tolerance = (bound_width - new_width) if not rescale_x else 0
    y_tolerance = (bound_height - new_height) if not rescale_y else 0

    # Random shifts along axes where space is available
    x_shift = np.random.uniform(0, x_tolerance) if x_tolerance > 0 else 0
    y_shift = np.random.uniform(0, y_tolerance) if y_tolerance > 0 else 0

    shift_vector = np.array([x_shift, y_shift])

    # Apply final shift
    adjusted_trajectory_1 += shift_vector
    if adjusted_trajectory_2 is not None:
        adjusted_trajectory_2 += shift_vector

    return adjusted_trajectory_1, adjusted_trajectory_2


def binocular_disparity(iod_cm, distance_cm):
    """
    Calculate the binocular disparity (in degrees) for an object at distance_cm
    from the midpoint between two eyes separated by iod_cm.
    """
    # theta = 2 * arctan( (IOD / 2) / distance )
    return 2 * math.degrees(math.atan((iod_cm / 2) / distance_cm))


def disparity_from_scaling_factor(
    scaling_factors,
    start_distance,
    end_distance,
    iod_cm,
    fix_disparity: float | None = None,
):
    """
    Given:
      - scaling_factors: a list of numbers (e.g., [1, 1.1, 1.1, 1.2, 1.4, ...])
      - start_distance and end_distance: e.g. 21 and 4
      - iod_cm: interocular distance (cm)
      - fix_disparity : float | None, optional
        • None (default): compute normal geometry-based disparities.  
        • float: return this disparity value for every element
          (array length = len(scaling_factors)).
    
    1) Maps each element in scaling_factors from [min_sf, max_sf]
       into [start_distance, end_distance].
       - If two scaling factors are the same, they map to the same distance.
    2) Computes the binocular disparity for each distance.
    3) Returns two arrays of equal length: 
       (distances_corresponding_to_scaling, disparities_in_degrees).
    """

    # Identify the range in scaling_factors
    min_sf = min(scaling_factors)
    max_sf = max(scaling_factors)
    
    # Edge case: if min_sf == max_sf, then all scaling_factors are the same
    # and we just treat every distance as start_distance (or end_distance).
    # For simplicity, we assume start_distance is the mapped distance.
    if min_sf == max_sf:
        distances = np.full(len(scaling_factors), start_distance)
    else:
        # Map scaling factor to distance via linear interpolation
        sf = np.asarray(scaling_factors, dtype=float)
        distances = start_distance + (end_distance - start_distance) * (
            sf - min_sf
        ) / (max_sf - min_sf)
    
    if fix_disparity is None:
        # normal computation
        disparities = np.array(
            [binocular_disparity(iod_cm, d) for d in distances]
        )
    else:
        # override with fixed value
        disparities = np.full(len(scaling_factors), float(fix_disparity))
    
    return disparities, distances


def convert_deg_to_pix(x, deg2um=32.5, pix2um=4.375, scaling=0.54):
    """
    Converts degrees to pixels using the provided scaling factors.

    Parameters:
    - x: numpy array of values to convert.
    - deg2um: conversion factor from degrees to micrometers (default: 32.5).
    - pix2um: conversion factor from pixels to micrometers (default: 4.375).
    - scaling: additional scaling factor (default: 0.54).

    Returns:
    - result: numpy array after conversion.
    """
    result = x * deg2um / (pix2um / scaling)
    return result


def plot_trajectories(bounds, original_1, original_2, adjusted_1, adjusted_2, save_folder, filename="trajectory_plot.png"):
    """
    Plot original and adjusted trajectories within the bounds and save the plot to a given folder.

    Parameters:
    - bounds: Tuple (x_min, x_max, y_min, y_max)
    - original_1: numpy array of shape (n, 2) for trajectory 1
    - original_2: numpy array of shape (n, 2) for trajectory 2 (optional)
    - adjusted_1: numpy array of shape (n, 2) for adjusted trajectory 1
    - adjusted_2: numpy array of shape (n, 2) for adjusted trajectory 2 (optional)
    - save_folder: Path to the folder where the plot will be saved
    - filename: Name of the output file (default: "trajectory_plot.png")
    """
    
    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot bounds
    x_min, x_max, y_min, y_max = bounds
    plt.plot([x_min, x_max, x_max, x_min, x_min], 
             [y_min, y_min, y_max, y_max, y_min], 
             'k--', linewidth=2, label="Bounds")
    
    # Plot original trajectories
    plt.plot(original_1[:, 0], original_1[:, 1], 'r-o', label="Original Trajectory 1")
    
    if original_2 is not None:
        plt.plot(original_2[:, 0], original_2[:, 1], 'b-o', label="Original Trajectory 2")
    
    # Plot adjusted trajectories
    plt.plot(adjusted_1[:, 0], adjusted_1[:, 1], 'r--', label="Adjusted Trajectory 1")
    
    if adjusted_2 is not None:
        plt.plot(adjusted_2[:, 0], adjusted_2[:, 1], 'b--', label="Adjusted Trajectory 2")
    
    plt.title("Trajectory Adjustment Within Bounds")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    plt.close()  # Close the plot to free up memory
    print(f"Plot saved to {save_path}")

