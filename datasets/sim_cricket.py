import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T


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


def random_movement(boundary_size, center_ratio, max_steps, a, b, initial_velocity=1.0, momentum_decay=0.95, velocity_randomness=0.02,
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
    half_width = boundary_size[0] / 2
    half_height = boundary_size[1] / 2

    # Calculate center region bounds based on center_ratio
    center_width = half_width * center_ratio
    center_height = half_height * center_ratio

    # Set initial position within the center region, with (0, 0) as the center
    x = np.random.uniform(-center_width, center_width)
    y = np.random.uniform(-center_height, center_height)
    path = [(x, y)]

    # State and movement parameters
    moving = False
    velocity = initial_velocity
    velocity_history = []
    angle = np.random.uniform(0, 2 * np.pi)

    for _ in range(max_steps):
        if moving:
            if np.random.rand() > b:
                moving = False  # Switch to staying
                velocity_history.append(0)
            else:
                # Adjust velocity by momentum decay and add a small random factor
                velocity = max(velocity * momentum_decay + np.random.uniform(-velocity_randomness, velocity_randomness), 0.01)

                # Slightly adjust direction to keep it mostly stable
                angle += np.random.uniform(-angle_range, angle_range)

                # Calculate new position with momentum applied
                x += velocity * np.cos(angle)
                y += velocity * np.sin(angle)
                velocity_history.append(velocity)

        else:
            angle = np.random.uniform(0, 2 * np.pi)
            velocity_history.append(0)
            if np.random.rand() > a:
                moving = True  # Switch to moving
                # Preserve momentum by reusing the last velocity and angle
                velocity = initial_velocity * momentum_decay + np.random.uniform(-velocity_randomness, velocity_randomness)

        # Check boundaries (with center at (0, 0))
        if not (-half_width <= x <= half_width and -half_height <= y <= half_height):
            break  # Stop if out of bounds

        path.append((x, y))

    return path, velocity_history