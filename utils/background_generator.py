import math
import random
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class WavePattern:
    """
    Abstract base class for a 1D wave pattern generator.
    """

    def generate(
        self, width: int, spatial_frequency: float, phase: float, gray_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate a 1D wave pattern.
        
        :param width: Number of pixels along the x-axis.
        :param spatial_frequency: Number of cycles across the image width.
        :param phase: Phase shift of the wave in the range [0, 1] (mapped to [0, 2π]).
        :param gray_range: Tuple (min, max) grayscale values in [0, 1].
        :return: 1D numpy array of length `width` with values in the specified gray_range.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class SineWave(WavePattern):
    def generate(
        self, width: int, spatial_frequency: float, phase: float, gray_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate a sine wave pattern.
        
        The sine wave is defined as:
            f(x) = offset + amplitude * sin(2π * spatial_frequency * x + phase*2π)
        where the offset is the middle of the gray range and the amplitude is half the gray range.
        
        :param width: Number of pixels along the x-axis.
        :param spatial_frequency: Number of cycles across the width.
        :param phase: Phase shift in [0, 1] (0 corresponds to starting at the middle intensity).
        :param gray_range: Tuple (min, max) grayscale values in [0, 1].
        :return: 1D numpy array of length `width`.
        """
        min_val, max_val = gray_range
        offset = (min_val + max_val) / 2.0
        amplitude = (max_val - min_val) / 2.0

        # Generate x values normalized in [0, 1)
        x = np.linspace(0, 1, width, endpoint=False)
        # Compute sine wave. (phase * 2π maps phase from [0,1] to [0,2π])
        wave = offset + amplitude * np.sin(2 * np.pi * spatial_frequency * x + phase * 2 * np.pi)
        return wave


class SquareWave(WavePattern):
    def generate(
        self, width: int, spatial_frequency: float, phase: float, gray_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate a square wave pattern.
        
        The square wave is derived from the sine wave; values are set to the high
        (max gray value) when sin(...) ≥ 0 and to the low (min gray value) otherwise.
        
        :param width: Number of pixels along the x-axis.
        :param spatial_frequency: Number of cycles across the width.
        :param phase: Phase shift in [0, 1].
        :param gray_range: Tuple (min, max) grayscale values in [0, 1].
        :return: 1D numpy array of length `width`.
        """
        min_val, max_val = gray_range
        # Generate a sine wave to threshold
        x = np.linspace(0, 1, width, endpoint=False)
        sine_component = np.sin(2 * np.pi * spatial_frequency * x + phase * 2 * np.pi)
        wave = np.where(sine_component >= 0, max_val, min_val)
        return wave


class PinkNoise(WavePattern):
    def generate(
        self, width: int, spatial_frequency: float, phase: float, gray_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Generate a 1D pink noise pattern.
        
        Note: The spatial_frequency and phase parameters are not used for pink noise.
        The algorithm creates white noise, applies a 1/f (pink) filter in the frequency domain,
        and then normalizes the result to the [min, max] gray value range.
        
        :param width: Number of pixels along the x-axis.
        :param spatial_frequency: Not used.
        :param phase: Not used.
        :param gray_range: Tuple (min, max) grayscale values in [0, 1].
        :return: 1D numpy array of length `width`.
        """
        n = width
        # Generate white noise
        white = np.random.randn(n)
        # Compute the FFT of the white noise
        fft = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(n)

        # Apply a 1/f^(0.5) scaling for pink noise.
        # For frequency 0 (DC), avoid division by zero by using a factor of 1.
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(frequencies == 0, 1.0, 1.0 / np.sqrt(frequencies))
        fft *= factor

        # Inverse FFT to get the pink noise in time domain
        pink = np.fft.irfft(fft, n)
        # Normalize the pink noise to the range [0, 1]
        pink_norm = (pink - np.min(pink)) / (np.max(pink) - np.min(pink))
        # Scale to the desired gray_range
        min_val, max_val = gray_range
        wave = min_val + pink_norm * (max_val - min_val)
        return wave


def replicate_wave(wave: np.ndarray, height: int) -> np.ndarray:
    """
    Replicate a 1D wave pattern vertically to create a 2D image.
    
    :param wave: 1D numpy array representing the wave pattern.
    :param height: Desired number of rows (image height).
    :return: 2D numpy array with the wave pattern repeated for each row.
    """
    return np.tile(wave, (height, 1))


def center_crop(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """
    Crop the image to the target size by taking the central region.
    
    :param image: PIL Image to crop.
    :param target_size: Desired (width, height).
    :return: Center-cropped PIL Image.
    """
    target_width, target_height = target_size
    width, height = image.size
    left = max((width - target_width) // 2, 0)
    top = max((height - target_height) // 2, 0)
    right = left + target_width
    bottom = top + target_height
    return image.crop((left, top, right, bottom))


class DriftingGrating:
    """
    Class to generate, transform (rotate), save, and visualize drifting grating images.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        spatial_frequency: float,
        phase: float,
        gray_range: Tuple[float, float],
        orientation: float,
        pattern: WavePattern,
    ) -> None:
        """
        Initialize the drifting grating with the provided parameters.
        
        :param image_size: Tuple (width, height) of the output image.
        :param spatial_frequency: Number of cycles across the image width.
        :param phase: Phase shift of the wave in [0, 1].
        :param gray_range: Tuple (min, max) grayscale values in [0, 1].
        :param orientation: Rotation angle in degrees.
        :param pattern: Instance of a WavePattern subclass.
        """
        self.image_size = image_size
        self.spatial_frequency = spatial_frequency
        self.phase = phase
        self.gray_range = gray_range
        self.orientation = orientation
        self.pattern = pattern

    def generate_pattern(self) -> np.ndarray:
        """
        Generate the drifting grating pattern as a 2D numpy array (dtype uint8).
        
        The process is as follows:
          1. Generate a 1D wave pattern.
          2. Replicate the 1D pattern along the vertical dimension.
          3. Convert the values from the specified gray_range to [0, 255].
          4. Create a PIL Image and apply the specified rotation.
          5. Crop the rotated image to the original size.
        
        :return: 2D numpy array representing the grating image.
        """
        width, height = (dim * 2 for dim in self.image_size)

        # Generate the 1D wave pattern.
        wave_1d = self.pattern.generate(width, self.spatial_frequency, self.phase, self.gray_range)
        # Replicate the pattern vertically.
        pattern_2d = replicate_wave(wave_1d, height)

        # Convert the 2D pattern to grayscale (0-255).
        # The wave functions output values in the range [gray_range[0], gray_range[1]],
        # so we map that linearly to [0, 255].
        img_array = (
            ((pattern_2d - self.gray_range[0]) / (self.gray_range[1] - self.gray_range[0])) * 255
        ).clip(0, 255).astype(np.uint8)

        # Create a PIL image.
        img = Image.fromarray(img_array, mode="L")

        # If an orientation is specified (not 0 mod 360), rotate the image.
        if self.orientation % 360 != 0:
            img = self._apply_orientation(img, self.orientation)

        return np.array(img)

    def _apply_orientation(self, image: Image.Image, angle: float) -> Image.Image:
        """
        Rotate the image by the specified angle and then center crop it to the original size.
        
        :param image: PIL Image to rotate.
        :param angle: Rotation angle in degrees.
        :return: Rotated and cropped PIL Image.
        """
        # Rotate with expansion to avoid clipping.
        rotated = image.rotate(angle, resample=Image.BILINEAR, expand=True)
        # Center crop to the original image size.
        cropped = center_crop(rotated, self.image_size)
        return cropped

    def save_image(self, filename: str) -> None:
        """
        Save the generated drifting grating image as a PNG file.
        
        :param filename: Output filename (should end with '.png').
        """
        pattern_array = self.generate_pattern()
        img = Image.fromarray(pattern_array, mode="L")
        img.save(filename, format="PNG")
        print(f"Image saved as {filename}")

    def show_image(self) -> None:
        """
        Display the generated drifting grating image for verification.
        """
        pattern_array = self.generate_pattern()
        plt.figure(figsize=(6, 6))
        plt.imshow(pattern_array, cmap="gray", vmin=0, vmax=255)
        plt.title("Drifting Grating Pattern")
        plt.axis("off")
        plt.show()


def generate_random_parameters(
    image_size: Tuple[int, int] = (256, 256), seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate random parameters for creating a drifting grating image.
    
    The following parameters are randomized:
      - Spatial frequency: Uniformly sampled between 0.5 and 5.0 cycles per image width.
      - Phase: Uniformly sampled between 0 and 1.
      - Gray range: The minimum is sampled in [0, 0.3] and the maximum in [0.7, 1.0].
      - Orientation: Uniformly sampled between 0° and 180°.
      - Wave pattern type: Randomly chosen among 'sine', 'square', and 'pink'.
    
    :param image_size: Tuple (width, height) for the image.
    :param seed: Optional seed for reproducibility.
    :return: Dictionary of parameters.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    spatial_frequency = random.uniform(2, 20)
    phase = random.uniform(0, 1)
    min_gray = random.uniform(0, 0.5)
    max_gray = random.uniform(0.5, 1.0)
    if min_gray > max_gray:
        min_gray, max_gray = max_gray, min_gray
    gray_range = (min_gray, max_gray)
    orientation = random.uniform(0, 180)
    pattern_type = random.choice(["sine", "square", "pink"])
    if pattern_type == "sine":
        pattern = SineWave()
    elif pattern_type == "square":
        pattern = SquareWave()
    elif pattern_type == "pink":
        pattern = PinkNoise()
    else:
        pattern = SineWave()

    params = {
        "image_size": image_size,
        "spatial_frequency": spatial_frequency,
        "phase": phase,
        "gray_range": gray_range,
        "orientation": orientation,
        "pattern": pattern,
        "pattern_type": pattern_type,
    }
    return params



