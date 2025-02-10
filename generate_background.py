import os
from utils.background_generator import generate_random_parameters, DriftingGrating

# Define the number of images to generate
num_images = 200  # Change this value to generate more images

# Define the folder for saving images
save_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images/cropped/drift-grating/'

# Ensure the save folder exists
os.makedirs(save_folder, exist_ok=True)

# -----------------------
# Deterministic example:
# -----------------------
image_size = (640, 480)  # (width, height)
# spatial_frequency = 2.0  # 2 cycles across the width
# phase = 0.0  # For sine waves: 0 ensures the wave starts at the middle intensity
# gray_range = (0.0, 1.0)  # Full grayscale range
# orientation = 45.0  # Rotate the pattern by 45 degrees
# pattern = SineWave()  # Use the sine wave pattern

# grating = DriftingGrating(image_size, spatial_frequency, phase, gray_range, orientation, pattern)
# grating.save_image(os.path.join(save_folder, "deterministic_grating.png"))
# grating.show_image()

# -----------------------
# Randomized examples:
# -----------------------
for i in range(num_images):
    random_params = generate_random_parameters(image_size)
    print(f"Random Parameters {i+1}:", random_params)

    grating_random = DriftingGrating(
        image_size=random_params["image_size"],
        spatial_frequency=random_params["spatial_frequency"],
        phase=random_params["phase"],
        gray_range=random_params["gray_range"],
        orientation=random_params["orientation"],
        pattern=random_params["pattern"],
    )

    random_filename = f"drifting_grating_{random_params['pattern_type']}_{i+1}.png"
    save_path = os.path.join(save_folder, random_filename)

    grating_random.save_image(save_path)
    # grating_random.show_image()
