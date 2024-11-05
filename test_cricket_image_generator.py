import os
import numpy as np 
from datasets.sim_cricket import overlay_images_with_jitter_and_scaling

rgc_canvas_size = np.array([640, 480])  # in micro
crop_extension_fac = 1.5  # 1.5 times larger than the target area, to be convolved with proper RGC RF
pixel_in_um = 4.375  # [task] make sure all recordings have the same value; if not, normalization is required
angular_in_um = 31.5
cricket_size_range = np.array([5, 20])  # visual angle (~ 20 cm for 1.5~2 cm cricket, ~ 5 cm for contact)

# calculation to pixel domain
cricket_size_range = cricket_size_range * angular_in_um / pixel_in_um
crop_size = rgc_canvas_size * crop_extension_fac / pixel_in_um
