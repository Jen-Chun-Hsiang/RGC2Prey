import os
import numpy as np 
from datasets.sim_cricket import overlay_images_with_jitter_and_scaling

# Below unit in pixel
image_size = np.array([640, 480])
crop_size = np.array([320, 240])
rgc_canvas_size = np.array([240, 180])  
cricket_size_range = np.array([40, 100])  # visual angle (~ 20 cm for 1.5~2 cm cricket, 5.56 to 13.89 degree)

pixel_in_um = 4.375  # [task] make sure all recordings have the same value; if not, normalization is required


