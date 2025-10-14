#!/bin/bash

pip install --upgrade pillow
pip install pandas
pip install scipy
pip install torchvision
pip install opencv-python-headless
pip install scikit-image
pip install matplotlib
# pip install openpyxl

#python $HOME/RGC2Prey/RGC2Prey/visual_preycapture_prediction.py --experiment_names 2025020301 --test_bg_folder 'black-bg' --test_ob_folder 'white-spot' --noise_levels 0.0 0.005 0.01 0.02 0.04

#python $HOME/RGC2Prey/RGC2Prey/visual_preycapture_prediction.py --experiment_names '2025100507' --epoch_number 200 --test_bg_folder 'blend' --boundary_size '(200, 120)' --fix_disparity_degrees 0.0 3.0 6.0 12.0 --noise_levels 0.016 0.032 0.064 0.128 0.256 --num_worker 24

python $HOME/RGC2Prey/RGC2Prey/visual_preycapture_prediction.py --experiment_names '2025100602' --epoch_number 200 --save_movie_frames --visual_sample_ids 53 --movie_eye 'right' --movie_input_channel 'second' --test_bg_folder 'blend' --boundary_size '(200, 120)' --noise_levels 0.0 0.016 0.032 0.064 0.128 0.256 --num_worker 24

#python $HOME/RGC2Prey/RGC2Prey/visual_preycapture_prediction.py --experiment_names 2025072001 --fix_disparity_degrees 0.0 1.0 2.0 4.0 8.0 --epoch_number 200 --test_bg_folder 'blend' --boundary_size '(200, 120)' --noise_levels 0.0 0.002 0.004 0.008 0.016 0.032