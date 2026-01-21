#!/bin/bash
pip install --upgrade pillow
pip install pandas
pip install scipy
pip install torchvision
pip install opencv-python-headless
pip install scikit-image
pip install matplotlib
pip install openpyxl

python $HOME/RGC2Prey/RGC2Prey/train_preycapture.py --experiment_name '2025100502' --rgc_noise_std 0.016 --fix_disparity 6.0 --is_two_grids --is_binocular --rectified_thr_ON 0.087 --set_surround_size_scalar 8.0 --is_rescale_diffgaussian --set_s_scale -0.09 --syn_params tf sf --sf_sheet_name 'SF_ON-T' --tf_sheet_name 'TF_ON-T' --is_rf_median_subtract --target_num_centers 475 --is_channel_normalization --bg_processing_type 'one-proj' --bg_info_type 'loc' --bg_info_cost_ratio 0.0 --min_lr 1e-6 --cnn_extractor_version 4 --seed '8901' --cnn_feature_dim 256 --lstm_hidden_size 128 --lstm_num_layers 4 --add_noise --is_rectified --sf_scalar 0.54 --sf_mask_radius 50 --mask_radius 17.7 --end_scaling 1.5 --start_scaling 0.3 --dynamic_scaling 0.15 --prob_stay_bg 0.5 --prob_mov_bg 0.975 --momentum_decay_ob 0.975 --velocity_randomness_bg 0.025 --velocity_randomness_ob 0.01 --sf_constraint_method 'circle' --coord_adj_dir -1.0 --coord_adj_type 'body' --batch_size 4 --accumulation_steps 128 --num_worker 24 --timer_tau 0.95 --timer_sample_cicle 1 --is_GPU --is_seq_reshape --num_samples 2000 --bg_folder 'drift-grating-blend' --grid_generate_method 'circle' --is_input_norm --is_norm_coords --num_epochs 200 --schedule_method 'CAWR' --grid_size_fac 0.5 --num_epoch_save 20