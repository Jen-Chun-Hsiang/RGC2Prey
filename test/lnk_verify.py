import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils.dynamic_fitting import predict_lnk_rate, LNKParams

# --- CONFIGURATION ---
mat_file_path = 'PATH/TO/YOUR/MATFILE.mat'  # Update with your .mat file path
params_file_path = 'PATH/TO/YOUR/PARAMS.npy'  # Update with your params file path (if needed)
dt = 0.001  # Time step in seconds (update as needed)

# --- LOAD DATA ---
mat_data = sio.loadmat(mat_file_path)
exp = mat_data['exp'].squeeze()  # Experimental data (1D array)
sim = mat_data['sim'].squeeze()  # Simulated data (1D array)

# --- LOAD FITTED PARAMS ---
# If params are saved as a dict or npy, update loading accordingly
params_dict = np.load(params_file_path, allow_pickle=True).item()
params = LNKParams(**params_dict)

# --- INFERENCE ---
rate_hat, _ = predict_lnk_rate(sim, params, dt)

# --- COMPARISON ---
plt.figure(figsize=(10, 4))
plt.plot(exp, label='Experimental')
plt.plot(rate_hat, label='Simulated (LNK Model)', alpha=0.7)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Rate')
plt.title('LNK Model Verification')
plt.tight_layout()
plt.show()
