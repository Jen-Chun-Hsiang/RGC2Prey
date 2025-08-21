# --- CORRELATION PRINTING ---
sim_exp_corr = np.corrcoef(sim, exp)[0, 1]
ratehat_exp_corr = np.corrcoef(rate_hat, exp)[0, 1]
print(f"Correlation between sim and exp: {sim_exp_corr:.4f}")
print(f"Correlation between rate_hat and exp: {ratehat_exp_corr:.4f}")
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils.dynamic_fitting import predict_lnk_rate, LNKParams

# --- CONFIGURATION ---
mat_file_path = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Results/temp_082125_e100724.mat'  # Update with your .mat file path
fig_save_file_path = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Results/temp_082125_e100724.png'  # Update with your .png file path

dt = 0.001  # Time step in seconds (update as needed)

# --- LOAD DATA ---
mat_data = sio.loadmat(mat_file_path)
exp = mat_data['exp'].squeeze()  # Experimental data (1D array)
sim = mat_data['sim'].squeeze()  # Simulated data (1D array)


# --- LOAD FITTED PARAMS FROM MAT FILE ---
# prm is a cell struct in the mat file
prm = mat_data['prm']
# If prm is a cell array, extract the first cell (adjust if needed)
if isinstance(prm, np.ndarray) and prm.dtype == 'O':
	prm_struct = prm[0, 0]
else:
	prm_struct = prm

params = LNKParams(
	tau=float(prm_struct['tau'].squeeze()),
	alpha_d=float(prm_struct['alpha_d'].squeeze()),
	sigma0=float(prm_struct['sigma0'].squeeze()),
	alpha=float(prm_struct['alpha'].squeeze()),
	beta=float(prm_struct['beta'].squeeze()),
	b_out=float(prm_struct['b_out'].squeeze()),
	g_out=float(prm_struct['g_out'].squeeze()),
	theta=float(prm_struct['theta'].squeeze())
)

# --- INFERENCE ---
rate_hat, _ = predict_lnk_rate(sim, params, dt)

# --- COMPARISON ---
plt.figure(figsize=(10, 4))
plt.plot(exp, label='Experimental')
plt.plot(rate_hat, label='Simulated (LNK Model)', alpha=0.7)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Rate')
plt.title(f'LNK Model Verification\nSim-Exp Corr: {sim_exp_corr:.4f}, RateHat-Exp Corr: {ratehat_exp_corr:.4f}')
plt.tight_layout()
plt.savefig(fig_save_file_path)
