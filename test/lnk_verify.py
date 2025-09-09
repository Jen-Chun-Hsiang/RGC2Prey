import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
import os
import time
from datetime import datetime

# Conditional import for torch (only needed if using new LNK model)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. New LNK model will be disabled.")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dynamic_fitting import predict_lnk_rate, predict_lnk_rate_two, LNKParams

if TORCH_AVAILABLE:
    from datasets.simple_lnk import compute_lnk_response_from_convolved

# --- CONFIGURATION ---
mat_file_path = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Results/lnk_verification/e100724_for_lnk_verification.mat'  # Update with your .mat file path

# Folder to save figures (change if you want a different location)
fig_save_file_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/lnk_verification/figures'

# Derive a timestamped figure filename from the mat file name so it doesn't need to be edited manually
mat_basename = os.path.splitext(os.path.basename(mat_file_path))[0]
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
fig_save_file_path = os.path.join(fig_save_file_folder, f"{mat_basename}_{timestamp}.png")

# Ensure output folder exists
os.makedirs(fig_save_file_folder, exist_ok=True)

dt = 0.001  # Time step in seconds (update as needed)

# Model selection: Choose which LNK implementation to use
USE_NEW_LNK_MODEL = True and TORCH_AVAILABLE  # Set to True to use compute_lnk_response_from_convolved, False for predict_lnk_rate_two

if USE_NEW_LNK_MODEL and not TORCH_AVAILABLE:
    print("Warning: New LNK model requested but PyTorch not available. Falling back to original model.")
    USE_NEW_LNK_MODEL = False

def lnk_params_to_dict(params: LNKParams, dt: float) -> dict:
    """Convert LNKParams dataclass to dictionary format expected by compute_lnk_response_from_convolved."""
    return {
        'tau': np.array([params.tau]),
        'alpha_d': np.array([params.alpha_d]),
        'theta': np.array([params.theta]),
        'sigma0': np.array([params.sigma0]),
        'alpha': np.array([params.alpha]),
        'beta': np.array([params.beta]),
        'b_out': np.array([params.b_out]),
        'g_out': np.array([params.g_out]),
        'w_xs': np.array([params.w_xs]),
        'dt': np.array([dt])
    }

if TORCH_AVAILABLE:
    def compute_lnk_new(x_center: np.ndarray, x_surround: np.ndarray, params: LNKParams, dt: float) -> np.ndarray:
        """
        Wrapper function to use compute_lnk_response_from_convolved with numpy arrays.
        
        Args:
            x_center: Center signal [T] - corresponds to sim
            x_surround: Surround signal [T] - corresponds to sim_s
            params: LNKParams object
            dt: Time step
            
        Returns:
            rate_hat: Predicted firing rate [T]
        """
        # Convert to torch tensors
        device = torch.device('cpu')
        dtype = torch.float32
        
        # Reshape to [N=1, T] format expected by the function
        x_center_torch = torch.from_numpy(x_center).unsqueeze(0).to(device=device, dtype=dtype)
        x_surround_torch = torch.from_numpy(x_surround).unsqueeze(0).to(device=device, dtype=dtype)
        
        # Convert parameters
        lnk_params_dict = lnk_params_to_dict(params, dt)
        
        # Compute response
        rgc_response = compute_lnk_response_from_convolved(
            x_center_torch, x_surround_torch, lnk_params_dict, device, dtype
        )
        
        # Convert back to numpy and squeeze to [T] shape
        return rgc_response.squeeze(0).detach().cpu().numpy()

# --- LOAD DATA ---
mat_data = sio.loadmat(mat_file_path)
exp = mat_data['exp'].squeeze()  # Experimental data (1D array)
sim = mat_data['sim'].squeeze()  # Simulated data (1D array)
sim_s = mat_data['sim_s'].squeeze()  # Simulated data (1D array)
r_hat = mat_data['r_hat'].squeeze()  # Fitted rate trajectory (1D array)
r_hat_s = mat_data['r_hat_s'].squeeze()  # Fitted rate trajectory (1D array)


# --- LOAD FITTED PARAMS FROM MAT FILE ---
# prm is a cell struct in the mat file
prm = mat_data['prm']
prm_s = mat_data['prm_s']
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


if isinstance(prm_s, np.ndarray) and prm_s.dtype == 'O':
    prm_s_struct = prm_s[0, 0]
else:
    prm_s_struct = prm_s
	
params_s = LNKParams(
	tau=float(prm_s_struct['tau'].squeeze()),
	alpha_d=float(prm_s_struct['alpha_d'].squeeze()),
	sigma0=float(prm_s_struct['sigma0'].squeeze()),
	alpha=float(prm_s_struct['alpha'].squeeze()),
	beta=float(prm_s_struct['beta'].squeeze()),
	b_out=float(prm_s_struct['b_out'].squeeze()),
	g_out=float(prm_s_struct['g_out'].squeeze()),
	theta=float(prm_s_struct['theta'].squeeze()),
	w_xs=float(prm_s_struct['w_xs'].squeeze())
)

# --- INFERENCE ---

# --- TIMING predict_lnk_rate ---

num_runs = 100

if USE_NEW_LNK_MODEL:
    print("Using new LNK model: compute_lnk_response_from_convolved")
    start_time = time.time()
    for _ in range(num_runs):
        rate_hat_s = compute_lnk_new(sim * 1e6, sim_s * 1e6, params_s, dt)
        # For single input case, create a params object without w_xs or set w_xs=0
        params_single = LNKParams(
            tau=params.tau, alpha_d=params.alpha_d, theta=params.theta,
            sigma0=params.sigma0, alpha=params.alpha, beta=params.beta,
            b_out=params.b_out, g_out=params.g_out, w_xs=0.0
        )
        rate_hat = compute_lnk_new(sim * 1e6, np.zeros_like(sim_s) * 1e6, params_single, dt)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    print(f"Average time per new LNK model run over {num_runs} runs: {avg_time:.6f} seconds")
else:
    print("Using original LNK model: predict_lnk_rate_two")
    start_time = time.time()
    for _ in range(num_runs):
        rate_hat_s, _ = predict_lnk_rate_two(sim * 1e6, sim_s * 1e6, params_s, dt)
        rate_hat, _ = predict_lnk_rate(sim * 1e6, params, dt)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    print(f"Average time per original LNK model run over {num_runs} runs: {avg_time:.6f} seconds")

# --- CORRELATION PRINTING ---
model_name = "New LNK Model" if USE_NEW_LNK_MODEL else "Original LNK Model"
print(f"\n--- {model_name} Results ---")

sim_exp_corr = np.corrcoef(sim, exp)[0, 1]
ratehat_exp_corr = np.corrcoef(rate_hat, exp)[0, 1]
ratehat_s_exp_corr = np.corrcoef(rate_hat_s, exp)[0, 1]
ratehat_r_hat_corr = np.corrcoef(rate_hat, r_hat)[0, 1]
ratehat_s_r_hat_corr = np.corrcoef(rate_hat_s, r_hat_s)[0, 1]
r_hat_s_exp_corr = np.corrcoef(exp, r_hat_s)[0, 1]

print(f"Correlation between sim and exp: {sim_exp_corr:.4f}")
print(f"Correlation between rate_hat and exp: {ratehat_exp_corr:.4f}")
print(f"Correlation between rate_hat_s and exp: {ratehat_s_exp_corr:.4f}")
print(f"Correlation between rate_hat and r_hat: {ratehat_r_hat_corr:.4f}")
print(f"Correlation between rate_hat_s and r_hat_s: {ratehat_s_r_hat_corr:.4f}")
print(f"Correlation between r_hat_s and exp: {r_hat_s_exp_corr:.4f}")

# --- COMPARISON ---
plt.figure(figsize=(10, 4))
plt.plot(exp, label='Experimental')
plt.plot(rate_hat_s, label=f'Simulated ({model_name})', alpha=0.7)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Rate')
plt.title(f'{model_name} Verification\nSim-Exp Corr: {sim_exp_corr:.4f}, RateHat-Exp Corr: {ratehat_exp_corr:.4f}')
plt.tight_layout()
plt.savefig(fig_save_file_path)
print(f"\nPlot saved to: {fig_save_file_path}")

# --- SUMMARY ---
print(f"\n--- Summary ---")
print(f"Model used: {model_name}")
print(f"Best correlation with experimental data: {max(ratehat_exp_corr, ratehat_s_exp_corr):.4f}")
print(f"Performance metric (rate_hat_s vs exp): {ratehat_s_exp_corr:.4f}")
if TORCH_AVAILABLE and USE_NEW_LNK_MODEL:
    print("✓ New LNK model executed successfully!")
elif not TORCH_AVAILABLE:
    print("⚠ PyTorch not available - using original model only")
else:
    print("Using original LNK model as requested")
