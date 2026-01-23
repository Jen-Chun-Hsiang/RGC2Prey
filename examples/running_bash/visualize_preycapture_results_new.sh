#!/bin/bash
set -euo pipefail

# Avoid picking up ~/.local packages
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Use a disposable venv inside the container FS (pick any writable path)
VENV_DIR="${SLURM_TMPDIR:-/tmp}/rgc2prey-venv"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Make sure pip is recent
python -m pip install --upgrade pip wheel setuptools

# Match your PyTorch 2.3.1 CUDA 12.1 image
# (torch itself is already in the image; no need to reinstall torch)
pip install "torchvision==0.18.1" "torchaudio==2.3.1" --extra-index-url https://download.pytorch.org/whl/cu121

# Headless OpenCV to avoid libGL.so.1
pip install --no-cache-dir "opencv-python-headless==4.10.0.84"

# The rest of your deps (pin if you like)
pip install --upgrade pillow
pip install pandas
pip install scipy
pip install scikit-image
pip install matplotlib
# pip install openpyxl  # if/when you need it

# Sanity check: ensure we're importing cv2 from the venv and that libGL isn't required
python - <<'PY'
import sys, cv2, subprocess
print("Python:", sys.executable)
print("cv2 path:", cv2.__file__)
print("Torch/TorchVision:", __import__("torch").__version__, __import__("torchvision").__version__)
try:
    out = subprocess.check_output(["ldd", cv2.__file__], text=True)
    print("ldd(cv2):\n", out)
except Exception as e:
    print("ldd check skipped:", e)
PY

# Run your job
python "$HOME/RGC2Prey/RGC2Prey/visual_preycapture_prediction.py" \
  --experiment_names '2025100602' \
  --epoch_number 200 \
  --save_movie_frames \
  --visual_sample_ids 53 \
  --movie_eye 'right' \
  --movie_input_channel 'second' \
  --test_bg_folder 'blend' \
  --boundary_size '(200, 120)' \
  --noise_levels 0.0 0.016 0.032 0.064 0.128 0.256 \
  --num_worker 24