#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- SAFEGUARD ---
# Check if a virtual environment is active. If not, exit.
if [ -z "$VIRTUAL_ENV" ]; then
  echo "ERROR: No virtual environment is active."
  echo "Please create and activate one before running this script."
  echo "Example: python3 -m venv .venv && source .venv/bin/activate"
  exit 1
fi
# --- END SAFEGUARD ---

echo "Checking for NVIDIA GPU..."

# The core logic: Check for the nvidia-smi command.
if nvidia-smi &> /dev/null; then
  # If it succeeds, a GPU is present.
  echo "✅ GPU found. Installing GPU-enabled JAX."
  
  # Step 1: Install the correct JAX version FIRST.
  uv pip install --upgrade \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    "jax[cuda12]"
  
  # Step 2: Downgrade nvidia-cublas-cu12 to the compatible version
  echo "Ensuring nvidia-cublas-cu12 has compatible version..."
  uv pip install nvidia-cublas-cu12==12.9.0.13 --force-reinstall
  
else
  # If it fails, no GPU is found.
  echo "ℹ️ No GPU found. Installing CPU-only JAX."
  
  # Step 1: Install the correct JAX version FIRST.
  uv pip install --upgrade "jax[cpu]"
fi

# Step 3: Now that JAX is installed, install all other dependencies.
echo "Installing remaining packages from requirements.txt..."
uv pip install -r requirements.txt

echo "✅ Installation complete."

# Optional: Verify JAX devices after installation
echo "Verifying JAX installation..."
python -c "import jax; print('JAX devices:', jax.devices())"

# Verify nvidia-cublas version for GPU installations
if nvidia-smi &> /dev/null; then
  echo "Verifying nvidia-cublas installation..."
  uv pip show nvidia-cublas-cu12 | grep Version || echo "Could not verify nvidia-cublas-cu12 version"
fi