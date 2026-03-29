#!/bin/bash

# Load required modules:
module load StdEnv/2023 intel/2023.2.1 openmpi/4.1.5
module load nwchem/7.2.3
module load xtb/6.6.1
module load cuda/12.6

# Set UV cache directory to a tmp folder on the scratch filesystem
mkdir -p /scratch/$USER/my_tmp_2
cache_dir=/scratch/$USER/my_tmp_2
export UV_CACHE_DIR=$cache_dir
export CUPY_CACHE_DIR=$cache_dir

# NVIDIA MPS settings: use a per-user pipe (and log) directory so CUDA
# MPS clients don't accidentally try to connect to another user's MPS
# control daemon. This avoids "MPS client failed to connect to the MPS
# control daemon" errors when a system-wide or other-user MPS is running.
export CUDA_MPS_PIPE_DIRECTORY="$cache_dir/.mps"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
chmod 700 "$CUDA_MPS_PIPE_DIRECTORY"
# Optional log directory for MPS (keeps logs out of /tmp)
export CUDA_MPS_LOG_DIRECTORY="$cache_dir/.mps/log"
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
chmod 700 "$CUDA_MPS_LOG_DIRECTORY"


source .venv/bin/activate

# Set thread / BLAS / OpenMP environment variables so Python libs
# (numpy, scipy, numexpr, etc.) respect the desired thread counts.
# These are exported here so any process started after sourcing this
# script will inherit them.
# export OMP_NUM_THREADS="4"
# export OMP_THREAD_LIMIT="4"
# export OPENBLAS_NUM_THREADS="1"
# export MKL_NUM_THREADS="1"
# export NUMEXPR_NUM_THREADS="1"
# export OMP_STACKSIZE="1M"

# Export CUDA-related dynamic libraries (these Python CUDA libs
# will be under site-packages/nvidia since we use uv for
# environment management):

export LD_LIBRARY_PATH="$(uv run python - <<'PY'
import site, pathlib
p = pathlib.Path(site.getsitepackages()[0]) / 'nvidia'
print(':'.join(str((p/sub/'lib')) for sub in p.iterdir() if (p/sub/'lib').exists()))
PY
):${LD_LIBRARY_PATH}"

# # this was necessary to get deno to work on trillium jupyter
# source /home/$USER/.deno/env
