#!/bin/bash

#SBATCH --job-name=peter_grl_n4
#SBATCH --output=logs/parallel_sweep_%j.out # Single output file for the job
#SBATCH --error=logs/parallel_sweep_%j.err
#SBATCH --cpus-per-task=16           # Number of CPUs, adjust as needed
#SBATCH --mem=32gb            # Memory per CPU, ensures total memory is sufficient
#SBATCH --time=08:00:00             # Max runtime for each agent job, adjust as needed
#SBATCH --gres=gpu:A6000:1  # Requesting one exclusive A6000 GPU.

# --- W&B Configuration ---
# Your W&B API Key
# This is the recommended way to provide credentials on a cluster.
export WANDB_API_KEY="397ebeb1fec1e8594788be5476bf0214807749a4"

# IMPORTANT: Paste the Sweep ID you get from the 'wandb sweep' command below.
export SWEEP_ID="pkoepernik-university-of-oxford/grl-gvd-analytical-experiments/q4z12ha0"

# Your W&B entity and project name
export WANDB_ENTITY="pkoepernik-university-of-oxford" # Or your team name
export WANDB_PROJECT="grl-gvd-analytical-experiments" # Must match the project in the python script


# --- JAX GPU Memory Configuration ---
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# --- End User Configuration ---

echo "Starting parallel agent manager..."

# Add the project root to Python's search path
PROJECT_ROOT="/nas/ucb/peterkoepernik/grl"
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH}

cd ${PROJECT_ROOT}
source .venv/bin/activate

# Execute the parallel agent manager.
# It will launch 16 parallel workers inside this single Slurm job.
uv run parallel_agent.py --sweep_id ${SWEEP_ID} --num_workers 16 --project_root ${PROJECT_ROOT}

echo "Parallel agent manager finished."

