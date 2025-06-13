#!/bin/bash

#SBATCH --job-name=peter_grl_n2
#SBATCH --output=logs/wandb_sweep_%A_%a.out # Log files for each agent
#SBATCH --error=logs/wandb_sweep_%A_%a.err
#SBATCH --cpus-per-task=1           # Number of CPUs, adjust as needed
#SBATCH --mem=4gb            # Memory per CPU, ensures total memory is sufficient
#SBATCH --gres=shard:2           # Request 2 GPU shards (1GB each) per agent
#SBATCH --time=08:00:00             # Max runtime for each agent job, adjust as needed
#SBATCH --array=1-1                 # Start 8 parallel agents

# --- W&B Configuration ---
# Your W&B API Key
# This is the recommended way to provide credentials on a cluster.
export WANDB_API_KEY="397ebeb1fec1e8594788be5476bf0214807749a4"

# IMPORTANT: Paste the Sweep ID you get from the 'wandb sweep' command below.
export SWEEP_ID="pkoepernik-university-of-oxford/grl-gvd-analytical-experiments/is2zrtzu"

# Your W&B entity and project name
export WANDB_ENTITY="pkoepernik-university-of-oxford" # Or your team name
export WANDB_PROJECT="grl-gvd-analytical-experiments" # Must match the project in the python script

# --- JAX GPU Memory Configuration ---
# By default, JAX pre-allocates ~90% of GPU memory.
# To run multiple jobs on one GPU, we must change this.
# 'platform' allocator tells JAX to allocate memory on-demand instead of pre-allocating.
# This is the most flexible way to share a GPU between multiple processes.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# --- End User Configuration ---

echo "Starting W&B agent for sweep: $SWEEP_ID"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

# Create log directory if it doesn't exist
mkdir -p logs

# Add the current directory (project root) to Python's search path.
# This ensures that imports from the root, like 'definitions', can be found.
# This is still good practice even when using the correct Python.
export PYTHONPATH=$PWD:$PYTHONPATH

# Execute the W&B agent
# The agent will automatically pick up hyperparameters from the sweep config.
# Without the --count flag, the agent will run continuously until the sweep is finished or stopped.
cd /nas/ucb/peterkoepernik/grl
source .venv/bin/activate
#python -m wandb agent --count 1 ${SWEEP_ID}
uv run wandb agent --count 1 ${SWEEP_ID}

echo "W&B agent finished."
