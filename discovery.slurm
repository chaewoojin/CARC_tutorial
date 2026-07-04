#!/bin/bash
# ===========================================================================
# Single-node, multi-core bandit experiment on CARC Discovery.
#
# One job, N CPUs. Python spreads the (algorithm x seed) experiments across
# those N cores with a process pool, then plots the averaged regret curves.
#
# Submit with:   sbatch discovery.slurm
# ===========================================================================
#SBATCH --account=<project_id>       # e.g. ttrojan_123  (run `myaccount` to see yours)
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # <- pool size; regret runs use this many cores
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=mab-regret
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job id

set -euo pipefail
mkdir -p logs results

# --- Environment ----------------------------------------------------------
# CARC provides Conda via a module; activate a pre-built env (see README).
module purge
eval "$(conda shell.bash hook)"
conda activate mab

echo "Host: $(hostname)  |  CPUs allocated: ${SLURM_CPUS_PER_TASK}"

# --- Run ------------------------------------------------------------------
# run.py reads SLURM_CPUS_PER_TASK automatically, but we pass it explicitly
# to make the tutorial connection obvious.
python run.py \
    --means 0.9,0.85,0.8,0.75,0.7,0.5 \
    --horizon 2000 \
    --n-seeds 200 \
    --algos ucb1 thompson epsilon_greedy \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --out results/all.npz

python plot.py results/all.npz --out results/regret.png

echo "Done. See results/regret.png"
