#!/bin/bash
# ===========================================================================
# Job-array version: distribute seeds across MANY independent Slurm tasks.
#
# This is the "scale beyond one node" lesson. Each array task owns a slice of
# the seeds, runs them (using its own few cores), and writes one part file.
# A separate aggregation step combines the parts into the final plot.
#
# Here: 10 tasks x 20 seeds/task = 200 seeds total.
#
# Submit with:
#   sbatch discovery_array.slurm
# Then, once the array finishes, aggregate (see the dependency tip in README):
#   python plot.py results/part_*.npz --out results/regret.png
# ===========================================================================
#SBATCH --account=<project_id>       # e.g. ttrojan_123
#SBATCH --partition=main
#SBATCH --array=0-9                   # 10 array tasks, IDs 0..9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # cores per array task
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=mab-array
#SBATCH --output=logs/%x-%A_%a.out    # %A = array job id, %a = array task id

set -euo pipefail
mkdir -p logs results

module purge
eval "$(conda shell.bash hook)"
conda activate mab

SEEDS_PER_TASK=20
OFFSET=$(( SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK ))

echo "Array task ${SLURM_ARRAY_TASK_ID}: seeds [${OFFSET}, $((OFFSET + SEEDS_PER_TASK)))"

python run.py \
    --means 0.9,0.85,0.8,0.75,0.7,0.5 \
    --horizon 2000 \
    --n-seeds "${SEEDS_PER_TASK}" \
    --seed-offset "${OFFSET}" \
    --algos ucb1 thompson epsilon_greedy \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --out "results/part_${SLURM_ARRAY_TASK_ID}.npz"
