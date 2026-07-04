# Multi-Armed Bandits on CARC Discovery — a parallel-computing tutorial

A minimal, self-contained example for a CARC tutorial: run several bandit
algorithms (**UCB1**, **Thompson sampling**, **ε-greedy**) across many random
seeds, then average the cumulative-regret curves. Seeds are independent, so the
work is embarrassingly parallel — a natural fit for HPC.

## Files

| File                    | Purpose                                                       |
|-------------------------|---------------------------------------------------------------|
| `mab.py`                | Bandit environment + algorithms + single-experiment runner.   |
| `run.py`                | Runs many `(algorithm, seed)` experiments in parallel; saves `.npz`. |
| `plot.py`               | Aggregates result file(s) across seeds and plots regret curves. |
| `discovery.slurm`       | **Approach A** — one node, many cores (Python process pool).  |
| `discovery_array.slurm` | **Approach B** — Slurm job array, seeds spread across tasks.  |
| `environment.yml`       | Conda environment (`numpy`, `matplotlib`).                    |

## One-time setup on Discovery

Log in, grab an interactive node for the install (don't build envs on the login
node), then create the environment:

```bash
salloc --partition=main --cpus-per-task=4 --mem=8G --time=1:00:00 --account=<project_id>
module purge
module load conda
conda env create -f environment.yml     # creates the "mab" env
exit                                     # release the interactive allocation
```

Replace `<project_id>` with your account (e.g. `ttrojan_123`; run `myaccount`
to list yours). Edit the `#SBATCH --account=` line in both `.slurm` files too.

## Approach A — single node, many cores

Each `(algorithm, seed)` experiment runs on one of the CPUs Slurm allocated.
`run.py` sizes its process pool from `SLURM_CPUS_PER_TASK`.

```bash
sbatch discovery.slurm
myqueue                     # watch it; empty output means finished
```

Output: `results/regret.png` plus `logs/mab-regret-<jobid>.out`. To feel the
speedup, try `--cpus-per-task=1` vs `16` and compare the reported runtime.

## Approach B — job array (scale past one node)

Ten tasks each handle 20 seeds and write their own part file. Tasks are
scheduled independently and can land on different nodes.

```bash
sbatch discovery_array.slurm
# after it finishes, aggregate all parts into one figure:
python plot.py results/part_*.npz --out results/regret.png
```

To aggregate automatically when the array completes, submit the plot as a
dependent job:

```bash
JID=$(sbatch --parsable discovery_array.slurm)
sbatch --dependency=afterok:$JID --account=<project_id> --partition=main \
       --wrap "module purge; eval \"\$(conda shell.bash hook)\"; conda activate mab; \
               python plot.py results/part_*.npz --out results/regret.png"
```

## Run locally first (recommended before the tutorial)

```bash
python run.py --horizon 500 --n-seeds 40 --out results/all.npz
python plot.py results/all.npz --out results/regret.png
```

## Tuning knobs

- `--means`   comma-separated true arm means (the gaps set the difficulty).
- `--horizon` pulls per experiment, `T`.
- `--n-seeds` / `--seed-offset` number of replications and where they start.
- `--algos`   any of `ucb1`, `thompson`, `epsilon_greedy`.

## Notes for the tutorial

- **Reproducible parallelism:** each run derives two independent RNG streams
  from its seed via `numpy.random.SeedSequence(seed).spawn(2)` — one for the
  environment, one for the algorithm. Same results whether run serially or on
  100 cores, and no shared global RNG state between workers.
- **Why regret averages cleanly:** regret is measured against the true best arm
  (`best_mean − mean[chosen]`), so curves are smooth and comparable across seeds.
- **Right-sizing requests:** more `--cpus-per-task` = shorter runtime but longer
  queue wait and more core-hours charged. Good live discussion point.
# CARC_tutorial
