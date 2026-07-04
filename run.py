"""Run bandit experiments across many seeds in parallel and save the results.

This is the "parallel computing" centerpiece of the tutorial. Each
(algorithm, seed) pair is an embarrassingly-parallel task, so we hand them to a
process pool. On CARC, the pool size comes from SLURM_CPUS_PER_TASK, i.e. the
CPUs Slurm allocated to the job.

Examples
--------
Local / single-node (use all allocated cores)::

    python run.py --means 0.9,0.85,0.8,0.75,0.7,0.5 --horizon 2000 \
        --n-seeds 200 --algos ucb1 thompson epsilon_greedy \
        --out results/all.npz

One shard of a Slurm job array (seeds [offset, offset+n_seeds))::

    python run.py --n-seeds 20 --seed-offset 40 --out results/part_2.npz
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mab import ALGORITHMS, run_experiment


def default_workers() -> int:
    """Cores to use: prefer Slurm's allocation, then affinity, then all CPUs."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return int(slurm)
    try:
        return len(os.sched_getaffinity(0))  # respects cgroup/affinity limits
    except AttributeError:
        return os.cpu_count() or 1


def _run_task(task):
    """Top-level worker function (must be picklable for ProcessPoolExecutor)."""
    algo_name, seed, means, horizon = task
    curve = run_experiment(algo_name, seed, means, horizon)
    return algo_name, seed, curve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--means", default="0.9,0.85,0.8,0.75,0.7,0.5",
                   help="Comma-separated true arm means.")
    p.add_argument("--horizon", type=int, default=2000,
                   help="Number of pulls per experiment (T).")
    p.add_argument("--n-seeds", type=int, default=200,
                   help="Number of seeds (replications) to run.")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="First seed; useful to shard seeds across array tasks.")
    p.add_argument("--algos", nargs="+", default=list(ALGORITHMS),
                   choices=list(ALGORITHMS), help="Algorithms to evaluate.")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel worker processes (default: SLURM_CPUS_PER_TASK).")
    p.add_argument("--out", default="results/all.npz",
                   help="Output .npz path for the regret curves.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    means = np.array([float(x) for x in args.means.split(",")], dtype=float)
    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))
    workers = args.workers or default_workers()

    tasks = [(algo, seed, means, args.horizon)
             for algo in args.algos for seed in seeds]

    print(f"[run] {len(args.algos)} algorithms x {len(seeds)} seeds "
          f"= {len(tasks)} experiments (T={args.horizon}) on {workers} workers")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_run_task, tasks, chunksize=1))
    elapsed = time.perf_counter() - t0
    print(f"[run] finished in {elapsed:.1f}s "
          f"({len(tasks) / elapsed:.1f} experiments/s)")

    # Assemble into arrays of shape (n_seeds, horizon), one per algorithm.
    seed_index = {seed: i for i, seed in enumerate(seeds)}
    regret = {algo: np.empty((len(seeds), args.horizon)) for algo in args.algos}
    for algo, seed, curve in results:
        regret[algo][seed_index[seed]] = curve

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        means=means,
        seeds=np.array(seeds),
        horizon=np.array(args.horizon),
        algos=np.array(args.algos),
        **{f"regret__{algo}": regret[algo] for algo in args.algos},
    )
    print(f"[run] saved -> {args.out}")


if __name__ == "__main__":
    main()
