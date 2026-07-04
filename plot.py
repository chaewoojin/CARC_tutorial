"""Aggregate one or more result files and plot averaged regret curves.

Works for both workflows:
  * single-node run  -> one .npz file
  * Slurm job array  -> many part_*.npz files (seeds concatenated per algorithm)

Usage::

    python plot.py results/all.npz --out results/regret.png
    python plot.py results/part_*.npz --out results/regret.png
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend: no display on compute nodes
import matplotlib.pyplot as plt  # noqa: E402


def load(paths):
    """Return {algo: regret array of shape (total_seeds, horizon)}."""
    collected: dict[str, list[np.ndarray]] = {}
    for path in paths:
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            if key.startswith("regret__"):
                algo = key[len("regret__"):]
                collected.setdefault(algo, []).append(data[key])
    return {algo: np.concatenate(chunks, axis=0)
            for algo, chunks in collected.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="Result .npz file(s) or glob(s).")
    p.add_argument("--out", default="results/regret.png", help="Output image.")
    p.add_argument("--title", default="Cumulative regret (mean over seeds)")
    args = p.parse_args()

    # Expand any globs the shell did not expand.
    paths = sorted({f for pattern in args.inputs for f in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No files matched: {args.inputs}")
    print(f"[plot] aggregating {len(paths)} file(s)")

    regret = load(paths)

    plt.figure(figsize=(8, 5))
    for algo in sorted(regret):
        curves = regret[algo]                       # (n_seeds, horizon)
        n = curves.shape[0]
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        steps = np.arange(1, mean.size + 1)
        line, = plt.plot(steps, mean, label=f"{algo} (n={n})")
        # 95% confidence band around the mean curve.
        plt.fill_between(steps, mean - 1.96 * sem, mean + 1.96 * sem,
                         color=line.get_color(), alpha=0.2)

    plt.xlabel("Time step")
    plt.ylabel("Cumulative regret")
    plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[plot] saved -> {args.out}")


if __name__ == "__main__":
    main()
