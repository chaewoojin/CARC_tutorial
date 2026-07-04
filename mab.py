"""Multi-armed bandit environments and algorithms.

Deliberately dependency-light (NumPy only) so a tutorial can focus on the
*parallel experiment workflow* rather than on a heavy RL stack.

Key ideas demonstrated here:
  * A Bernoulli bandit whose arm means are fixed and known to the environment.
  * Three classic algorithms: epsilon-greedy, UCB1, Thompson sampling.
  * "Expected regret" measured against the true best arm, which yields smooth,
    averageable curves -- ideal for comparing algorithms across many seeds.
  * Correct seeding for parallel runs via numpy.random.SeedSequence so that
    every (algorithm, seed) run is independent AND reproducible.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class BernoulliBandit:
    """A K-armed Bernoulli bandit.

    Each arm ``i`` returns reward 1 with probability ``means[i]`` and 0 otherwise.
    """

    def __init__(self, means, rng: np.random.Generator):
        self.means = np.asarray(means, dtype=float)
        self.k = int(self.means.size)
        self.rng = rng
        self.best_mean = float(self.means.max())

    def pull(self, arm: int) -> float:
        """Draw one reward from the chosen arm."""
        return float(self.rng.random() < self.means[arm])

    def gap(self, arm: int) -> float:
        """Per-step *expected* regret of choosing ``arm`` (best_mean - mean[arm])."""
        return self.best_mean - self.means[arm]


# --------------------------------------------------------------------------- #
# Small helper: argmax with random tie-breaking
# --------------------------------------------------------------------------- #
def argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    values = np.asarray(values)
    candidates = np.flatnonzero(values == values.max())
    return int(rng.choice(candidates))


# --------------------------------------------------------------------------- #
# Algorithms. Each exposes select(t) -> arm and update(arm, reward).
# --------------------------------------------------------------------------- #
class EpsilonGreedy:
    """With probability epsilon explore uniformly, otherwise exploit."""

    def __init__(self, k: int, rng: np.random.Generator, epsilon: float = 0.1):
        self.k = k
        self.rng = rng
        self.epsilon = epsilon
        self.counts = np.zeros(k)
        self.values = np.zeros(k)  # running mean reward per arm

    def select(self, t: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        return argmax_random(self.values, self.rng)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        # Incremental sample mean.
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class UCB1:
    """UCB1: pick argmax of empirical mean + sqrt(c * ln t / n_i)."""

    def __init__(self, k: int, rng: np.random.Generator, c: float = 2.0):
        self.k = k
        self.rng = rng
        self.c = c
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def select(self, t: int) -> int:
        # Play every arm once before trusting the confidence bound.
        for a in range(self.k):
            if self.counts[a] == 0:
                return a
        bonus = np.sqrt(self.c * np.log(t) / self.counts)
        return argmax_random(self.values + bonus, self.rng)

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class ThompsonSampling:
    """Beta-Bernoulli Thompson sampling with a uniform Beta(1, 1) prior."""

    def __init__(self, k: int, rng: np.random.Generator):
        self.k = k
        self.rng = rng
        self.alpha = np.ones(k)  # successes + 1
        self.beta = np.ones(k)   # failures + 1

    def select(self, t: int) -> int:
        theta = self.rng.beta(self.alpha, self.beta)
        return argmax_random(theta, self.rng)

    def update(self, arm: int, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward


# Registry so scripts can select algorithms by name from the command line.
ALGORITHMS = {
    "epsilon_greedy": EpsilonGreedy,
    "ucb1": UCB1,
    "thompson": ThompsonSampling,
}


# --------------------------------------------------------------------------- #
# One experiment = one (algorithm, seed) pair. This is the unit of parallelism.
# --------------------------------------------------------------------------- #
def run_experiment(algo_name: str, seed: int, means, horizon: int,
                   algo_kwargs: dict | None = None) -> np.ndarray:
    """Run a single bandit experiment and return its cumulative-regret curve.

    We derive two *independent* RNG streams from one seed:
      * env_rng   -> reward realizations
      * algo_rng  -> algorithm's internal randomness (exploration, TS samples)

    Because exactly one env draw happens per step, the environment's reward
    stream for a given seed is identical across algorithms, giving a clean
    paired comparison with lower variance.
    """
    env_seq, algo_seq = np.random.SeedSequence(seed).spawn(2)
    env_rng = np.random.default_rng(env_seq)
    algo_rng = np.random.default_rng(algo_seq)

    bandit = BernoulliBandit(means, env_rng)
    algo = ALGORITHMS[algo_name](bandit.k, algo_rng, **(algo_kwargs or {}))

    regret = np.empty(horizon, dtype=float)
    cumulative = 0.0
    for t in range(1, horizon + 1):
        arm = algo.select(t)
        reward = bandit.pull(arm)
        algo.update(arm, reward)
        cumulative += bandit.gap(arm)
        regret[t - 1] = cumulative
    return regret
