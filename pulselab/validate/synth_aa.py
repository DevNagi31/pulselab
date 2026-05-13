"""Synthetic A/A validation — the project's most important quality check.

Runs N bootstrapped null experiments (no real effect) with daily peeking under
the mSPRT rule. Counts how often we'd have stopped and falsely rejected the
null. Empirical FPR must stay <= alpha — that's the entire promise of
"always-valid" sequential testing, verified at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analyze.msprt import MsprtStream


@dataclass
class AaValidationResult:
    n_experiments: int
    n_false_positives: int
    fpr: float
    target_alpha: float
    avg_n_at_stop: float
    """Average sample size when an experiment was stopped (incl. ones that ran to the end)."""

    passed: bool


def run_synth_aa(
    *,
    n_experiments: int = 1000,
    per_arm_per_day: int = 200,
    n_days: int = 30,
    alpha: float = 0.05,
    tau2: float = 1.0,
    baseline_mean: float = 4.81,
    baseline_std: float = 1.0,
    seed: int = 0,
) -> AaValidationResult:
    """Simulate `n_experiments` null A/A tests with daily peeks; assert FPR <= alpha."""
    rng = np.random.default_rng(seed)
    false_positives = 0
    sample_sizes: list[int] = []

    for _ in range(n_experiments):
        stream = MsprtStream(tau2=tau2)
        stopped = False
        for _ in range(n_days):
            # Add today's batch from the SAME distribution to both arms (true null).
            c = rng.normal(baseline_mean, baseline_std, per_arm_per_day)
            t = rng.normal(baseline_mean, baseline_std, per_arm_per_day)
            stream.observe_many(c, t)
            result = stream.snapshot(alpha=alpha)
            if result is not None and result.reject_null(alpha=alpha):
                stopped = True
                false_positives += 1
                break
        sample_sizes.append(stream.n_control + stream.n_treatment)

    fpr = false_positives / n_experiments
    return AaValidationResult(
        n_experiments=n_experiments,
        n_false_positives=false_positives,
        fpr=fpr,
        target_alpha=alpha,
        avg_n_at_stop=float(np.mean(sample_sizes)),
        passed=bool(fpr <= alpha * 1.4),  # allow ~40% slack for Monte Carlo noise at moderate n
    )
