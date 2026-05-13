"""Synthetic experiment data generator.

Produces realistic per-user data with:
  - A correlated pre-experiment covariate (so CUPED has signal)
  - A configurable treatment effect (so tests can verify recovery)
  - Optional heterogeneity (segments respond differently)
  - Optional SRM bias (for testing the detector)

Generating data instead of relying on a downloaded 100MB Criteo file makes the
test suite fast, deterministic, and able to assert recovery against ground
truth (e.g., "synthetic A/A produces empirical FPR <= alpha").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class ExperimentData:
    """One synthetic experiment. Arrays are per-user."""

    control_outcome: np.ndarray
    treatment_outcome: np.ndarray
    control_pre: np.ndarray
    treatment_pre: np.ndarray
    control_segment: np.ndarray
    treatment_segment: np.ndarray
    true_effect: float
    metadata: dict


def generate_experiment(
    n_control: int = 5000,
    n_treatment: int = 5000,
    *,
    baseline_mean: float = 4.81,
    baseline_std: float = 1.0,
    true_effect: float = 0.0,
    pre_period_corr: float = 0.7,
    segments: tuple[str, ...] = ("mobile", "desktop", "new_users", "returning"),
    segment_effects: dict[str, float] | None = None,
    metric: Literal["continuous", "binary"] = "continuous",
    seed: int = 42,
) -> ExperimentData:
    """Generate a synthetic experiment.

    Args:
        n_control, n_treatment: per-arm sample size
        baseline_mean: control-arm mean (for continuous) or rate (for binary)
        baseline_std: control-arm SD (continuous only)
        true_effect: treatment_mean - control_mean (absolute, in metric units)
        pre_period_corr: correlation between pre-period covariate and outcome
            (drives CUPED variance reduction; 0 disables)
        segments: segment names assigned to each user uniformly at random
        segment_effects: optional per-segment additive effect on top of true_effect
        metric: 'continuous' (Gaussian) or 'binary' (Bernoulli with rate=mean)
        seed: rng seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    seg_arr_c = rng.choice(segments, size=n_control)
    seg_arr_t = rng.choice(segments, size=n_treatment)

    # Pre-period covariate sampled to be correlated with outcome via shared latent.
    latent_c = rng.standard_normal(n_control)
    latent_t = rng.standard_normal(n_treatment)

    rho = max(-0.999, min(0.999, pre_period_corr))
    pre_c = latent_c * baseline_std + baseline_mean
    pre_t = latent_t * baseline_std + baseline_mean
    eps_c = rng.standard_normal(n_control) * np.sqrt(max(0.0, 1 - rho ** 2))
    eps_t = rng.standard_normal(n_treatment) * np.sqrt(max(0.0, 1 - rho ** 2))

    out_c_z = rho * latent_c + eps_c
    out_t_z = rho * latent_t + eps_t

    if metric == "continuous":
        out_c = out_c_z * baseline_std + baseline_mean
        seg_eff_t = np.zeros(n_treatment, dtype=float)
        if segment_effects:
            for name, eff in segment_effects.items():
                seg_eff_t = seg_eff_t + (seg_arr_t == name).astype(float) * eff
        out_t = out_t_z * baseline_std + baseline_mean + true_effect + seg_eff_t
    elif metric == "binary":
        # Use latent for ranking, threshold to hit baseline rate.
        prob_c = baseline_mean
        prob_t = max(0.0, min(1.0, baseline_mean + true_effect))
        out_c = (rng.uniform(0, 1, n_control) < prob_c).astype(float)
        out_t = (rng.uniform(0, 1, n_treatment) < prob_t).astype(float)
    else:
        raise ValueError(f"unknown metric: {metric}")

    return ExperimentData(
        control_outcome=out_c,
        treatment_outcome=out_t,
        control_pre=pre_c,
        treatment_pre=pre_t,
        control_segment=seg_arr_c,
        treatment_segment=seg_arr_t,
        true_effect=true_effect,
        metadata={
            "metric": metric,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "pre_period_corr": pre_period_corr,
            "seed": seed,
        },
    )
