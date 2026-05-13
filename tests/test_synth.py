"""Synthetic data generator must produce data with the claimed properties."""
from __future__ import annotations

import numpy as np

from pulselab.data.synth import generate_experiment


def test_synth_recovers_true_effect():
    exp = generate_experiment(
        n_control=5000, n_treatment=5000, true_effect=0.2, seed=0
    )
    observed = exp.treatment_outcome.mean() - exp.control_outcome.mean()
    assert abs(observed - 0.2) < 0.05


def test_synth_correlation_matches_request():
    exp = generate_experiment(
        n_control=5000, n_treatment=5000, pre_period_corr=0.7, seed=0
    )
    # Correlation should be close to 0.7 in control arm.
    r = np.corrcoef(exp.control_outcome, exp.control_pre)[0, 1]
    assert 0.6 < r < 0.8


def test_synth_binary_metric_respects_rate():
    exp = generate_experiment(
        n_control=10000,
        n_treatment=10000,
        baseline_mean=0.1,
        true_effect=0.0,
        metric="binary",
        seed=0,
    )
    assert 0.08 < exp.control_outcome.mean() < 0.12
    # Outcomes are 0/1
    assert set(np.unique(exp.control_outcome)) <= {0.0, 1.0}


def test_synth_segment_effects_applied():
    exp = generate_experiment(
        n_control=2000,
        n_treatment=2000,
        true_effect=0.0,
        segment_effects={"mobile": 0.5},
        seed=0,
    )
    mobile_mask = exp.treatment_segment == "mobile"
    other_mask = ~mobile_mask
    assert exp.treatment_outcome[mobile_mask].mean() - exp.control_outcome.mean() > 0.3
    assert abs(exp.treatment_outcome[other_mask].mean() - exp.control_outcome.mean()) < 0.15


def test_synth_seed_reproducible():
    a = generate_experiment(n_control=100, n_treatment=100, seed=42)
    b = generate_experiment(n_control=100, n_treatment=100, seed=42)
    assert np.allclose(a.control_outcome, b.control_outcome)
    assert np.allclose(a.treatment_outcome, b.treatment_outcome)
