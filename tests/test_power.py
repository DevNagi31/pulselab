"""Power / MDE calculators must match textbook formulas."""
from __future__ import annotations

import math

from pulselab.design.power import (
    mde_for_means,
    sample_size_for_means,
    sample_size_for_proportions,
)


def test_sample_size_means_matches_textbook():
    """Cohen 1988: per-arm n for 0.5 SD effect at alpha=0.05, power=0.8 is ~64."""
    r = sample_size_for_means(baseline_mean=10.0, baseline_std=1.0, effect=0.5)
    # (1.96 + 0.842)^2 * 2 / 0.25 ≈ 62.8, ceil → 63 (with two-sided test).
    assert 60 <= r.per_arm_n <= 70
    assert r.total_n == 2 * r.per_arm_n
    assert r.relative_effect == 0.5 / 10.0


def test_sample_size_proportions_5pct_baseline():
    """Detecting 0.5pp lift on a 5% baseline: ~31K per arm at 80% power."""
    r = sample_size_for_proportions(baseline_rate=0.05, absolute_lift=0.005)
    assert 25_000 < r.per_arm_n < 40_000


def test_mde_decreases_as_n_grows():
    """Larger n → smaller MDE."""
    a = mde_for_means(baseline_std=1.0, per_arm_n=100)
    b = mde_for_means(baseline_std=1.0, per_arm_n=10_000)
    assert b < a
    # Specifically, MDE scales as 1/sqrt(n), so 100x more samples → ~10x smaller.
    assert b * 8 < a < b * 12


def test_mde_matches_inverse_of_sample_size():
    """If we compute MDE for n, then ask for sample size to detect that MDE, we recover n."""
    n0 = 1000
    mde = mde_for_means(baseline_std=1.0, per_arm_n=n0)
    r = sample_size_for_means(baseline_mean=1.0, baseline_std=1.0, effect=mde)
    # Allow a few samples of slack due to ceiling.
    assert abs(r.per_arm_n - n0) <= 5
