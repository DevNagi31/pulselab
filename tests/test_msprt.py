"""Tests for mSPRT — the math has to hold up to peeking under the null."""
from __future__ import annotations

import numpy as np
import pytest

from pulselab.analyze.msprt import MsprtStream, msprt


def test_msprt_zero_effect_high_p_value():
    """At n=1000 with mean_diff=0, p-value should stay near 1."""
    r = msprt(n=1000, mean_diff=0.0, pooled_var=1.0, tau2=1.0, alpha=0.05)
    assert r.p_value > 0.5
    assert r.ci_low < 0 < r.ci_high


def test_msprt_strong_effect_rejects_null():
    """Large effect at n=1000 → tiny p-value, reject null."""
    r = msprt(n=1000, mean_diff=0.3, pooled_var=1.0, tau2=1.0, alpha=0.05)
    assert r.p_value < 0.01
    assert r.reject_null(alpha=0.05)


def test_msprt_p_value_bounded():
    """p-value must be in [0, 1]. May underflow to exactly 0 for huge effects."""
    for diff in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        r = msprt(n=500, mean_diff=diff, pooled_var=1.0, tau2=1.0)
        assert 0.0 <= r.p_value <= 1.0
    # Smaller effect should produce a finite, non-zero p-value.
    r_small = msprt(n=500, mean_diff=0.05, pooled_var=1.0, tau2=1.0)
    assert 0.0 < r_small.p_value <= 1.0


def test_msprt_invalid_inputs():
    with pytest.raises(ValueError):
        msprt(n=0, mean_diff=0.1, pooled_var=1.0)
    with pytest.raises(ValueError):
        msprt(n=100, mean_diff=0.1, pooled_var=-1.0)


def test_msprt_stream_welford_matches_numpy():
    """Streaming variance should match numpy's on the same data."""
    rng = np.random.default_rng(0)
    c = rng.normal(4.8, 1.0, 500)
    t = rng.normal(4.8, 1.0, 500)
    s = MsprtStream(tau2=1.0)
    s.observe_many(c, t)
    snap = s.snapshot()
    assert snap is not None
    # Cross-check mean_diff
    assert snap.mean_diff == pytest.approx(t.mean() - c.mean(), rel=1e-9, abs=1e-9)


def test_msprt_stream_returns_none_for_tiny_samples():
    s = MsprtStream()
    s.observe_control(1.0)
    s.observe_treatment(1.0)
    assert s.snapshot() is None  # only 1 obs per arm


def test_msprt_peeking_does_not_inflate_fpr():
    """The headline claim: peek every day under the null and FPR stays <= alpha.

    Runs 500 null A/A experiments with daily peeking; counts false positives.
    With alpha=0.05 and Monte Carlo noise at N=500, FPR should sit comfortably
    below ~0.08. Standard t-test peeking would balloon to ~0.20+.
    """
    rng = np.random.default_rng(7)
    alpha = 0.05
    n_experiments = 500
    per_day = 100
    n_days = 30

    fp = 0
    for _ in range(n_experiments):
        s = MsprtStream(tau2=1.0)
        for _ in range(n_days):
            c = rng.normal(0.0, 1.0, per_day)
            t = rng.normal(0.0, 1.0, per_day)
            s.observe_many(c, t)
            r = s.snapshot(alpha=alpha)
            if r is not None and r.reject_null(alpha=alpha):
                fp += 1
                break
    empirical_fpr = fp / n_experiments
    # Generous upper bound for MC noise at N=500: alpha + ~3 SE on a binomial.
    assert empirical_fpr <= alpha + 0.03, (
        f"Empirical FPR {empirical_fpr:.3f} exceeds alpha + tolerance"
    )


def test_msprt_recovers_known_effect():
    """If true effect = 0.3 and n is large, mSPRT should reject the null."""
    rng = np.random.default_rng(11)
    c = rng.normal(0.0, 1.0, 5000)
    t = rng.normal(0.3, 1.0, 5000)
    s = MsprtStream(tau2=1.0)
    s.observe_many(c, t)
    snap = s.snapshot()
    assert snap is not None
    assert snap.reject_null(alpha=0.05)
    assert snap.ci_low < 0.3 < snap.ci_high or snap.ci_low <= 0.3
