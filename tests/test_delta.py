"""Delta-method variance for ratio metrics."""
from __future__ import annotations

import numpy as np
import pytest

from pulselab.analyze.delta import ratio_variance


def test_ratio_variance_reduces_to_proportion_for_unit_denom():
    """When denominator is all 1s, ratio = sample mean; variance is var/n."""
    rng = np.random.default_rng(0)
    y = (rng.uniform(0, 1, 5000) < 0.1).astype(float)
    x = np.ones_like(y)
    r = ratio_variance(y, x)
    # Sample mean ~ 0.1; variance ~ p*(1-p)/n
    expected_var = float(np.var(y, ddof=1) / len(y))
    assert r.variance == pytest.approx(expected_var, rel=0.02)
    assert r.ratio == pytest.approx(float(y.mean()), rel=1e-9)


def test_ratio_variance_session_based_conversion_differs_from_user_mean():
    """When users have variable session counts, ratio-of-sums variance differs."""
    rng = np.random.default_rng(1)
    sessions_per_user = rng.poisson(3, 1000) + 1
    conversions = rng.binomial(sessions_per_user, 0.05)
    r = ratio_variance(conversions, sessions_per_user)
    # Sanity: variance must be positive and standard error reasonable.
    assert r.variance > 0
    assert r.ratio == pytest.approx(conversions.sum() / sessions_per_user.sum(), rel=1e-9)


def test_ratio_variance_invariants():
    rng = np.random.default_rng(2)
    y = rng.normal(0.5, 0.1, 100)
    x = rng.normal(1.0, 0.1, 100)
    r = ratio_variance(y, x)
    assert r.standard_error >= 0
    assert r.variance >= 0


def test_ratio_variance_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ratio_variance(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError):
        ratio_variance(np.zeros(2), np.zeros(2))  # denominator sums to zero
    with pytest.raises(ValueError):
        ratio_variance(np.ones(1), np.ones(1))  # n < 2
