"""CUPED must produce smaller CIs when the pre-period covariate is correlated."""
from __future__ import annotations

import numpy as np
import pytest

from pulselab.analyze.cuped import cuped_adjust
from pulselab.data.synth import generate_experiment


def test_cuped_shrinks_variance_when_correlated():
    """At rho=0.7, expect ~50% variance reduction (1 - 0.7^2 = 0.51)."""
    exp = generate_experiment(
        n_control=5000,
        n_treatment=5000,
        true_effect=0.1,
        pre_period_corr=0.7,
        seed=42,
    )
    r = cuped_adjust(
        exp.treatment_outcome,
        exp.control_outcome,
        exp.treatment_pre,
        exp.control_pre,
    )
    # The naive estimator's variance must be reduced.
    assert r.cuped_var < r.naive_var
    # Variance reduction should be close to rho^2 = 0.49.
    assert 0.3 < r.variance_reduction < 0.7


def test_cuped_no_reduction_when_uncorrelated():
    """With rho=0 the pre-period covariate is noise; reduction should be ~0."""
    exp = generate_experiment(
        n_control=3000,
        n_treatment=3000,
        true_effect=0.0,
        pre_period_corr=0.0,
        seed=1,
    )
    r = cuped_adjust(
        exp.treatment_outcome,
        exp.control_outcome,
        exp.treatment_pre,
        exp.control_pre,
    )
    # Variance reduction should hover near zero (within MC noise).
    assert abs(r.variance_reduction) < 0.05


def test_cuped_effect_estimate_remains_unbiased():
    """Centering the covariate keeps the effect estimate unbiased."""
    true_effect = 0.2
    exp = generate_experiment(
        n_control=10000,
        n_treatment=10000,
        true_effect=true_effect,
        pre_period_corr=0.6,
        seed=99,
    )
    r = cuped_adjust(
        exp.treatment_outcome,
        exp.control_outcome,
        exp.treatment_pre,
        exp.control_pre,
    )
    # CUPED effect should be within ~3 SE of the true effect.
    naive_se = np.sqrt(r.naive_var)
    assert abs(r.cuped_effect - true_effect) < 3 * naive_se


def test_cuped_handles_constant_covariate():
    """Zero-variance covariate must not blow up; theta should be 0."""
    n = 100
    y_t = np.random.default_rng(0).normal(0, 1, n)
    y_c = np.random.default_rng(1).normal(0, 1, n)
    pre = np.ones(n) * 5.0
    r = cuped_adjust(y_t, y_c, pre, pre)
    assert r.theta == 0.0
    # No adjustment → cuped_var equals naive_var
    assert r.cuped_var == pytest.approx(r.naive_var, rel=1e-9)


def test_cuped_rejects_misaligned_arrays():
    with pytest.raises(ValueError):
        cuped_adjust(np.zeros(10), np.zeros(10), np.zeros(5), np.zeros(10))
