"""SRM detector must flag biased splits but not panic on healthy ones."""
from __future__ import annotations

import pytest

from pulselab.analyze.srm import srm_check


def test_srm_healthy_50_50_split_passes():
    r = srm_check([5012, 4988])
    assert r.healthy
    assert r.p_value > 0.01


def test_srm_obvious_mismatch_caught():
    """49.2 / 50.8 on n=200K should be flagged — that's the canonical real-world bug."""
    r = srm_check([98400, 101600])
    assert not r.healthy
    assert r.p_value < 1e-3


def test_srm_extreme_mismatch_caught():
    r = srm_check([4000, 6000])
    assert not r.healthy
    assert r.p_value < 1e-9


def test_srm_three_arm_split():
    """Three-arm 33/33/33 traffic should pass when balanced."""
    r = srm_check([3340, 3320, 3340], expected_ratio=[1, 1, 1])
    assert r.healthy


def test_srm_three_arm_unbalanced_expected():
    """50/25/25 design with matching observed allocation should pass."""
    r = srm_check([10000, 5050, 4950], expected_ratio=[2, 1, 1])
    assert r.healthy


def test_srm_rejects_bad_inputs():
    with pytest.raises(ValueError):
        srm_check([100])  # need >= 2 arms
    with pytest.raises(ValueError):
        srm_check([-1, 100])  # negative
    with pytest.raises(ValueError):
        srm_check([0, 0])  # total zero
    with pytest.raises(ValueError):
        srm_check([100, 100], expected_ratio=[1])  # length mismatch


def test_srm_summary_has_status():
    r = srm_check([4000, 6000])
    s = r.summary()
    assert "MISMATCH" in s and "χ²" in s
