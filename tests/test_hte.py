"""Heterogeneous treatment effects + Benjamini-Hochberg."""
from __future__ import annotations

import numpy as np
import pytest

from pulselab.analyze.hte import benjamini_hochberg, segment_effects


def test_bh_monotone():
    """Sorted input → sorted adjusted output."""
    p = [0.001, 0.01, 0.03, 0.04, 0.5]
    adj = benjamini_hochberg(p, q=0.05)
    assert all(adj[i] <= adj[i + 1] for i in range(len(adj) - 1))
    assert all(0 <= a <= 1 for a in adj)


def test_bh_empty():
    assert benjamini_hochberg([]) == []


def test_bh_all_significant_when_small():
    """Five p-values all at 0.001 — BH should keep them significant at q=0.05."""
    adj = benjamini_hochberg([0.001] * 5, q=0.05)
    assert all(a <= 0.05 for a in adj)


def test_bh_no_false_discoveries_under_null():
    """Uniform null p-values rarely produce BH-significant results."""
    rng = np.random.default_rng(0)
    n_runs = 200
    false_disc = 0
    for _ in range(n_runs):
        ps = list(rng.uniform(0, 1, 10))
        adj = benjamini_hochberg(ps, q=0.05)
        false_disc += sum(1 for a in adj if a <= 0.05)
    # Across 200 runs of 10 null p-values, expect ~10 false discoveries on average (q=0.05).
    assert false_disc < 100  # very generous bound


def test_segment_effects_detects_real_lift():
    rng = np.random.default_rng(42)
    segments = {
        "mobile": (rng.normal(0, 1, 1000), rng.normal(0.3, 1, 1000)),  # real lift
        "desktop": (rng.normal(0, 1, 1000), rng.normal(0, 1, 1000)),  # no lift
    }
    out = segment_effects(segments, q=0.05)
    mobile = next(e for e in out if e.segment == "mobile")
    desktop = next(e for e in out if e.segment == "desktop")
    assert mobile.significant
    assert mobile.effect > 0
    assert not desktop.significant


def test_segment_effects_skips_tiny_segments():
    """Segments with <2 obs per arm get silently dropped (not errored)."""
    segments = {
        "tiny": (np.array([1.0]), np.array([1.0, 2.0])),  # control has 1 obs
        "normal": (np.zeros(100), np.ones(100)),
    }
    out = segment_effects(segments, q=0.05)
    names = [e.segment for e in out]
    assert "tiny" not in names
    assert "normal" in names
