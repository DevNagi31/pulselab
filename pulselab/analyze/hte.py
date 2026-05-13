"""Heterogeneous treatment effects with Benjamini-Hochberg FDR correction.

When you slice an experiment by 8 segments and check each for significance,
the family-wise false-positive rate explodes. Bonferroni is too conservative;
Benjamini-Hochberg (BH) controls the false-discovery rate instead and is
the modern default for pre-registered subgroup analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import stats


@dataclass
class SegmentEffect:
    segment: str
    n_control: int
    n_treatment: int
    effect: float
    se: float
    p_value: float
    p_adjusted: float
    """Benjamini-Hochberg adjusted p-value (FDR-controlled at q=alpha)."""

    significant: bool


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[float]:
    """Return BH-adjusted p-values. Values <= q are significant under FDR control."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.array(p_values, dtype=float)[order]
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out.tolist()


def segment_effects(
    segments: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    q: float = 0.05,
) -> list[SegmentEffect]:
    """Compute per-segment Welch-t effects and BH-adjust the p-values.

    Args:
        segments: name -> (control_values, treatment_values) arrays.
        q: FDR control level. Segments with p_adjusted <= q are significant.
    """
    raw: list[SegmentEffect] = []
    for name, (ctrl, treat) in segments.items():
        c = np.asarray(ctrl, dtype=float)
        t = np.asarray(treat, dtype=float)
        if len(c) < 2 or len(t) < 2:
            continue
        mean_c = float(np.mean(c))
        mean_t = float(np.mean(t))
        se = float(np.sqrt(np.var(c, ddof=1) / len(c) + np.var(t, ddof=1) / len(t)))
        if se == 0.0:
            p = 1.0
        else:
            tstat, p = stats.ttest_ind(t, c, equal_var=False)
            p = float(p)
        raw.append(
            SegmentEffect(
                segment=name,
                n_control=len(c),
                n_treatment=len(t),
                effect=mean_t - mean_c,
                se=se,
                p_value=p,
                p_adjusted=p,  # placeholder, overwritten below
                significant=False,
            )
        )
    if not raw:
        return raw

    adj = benjamini_hochberg([e.p_value for e in raw], q=q)
    for e, a in zip(raw, adj):
        e.p_adjusted = a
        e.significant = a <= q
    return raw
