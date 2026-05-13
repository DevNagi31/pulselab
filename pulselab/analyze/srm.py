"""Sample Ratio Mismatch (SRM) detection.

When you want 50/50 traffic and actually observe 49.2/50.8, that's a *bug
indicator* — usually a logging or assignment failure. Catching this in the
analysis layer is one of the most valuable single features an experimentation
platform can ship; ignoring it produces real-but-fake "wins" all year.

Test: chi-square goodness-of-fit against the expected allocation. Convention
is to refuse to return any treatment-effect numbers if p_srm < 0.001.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scipy import stats


@dataclass
class SrmResult:
    observed: tuple[int, ...]
    expected: tuple[float, ...]
    chi2: float
    p_value: float
    healthy: bool
    """True iff p_value >= threshold (default 0.001). When False, do NOT compute
    treatment-effect estimates — investigate the assignment pipeline first."""

    def summary(self) -> str:
        obs = ", ".join(str(o) for o in self.observed)
        exp = ", ".join(f"{e:.0f}" for e in self.expected)
        status = "OK" if self.healthy else "MISMATCH"
        return f"SRM {status}: observed=[{obs}] expected=[{exp}] χ²={self.chi2:.2f} p={self.p_value:.4f}"


def srm_check(
    observed: Sequence[int],
    expected_ratio: Sequence[float] | None = None,
    *,
    threshold: float = 1e-3,
) -> SrmResult:
    """Chi-square goodness-of-fit against the expected traffic allocation.

    Args:
        observed: Per-arm observed counts (e.g., [n_control, n_treatment]).
        expected_ratio: Per-arm expected proportions. Defaults to equal split.
        threshold: p-value below which we flag SRM. Default 0.001 (Bing/Meta convention).
    """
    obs = tuple(int(x) for x in observed)
    if len(obs) < 2:
        raise ValueError("need at least two arms")
    if any(x < 0 for x in obs):
        raise ValueError("observed counts must be non-negative")

    total = sum(obs)
    if total == 0:
        raise ValueError("total count is zero")

    if expected_ratio is None:
        ratio = [1.0 / len(obs)] * len(obs)
    else:
        if len(expected_ratio) != len(obs):
            raise ValueError("expected_ratio length must match observed")
        s = sum(expected_ratio)
        if s <= 0:
            raise ValueError("expected_ratio must sum to a positive value")
        ratio = [r / s for r in expected_ratio]

    expected = tuple(total * r for r in ratio)
    chi2, p_value = stats.chisquare(obs, f_exp=expected)
    return SrmResult(
        observed=obs,
        expected=expected,
        chi2=float(chi2),
        p_value=float(p_value),
        healthy=bool(p_value >= threshold),
    )
