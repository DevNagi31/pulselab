"""Delta-method variance for ratio metrics.

Conversion rate = sum(conversions) / sum(exposures), ARPU = sum(revenue) /
sum(users), etc. These are NOT sample means — they're ratios of sums.
A naive t-test on per-user binary outcomes gives wrong CIs at high traffic
concentration. The delta method linearizes around the population ratio and
returns the correct variance.

Formula (univariate ratio R = Y/X):
    Var(R) ≈ (1/X^2) * Var(Y) - 2*(Y/X^3)*Cov(X,Y) + (Y^2/X^4)*Var(X)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DeltaResult:
    ratio: float
    variance: float
    standard_error: float


def ratio_variance(
    numerator: np.ndarray, denominator: np.ndarray, *, ddof: int = 1
) -> DeltaResult:
    """Delta-method variance for sum(num) / sum(den).

    Each input is per-user (per-randomization-unit) values. For conversion rate
    on a binary outcome, numerator is 0/1 conversion flags and denominator is
    all 1s (or session counts when measuring per-session rates).
    """
    y = np.asarray(numerator, dtype=float)
    x = np.asarray(denominator, dtype=float)
    if y.shape != x.shape:
        raise ValueError("numerator and denominator must align")
    n = len(y)
    if n < 2:
        raise ValueError("need at least two observations")

    sum_y = float(np.sum(y))
    sum_x = float(np.sum(x))
    if sum_x == 0:
        raise ValueError("denominator sums to zero")

    ratio = sum_y / sum_x
    mean_y = sum_y / n
    mean_x = sum_x / n

    var_y = float(np.var(y, ddof=ddof))
    var_x = float(np.var(x, ddof=ddof))
    cov_xy = float(np.cov(x, y, ddof=ddof)[0, 1])

    # Variance of the ratio of sample means is var of ratio / n.
    # Standard delta-method expansion:
    variance = (
        var_y / (mean_x ** 2)
        - 2.0 * (mean_y / mean_x ** 3) * cov_xy
        + (mean_y ** 2 / mean_x ** 4) * var_x
    ) / n
    variance = max(0.0, variance)
    return DeltaResult(ratio=ratio, variance=variance, standard_error=np.sqrt(variance))
