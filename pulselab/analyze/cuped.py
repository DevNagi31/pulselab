"""CUPED variance reduction (Deng, Xu, Kohavi, Walker 2013).

If users have a pre-experiment covariate Y' that's correlated with the metric Y
during the experiment, we can shrink the variance of the treatment effect
estimate by a factor of (1 - rho^2) where rho = corr(Y, Y'). For typical
products, retention/spend has rho ~ 0.6 with its pre-period value, giving
~36% smaller CIs — for free, no statistical cost.

Formula:
    Y_cuped = Y - theta * (Y' - mean(Y'))
    theta_hat = Cov(Y, Y') / Var(Y')

The treatment-effect estimate on Y_cuped is unbiased and has lower variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CupedResult:
    theta: float
    """Optimal coefficient — typically a number near corr(Y, Y') * sd(Y)/sd(Y')."""

    naive_effect: float
    naive_var: float

    cuped_effect: float
    cuped_var: float

    variance_reduction: float
    """Fractional reduction: 1 - cuped_var / naive_var. Same as rho^2 in expectation."""


def cuped_adjust(
    y_treatment: np.ndarray,
    y_control: np.ndarray,
    pre_treatment: np.ndarray,
    pre_control: np.ndarray,
) -> CupedResult:
    """Compute the CUPED-adjusted treatment effect and naive baseline.

    Inputs are 1-D arrays of equal-length-within-arm:
      y_treatment, y_control:   in-experiment metric values per user
      pre_treatment, pre_control: pre-experiment covariate per user (same units)
    """
    y_t = np.asarray(y_treatment, dtype=float)
    y_c = np.asarray(y_control, dtype=float)
    p_t = np.asarray(pre_treatment, dtype=float)
    p_c = np.asarray(pre_control, dtype=float)
    if y_t.shape != p_t.shape or y_c.shape != p_c.shape:
        raise ValueError("pre-period and in-experiment arrays must align per user")

    # Pool everyone to estimate theta. Center the covariate to keep effect
    # estimates unbiased after subtraction.
    y_all = np.concatenate([y_t, y_c])
    p_all = np.concatenate([p_t, p_c])
    p_mean = float(np.mean(p_all))

    p_centered = p_all - p_mean
    var_p = float(np.var(p_centered, ddof=1))
    if var_p == 0.0:
        theta = 0.0
    else:
        cov = float(np.cov(y_all, p_centered, ddof=1)[0, 1])
        theta = cov / var_p

    y_t_adj = y_t - theta * (p_t - p_mean)
    y_c_adj = y_c - theta * (p_c - p_mean)

    naive_effect = float(np.mean(y_t) - np.mean(y_c))
    cuped_effect = float(np.mean(y_t_adj) - np.mean(y_c_adj))

    naive_var = float(np.var(y_t, ddof=1) / len(y_t) + np.var(y_c, ddof=1) / len(y_c))
    cuped_var = float(
        np.var(y_t_adj, ddof=1) / len(y_t_adj) + np.var(y_c_adj, ddof=1) / len(y_c_adj)
    )

    reduction = 1.0 - cuped_var / naive_var if naive_var > 0 else 0.0
    return CupedResult(
        theta=theta,
        naive_effect=naive_effect,
        naive_var=naive_var,
        cuped_effect=cuped_effect,
        cuped_var=cuped_var,
        variance_reduction=reduction,
    )
