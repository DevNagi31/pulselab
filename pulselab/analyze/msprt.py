"""Always-valid sequential testing via the mixture Sequential Probability Ratio Test.

The single most important method in PulseLab. Standard A/B p-values are valid
exactly once per experiment: if you check the dashboard daily and stop on the
first day p < 0.05, the false-positive rate balloons from 5% to 20-30%+.

mSPRT (Robbins 1970, Howard et al. 2021) produces *always-valid* p-values and
confidence intervals: under the null hypothesis, the cumulative likelihood
ratio is a martingale, so peeking does not inflate Type-I error.

Reference: Howard, Ramdas, McAuliffe, Sekhon. "Time-uniform, nonparametric,
nonasymptotic confidence sequences" (Annals of Statistics, 2021).

This module implements the Gaussian-mixture variant — appropriate for
continuous metrics (means, conversion rates after normal approximation) and
for binary outcomes where np > 10.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass
class MsprtResult:
    """One observation along an mSPRT sequence."""

    n: int
    mean_diff: float
    """Treatment mean minus control mean (or analogous effect statistic)."""

    pooled_var: float
    """Pooled variance estimate for the difference."""

    log_lr: float
    """Log-likelihood ratio under the mixture prior. Larger → more evidence."""

    p_value: float
    """Always-valid p-value in (0, 1]. Safe to inspect at any n."""

    ci_low: float
    ci_high: float

    def reject_null(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def _mixture_log_lr(diff: float, var: float, n: int, tau2: float) -> float:
    """Log-likelihood ratio under a N(0, tau2) mixing prior over the effect.

    Standard derivation: for X-bar ~ N(theta, var/n) and theta ~ N(0, tau2),
    the marginal under H1 is N(0, var/n + tau2). Under H0 (theta=0) the
    likelihood is N(0, var/n). The log ratio is:

      log L1/L0 = 0.5 * [log(var/n) - log(var/n + tau2)]
                + 0.5 * (X-bar)^2 * (1/(var/n) - 1/(var/n + tau2))
    """
    if n <= 0 or var <= 0:
        return 0.0
    se2 = var / n  # variance of the sample mean diff
    inv = 1.0 / se2 - 1.0 / (se2 + tau2)
    return 0.5 * (math.log(se2) - math.log(se2 + tau2)) + 0.5 * diff * diff * inv


def msprt(
    n: int,
    mean_diff: float,
    pooled_var: float,
    *,
    tau2: float = 1.0,
    alpha: float = 0.05,
) -> MsprtResult:
    """Single-snapshot mSPRT computation.

    Args:
        n: Total sample size per arm (assumes balanced; for unbalanced pass
           the harmonic mean).
        mean_diff: Observed treatment - control mean difference.
        pooled_var: Pooled estimate of variance for the difference statistic.
        tau2: Variance of the Gaussian mixing prior over the effect size.
              Larger tau2 = expect bigger effects = more aggressive early stopping.
              Default 1.0 is a reasonable wide prior on standardized effects.
        alpha: Nominal Type-I error rate (used for CI width and reject decision).

    Returns:
        MsprtResult with always-valid p-value and confidence interval.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if pooled_var <= 0:
        raise ValueError("pooled_var must be positive")

    log_lr = _mixture_log_lr(mean_diff, pooled_var, n, tau2)
    # Always-valid p-value bound (Ville's inequality): P(LR > 1/alpha) <= alpha
    # implies always-valid p_value = min(1, 1 / LR).
    p_value = min(1.0, math.exp(-log_lr)) if log_lr > 0 else 1.0

    # Always-valid CI for the mean difference. Width derived from the same
    # mixture prior; see Howard et al. eq. (8).
    se2 = pooled_var / n
    if tau2 <= 0:
        half_width = 0.0
    else:
        radius_sq = (
            (se2 + tau2)
            * (
                math.log(1.0 / (alpha * alpha)) / 2.0
                + math.log((se2 + tau2) / se2) / 2.0
            )
        ) * 2.0 / n  # converted from total-sum to per-mean form
        radius_sq = max(0.0, radius_sq)
        half_width = math.sqrt(radius_sq)
    return MsprtResult(
        n=n,
        mean_diff=mean_diff,
        pooled_var=pooled_var,
        log_lr=log_lr,
        p_value=p_value,
        ci_low=mean_diff - half_width,
        ci_high=mean_diff + half_width,
    )


class MsprtStream:
    """Streaming mSPRT for an experiment receiving observations over time.

    Updates running means and variances using Welford's algorithm so any
    snapshot at any sample size n produces an always-valid p-value.

    Example:
        s = MsprtStream(tau2=0.5)
        for control_obs, treatment_obs in batches:
            for x in control_obs:
                s.observe_control(x)
            for x in treatment_obs:
                s.observe_treatment(x)
            result = s.snapshot()
            if result.reject_null(alpha=0.05):
                break  # peeking is safe — FPR is still <= alpha
    """

    def __init__(self, *, tau2: float = 1.0):
        self.tau2 = tau2
        self._n_c = 0
        self._m_c = 0.0
        self._s_c = 0.0
        self._n_t = 0
        self._m_t = 0.0
        self._s_t = 0.0

    def _push(self, x: float, n: int, m: float, s: float) -> tuple[int, float, float]:
        n += 1
        old_m = m
        m += (x - old_m) / n
        s += (x - old_m) * (x - m)
        return n, m, s

    def observe_control(self, x: float) -> None:
        self._n_c, self._m_c, self._s_c = self._push(x, self._n_c, self._m_c, self._s_c)

    def observe_treatment(self, x: float) -> None:
        self._n_t, self._m_t, self._s_t = self._push(x, self._n_t, self._m_t, self._s_t)

    def observe_many(self, control: Iterable[float], treatment: Iterable[float]) -> None:
        for x in control:
            self.observe_control(x)
        for x in treatment:
            self.observe_treatment(x)

    @property
    def n_control(self) -> int:
        return self._n_c

    @property
    def n_treatment(self) -> int:
        return self._n_t

    def snapshot(self, *, alpha: float = 0.05) -> MsprtResult | None:
        """Returns None if too few observations to compute (n<2 in either arm)."""
        if self._n_c < 2 or self._n_t < 2:
            return None
        var_c = self._s_c / (self._n_c - 1)
        var_t = self._s_t / (self._n_t - 1)
        # Pooled variance for the difference of means.
        pooled = var_c / self._n_c + var_t / self._n_t
        # Convert to "variance per sample" for the msprt() interface.
        n_harm = 2 / (1 / self._n_c + 1 / self._n_t)
        pooled_per_sample = pooled * n_harm
        return msprt(
            n=int(round(n_harm)),
            mean_diff=self._m_t - self._m_c,
            pooled_var=pooled_per_sample,
            tau2=self.tau2,
            alpha=alpha,
        )
