"""Power analysis: sample size for a target effect, MDE for a fixed sample.

These are the calculators stakeholders use before launching an experiment.
Built around the standard two-sample z-test approximation, which is accurate
for n > 30 per arm in practice.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from scipy import stats


@dataclass
class PowerResult:
    per_arm_n: int
    total_n: int
    detectable_effect: float  # absolute units (same as mean_diff input)
    relative_effect: float    # detectable_effect / baseline_mean
    alpha: float
    power: float


def sample_size_for_means(
    baseline_mean: float,
    baseline_std: float,
    effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> PowerResult:
    """Return per-arm sample size to detect `effect` (absolute) with given alpha/power."""
    if baseline_std <= 0:
        raise ValueError("baseline_std must be positive")
    if effect == 0:
        raise ValueError("effect must be nonzero")
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2) * (2.0 * baseline_std ** 2) / (effect ** 2)
    per_arm = int(-(-n // 1))  # ceil
    return PowerResult(
        per_arm_n=per_arm,
        total_n=2 * per_arm,
        detectable_effect=effect,
        relative_effect=effect / baseline_mean if baseline_mean != 0 else float("inf"),
        alpha=alpha,
        power=power,
    )


def sample_size_for_proportions(
    baseline_rate: float,
    absolute_lift: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> PowerResult:
    """Sample size to detect an absolute conversion-rate lift.

    Example: baseline 5%, want to detect a 0.5pp lift to 5.5%.
        sample_size_for_proportions(0.05, 0.005) → ~31K per arm.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if absolute_lift == 0:
        raise ValueError("absolute_lift must be nonzero")
    p1 = baseline_rate
    p2 = baseline_rate + absolute_lift
    if not 0 < p2 < 1:
        raise ValueError("baseline_rate + absolute_lift must be in (0, 1)")
    p_bar = (p1 + p2) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    numerator = (
        z_alpha * sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n = numerator / (absolute_lift ** 2)
    per_arm = int(-(-n // 1))
    return PowerResult(
        per_arm_n=per_arm,
        total_n=2 * per_arm,
        detectable_effect=absolute_lift,
        relative_effect=absolute_lift / baseline_rate,
        alpha=alpha,
        power=power,
    )


def mde_for_means(
    baseline_std: float,
    per_arm_n: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> float:
    """Minimum detectable effect (absolute) given a fixed per-arm sample size."""
    if per_arm_n <= 0:
        raise ValueError("per_arm_n must be positive")
    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    return (z_alpha + z_beta) * baseline_std * sqrt(2.0 / per_arm_n)
