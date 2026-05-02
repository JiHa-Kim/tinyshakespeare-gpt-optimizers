import math
from dataclasses import dataclass

__all__ = [
    "InvariantAction",
    "RMSCorrection",
    "halving_factor",
    "invariant_eta_unit",
    "rms_additive_eta",
    "rms_eta_unit",
    "scheduled_invariant_action",
    "scheduled_ratio",
    "scheduled_shrink",
    "resolve_schedule",
    "schedule_at_step",
]


@dataclass(frozen=True)
class RMSCorrection:
    atom_sq: float = 1.0
    keep_fraction: float = 1.0
    momentum_factor: float = 1.0

    def validate(self) -> None:
        if self.atom_sq <= 0.0:
            raise ValueError(f"invalid RMS atom scale: {self.atom_sq}")
        if self.keep_fraction <= 0.0:
            raise ValueError(f"invalid RMS keep fraction: {self.keep_fraction}")
        if self.momentum_factor <= 0.0:
            raise ValueError(f"invalid RMS momentum factor: {self.momentum_factor}")


@dataclass(frozen=True, slots=True)
class InvariantAction:
    shrink: float
    eta_unit: float
    eta: float
    ratio: float


def halving_factor(delta_tau: float, half_life: float, name: str) -> float:
    if delta_tau <= 0.0:
        raise ValueError(f"invalid count increment: {delta_tau}")
    if half_life <= 0.0:
        raise ValueError(f"invalid {name}: {half_life}")
    if math.isinf(half_life):
        return 1.0
    return 2.0 ** (-delta_tau / half_life)


def resolve_schedule(
    max_steps: int, warmup_steps: int, decay_steps: int
) -> tuple[int, int, int]:
    if max_steps <= 0:
        raise ValueError(f"invalid max_steps: {max_steps}")
    warmup_steps = max(0, min(warmup_steps, max_steps))
    decay_steps = max(0, min(decay_steps, max_steps - warmup_steps))
    stable_steps = max_steps - warmup_steps - decay_steps
    return warmup_steps, stable_steps, decay_steps


def schedule_at_step(
    step: int,
    max_steps: int,
    peak: float,
    floor: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Return the per-update schedule value for a piecewise-constant update."""
    warmup_steps, stable_steps, decay_steps = resolve_schedule(
        max_steps, warmup_steps, decay_steps
    )

    if warmup_steps > 0 and step < warmup_steps:
        return peak * (step + 1) / warmup_steps

    decay_start = warmup_steps + stable_steps
    if decay_steps == 0 or step < decay_start:
        return peak
    if decay_steps == 1:
        return floor

    progress = (step - decay_start) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return peak + (floor - peak) * progress


def scheduled_ratio(
    scheduled_scale: float, peak_scale: float, name: str = "schedule"
) -> float:
    if not math.isfinite(scheduled_scale) or scheduled_scale < 0.0:
        raise ValueError(f"invalid {name} scheduled scale: {scheduled_scale}")
    if not math.isfinite(peak_scale) or peak_scale < 0.0:
        raise ValueError(f"invalid {name} peak scale: {peak_scale}")
    if peak_scale == 0.0:
        return 0.0
    ratio = scheduled_scale / peak_scale
    if not math.isfinite(ratio):
        raise ValueError(f"invalid {name} schedule ratio: {ratio}")
    return ratio


def scheduled_shrink(peak_shrink: float, ratio: float, name: str = "shrink") -> float:
    if not (0.0 < peak_shrink <= 1.0):
        raise ValueError(f"invalid {name} peak shrink: {peak_shrink}")
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError(f"invalid {name} schedule ratio: {ratio}")
    return peak_shrink**ratio


def invariant_eta_unit(rho: float, shrink: float) -> float:
    if rho <= 0.0:
        raise ValueError(f"invalid reference radius: {rho}")
    if not (0.0 < shrink <= 1.0):
        raise ValueError(f"invalid shrink: {shrink}")
    return (1.0 - shrink) * rho


def scheduled_invariant_action(
    rho: float,
    peak_shrink: float,
    peak_scale: float,
    scheduled_scale: float,
    name: str = "group",
) -> InvariantAction:
    ratio = scheduled_ratio(scheduled_scale, peak_scale, name)
    shrink = scheduled_shrink(peak_shrink, ratio, name)
    eta_unit = invariant_eta_unit(rho, shrink)
    return InvariantAction(
        shrink=shrink,
        eta_unit=eta_unit,
        eta=peak_scale * eta_unit,
        ratio=ratio,
    )


def rms_eta_unit(
    rho: float,
    shrink: float,
    correction: RMSCorrection,
    eps: float = 1e-12,
) -> float:
    if rho <= 0.0:
        raise ValueError(f"invalid RMS radius: {rho}")
    if not (0.0 < shrink < 1.0):
        raise ValueError(
            f"cannot derive additive eta from shrink={shrink}; "
            "use a finite shrink half-life"
        )
    correction.validate()
    variance = (
        correction.keep_fraction
        * max(1.0 - shrink * shrink, eps)
        / (correction.atom_sq * correction.momentum_factor)
    )
    return rho * math.sqrt(variance)


def rms_additive_eta(
    rho: float,
    shrink: float,
    step_scale: float,
    correction: RMSCorrection,
    eps: float = 1e-12,
) -> float:
    return step_scale * rms_eta_unit(rho, shrink, correction, eps)
