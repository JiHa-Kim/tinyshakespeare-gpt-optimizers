import math
from dataclasses import dataclass

__all__ = [
    "RMSCorrection",
    "additive_eta",
    "eta_unit",
    "halving_factor",
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
    if peak > 0.0 and floor > 0.0:
        return 1.0 / ((1.0 - progress) / peak + progress / floor)
    return peak + (floor - peak) * progress


def eta_unit(
    rho: float,
    shrink: float,
    correction: RMSCorrection,
    eps: float = 1e-12,
) -> float:
    if rho <= 0.0:
        raise ValueError(f"invalid steady radius: {rho}")
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


def additive_eta(
    rho: float,
    shrink: float,
    step_scale: float,
    correction: RMSCorrection,
    eps: float = 1e-12,
) -> float:
    return step_scale * eta_unit(rho, shrink, correction, eps)
