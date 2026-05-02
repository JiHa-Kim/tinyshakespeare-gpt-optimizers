from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch.optim import Optimizer

__all__ = [
    "StepStatSnapshot",
    "accumulate_step_stats",
    "capture_step_stats",
    "consume_step_stats",
]


@dataclass
class StepStatSnapshot:
    group: str
    lr: float
    eta: float | None
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]]


def capture_step_stats(optimizer: Optimizer) -> list[StepStatSnapshot]:
    snapshots = []
    for group in optimizer.param_groups:
        items = []
        lr = float(group["lr"])
        eta = group.get("eta")
        if group.get("cwd", False) or group.get("phi", 0.0):
            eta = None
        elif group.get("shrink") is not None:
            eta = 1.0 - float(group["shrink"])
        elif eta is None and group.get("theta2") is None:
            eta = lr * float(group.get("weight_decay", 0.0))
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            items.append(
                (
                    p,
                    p.detach().clone(memory_format=torch.preserve_format),
                    g.detach().clone(memory_format=torch.preserve_format),
                    optimizer.state[p],
                )
            )
        if items:
            snapshots.append(
                StepStatSnapshot(group.get("name", "group"), lr, eta, items)
            )
    return snapshots


def _stat_add(stats: dict, name: str, value: torch.Tensor) -> None:
    value = value.detach()
    current = stats.get(name)
    stats[name] = value if current is None else current + value


@torch.no_grad()
def accumulate_step_stats(
    accum: dict[str, dict], snapshots: Iterable[StepStatSnapshot]
) -> None:
    for snapshot in snapshots:
        stats = accum.setdefault(
            snapshot.group, {"steps": 0, "params": 0, "numel": 0}
        )
        stats["steps"] += 1
        stats["params"] += len(snapshot.items)

        for p, p_before, grad, state in snapshot.items:
            grad = grad.float()
            delta = p.detach().float() - p_before.float()
            p32 = p_before.float()
            momentum = state.get("m")
            mom = momentum.detach().float() if momentum is not None else None
            atom = None
            if snapshot.eta is not None and snapshot.lr > 0.0:
                atom = delta.add(p32, alpha=float(snapshot.eta)) / snapshot.lr

            update_sq = delta.square().sum()
            grad_sq = grad.square().sum()
            param_sq = p32.square().sum()
            descent = -(grad * delta).sum()
            param_grad = (p32 * grad).sum()
            param_update = (p32 * delta).sum()
            stats["numel"] += p.numel()
            _stat_add(stats, "grad_sq", grad_sq)
            _stat_add(stats, "update_sq", update_sq)
            _stat_add(stats, "param_sq", param_sq)
            _stat_add(stats, "descent", descent)
            _stat_add(stats, "param_grad", param_grad)
            _stat_add(stats, "param_update", param_update)
            _stat_add(stats, "grad_abs", grad.abs().sum())
            _stat_add(stats, "update_abs", delta.abs().sum())
            _stat_add(stats, "param_abs", p32.abs().sum())
            _stat_add(stats, "grad_fourth", grad.square().square().sum())
            _stat_add(stats, "update_fourth", delta.square().square().sum())
            _stat_add(stats, "param_fourth", p32.square().square().sum())
            if atom is not None:
                atom_sq = atom.square().sum()
                _stat_add(stats, "atom_sq", atom_sq)
                _stat_add(stats, "atom_abs", atom.abs().sum())
                _stat_add(stats, "atom_fourth", atom.square().square().sum())
                _stat_add(stats, "grad_atom", -(grad * atom).sum())
                _stat_add(stats, "param_atom", (p32 * atom).sum())
                _stat_add(stats, "update_atom", (delta * atom).sum())
            if mom is not None:
                mom_sq = mom.square().sum()
                _stat_add(stats, "mom_sq", mom_sq)
                _stat_add(stats, "mom_abs", mom.abs().sum())
                _stat_add(stats, "mom_fourth", mom.square().square().sum())
                _stat_add(stats, "grad_mom", (grad * mom).sum())
                _stat_add(stats, "param_mom", (p32 * mom).sum())
                _stat_add(stats, "update_mom", (delta * mom).sum())
                if atom is not None:
                    _stat_add(stats, "mom_atom", -(mom * atom).sum())
            _stat_add(stats, "lr_descent", descent)
            _stat_add(stats, "raw_descent", descent)
            _stat_add(stats, "lr_raw_descent", descent)
            _stat_add(stats, "lr2_update_sq", update_sq)


def consume_step_stats(
    accum: dict[str, dict], eps: float = 1e-12
) -> dict[str, dict[str, float]]:
    out = {}
    for name, stats in list(accum.items()):
        grad_sq = float(stats.get("grad_sq", 0.0))
        update_sq = float(stats.get("update_sq", 0.0))
        param_sq = float(stats.get("param_sq", 0.0))
        descent = float(stats.get("descent", 0.0))
        param_grad = float(stats.get("param_grad", 0.0))
        param_update = float(stats.get("param_update", 0.0))
        lr_descent = float(stats.get("lr_descent", descent))
        raw_descent = float(stats.get("raw_descent", descent))
        lr_raw_descent = float(stats.get("lr_raw_descent", descent))
        lr2_update_sq = float(stats.get("lr2_update_sq", update_sq))
        numel = max(int(stats["numel"]), 1)
        grad_rms = (grad_sq / numel) ** 0.5
        update_rms = (update_sq / numel) ** 0.5
        param_rms = (param_sq / numel) ** 0.5
        grad_abs = float(stats.get("grad_abs", 0.0)) / numel
        update_abs = float(stats.get("update_abs", 0.0)) / numel
        param_abs = float(stats.get("param_abs", 0.0)) / numel
        grad_kurt = numel * float(stats.get("grad_fourth", 0.0)) / (grad_sq**2 + eps)
        update_kurt = (
            numel * float(stats.get("update_fourth", 0.0)) / (update_sq**2 + eps)
        )
        param_kurt = (
            numel * float(stats.get("param_fourth", 0.0)) / (param_sq**2 + eps)
        )
        out[name] = {
            "steps": int(stats["steps"]),
            "params": int(stats["params"]),
            "grad_rms": grad_rms,
            "raw_grad_rms": grad_rms,
            "update_rms": update_rms,
            "param_rms": param_rms,
            "grad_abs_mean": grad_abs,
            "update_abs_mean": update_abs,
            "param_abs_mean": param_abs,
            "grad_kurtosis": grad_kurt,
            "update_kurtosis": update_kurt,
            "param_kurtosis": param_kurt,
            "descent": descent,
            "lr_descent": lr_descent,
            "raw_descent": raw_descent,
            "lr_raw_descent": lr_raw_descent,
            "lr2_update_sq": lr2_update_sq,
            "cos": descent / ((grad_sq * update_sq) ** 0.5 + eps),
            "raw_cos": raw_descent / ((grad_sq * update_sq) ** 0.5 + eps),
            "param_grad_cos": param_grad / ((param_sq * grad_sq) ** 0.5 + eps),
            "param_update_cos": param_update / ((param_sq * update_sq) ** 0.5 + eps),
            "update_grad_rms": update_rms / (grad_rms + eps),
            "grad_param_rms": grad_rms / (param_rms + eps),
            "update_param_rms": update_rms / (param_rms + eps),
            "grad_abs_rms": grad_abs / (grad_rms + eps),
            "update_abs_rms": update_abs / (update_rms + eps),
            "param_abs_rms": param_abs / (param_rms + eps),
        }
        add_optional_object_stats(out[name], stats, numel, eps)
    accum.clear()
    return out


def add_optional_object_stats(
    out: dict[str, float | int], stats: dict, numel: int, eps: float
) -> None:
    grad_sq = float(stats.get("grad_sq", 0.0))
    update_sq = float(stats.get("update_sq", 0.0))
    param_sq = float(stats.get("param_sq", 0.0))
    grad_rms = (grad_sq / numel) ** 0.5
    update_rms = (update_sq / numel) ** 0.5
    param_rms = (param_sq / numel) ** 0.5

    atom_sq = float(stats.get("atom_sq", 0.0))
    if atom_sq > 0.0:
        atom_rms = (atom_sq / numel) ** 0.5
        atom_abs = float(stats.get("atom_abs", 0.0)) / numel
        out.update(
            {
                "atom_rms": atom_rms,
                "atom_grad_rms": atom_rms / (grad_rms + eps),
                "atom_param_rms": atom_rms / (param_rms + eps),
                "atom_update_rms": atom_rms / (update_rms + eps),
                "atom_abs_rms": atom_abs / (atom_rms + eps),
                "atom_kurtosis": numel
                * float(stats.get("atom_fourth", 0.0))
                / (atom_sq**2 + eps),
                "grad_atom_cos": float(stats.get("grad_atom", 0.0))
                / ((grad_sq * atom_sq) ** 0.5 + eps),
                "param_atom_cos": float(stats.get("param_atom", 0.0))
                / ((param_sq * atom_sq) ** 0.5 + eps),
                "update_atom_cos": float(stats.get("update_atom", 0.0))
                / ((update_sq * atom_sq) ** 0.5 + eps),
            }
        )

    mom_sq = float(stats.get("mom_sq", 0.0))
    if mom_sq > 0.0:
        mom_rms = (mom_sq / numel) ** 0.5
        mom_abs = float(stats.get("mom_abs", 0.0)) / numel
        out.update(
            {
                "mom_rms": mom_rms,
                "mom_grad_rms": mom_rms / (grad_rms + eps),
                "mom_param_rms": mom_rms / (param_rms + eps),
                "mom_update_rms": mom_rms / (update_rms + eps),
                "mom_abs_rms": mom_abs / (mom_rms + eps),
                "mom_kurtosis": numel
                * float(stats.get("mom_fourth", 0.0))
                / (mom_sq**2 + eps),
                "grad_mom_cos": float(stats.get("grad_mom", 0.0))
                / ((grad_sq * mom_sq) ** 0.5 + eps),
                "param_mom_cos": float(stats.get("param_mom", 0.0))
                / ((param_sq * mom_sq) ** 0.5 + eps),
                "update_mom_cos": float(stats.get("update_mom", 0.0))
                / ((update_sq * mom_sq) ** 0.5 + eps),
            }
        )
        atom_sq = float(stats.get("atom_sq", 0.0))
        if atom_sq > 0.0:
            out["mom_atom_cos"] = float(stats.get("mom_atom", 0.0)) / (
                (mom_sq * atom_sq) ** 0.5 + eps
            )
