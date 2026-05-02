import torch


def parse_line_scales(text: str) -> list[float]:
    if not text:
        return []
    values = sorted({float(item) for item in text.split(",") if item.strip()})
    values = sorted({0.0, 1.0, *values})
    return values


def capture_rng(device: torch.device):
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.random.get_rng_state(), cuda_state


def restore_rng(state, device: torch.device) -> None:
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


@torch.no_grad()
def capture_params(params) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (p, p.detach().clone(memory_format=torch.preserve_format))
        for p in params
        if p.requires_grad
    ]


@torch.no_grad()
def finish_line_snapshot(
    before: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        (p, start, p.detach().clone(memory_format=torch.preserve_format))
        for p, start in before
    ]


@torch.no_grad()
def apply_line_scale(
    snapshot: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    scale: float,
) -> None:
    for p, start, end in snapshot:
        p.copy_(start).add_(end - start, alpha=scale)


def line_probe_text(
    step: int,
    loss_before: float,
    loss_after: float,
    stats: dict[str, dict],
    eps: float = 1e-12,
) -> str:
    pred = sum(
        values.get("lr_raw_descent", values.get("lr_descent", 0.0))
        for values in stats.values()
    )
    if pred <= eps:
        return ""
    actual = loss_before - loss_after
    ratio = actual / pred
    quadratic = 2.0 * (1.0 - ratio)
    lr2_update_sq = sum(values.get("lr2_update_sq", 0.0) for values in stats.values())
    curvature = (
        2.0 * (loss_after - loss_before + pred) / lr2_update_sq
        if lr2_update_sq > eps
        else float("nan")
    )
    obj = line_object_stats_text(stats)
    return (
        f"line_probe step {step:5d} | "
        f"pred {pred:.3e} | actual {actual:.3e} | "
        f"ratio {ratio:.3f} | quad {quadratic:.3f} | curv {curvature:.3e}"
        f"{obj}"
    )


def line_object_stats_text(stats: dict[str, dict]) -> str:
    parts = []
    for name, values in stats.items():
        parts.append(
            f"{name}:cos={values['cos']:.3f},"
            f"u/p={values['update_param_rms']:.2e},"
            f"u/g={values['update_grad_rms']:.2e},"
            f"g/p={values['grad_param_rms']:.2e},"
            f"xg={values['param_grad_cos']:.3f},"
            f"xu={values['param_update_cos']:.3f},"
            f"ga/r={values['grad_abs_rms']:.3f},"
            f"ua/r={values['update_abs_rms']:.3f},"
            f"gk={values['grad_kurtosis']:.2e},"
            f"uk={values['update_kurtosis']:.2e}"
            f"{optional_object_text(values)}"
        )
    return " | obj_stats " + "; ".join(parts) if parts else ""


def optional_object_text(values: dict) -> str:
    parts = []
    fields = (
        ("v/g", "atom_grad_rms", ".2e"),
        ("v/p", "atom_param_rms", ".2e"),
        ("gv", "grad_atom_cos", ".3f"),
        ("xv", "param_atom_cos", ".3f"),
        ("uv", "update_atom_cos", ".3f"),
        ("vk", "atom_kurtosis", ".2e"),
        ("m/g", "mom_grad_rms", ".2e"),
        ("m/p", "mom_param_rms", ".2e"),
        ("gm", "grad_mom_cos", ".3f"),
        ("xm", "param_mom_cos", ".3f"),
        ("um", "update_mom_cos", ".3f"),
        ("mv", "mom_atom_cos", ".3f"),
        ("mk", "mom_kurtosis", ".2e"),
    )
    for label, key, fmt in fields:
        value = values.get(key)
        if value is not None:
            parts.append(f"{label}={value:{fmt}}")
    return "," + ",".join(parts) if parts else ""


def line_curve_text(step: int, losses: list[tuple[float, float]]) -> str:
    if len(losses) < 3:
        return ""

    xs = torch.tensor([scale for scale, _ in losses], dtype=torch.float64)
    ys = torch.tensor([loss for _, loss in losses], dtype=torch.float64)
    zero = int(torch.argmin(xs.abs()).item())
    y0 = ys[zero]
    mask = xs.abs() > 1e-12
    design = torch.stack([xs[mask], xs[mask].square()], dim=1)
    target = ys[mask] - y0
    if design.size(0) < design.size(1):
        return ""

    coeff = torch.linalg.lstsq(design, target).solution
    predicted = -float(coeff[0])
    curvature = float(2.0 * coeff[1])
    a_star = predicted / curvature if curvature > 1e-12 else float("nan")
    best_scale, best_loss = min(losses, key=lambda item: item[1])
    points = ",".join(f"{scale:g}:{loss:.4f}" for scale, loss in losses)
    return (
        f"line_curve step {step:5d} | "
        f"Pfit {predicted:.3e} | Cfit {curvature:.3e} | "
        f"a* {a_star:.3f} | best {best_scale:g}:{best_loss:.4f} | "
        f"losses {points}"
    )
